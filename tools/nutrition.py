"""
Recipe nutrition enrichment — OpenAI estimates macros; veg/meal type is deterministic.

Classes:
    NutritionEstimate — calories, carbs, protein, fat, sugar
    NutritionEstimator  — OpenAI client with rate-limit retries

Functions:
    classify_veg_deterministic      — ingredient keyword → veg/non-veg
    classify_meal_type_deterministic — recipe name → breakfast/lunch/dinner/snack
    normalize_meal_type              — canonical meal type string
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack", "other")

_NON_VEG_PATTERNS = (
    r"\bchicken\b",
    r"\bfish\b",
    r"\bprawn\b",
    r"\bshrimp\b",
    r"\begg\b",
    r"\beggs\b",
    r"\bmutton\b",
    r"\blamb\b",
    r"\bbeef\b",
    r"\bpork\b",
    r"\bbacon\b",
    r"\bcrab\b",
    r"\blobster\b",
    r"\bmeat\b",
    r"\bliver\b",
    r"\bseafood\b",
    r"\btuna\b",
    r"\bsalmon\b",
    r"\banchovy\b",
)

_MEAL_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "breakfast": ("breakfast", "brunch", "oat", "cereal", "pancake", "upma", "poha", "idli", "dosa"),
    "lunch": ("lunch", "rice", "biryani", "pulao", "thali", "curry"),
    "dinner": ("dinner", "supper", "stew", "roast"),
    "snack": ("snack", "appetizer", "starter", "chaat", "pakora", "samosa", "dip"),
}


class NutritionEstimate(BaseModel):
    """Structured LLM response for per-serving nutrition."""

    calories: float = Field(ge=0)
    protein: float = Field(ge=0)
    carbs: float = Field(ge=0)
    fat: float = Field(ge=0)
    fiber: float = Field(ge=0)
    sugar: float = Field(ge=0)
    summary: str = Field(min_length=10, max_length=500)
    meal_type: str = Field(description="One of: breakfast, lunch, dinner, snack, other")


class NutritionEstimator:
    """
    OpenAI client wrapper with retries and rate-limit handling.

    Deterministic classification (veg, fallback meal_type) runs outside this class.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.openai_model
        self._max_retries = settings.enrich_max_retries
        self._base_delay = settings.enrich_retry_base_delay
        self._min_interval = settings.enrich_min_request_interval
        self._last_request_at: float = 0.0
        self._client: OpenAI | None = None
        if settings.openai_api_key:
            self._client = OpenAI(api_key=settings.openai_api_key)

    def estimate(
        self,
        name: str,
        ingredients: list[str],
        steps: list[str],
        *,
        cuisine: str | None = None,
        servings: int = 1,
    ) -> NutritionEstimate | None:
        """
        Estimate nutrition and metadata via OpenAI structured JSON output.

        Returns:
            NutritionEstimate or None if API unavailable / all retries failed.
        """
        if not self._client:
            logger.error("OPENAI_API_KEY not set; cannot estimate nutrition")
            return None

        prompt = _build_estimation_prompt(name, ingredients, steps, cuisine=cuisine, servings=servings)
        schema = NutritionEstimate.model_json_schema()

        for attempt in range(1, self._max_retries + 1):
            self._throttle()
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a clinical nutrition assistant. "
                                "Estimate per-serving macronutrients for diabetes meal planning. "
                                "Return realistic numbers as JSON matching the schema."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "nutrition_estimate",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                    temperature=0.2,
                )
                raw = response.choices[0].message.content or "{}"
                data = json.loads(raw)
                estimate = NutritionEstimate.model_validate(data)
                estimate.meal_type = normalize_meal_type(estimate.meal_type)
                return estimate

            except RateLimitError as exc:
                delay = _retry_delay(exc, attempt, self._base_delay)
                logger.warning(
                    "Rate limit hit (attempt %d/%d); sleeping %.1fs",
                    attempt,
                    self._max_retries,
                    delay,
                )
                time.sleep(delay)
            except (APITimeoutError, APIError) as exc:
                delay = self._base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI API error (attempt %d/%d): %s; sleeping %.1fs",
                    attempt,
                    self._max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Invalid nutrition JSON (attempt %d): %s", attempt, exc)
                time.sleep(self._base_delay)
            except Exception as exc:
                if "json_schema" in str(exc).lower() and attempt == 1:
                    logger.warning("json_schema unsupported; retrying with json_object mode")
                    estimate = self._estimate_json_object(prompt)
                    if estimate:
                        return estimate
                logger.exception("Unexpected nutrition estimation error: %s", exc)
                break

        return None

    def _estimate_json_object(self, prompt: str) -> NutritionEstimate | None:
        """Fallback when strict json_schema is unavailable."""
        if not self._client:
            return None
        self._throttle()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "Respond with JSON only matching NutritionEstimate fields.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        estimate = NutritionEstimate.model_validate(data)
        estimate.meal_type = normalize_meal_type(estimate.meal_type)
        return estimate

    def _throttle(self) -> None:
        """Enforce minimum interval between API calls."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()


def classify_veg_deterministic(ingredients: list[str], name: str = "") -> bool:
    """
    Classify recipe as vegetarian (True) or non-veg (False) from ingredients.

    Args:
        ingredients: Ingredient strings.
        name: Optional recipe title for extra signal.

    Returns:
        True if vegetarian, False if non-vegetarian.
    """
    blob = " ".join(ingredients + [name]).lower()
    for pattern in _NON_VEG_PATTERNS:
        if re.search(pattern, blob):
            return False
    return True


def classify_meal_type_deterministic(name: str) -> str:
    """
    Classify meal type from recipe title keywords.

    Args:
        name: Recipe name.

    Returns:
        One of breakfast, lunch, dinner, snack, other.
    """
    lower = name.lower()
    for meal_type, keywords in _MEAL_TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return meal_type
    return "other"


def normalize_meal_type(value: str) -> str:
    """Normalize meal type to allowed enum values."""
    cleaned = value.strip().lower()
    if cleaned in MEAL_TYPES:
        return cleaned
    for meal_type in MEAL_TYPES:
        if meal_type in cleaned:
            return meal_type
    return "other"


def _build_estimation_prompt(
    name: str,
    ingredients: list[str],
    steps: list[str],
    *,
    cuisine: str | None,
    servings: int,
) -> str:
    ing = "\n".join(f"- {i}" for i in ingredients[:40])
    st = "\n".join(f"- {s}" for s in steps[:15])
    return (
        f"Recipe: {name}\n"
        f"Cuisine: {cuisine or 'unknown'}\n"
        f"Servings: {servings}\n\n"
        f"Ingredients:\n{ing}\n\n"
        f"Steps (excerpt):\n{st}\n\n"
        "Estimate per-serving: calories (kcal), protein/carbs/fat/fiber/sugar (grams). "
        "Write a concise 1-2 sentence summary for someone managing diabetes. "
        "Classify meal_type as breakfast, lunch, dinner, snack, or other."
    )


def _retry_delay(exc: RateLimitError, attempt: int, base_delay: float) -> float:
    """Parse Retry-After header or use exponential backoff."""
    retry_after: float | None = None
    if exc.response and exc.response.headers:
        header = exc.response.headers.get("retry-after")
        if header:
            try:
                retry_after = float(header)
            except ValueError:
                pass
    return retry_after if retry_after is not None else base_delay * (2 ** (attempt - 1))
