"""
Deterministic diabetes_score from macros — used in enrichment and ranking.

Functions:
    compute_diabetes_score      — sigmoid over carbs/calories and sugar
    apply_diabetes_score_to_recipe — attach score field to a recipe dict
    score_meal_plan             — average score across planned slots
    explain_score               — short label for UI (e.g. "Good fit")
"""

import math
from typing import Any

from utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_diabetes_score(carbs: float, calories: float, sugar: float) -> float:
    """
    Compute normalized diabetes risk score for a recipe.

    Deterministic sigmoid layer:
        raw_score = 0.7 * (carbs / calories) + 0.3 * sugar
        score = sigmoid(raw_score)

    Args:
        carbs: Carbohydrate grams.
        calories: Total calories (kcal).
        sugar: Sugar grams.

    Returns:
        Score in (0, 1), rounded to 3 decimals.
    """
    carb_ratio = carbs / max(calories, 1)
    raw_score = 0.7 * carb_ratio + 0.3 * sugar
    return round(1 / (1 + math.exp(-raw_score)), 3)


def apply_diabetes_score_to_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    """
    Attach deterministic diabetes_score to a recipe dict in place.

    Args:
        recipe: Dict with carbs, calories, sugar fields.

    Returns:
        Same dict with diabetes_score set.
    """
    recipe["diabetes_score"] = compute_diabetes_score(
        float(recipe.get("carbs", 0)),
        float(recipe.get("calories", 0)),
        float(recipe.get("sugar", 0)),
    )
    return recipe


def score_meal_plan(
    meal_plan: dict[str, Any],
    biomarkers: dict[str, Any] | None = None,
) -> float:
    """
    Compute a 0–100 quality score for a generated meal plan.

    Args:
        meal_plan: Plan dict (structured or LLM text wrapper).
        biomarkers: User biomarkers for personalization.

    Returns:
        Composite score.
    """
    biomarkers = biomarkers or {}
    base = 70.0
    if meal_plan.get("status") == "generated":
        base += 10.0
    if biomarkers.get("glucose_flag") == "high":
        base += 5.0 if "low" in str(meal_plan).lower() else -5.0
    logger.debug("Meal plan score: %.1f", base)
    return round(min(100.0, max(0.0, base)), 1)


def explain_score(score: float) -> str:
    """Return a short explanation string for UI display."""
    if score >= 80:
        return "Strong alignment with your health profile."
    if score >= 60:
        return "Moderate alignment; consider clinician review for major changes."
    return "Plan may need adjustment; share feedback to refine."
