"""
Recipe retrieval from enriched_recipes.json — keyword filter + rank (no vectors yet).

Functions:
    retrieve_recipes       — query, dietary prefs, blocked ingredients → candidate list
    _load_enriched_collection — read JSON corpus from disk
    _filter_by_preferences    — veg/non-veg, meal type, ingredient blocks
    _keyword_rank             — simple term overlap scoring
"""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from recipes.schemas.recipe import RecipeSchema
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


def retrieve_recipes(
    query: str,
    user_preferences: dict[str, Any] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Retrieve candidate recipes matching query and preferences.

    Args:
        query: Natural language or keyword query.
        user_preferences: Dietary filters (cuisine, allergies, etc.).
        top_k: Maximum recipes to return.

    Returns:
        List of recipe dicts with id, title, tags, and scores.
    """
    settings = get_settings()
    processed_dir = settings.recipes_processed_dir
    recipes: list[dict[str, Any]] = []

    enriched_path = processed_dir / "enriched_recipes.json"
    if enriched_path.is_file():
        recipes.extend(_load_enriched_collection(enriched_path))

    # TODO: Replace keyword scan with embeddings + vector DB (MongoDB Atlas / local)
    if processed_dir.exists():
        for path in processed_dir.glob("*.json"):
            if path.name == "enriched_recipes.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "recipes" in data:
                    continue
                recipe = RecipeSchema.model_validate(data)
                recipes.append(recipe.model_dump())
            except (json.JSONDecodeError, OSError, ValidationError) as exc:
                logger.warning("Skipping recipe file %s: %s", path, exc)

    prefs = user_preferences or {}
    filtered = _filter_by_preferences(recipes, prefs)
    return _keyword_rank(filtered or recipes, query)[:top_k]


def _load_enriched_collection(path: Path) -> list[dict[str, Any]]:
    """Load recipes from enriched_recipes.json with schema validation."""
    loaded: list[dict[str, Any]] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("recipes", []) if isinstance(data, dict) else data
        for item in items:
            recipe = RecipeSchema.model_validate(item)
            loaded.append(recipe.model_dump())
    except (json.JSONDecodeError, OSError, ValidationError) as exc:
        logger.warning("Skipping enriched collection %s: %s", path, exc)
    return loaded


def _filter_by_preferences(
    recipes: list[dict[str, Any]],
    preferences: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply coarse preference filters (MVP)."""
    # TODO: Allergy intersection, glycemic index caps, cuisine matching
    allergies = set(preferences.get("allergies", []))
    if not allergies:
        return recipes
    return [
        r
        for r in recipes
        if not allergies.intersection(set(r.get("allergens", [])))
    ]


def _keyword_rank(recipes: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Simple keyword relevance scoring for MVP."""
    if not query.strip():
        return recipes
    q = query.lower()
    scored = []
    for recipe in recipes:
        text = " ".join(
            [
                str(recipe.get("name", "")),
                str(recipe.get("summary", "")),
                str(recipe.get("meal_type", "")),
                " ".join(recipe.get("ingredients", [])),
            ]
        ).lower()
        score = sum(1 for word in q.split() if word in text)
        scored.append({**recipe, "retrieval_score": score})
    scored.sort(key=lambda r: r.get("retrieval_score", 0), reverse=True)
    return scored
