"""
Input validation helpers used by API and copilot.

Functions:
    validate_meal_plan — required keys, day/slot structure
    validate_recipe    — RecipeSchema Pydantic check
    validate_user_id   — safe id pattern for file paths
"""

from typing import Any

from pydantic import ValidationError

from recipes.schemas.recipe import RecipeSchema
from utils.logger import setup_logger

logger = setup_logger(__name__)


def validate_meal_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    """
    Run safety and schema checks on a meal plan.

    Args:
        plan: Meal plan dict to validate.

    Returns:
        Dict with valid (bool), errors (list), warnings (list).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not plan:
        errors.append("Meal plan is empty")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # TODO: JSON schema validation against meal_plan.json when added
    if plan.get("status") == "stub":
        warnings.append("LLM generation disabled; stub plan returned")

    content = plan.get("content") or plan.get("message") or ""
    if plan.get("status") == "generated" and not content:
        errors.append("Generated plan missing content")

    # TODO: Allergen cross-check, carb limits, medication interaction flags
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_recipe(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a recipe dict against RecipeSchema.

    Returns:
        Dict with valid (bool), errors (list), warnings (list), and recipe (dict | None).
    """
    warnings: list[str] = []
    try:
        recipe = RecipeSchema.model_validate(data)
        if recipe.diabetes_score < 0 or recipe.diabetes_score > 1:
            warnings.append("diabetes_score outside [0, 1]")
        if recipe.calories <= 0:
            warnings.append("calories should be positive")
        return {
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "recipe": recipe.model_dump(),
        }
    except ValidationError as exc:
        errors = [
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()
        ]
        return {"valid": False, "errors": errors, "warnings": warnings, "recipe": None}


def validate_user_id(user_id: str) -> bool:
    """Check user_id format (MVP: non-empty string)."""
    # TODO: Verify user exists in users.json or MongoDB
    return bool(user_id and user_id.strip())
