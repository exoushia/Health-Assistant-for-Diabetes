"""
RecipeSchema — contract for enriched recipes (macros, diabetes_score, ingredients).

Classes:
    RecipeSchema — validated recipe document

Functions:
    parse_recipe — dict → RecipeSchema with ValidationError on bad data
"""

from typing import List

from pydantic import BaseModel


class RecipeSchema(BaseModel):
    """Validated recipe document stored under recipes/processed/."""

    recipe_id: str
    name: str
    meal_type: str
    veg: bool
    calories: float
    protein: float
    fat: float
    carbs: float
    fiber: float
    sugar: float
    diabetes_score: float
    summary: str
    ingredients: List[str]
    steps: List[str]
    source_url: str


def parse_recipe(data: dict) -> RecipeSchema:
    """
    Parse and validate a raw dict as RecipeSchema.

    Raises:
        pydantic.ValidationError: If required fields are missing or invalid.
    """
    return RecipeSchema.model_validate(data)
