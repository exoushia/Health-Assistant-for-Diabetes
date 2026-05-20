"""
Deterministic recipe ranking by diabetes_score and biomarker context.

Functions:
    rank_recipes       — sort candidates best-first for the user's risk profile
    _composite_score   — blend recipe diabetes_score with HbA1c-based weighting
"""

from typing import Any

from utils.logger import setup_logger

logger = setup_logger(__name__)


def rank_recipes(
    recipes: list[dict[str, Any]],
    biomarkers: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Re-rank recipes using biomarker-aware heuristics.

    Args:
        recipes: Candidates from retrieval.
        biomarkers: Parsed user biomarkers.

    Returns:
        Recipes sorted by composite rank_score (descending).
    """
    biomarkers = biomarkers or {}
    ranked = []
    for recipe in recipes:
        score = _composite_score(recipe, biomarkers)
        ranked.append({**recipe, "rank_score": score})
    ranked.sort(key=lambda r: r.get("rank_score", 0), reverse=True)
    logger.debug("Ranked %d recipes", len(ranked))
    return ranked


def _composite_score(recipe: dict[str, Any], biomarkers: dict[str, Any]) -> float:
    """Compute MVP composite score from retrieval score and glycemic hints."""
    # TODO: Learned ranker, carb budgets from biomarkers, meal timing
    base = float(recipe.get("retrieval_score", 0))
    diabetes_score = float(recipe.get("diabetes_score", 0))
    penalty = 0.0
    risk = biomarkers.get("diabetes_risk")
    if risk in ("high", "medium") and diabetes_score < 0.6:
        penalty = 2.0
    elif biomarkers.get("glucose_flag") == "high" and diabetes_score < 0.6:
        penalty = 2.0
    score_bonus = diabetes_score * 2.0
    return base - penalty + score_bonus
