"""
Legacy/simple feedback helpers — main flow uses orchestrator/agent.py instead.

Functions:
    apply_feedback       — append feedback item to session list (lightweight)
    summarize_feedback   — concatenate recent feedback strings for display
"""

from typing import Any

from utils.logger import setup_logger

logger = setup_logger(__name__)


def apply_feedback(
    meal_plan: dict[str, Any] | None,
    feedback_text: str,
) -> dict[str, Any]:
    """
    Merge user feedback into the current meal plan (MVP: annotate only).

    Args:
        meal_plan: Existing plan dict or None.
        feedback_text: Free-text user feedback.

    Returns:
        Updated meal plan dict.
    """
    plan = dict(meal_plan or {"status": "empty"})
    history = plan.setdefault("feedback_history", [])
    history.append({"text": feedback_text})
    plan["last_feedback"] = feedback_text
    # TODO: Re-run orchestrator with FEEDBACK_PROMPT; patch specific days/meals
    logger.info("Feedback recorded: %s", feedback_text[:80])
    return plan


def summarize_feedback(feedback_items: list[dict[str, Any]]) -> str:
    """Aggregate feedback for ranking or prompt context."""
    # TODO: Sentiment and theme extraction (dislikes, portion size, timing)
    texts = [f.get("text", "") for f in feedback_items]
    return "; ".join(t for t in texts if t)
