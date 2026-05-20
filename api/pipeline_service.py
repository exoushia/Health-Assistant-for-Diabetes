"""
Service layer between HTTP routes and HealthCopilot — HTML formatting + user CRUD.

Functions:
    steps_to_html              — render pipeline steps for Gradio
    register_user              — append to data/users.json
    get_health_profile_summary — load biomarkers/risk for display
    format_meal_cards_html     — meal plan as HTML cards
    run_full_pipeline          — onboarding steps via get_copilot()
    run_e2e_pipeline           — full 9-step copilot for API/Gradio
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestrator.copilot import HealthCopilot, STEP_LABELS, get_copilot
from scheduler.jobs import save_user_profile
from tools.biomarkers import load_health_profile
from scheduler.jobs import load_user_profile
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Re-export step constants for backward compatibility
STEP_EXTRACT = "extracting_biomarkers"
STEP_FILTER = "filtering_recipes"
STEP_RANK = "ranking_meals"
STEP_TRANSLATE = "translating_hindi"
STEP_WHATSAPP = "sending_whatsapp"
STEP_LABELS = STEP_LABELS


def steps_to_html(steps: list[dict[str, Any]]) -> str:
    """Render orchestration steps as HTML for Gradio Markdown."""
    icons = {"pending": "⏳", "running": "🔄", "done": "✅", "error": "❌", "skipped": "⏭️"}
    lines = ["<div class='orch-steps'>"]
    for s in steps:
        icon = icons.get(s.get("status", "pending"), "•")
        label = s.get("label", s.get("id", ""))
        msg = s.get("message", "")
        lines.append(
            f"<p class='step-{s.get('status')}'>{icon} <strong>{label}</strong>"
            f"{': ' + msg if msg else ''}</p>"
        )
    lines.append("</div>")
    return "\n".join(lines)


def register_user(
    user_id: str,
    name: str,
    phone: str,
    diabetes_type: str = "type_2",
    locale: str = "en",
) -> dict[str, Any]:
    """Register or update a user in users.json."""
    profile = {
        "id": user_id,
        "name": name.strip(),
        "phone": phone.strip(),
        "diabetes_type": diabetes_type,
        "locale": locale,
        "preferences": {
            "cuisine": ["indian"],
            "allergies": [],
            "dietary_restrictions": ["low_glycemic"],
        },
        "biomarkers": {},
        "medications": [],
        "meal_plan": None,
    }
    save_user_profile(user_id, profile)
    return {"status": "registered", "user": profile}


def get_health_profile_summary(user_id: str) -> dict[str, Any]:
    """Return health profile for UI display."""
    profile = load_health_profile(user_id)
    if profile:
        return {"user_id": user_id, "profile": profile, "source": "health_store"}
    user = load_user_profile(user_id)
    return {
        "user_id": user_id,
        "profile": user.get("biomarkers", {}),
        "user": user,
        "source": "users_json",
    }


def format_meal_cards_html(meal_plan: dict[str, Any] | None) -> str:
    """Render meal plan as HTML cards."""
    if not meal_plan or not meal_plan.get("days"):
        return "<p class='muted'>No meal plan generated yet.</p>"

    cards = ["<div class='meal-cards'>"]
    for day_block in meal_plan.get("days", [])[:7]:
        day = day_block.get("day", "").capitalize()
        cards.append(f"<div class='meal-day'><h4>📅 {day}</h4>")
        for mt in ("breakfast", "lunch", "dinner"):
            meal = (day_block.get("meals") or {}).get(mt)
            if not isinstance(meal, dict):
                continue
            name = meal.get("name", "—")
            cal = meal.get("calories")
            score = meal.get("diabetes_score")
            cal_s = f"{cal:.0f} kcal" if cal else ""
            score_s = f" · ✅ {score:.2f}" if score is not None else ""
            cards.append(
                f"<div class='meal-card'><span class='meal-type'>{mt}</span>"
                f"<strong>{name}</strong><br><small>{cal_s}{score_s}</small></div>"
            )
        cards.append("</div>")
    cards.append("</div>")
    return "\n".join(cards)


def run_full_pipeline(
    user_id: str,
    *,
    pdf_path: str | Path | None = None,
    plan_message: str = "Create a 7-day diabetes-friendly meal plan",
    locale: str = "en",
    send_whatsapp: bool = False,
    phone: str | None = None,
) -> dict[str, Any]:
    """Run onboarding pipeline (steps 1–6) via HealthCopilot."""
    result = get_copilot().run_onboarding_pipeline(
        user_id,
        pdf_path=pdf_path,
        plan_message=plan_message,
        locale=locale,
        send_whatsapp=send_whatsapp,
        phone=phone,
    )
    return result.to_dict()


def run_e2e_pipeline(
    user_id: str,
    *,
    pdf_path: str | Path | None = None,
    plan_message: str = "Create a 7-day diabetes-friendly meal plan",
    feedback_text: str = "No paneer tonight",
    locale: str = "en",
    send_whatsapp: bool = False,
    phone: str | None = None,
) -> dict[str, Any]:
    """Run full end-to-end copilot flow (steps 1–9)."""
    result = get_copilot().run_end_to_end(
        user_id,
        pdf_path=pdf_path,
        plan_message=plan_message,
        feedback_text=feedback_text,
        locale=locale,
        send_whatsapp=send_whatsapp,
        phone=phone,
    )
    return result.to_dict()
