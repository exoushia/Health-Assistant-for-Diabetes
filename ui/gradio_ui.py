"""
Gradio UI — register, upload PDF, generate plan, feedback, health profile refresh.

Functions:
    _use_api / get_client path  — HTTP to FastAPI when API_BASE_URL is up, else in-process
    _format_health_md           — biomarker table for Markdown panel
    handle_register             — create user via API or scheduler.jobs
    handle_pdf_upload           — POST /ocr/upload or local OCR
    handle_generate_plan        — pipeline with live step HTML
    handle_feedback             — POST /feedback, show patched plan + WhatsApp preview
    handle_refresh_health       — reload profile JSON
    build_ui / launch           — tabs, theme, CSS, start server on GRADIO_PORT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from api.pipeline_service import (
    format_meal_cards_html,
    run_full_pipeline,
    steps_to_html,
)
from ui.api_client import get_client
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
.hero { text-align: center; padding: 0.5rem 0 1.5rem; }
.hero h1 { font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.25rem; }
.hero p { color: #64748b; font-size: 0.95rem; }
.panel { border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem 1.25rem; background: #fafafa; }
.orch-steps p { margin: 0.35rem 0; font-size: 0.9rem; }
.step-running { color: #0ea5e9; }
.step-done { color: #16a34a; }
.step-error { color: #dc2626; }
.step-skipped { color: #94a3b8; }
.meal-cards { display: grid; gap: 0.75rem; }
.meal-day { background: #fff; border-radius: 10px; padding: 0.75rem 1rem; border: 1px solid #e2e8f0; }
.meal-card { padding: 0.4rem 0; border-bottom: 1px solid #f1f5f9; }
.meal-card:last-child { border-bottom: none; }
.meal-type { text-transform: capitalize; color: #64748b; font-size: 0.8rem; display: block; }
.muted { color: #94a3b8; }
.wa-preview { font-family: system-ui, sans-serif; white-space: pre-wrap; background: #e8f5e9;
  border-radius: 12px; padding: 1rem; border: 1px solid #c8e6c9; font-size: 0.9rem; max-height: 400px; overflow-y: auto; }
footer { display: none !important; }
@media (max-width: 768px) {
  .gradio-container { padding: 0.5rem !important; }
}
"""

PENDING_STEPS_HTML = steps_to_html([
    {"id": "extracting_biomarkers", "label": "Extracting biomarkers", "status": "pending"},
    {"id": "filtering_recipes", "label": "Filtering recipes", "status": "pending"},
    {"id": "ranking_meals", "label": "Ranking meals", "status": "pending"},
    {"id": "translating_hindi", "label": "Translating Hindi", "status": "pending"},
    {"id": "sending_whatsapp", "label": "Sending WhatsApp", "status": "pending"},
])


def _use_api() -> bool:
    """Return True if FastAPI backend is reachable."""
    try:
        get_client().health()
        return True
    except Exception:
        return False


def _format_health_md(data: dict[str, Any]) -> str:
    profile = data.get("profile") or data.get("health_profile") or {}
    if isinstance(profile, dict) and "biomarkers" in profile:
        bio = profile.get("biomarkers", {})
        risk = profile.get("diabetes_risk", "—")
        conditions = profile.get("conditions", [])
        meds = profile.get("medications", [])
    else:
        bio = profile
        risk = bio.get("diabetes_risk", "—") if isinstance(bio, dict) else "—"
        conditions = profile.get("conditions", []) if isinstance(profile, dict) else []
        meds = profile.get("medications", []) if isinstance(profile, dict) else []

    lines = [
        "### Health profile",
        f"- **Diabetes risk:** `{risk}`",
        f"- **HbA1c:** {bio.get('hba1c_percent') or bio.get('hba1c') or '—'} %",
        f"- **Fasting glucose:** {bio.get('fasting_glucose_mg_dl', '—')} mg/dL",
        f"- **Cholesterol:** {bio.get('cholesterol_mg_dl', '—')} mg/dL",
        f"- **Triglycerides:** {bio.get('triglycerides_mg_dl', '—')} mg/dL",
        f"- **Vitamin D:** {bio.get('vitamin_d_ng_ml', '—')} ng/mL",
    ]
    if conditions:
        lines.append(f"- **Conditions:** {', '.join(conditions)}")
    if meds:
        lines.append(f"- **Medications:** {', '.join(meds)}")
    return "\n".join(lines)


def handle_register(
    user_id: str,
    name: str,
    phone: str,
    diabetes_type: str,
    locale: str,
) -> tuple[str, str]:
    """Registration form handler."""
    if not user_id.strip() or not name.strip():
        return "❌ User ID and name are required.", ""
    try:
        if _use_api():
            result = get_client().register(
                user_id.strip(), name.strip(), phone.strip(),
                diabetes_type=diabetes_type, locale=locale,
            )
        else:
            from api.pipeline_service import register_user
            result = register_user(
                user_id.strip(), name.strip(), phone.strip(),
                diabetes_type=diabetes_type, locale=locale,
            )
        return f"✅ Registered **{name}** (`{user_id}`)", json.dumps(result, indent=2)
    except Exception as exc:
        logger.exception("Registration failed")
        return f"❌ Registration failed: {exc}", ""


def handle_pdf_upload(user_id: str, pdf_file: str | None) -> tuple[str, str, str]:
    """PDF upload → OCR → health profile."""
    if not user_id.strip():
        return "❌ Enter User ID first.", "", PENDING_STEPS_HTML
    if not pdf_file:
        return "❌ Upload a PDF lab report.", "", PENDING_STEPS_HTML

    steps_html = steps_to_html([
        {"id": "extracting_biomarkers", "label": "Extracting biomarkers", "status": "running"},
    ])
    try:
        if _use_api():
            result = get_client().upload_pdf(user_id.strip(), pdf_file)
            profile = result.get("health_profile", {})
        else:
            from tools.ocr import extract_text_from_pdf
            from tools.biomarkers import build_health_profile
            ocr = extract_text_from_pdf(pdf_file, user_id=user_id.strip(), store=True)
            profile = build_health_profile(ocr.get("text", ""), user_id=user_id.strip(), store=True)
            result = {"ocr": ocr, "health_profile": profile}

        steps_html = steps_to_html([
            {"id": "extracting_biomarkers", "label": "Extracting biomarkers", "status": "done",
             "message": "PDF processed"},
        ])
        health_md = _format_health_md({"profile": profile})
        return "✅ Lab report processed.", health_md, steps_html
    except Exception as exc:
        logger.exception("PDF upload failed")
        err_steps = steps_to_html([
            {"id": "extracting_biomarkers", "label": "Extracting biomarkers", "status": "error",
             "message": str(exc)},
        ])
        return f"❌ {exc}", "", err_steps


def handle_generate_plan(
    user_id: str,
    message: str,
    locale: str,
    pdf_file: str | None,
    send_wa: bool,
    phone: str,
) -> tuple[str, str, str, str, str, str]:
    """
    Full pipeline: biomarkers → filter → rank → Hindi → WhatsApp preview.

    Returns:
        status, steps_html, health_md, meal_cards_html, wa_preview, raw_json
    """
    if not user_id.strip():
        empty = "❌ User ID required."
        return empty, PENDING_STEPS_HTML, "", "", "", ""

    try:
        if _use_api():
            result = get_client().run_pipeline(
                user_id.strip(),
                message=message or "Create a 7-day diabetes-friendly meal plan",
                locale=locale,
                pdf_path=pdf_file,
                send_whatsapp=send_wa,
                phone=phone.strip() or None,
            )
        else:
            result = run_full_pipeline(
                user_id.strip(),
                pdf_path=pdf_file,
                plan_message=message,
                locale=locale,
                send_whatsapp=send_wa,
                phone=phone.strip() or None,
            )

        steps_html = result.get("steps_html", "")
        health_md = _format_health_md(result.get("health_profile", {}))
        cards = result.get("meal_cards_html", format_meal_cards_html(result.get("meal_plan")))
        wa = result.get("whatsapp_preview", result.get("whatsapp_preview_en", ""))
        status = "✅ Meal plan ready."
        if send_wa:
            send_st = (result.get("whatsapp_send") or {}).get("status", "")
            status += f" WhatsApp: {send_st}"
        return status, steps_html, health_md, cards, wa, json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("Pipeline failed")
        err = steps_to_html([
            {"id": "pipeline", "label": "Pipeline", "status": "error", "message": str(exc)},
        ])
        return f"❌ {exc}", err, "", "", "", ""


def handle_feedback(
    user_id: str,
    feedback: str,
    locale: str,
) -> tuple[str, str, str, str, str]:
    """Conversational feedback loop."""
    if not user_id.strip() or not feedback.strip():
        return "❌ User ID and feedback required.", "", "", "", ""

    try:
        if _use_api():
            result = get_client().feedback(user_id.strip(), feedback.strip(), locale=locale)
        else:
            from orchestrator.agent import WorkflowOrchestrator
            result = WorkflowOrchestrator().process_feedback(user_id.strip(), feedback.strip())
            result["meal_cards_html"] = format_meal_cards_html(result.get("meal_plan"))
            result["steps_html"] = steps_to_html([
                {"id": "feedback", "label": "Feedback", "status": "done",
                 "message": result.get("intent", "")},
            ])
            if locale.startswith("hi"):
                from tools.localization import localize_meal_plan_whatsapp
                result["whatsapp_preview"] = localize_meal_plan_whatsapp(
                    result.get("meal_plan") or {}, locale="hi"
                )
            else:
                result["whatsapp_preview"] = result.get("whatsapp_message", "")

        steps = result.get("steps_html") or steps_to_html([
            {"id": "feedback", "label": "Processing feedback", "status": "done",
             "message": f"Intent: {result.get('intent', '')}"},
        ])
        cards = result.get("meal_cards_html", format_meal_cards_html(result.get("meal_plan")))
        wa = result.get("whatsapp_preview", result.get("whatsapp_message", ""))
        return (
            f"✅ Feedback applied (`{result.get('intent', '')}`)",
            steps,
            _format_health_md({"profile": result}),
            cards,
            wa,
        )
    except Exception as exc:
        logger.exception("Feedback failed")
        return f"❌ {exc}", "", "", "", ""


def handle_refresh_health(user_id: str) -> str:
    """Reload health profile from API."""
    if not user_id.strip():
        return "❌ Enter User ID."
    try:
        if _use_api():
            data = get_client().health_profile(user_id.strip())
        else:
            from api.pipeline_service import get_health_profile_summary
            data = get_health_profile_summary(user_id.strip())
        return _format_health_md(data)
    except Exception as exc:
        return f"❌ {exc}"


def _app_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#f8fafc",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_label_text_weight="600",
        button_primary_background_fill="linear-gradient(135deg, #0d9488 0%, #14b8a6 100%)",
    )


def build_ui() -> gr.Blocks:
    """Construct the modular Gradio interface."""
    with gr.Blocks(title="Health Meal Planner") as demo:
        gr.HTML(
            "<div class='hero'>"
            "<h1>🩺 Health Meal Planner</h1>"
            "<p>Diabetes-aware meals · Lab OCR · Hindi WhatsApp · Conversational feedback</p>"
            "</div>"
        )

        with gr.Row():
            api_status = gr.Markdown(
                "🟢 API connected" if _use_api() else "🟡 API offline — using direct mode"
            )

        # Shared session state
        with gr.Row(equal_height=True):
            session_user = gr.Textbox(
                label="User ID",
                placeholder="user_demo_001",
                value="user_demo_001",
                scale=2,
            )
            session_locale = gr.Dropdown(
                label="Locale",
                choices=[("English", "en"), ("Hindi", "hi")],
                value="en",
                scale=1,
            )

        with gr.Tabs():
            # --- 1. Registration ---
            with gr.Tab("📝 Register"):
                gr.Markdown("Create or update your profile. Links WhatsApp and health data.")
                with gr.Row():
                    reg_name = gr.Textbox(label="Full name", placeholder="Priya Sharma")
                    reg_phone = gr.Textbox(label="Phone (E.164)", placeholder="+919876543210")
                with gr.Row():
                    reg_diabetes = gr.Dropdown(
                        label="Diabetes type",
                        choices=["type_1", "type_2", "prediabetes", "gestational"],
                        value="type_2",
                    )
                    reg_locale = gr.Dropdown(
                        label="Preferred language",
                        choices=[("English", "en"), ("Hindi", "hi")],
                        value="en",
                    )
                reg_btn = gr.Button("Register", variant="primary")
                reg_status = gr.Markdown()
                reg_json = gr.JSON(label="Registration response", visible=False)

                reg_btn.click(
                    handle_register,
                    inputs=[session_user, reg_name, reg_phone, reg_diabetes, reg_locale],
                    outputs=[reg_status, reg_json],
                    show_progress="full",
                )

            # --- 2. PDF Upload ---
            with gr.Tab("📄 Lab report"):
                gr.Markdown("Upload a lab PDF to extract biomarkers (HbA1c, glucose, lipids…).")
                pdf_input = gr.File(
                    label="Lab report PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                pdf_btn = gr.Button("Extract biomarkers", variant="primary")
                pdf_status = gr.Markdown()
                pdf_steps = gr.HTML(label="Pipeline steps", value=PENDING_STEPS_HTML)
                pdf_health = gr.Markdown(label="Extracted health profile")

                pdf_btn.click(
                    handle_pdf_upload,
                    inputs=[session_user, pdf_input],
                    outputs=[pdf_status, pdf_health, pdf_steps],
                    show_progress="full",
                )

            # --- 3–6. Dashboard ---
            with gr.Tab("🍽 Meal plan"):
                with gr.Row():
                    plan_message = gr.Textbox(
                        label="Meal plan request",
                        placeholder="Create a low glycemic 7-day meal plan",
                        lines=2,
                        scale=2,
                    )
                with gr.Row():
                    plan_phone = gr.Textbox(
                        label="WhatsApp phone (optional send)",
                        placeholder="+10000000000",
                        scale=1,
                    )
                    plan_send_wa = gr.Checkbox(label="Send via WhatsApp", value=False)
                plan_btn = gr.Button("Generate meal plan", variant="primary", size="lg")
                plan_status = gr.Markdown()

                gr.Markdown("### Orchestration progress")
                plan_steps = gr.HTML(value=PENDING_STEPS_HTML)

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.Markdown("### Health profile")
                        plan_health = gr.Markdown()
                        refresh_health_btn = gr.Button("Refresh profile", size="sm")
                    with gr.Column(scale=1):
                        gr.Markdown("### Meal cards")
                        plan_meals = gr.HTML()

                gr.Markdown("### WhatsApp preview")
                plan_wa = gr.Textbox(
                    label="Message preview",
                    lines=12,
                    elem_classes=["wa-preview"],
                )
                plan_raw = gr.Accordion("Raw API response", open=False)
                with plan_raw:
                    plan_json = gr.Code(language="json")

                plan_btn.click(
                    handle_generate_plan,
                    inputs=[
                        session_user,
                        plan_message,
                        session_locale,
                        pdf_input,
                        plan_send_wa,
                        plan_phone,
                    ],
                    outputs=[plan_status, plan_steps, plan_health, plan_meals, plan_wa, plan_json],
                    show_progress="full",
                )
                refresh_health_btn.click(
                    handle_refresh_health,
                    inputs=[session_user],
                    outputs=[plan_health],
                )

            # --- Feedback ---
            with gr.Tab("💬 Feedback"):
                gr.Markdown(
                    "Examples: *No paneer tonight* · *Don't feel like idli* · *Replace breakfast*"
                )
                feedback_box = gr.Textbox(
                    label="Your feedback",
                    placeholder="No paneer tonight",
                    lines=3,
                )
                feedback_btn = gr.Button("Apply feedback", variant="primary")
                feedback_status = gr.Markdown()
                feedback_steps = gr.HTML(value=PENDING_STEPS_HTML)
                with gr.Row():
                    feedback_health = gr.Markdown()
                    feedback_meals = gr.HTML()
                feedback_wa = gr.Textbox(label="WhatsApp reply preview", lines=8)

                feedback_btn.click(
                    handle_feedback,
                    inputs=[session_user, feedback_box, session_locale],
                    outputs=[
                        feedback_status,
                        feedback_steps,
                        feedback_health,
                        feedback_meals,
                        feedback_wa,
                    ],
                    show_progress="full",
                )

        gr.Markdown(
            "<p class='muted' style='text-align:center;margin-top:1rem'>"
            "Start FastAPI: <code>python app.py</code> · UI uses <code>/api/v1</code> endpoints"
            "</p>"
        )

    return demo


def launch() -> None:
    """Launch Gradio with settings from environment."""
    settings = get_settings()
    demo = build_ui()
    demo.launch(
        server_name=settings.gradio_host,
        server_port=settings.gradio_port,
        share=False,
        show_error=True,
        theme=_app_theme(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    launch()
