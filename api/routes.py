"""
REST API under /api/v1 — registration, OCR upload, pipeline, feedback, WhatsApp.

Request models: RegisterRequest, PlanRequest, FeedbackRequest, WhatsAppRequest,
                 PipelineRequest, E2ERequest

Route handlers:
    health_check            — GET /health
    register                — POST /register
    health_profile          — GET /users/{id}/health-profile
    upload_lab_pdf          — POST /ocr/upload (PDF → OCR → biomarkers)
    run_pipeline            — POST /pipeline (no file)
    run_pipeline_with_pdf   — POST /pipeline/upload
    create_meal_plan        — POST /plan
    run_copilot_e2e         — POST /copilot/e2e (full 9 steps)
    submit_feedback         — POST /feedback
    send_plan_whatsapp      — POST /whatsapp/send
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.pipeline_service import (
    format_meal_cards_html,
    get_health_profile_summary,
    register_user,
    run_e2e_pipeline,
    run_full_pipeline,
    steps_to_html,
)
from orchestrator.copilot import get_copilot
from orchestrator.agent import MealPlanningAgent
from utils.errors import CopilotError
from tools.biomarkers import build_health_profile
from tools.ocr import extract_text_from_pdf
from tools.validation import validate_user_id
from tools.whatsapp import format_meal_plan_for_whatsapp, send_whatsapp_message
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["meal-planner"])

_agent = MealPlanningAgent()


class RegisterRequest(BaseModel):
    """User registration payload."""

    user_id: str = Field(..., min_length=3)
    name: str
    phone: str = Field(..., description="E.164 phone, e.g. +919876543210")
    diabetes_type: str = "type_2"
    locale: str = "en"


class PlanRequest(BaseModel):
    """Request body for meal plan generation."""

    user_id: str = Field(..., description="Unique user identifier")
    message: str = Field(..., description="User message or planning request")
    locale: str = "en"


class FeedbackRequest(BaseModel):
    """Request body for meal plan feedback."""

    user_id: str
    feedback: str
    locale: str = "en"


class WhatsAppRequest(BaseModel):
    """Request body to push a meal plan via WhatsApp."""

    user_id: str
    phone: str = Field(..., description="E.164 phone number")
    locale: str = "en"
    send: bool = True


class PipelineRequest(BaseModel):
    """Run full orchestration pipeline (without PDF — use upload endpoint for PDF)."""

    user_id: str
    message: str = "Create a 7-day diabetes-friendly meal plan"
    locale: str = "en"
    send_whatsapp: bool = False
    phone: str | None = None


class E2ERequest(BaseModel):
    """Full copilot demonstration: onboarding + feedback loop (steps 1–9)."""

    user_id: str
    message: str = "Create a 7-day diabetes-friendly meal plan"
    feedback: str = "No paneer tonight"
    locale: str = "en"
    send_whatsapp: bool = False
    phone: str | None = None


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe for load balancers and monitoring."""
    return {"status": "ok"}


@router.post("/register")
async def register(body: RegisterRequest) -> dict[str, Any]:
    """Register or update a user profile."""
    if not validate_user_id(body.user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    return register_user(
        body.user_id,
        body.name,
        body.phone,
        diabetes_type=body.diabetes_type,
        locale=body.locale,
    )


@router.get("/users/{user_id}/health-profile")
async def health_profile(user_id: str) -> dict[str, Any]:
    """Get health profile summary for a user."""
    if not validate_user_id(user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    return get_health_profile_summary(user_id)


@router.post("/ocr/upload")
async def upload_lab_pdf(
    user_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Upload a lab report PDF; extract biomarkers via OCR pipeline.

    Returns OCR text excerpt, health profile, and extraction metadata.
    """
    if not validate_user_id(user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF file required")

    settings = get_settings()
    upload_dir = settings.data_dir / "uploads" / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    content = await file.read()
    dest.write_bytes(content)

    ocr_result = extract_text_from_pdf(dest, user_id=user_id, store=True)
    profile = build_health_profile(
        ocr_result.get("text", ""),
        user_id=user_id,
        store=True,
    )

    return {
        "user_id": user_id,
        "filename": file.filename,
        "ocr": ocr_result,
        "health_profile": profile,
    }


@router.post("/pipeline")
async def run_pipeline(body: PipelineRequest) -> dict[str, Any]:
    """Run full orchestration pipeline with step tracking (no PDF)."""
    if not validate_user_id(body.user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    return run_full_pipeline(
        body.user_id,
        plan_message=body.message,
        locale=body.locale,
        send_whatsapp=body.send_whatsapp,
        phone=body.phone,
    )


@router.post("/pipeline/upload")
async def run_pipeline_with_pdf(
    user_id: str = Form(...),
    message: str = Form("Create a 7-day diabetes-friendly meal plan"),
    locale: str = Form("en"),
    send_whatsapp: bool = Form(False),
    phone: str | None = Form(None),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Full pipeline including PDF biomarker extraction."""
    if not validate_user_id(user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")

    settings = get_settings()
    upload_dir = settings.data_dir / "uploads" / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / (file.filename or "report.pdf")
    dest.write_bytes(await file.read())

    return run_full_pipeline(
        user_id,
        pdf_path=dest,
        plan_message=message,
        locale=locale,
        send_whatsapp=send_whatsapp,
        phone=phone,
    )


@router.post("/plan")
async def create_meal_plan(body: PlanRequest) -> dict[str, Any]:
    """Generate or update a meal plan for a user."""
    if not validate_user_id(body.user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    logger.info("Plan request for user=%s", body.user_id)
    result = _agent.run(body.user_id, body.message)
    result["steps_html"] = steps_to_html([
        {"id": "ranking_meals", "label": "Ranking meals", "status": "done", "message": "Complete"},
    ])
    return result


@router.post("/copilot/e2e")
async def run_copilot_e2e(body: E2ERequest) -> dict[str, Any]:
    """
    End-to-end copilot: upload/OCR → risk → plan → Hindi → WhatsApp → feedback → rerank → update.

    See README.md for the full 9-step flow diagram.
    """
    if not validate_user_id(body.user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    logger.info("E2E copilot for user=%s", body.user_id)
    try:
        return run_e2e_pipeline(
            body.user_id,
            plan_message=body.message,
            feedback_text=body.feedback,
            locale=body.locale,
            send_whatsapp=body.send_whatsapp,
            phone=body.phone,
        )
    except CopilotError as exc:
        raise HTTPException(status_code=422, detail=exc.message) from exc


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest) -> dict[str, Any]:
    """Run feedback workflow (steps 7–9): classify → rerank → updated WhatsApp."""
    if not validate_user_id(body.user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    logger.info("Feedback workflow for user=%s", body.user_id)
    try:
        result = get_copilot().run_feedback_pipeline(
            body.user_id,
            body.feedback,
            locale=body.locale,
            send_whatsapp=False,
        )
        return result.to_dict()
    except CopilotError as exc:
        raise HTTPException(status_code=422, detail=exc.message) from exc


@router.post("/whatsapp/send")
async def send_plan_whatsapp(body: WhatsAppRequest) -> dict[str, Any]:
    """Send the user's current meal plan via WhatsApp (outbound API)."""
    result = _agent.run(body.user_id, "Send my current meal plan")
    plan = result.get("meal_plan", {})
    text = result.get("whatsapp_message") or format_meal_plan_for_whatsapp(
        plan, locale=body.locale
    )
    outbound = send_whatsapp_message(body.phone, text) if body.send else {"status": "skipped"}
    return {"orchestrator": result, "send": outbound, "preview": text}
