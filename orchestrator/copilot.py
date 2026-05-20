"""
End-to-end Health Copilot — the integration layer that runs the full pipeline.

Classes:
    PipelineStep      — one logged step (id, status, message, detail)
    CopilotResult     — onboarding/feedback/E2E result bundle
    HealthCopilot     — main orchestrator

HealthCopilot methods:
    run_onboarding_pipeline — steps 1–6: PDF/OCR → biomarkers → risk → rank → plan → Hindi → WhatsApp
    run_feedback_pipeline   — steps 7–9: feedback → OpenAI tools → rerank → updated WhatsApp
    run_end_to_end          — chains onboarding + feedback for demos/tests
    _run_step / _finish     — step logging helpers

Functions:
    get_copilot — singleton accessor used by API and scripts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.agent import WorkflowOrchestrator
from orchestrator.state import ensure_meal_plan, load_state, save_state
from scheduler.jobs import load_user_profile, save_user_profile
from tools.biomarkers import assign_diabetes_risk, build_health_profile, load_health_profile
from tools.localization import localize_meal_plan, localize_meal_plan_whatsapp
from tools.ocr import extract_text_from_pdf
from tools.ranking import rank_recipes
from tools.retrieval import retrieve_recipes
from tools.scoring import explain_score
from tools.whatsapp import format_meal_plan_for_whatsapp, send_whatsapp_message
from utils.config import get_settings
from utils.errors import BiomarkerError, CopilotError, DeliveryError, OCRError
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Pipeline step identifiers (stable for UI + logs)
STEP_UPLOAD = "upload_report"
STEP_EXTRACT = "extracting_biomarkers"
STEP_RISK = "assign_diabetes_risk"
STEP_FILTER = "filtering_recipes"
STEP_RANK = "ranking_meals"
STEP_PLAN = "generate_meal_plan"
STEP_TRANSLATE = "translating_hindi"
STEP_WHATSAPP = "sending_whatsapp"
STEP_FEEDBACK = "receive_feedback"
STEP_RERANK = "rerank_meals"
STEP_UPDATED = "send_updated_plan"

STEP_LABELS = {
    STEP_UPLOAD: "Upload report",
    STEP_EXTRACT: "Extract biomarkers",
    STEP_RISK: "Assign diabetes risk",
    STEP_FILTER: "Filter recipes",
    STEP_RANK: "Rank meals (deterministic)",
    STEP_PLAN: "Generate meal plan (OpenAI)",
    STEP_TRANSLATE: "Translate Hindi",
    STEP_WHATSAPP: "Send WhatsApp",
    STEP_FEEDBACK: "Receive feedback",
    STEP_RERANK: "Rerank affected meals",
    STEP_UPDATED: "Send updated meal plan",
}


@dataclass
class PipelineStep:
    """Single orchestration step with status and metadata."""

    id: str
    status: str  # pending | running | done | error | skipped
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": STEP_LABELS.get(self.id, self.id),
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


@dataclass
class CopilotResult:
    """Complete copilot run result."""

    success: bool
    user_id: str
    steps: list[PipelineStep] = field(default_factory=list)
    health_profile: dict[str, Any] | None = None
    meal_plan: dict[str, Any] | None = None
    whatsapp_preview: str = ""
    feedback_result: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from api.pipeline_service import format_meal_cards_html, steps_to_html

        steps_dicts = [s.to_dict() for s in self.steps]
        return {
            "success": self.success,
            "user_id": self.user_id,
            "steps": steps_dicts,
            "steps_html": steps_to_html(steps_dicts),
            "health_profile": self.health_profile,
            "meal_plan": self.meal_plan,
            "meal_cards_html": format_meal_cards_html(self.meal_plan),
            "whatsapp_preview": self.whatsapp_preview,
            "feedback_result": self.feedback_result,
            "errors": self.errors,
        }


class HealthCopilot:
    """
    End-to-end health copilot integrating all subsystems.

    Architecture:
        tools/ocr          → lab PDF text
        tools/biomarkers   → structured profile + risk
        tools/retrieval    → candidate recipes
        tools/ranking      → deterministic diabetes scoring
        orchestrator/agent → OpenAI tool-calling feedback & plan patches
        tools/localization → Hindi WhatsApp text
        tools/whatsapp     → Twilio delivery + inbound webhook
        ui/gradio_ui       → FastAPI-backed frontend
    """

    def __init__(self) -> None:
        self._orchestrator = WorkflowOrchestrator()
        self._settings = get_settings()

    def run_onboarding_pipeline(
        self,
        user_id: str,
        *,
        pdf_path: str | Path | None = None,
        plan_message: str = "Create a 7-day diabetes-friendly meal plan",
        locale: str | None = None,
        send_whatsapp: bool = False,
        phone: str | None = None,
    ) -> CopilotResult:
        """
        Steps 1–6: upload → extract → risk → plan → Hindi → WhatsApp.

        Args:
            user_id: Registered user id.
            pdf_path: Optional lab report PDF.
            plan_message: Meal plan generation prompt.
            locale: ``en`` or ``hi`` (defaults to user profile / settings).
            send_whatsapp: Whether to send via Twilio.
            phone: Override phone from profile.

        Returns:
            CopilotResult with steps and artifacts.
        """
        locale = locale or self._resolve_locale(user_id)
        steps: list[PipelineStep] = []
        errors: list[str] = []

        profile = load_user_profile(user_id)
        state = load_state(user_id)
        state.user_profile = profile

        # 1. Upload report
        steps.append(self._run_step(STEP_UPLOAD, "running", "Receiving document…"))
        if pdf_path and Path(pdf_path).is_file():
            steps[-1] = self._finish(
                steps[-1], "done", Path(pdf_path).name, {"path": str(pdf_path)}
            )
        else:
            steps[-1] = self._finish(steps[-1], "skipped", "No PDF provided")

        # 2. Extract biomarkers (OCR + deterministic parse)
        health_profile: dict[str, Any] | None = None
        steps.append(self._run_step(STEP_EXTRACT, "running", "OCR + parsing…"))
        try:
            if pdf_path and Path(pdf_path).is_file():
                ocr = extract_text_from_pdf(pdf_path, user_id=user_id, store=True)
                if not ocr.get("success") and not ocr.get("text"):
                    raise OCRError("OCR produced no text", detail=ocr)
                health_profile = build_health_profile(
                    ocr.get("text", ""),
                    user_id=user_id,
                    store=True,
                )
                steps[-1] = self._finish(
                    steps[-1],
                    "done",
                    f"{ocr.get('page_count', 0)} pages processed",
                    {"storage": ocr.get("storage_path")},
                )
            else:
                health_profile = load_health_profile(user_id)
                if health_profile:
                    steps[-1] = self._finish(steps[-1], "done", "Loaded stored profile")
                else:
                    steps[-1] = self._finish(steps[-1], "skipped", "No lab data")
        except Exception as exc:
            steps[-1] = self._finish(steps[-1], "error", str(exc))
            errors.append(f"extract: {exc}")
            logger.exception("Biomarker extraction failed for %s", user_id)

        # 3. Assign diabetes risk (deterministic)
        steps.append(self._run_step(STEP_RISK, "running"))
        try:
            if health_profile:
                risk = health_profile.get("diabetes_risk") or assign_diabetes_risk(
                    health_profile.get("biomarkers", health_profile)
                )
                state.biomarkers = health_profile.get("biomarkers", {})
                state.health_constraints = {
                    "diabetes_type": profile.get("diabetes_type"),
                    "diabetes_risk": risk,
                }
                steps[-1] = self._finish(steps[-1], "done", f"Risk level: {risk}", {"risk": risk})
            else:
                steps[-1] = self._finish(steps[-1], "skipped", "No biomarkers available")
        except Exception as exc:
            steps[-1] = self._finish(steps[-1], "error", str(exc))
            errors.append(f"risk: {exc}")

        # 4. Filter + rank recipes (deterministic)
        retrieved: list[dict[str, Any]] = []
        steps.append(self._run_step(STEP_FILTER, "running"))
        try:
            prefs = profile.get("preferences", {})
            retrieved = retrieve_recipes(plan_message, user_preferences=prefs, top_k=25)
            state.retrieved_recipes = retrieved
            steps[-1] = self._finish(
                steps[-1],
                "done",
                f"{len(retrieved)} candidates",
                {"count": len(retrieved)},
            )
        except Exception as exc:
            steps[-1] = self._finish(steps[-1], "error", str(exc))
            errors.append(f"filter: {exc}")
            retrieved = []

        steps.append(self._run_step(STEP_RANK, "running", "Deterministic diabetes_score…"))
        try:
            state.ranked_recipes = rank_recipes(retrieved, state.biomarkers)
            steps[-1] = self._finish(
                steps[-1],
                "done",
                f"Ranked {len(state.ranked_recipes)} recipes",
                {"top_id": (state.ranked_recipes[0] or {}).get("recipe_id")},
            )
        except Exception as exc:
            steps[-1] = self._finish(steps[-1], "error", str(exc))
            errors.append(f"rank: {exc}")

        save_state(state)

        # 5. Generate meal plan (OpenAI orchestrator bootstrap)
        meal_plan: dict[str, Any] | None = None
        steps.append(self._run_step(STEP_PLAN, "running", "Building weekly plan…"))
        plan_result: dict[str, Any] = {}
        try:
            plan_result = self._orchestrator.run(user_id, plan_message)
            state = load_state(user_id)
            ensure_meal_plan(state)
            meal_plan = plan_result.get("meal_plan") or state.current_meal_plan
            if meal_plan:
                state.current_meal_plan = meal_plan
                score = meal_plan.get("score")
                if score is not None:
                    meal_plan["score_explanation"] = explain_score(float(score))
                save_state(state)
            steps[-1] = self._finish(steps[-1], "done", "Plan generated")
        except Exception as exc:
            steps[-1] = self._finish(steps[-1], "error", str(exc))
            errors.append(f"plan: {exc}")
            meal_plan = None

        # 6. Hindi localization
        steps.append(self._run_step(STEP_TRANSLATE, "running"))
        preview = ""
        try:
            meal_plan = meal_plan or {}
            if locale.lower().startswith("hi"):
                localized = localize_meal_plan(meal_plan, locale="hi")
                meal_plan = localized
                preview = localized.get("whatsapp_text") or localize_meal_plan_whatsapp(
                    meal_plan, locale="hi"
                )
                steps[-1] = self._finish(steps[-1], "done", "Hindi translation complete")
            else:
                preview = format_meal_plan_for_whatsapp(meal_plan, locale="en")
                steps[-1] = self._finish(steps[-1], "skipped", "English locale")
        except Exception as exc:
            steps[-1] = self._finish(steps[-1], "error", str(exc))
            errors.append(f"translate: {exc}")
            preview = format_meal_plan_for_whatsapp(meal_plan or {}, locale=locale)

        # 7. Send WhatsApp
        steps.append(self._run_step(STEP_WHATSAPP, "running"))
        target = phone or profile.get("phone")
        try:
            if send_whatsapp and target:
                send_result = send_whatsapp_message(target, preview)
                ok = send_result.get("status") == "sent"
                steps[-1] = self._finish(
                    steps[-1],
                    "done" if ok else "error",
                    send_result.get("status", ""),
                    send_result,
                )
                if not ok:
                    errors.append(f"whatsapp: {send_result.get('detail', 'send failed')}")
            else:
                steps[-1] = self._finish(steps[-1], "done", "Preview only (not sent)")
        except Exception as exc:
            steps[-1] = self._finish(steps[-1], "error", str(exc))
            errors.append(f"whatsapp: {exc}")

        return CopilotResult(
            success=len(errors) == 0,
            user_id=user_id,
            steps=steps,
            health_profile=health_profile,
            meal_plan=meal_plan,
            whatsapp_preview=preview,
            errors=errors,
        )

    def run_feedback_pipeline(
        self,
        user_id: str,
        feedback_text: str,
        *,
        locale: str | None = None,
        send_whatsapp: bool = True,
        phone: str | None = None,
    ) -> CopilotResult:
        """
        Steps 7–9: receive feedback → rerank/patch meals → send updated plan.

        Uses OpenAI tool-calling WorkflowOrchestrator (not a chatbot).
        """
        locale = locale or self._resolve_locale(user_id)
        steps: list[PipelineStep] = []
        errors: list[str] = []
        profile = load_user_profile(user_id)

        # 7. Receive feedback
        steps.append(self._run_step(STEP_FEEDBACK, "running", feedback_text[:60]))
        feedback_result: dict[str, Any] = {}
        try:
            feedback_result = self._orchestrator.process_feedback(user_id, feedback_text)
            intent = feedback_result.get("intent", "unknown")
            steps[-1] = self._finish(
                steps[-1],
                "done",
                f"Intent: {intent}",
                {
                    "intent": intent,
                    "affected_slots": feedback_result.get("affected_slots", []),
                },
            )
        except Exception as exc:
            steps[-1] = self._finish(steps[-1], "error", str(exc))
            errors.append(f"feedback: {exc}")
            logger.exception("Feedback pipeline failed")

        # 8. Rerank meals (logged in orchestrator tool_execution_log)
        steps.append(self._run_step(STEP_RERANK, "running", "Reranking affected meals…"))
        rerank_status = "done" if feedback_result else "skipped"
        steps[-1] = self._finish(
            steps[-1],
            rerank_status,
            "Affected slots reranked via orchestrator tools",
            {
                "tools": [
                    t.get("tool")
                    for t in (feedback_result.get("tool_execution_log") or [])[-5:]
                ],
            },
        )

        meal_plan = feedback_result.get("meal_plan")
        preview = feedback_result.get("whatsapp_message", "")

        # Hindi refresh for updated plan
        if locale.lower().startswith("hi") and meal_plan:
            try:
                preview = localize_meal_plan_whatsapp(meal_plan, locale="hi")
            except Exception as exc:
                errors.append(f"translate: {exc}")

        # 9. Send updated plan
        steps.append(self._run_step(STEP_UPDATED, "running"))
        target = phone or profile.get("phone")
        try:
            if send_whatsapp and target and preview:
                send_result = send_whatsapp_message(target, preview)
                ok = send_result.get("status") == "sent"
                steps[-1] = self._finish(
                    steps[-1],
                    "done" if ok else "error",
                    send_result.get("status", ""),
                    send_result,
                )
            else:
                steps[-1] = self._finish(steps[-1], "done", "Updated preview ready")
        except Exception as exc:
            steps[-1] = self._finish(steps[-1], "error", str(exc))
            errors.append(f"updated_plan: {exc}")

        return CopilotResult(
            success=len(errors) == 0 and bool(feedback_result),
            user_id=user_id,
            steps=steps,
            meal_plan=meal_plan,
            whatsapp_preview=preview,
            feedback_result=feedback_result,
            errors=errors,
        )

    def run_end_to_end(
        self,
        user_id: str,
        *,
        pdf_path: str | Path | None = None,
        plan_message: str = "Create a 7-day diabetes-friendly meal plan",
        feedback_text: str = "No paneer tonight",
        locale: str | None = None,
        send_whatsapp: bool = False,
        phone: str | None = None,
    ) -> CopilotResult:
        """
        Full demonstration flow (steps 1–9).

        Combines onboarding pipeline and feedback pipeline in one call.
        """
        logger.info("=== E2E copilot start user=%s ===", user_id)
        onboard = self.run_onboarding_pipeline(
            user_id,
            pdf_path=pdf_path,
            plan_message=plan_message,
            locale=locale,
            send_whatsapp=send_whatsapp,
            phone=phone,
        )
        feedback = self.run_feedback_pipeline(
            user_id,
            feedback_text,
            locale=locale,
            send_whatsapp=send_whatsapp,
            phone=phone,
        )

        all_steps = onboard.steps + feedback.steps
        all_errors = onboard.errors + feedback.errors

        result = CopilotResult(
            success=onboard.success and feedback.success and len(all_errors) == 0,
            user_id=user_id,
            steps=all_steps,
            health_profile=onboard.health_profile,
            meal_plan=feedback.meal_plan or onboard.meal_plan,
            whatsapp_preview=feedback.whatsapp_preview or onboard.whatsapp_preview,
            feedback_result=feedback.feedback_result,
            errors=all_errors,
        )
        logger.info(
            "=== E2E copilot done user=%s success=%s errors=%d ===",
            user_id,
            result.success,
            len(all_errors),
        )
        return result

    def _resolve_locale(self, user_id: str) -> str:
        profile = load_user_profile(user_id)
        return profile.get("locale") or self._settings.default_locale

    @staticmethod
    def _run_step(step_id: str, status: str, message: str = "") -> PipelineStep:
        step = PipelineStep(id=step_id, status=status, message=message)
        logger.info("STEP [%s] %s — %s", step_id, status, message)
        return step

    @staticmethod
    def _finish(
        step: PipelineStep,
        status: str,
        message: str = "",
        detail: dict[str, Any] | None = None,
    ) -> PipelineStep:
        step.status = status
        step.message = message
        step.detail = detail or {}
        step.timestamp = datetime.now(timezone.utc).isoformat()
        logger.info("STEP [%s] %s — %s", step.id, status, message)
        return step


def get_copilot() -> HealthCopilot:
    """Return singleton copilot instance."""
    return HealthCopilot()
