"""
OpenAI tool-calling workflow for meal-plan updates — not a free-form chatbot.

Classes:
    FeedbackClassification — structured intent + affected slots from the model
    WorkflowOrchestrator   — runs classify → tool loop → patched plan
    MealPlanningAgent      — alias for WorkflowOrchestrator (API compatibility)

WorkflowOrchestrator methods:
    run              — bootstrap a new weekly meal plan into session state
    process_feedback — classify user text, call tools, return patched plan + WhatsApp text
    _execute_tool    — dispatch one registered tool (rerank, block ingredient, etc.)

Module helpers:
    _classify_feedback_deterministic — regex fallback when OpenAI is unavailable
    _tools_for_intent                — which tools each intent may call
    _filter_blocked_recipes          — drop recipes containing blocked ingredients
    _patch_plan_slots                — swap only affected day/meal slots
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from openai import APIError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from orchestrator.prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    TOOL_ORCHESTRATION_PROMPT,
    WHATSAPP_SUMMARY_PROMPT,
    WORKFLOW_SYSTEM_PROMPT,
)
from orchestrator.state import (
    MEAL_TYPES,
    WEEKDAYS,
    AgentState,
    ensure_meal_plan,
    load_state,
    save_state,
)
from tools.biomarkers import load_health_profile, parse_biomarkers
from tools.localization import localize_meal_plan, localize_recipe
from tools.ranking import rank_recipes
from tools.retrieval import retrieve_recipes
from tools.scoring import score_meal_plan
from tools.validation import validate_meal_plan
from tools.whatsapp import format_meal_plan_for_whatsapp
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

IntentType = Literal[
    "ingredient_issue",
    "mood_change",
    "meal_replacement",
    "insufficient_query",
]

_MAX_TOOL_ROUNDS = 8


class FeedbackClassification(BaseModel):
    """Structured output from feedback classification step."""

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    affected_meal_types: list[str] = Field(default_factory=list)
    affected_days: list[str] = Field(default_factory=list)
    blocked_ingredients: list[str] = Field(default_factory=list)
    disliked_foods: list[str] = Field(default_factory=list)
    replacement_query: str = ""
    reasoning: str = ""


class WorkflowOrchestrator:
    """
    Workflow engine: classify → identify slots → rerank affected → patch plan → optional tools.

    Uses OpenAI structured outputs for classification and tool calling for dynamic execution.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.openai_model
        self._client: OpenAI | None = None
        if settings.openai_api_key:
            self._client = OpenAI(api_key=settings.openai_api_key)
        self._tools = self._build_tool_definitions()
        self._tool_handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "add_blocked_ingredients": self._tool_add_blocked_ingredients,
            "rerank_affected_meals": self._tool_rerank_affected_meals,
            "update_meal_slots": self._tool_update_meal_slots,
            "localize_meal_plan": self._tool_localize_meal_plan,
            "prepare_whatsapp_response": self._tool_prepare_whatsapp_response,
        }

    def process_feedback(self, user_id: str, feedback_text: str) -> dict[str, Any]:
        """
        Run the full feedback workflow for a user.

        Steps:
            1. Classify intent (structured output)
            2. Identify affected meals
            3. Rerank recipes for affected slots only
            4. Preserve unchanged meals
            5. Return updated plan + tool log
        """
        state = self._hydrate_state(user_id)
        ensure_meal_plan(state)

        classification = self.classify_feedback(state, feedback_text)
        state.last_intent = classification.intent
        state.record_feedback(
            {
                "text": feedback_text,
                "intent": classification.intent,
                "confidence": classification.confidence,
                "affected_meal_types": classification.affected_meal_types,
                "affected_days": classification.affected_days,
                "blocked_ingredients": classification.blocked_ingredients,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        if classification.intent == "insufficient_query":
            result = self._handle_insufficient_query(state, classification, feedback_text)
            save_state(state)
            return result

        affected_slots = self.identify_affected_meals(state, classification)
        affected_slots = _narrow_slots_to_disliked(
            affected_slots,
            classification.blocked_ingredients + classification.disliked_foods,
        )
        state.last_affected_slots = affected_slots
        state.log_tool(
            "identify_affected_meals",
            {"classification": classification.model_dump(), "feedback": feedback_text},
            f"Resolved {len(affected_slots)} slot(s): {affected_slots}",
        )

        if classification.blocked_ingredients or classification.disliked_foods:
            state.add_blocked_ingredients(
                classification.blocked_ingredients + classification.disliked_foods
            )

        if self._openai_available():
            try:
                workflow_result = self._run_tool_orchestration_loop(
                    state, classification, feedback_text, affected_slots
                )
            except (AuthenticationError, APIError, RateLimitError) as exc:
                logger.warning("Tool orchestration API failed: %s; deterministic fallback", exc)
                workflow_result = self._run_deterministic_workflow(
                    state, classification, feedback_text, affected_slots
                )
        else:
            workflow_result = self._run_deterministic_workflow(
                state, classification, feedback_text, affected_slots
            )

        validation = validate_meal_plan(state.current_meal_plan)
        if state.current_meal_plan:
            state.current_meal_plan["score"] = score_meal_plan(
                state.current_meal_plan,
                state.biomarkers,
            )

        save_state(state)
        return {
            "user_id": user_id,
            "intent": classification.intent,
            "affected_slots": affected_slots,
            "meal_plan": state.current_meal_plan,
            "whatsapp_message": workflow_result.get("whatsapp_message"),
            "validation": validation,
            "tool_execution_log": [t.to_dict() for t in state.tool_execution_log[-20:]],
            "blocked_ingredients": state.blocked_ingredients,
            "reasoning": classification.reasoning,
        }

    def classify_feedback(
        self,
        state: AgentState,
        feedback_text: str,
    ) -> FeedbackClassification:
        """
        Step 1: Classify user feedback intent via OpenAI structured output.

        Falls back to deterministic rules when API is unavailable.
        """
        if self._openai_available():
            try:
                schema = FeedbackClassification.model_json_schema()
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": WORKFLOW_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"{INTENT_CLASSIFICATION_PROMPT}\n\n"
                            f"Blocked so far: {state.blocked_ingredients}\n"
                            f"Health constraints: {state.health_constraints}\n"
                            f"Feedback: {feedback_text}"
                        )},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "feedback_classification",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                    temperature=0.1,
                )
                raw = response.choices[0].message.content or "{}"
                result = FeedbackClassification.model_validate(json.loads(raw))
                state.log_tool(
                    "classify_feedback",
                    {"feedback": feedback_text},
                    f"intent={result.intent} confidence={result.confidence:.2f}",
                )
                return result
            except (AuthenticationError, APIError, RateLimitError, Exception) as exc:
                logger.warning("Structured classification failed: %s; using fallback", exc)

        result = _classify_feedback_deterministic(feedback_text)
        state.log_tool(
            "classify_feedback",
            {"feedback": feedback_text, "mode": "deterministic"},
            f"intent={result.intent}",
        )
        return result

    def identify_affected_meals(
        self,
        state: AgentState,
        classification: FeedbackClassification,
    ) -> list[dict[str, str]]:
        """
        Step 2: Map classification to concrete plan slots {day, meal_type}.

        Preserves granularity so only these slots are modified later.
        """
        ensure_meal_plan(state)
        plan = state.current_meal_plan or {}
        all_slots: list[dict[str, str]] = []

        meal_types = _normalize_meal_types(classification.affected_meal_types)
        days = _resolve_days(classification.affected_days)

        if not meal_types and classification.intent == "meal_replacement":
            meal_types = ["breakfast"]
        if not meal_types and classification.intent in ("ingredient_issue", "mood_change"):
            meal_types = list(MEAL_TYPES)

        if not days:
            days = [_today_weekday()]

        for day_block in plan.get("days", []):
            day_name = day_block.get("day", "").lower()
            if day_name not in days:
                continue
            for meal_type, meal in (day_block.get("meals") or {}).items():
                if meal_type not in meal_types:
                    continue
                if isinstance(meal, dict):
                    all_slots.append({
                        "day": day_name,
                        "meal_type": meal_type,
                        "recipe_id": meal.get("recipe_id", ""),
                        "name": meal.get("name", ""),
                    })

        if not all_slots:
            for day_block in plan.get("days", []):
                day_name = day_block.get("day", "").lower()
                for meal_type, meal in (day_block.get("meals") or {}).items():
                    if isinstance(meal, dict):
                        all_slots.append({
                            "day": day_name,
                            "meal_type": meal_type,
                            "recipe_id": meal.get("recipe_id", ""),
                            "name": meal.get("name", ""),
                        })
                break

        return all_slots

    def run(self, user_id: str, user_message: str) -> dict[str, Any]:
        """
        Entry point: route feedback-like messages to workflow; else bootstrap plan.
        """
        if _looks_like_feedback(user_message):
            return self.process_feedback(user_id, user_message)

        state = self._hydrate_state(user_id)
        state.biomarkers = parse_biomarkers(state.biomarkers)
        query = user_message.strip() or "diabetes friendly weekly meal plan"
        state.retrieved_recipes = retrieve_recipes(
            query=query,
            user_preferences=state.user_profile.get("preferences", {}),
        )
        state.ranked_recipes = rank_recipes(state.retrieved_recipes, state.biomarkers)
        state.current_meal_plan = self._bootstrap_plan(state)
        validation = validate_meal_plan(state.current_meal_plan)
        save_state(state)
        return {
            "user_id": user_id,
            "meal_plan": state.current_meal_plan,
            "validation": validation,
            "recipe_count": len(state.ranked_recipes),
        }

    def incorporate_feedback(self, user_id: str, feedback_text: str) -> dict[str, Any]:
        """Backward-compatible alias for process_feedback."""
        return self.process_feedback(user_id, feedback_text)

    # -------------------------------------------------------------------------
    # Tool orchestration (OpenAI tool calling)
    # -------------------------------------------------------------------------

    def _run_tool_orchestration_loop(
        self,
        state: AgentState,
        classification: FeedbackClassification,
        feedback_text: str,
        affected_slots: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Dynamic tool selection via OpenAI tool calling."""
        assert self._client is not None
        allowed = _tools_for_intent(classification.intent)
        tool_defs = [t for t in self._tools if t["function"]["name"] in allowed]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": WORKFLOW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{TOOL_ORCHESTRATION_PROMPT}\n\n"
                    f"Intent: {classification.intent}\n"
                    f"Feedback: {feedback_text}\n"
                    f"Affected slots: {json.dumps(affected_slots)}\n"
                    f"Replacement query: {classification.replacement_query or feedback_text}\n"
                    f"Blocked ingredients: {state.blocked_ingredients}\n"
                    f"Locale: {state.locale}"
                ),
            },
        ]

        context = {"affected_slots": affected_slots, "classification": classification}
        whatsapp_message: str | None = None

        for round_num in range(_MAX_TOOL_ROUNDS):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tool_defs,
                tool_choice="auto",
                temperature=0.1,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                whatsapp_message = whatsapp_message or message.content
                break

            messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                args["_affected_slots"] = affected_slots
                args["_replacement_query"] = (
                    classification.replacement_query or feedback_text
                )
                result = self._dispatch_tool(state, name, args, context)

                if name == "prepare_whatsapp_response":
                    whatsapp_message = result.get("message")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

            logger.info("Tool round %d complete (%d calls)", round_num + 1, len(message.tool_calls))

        if not whatsapp_message:
            whatsapp_message = self._tool_prepare_whatsapp_response(
                state, {"summary": "Meal plan updated."}, context
            ).get("message")

        return {"whatsapp_message": whatsapp_message}

    def _run_deterministic_workflow(
        self,
        state: AgentState,
        classification: FeedbackClassification,
        feedback_text: str,
        affected_slots: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Execute tools in fixed order when OpenAI is unavailable."""
        context: dict[str, Any] = {"affected_slots": affected_slots}
        if classification.blocked_ingredients or classification.disliked_foods:
            self._tool_add_blocked_ingredients(
                state,
                {"ingredients": classification.blocked_ingredients + classification.disliked_foods},
                context,
            )
        ranked = self._tool_rerank_affected_meals(
            state,
            {
                "meal_types": classification.affected_meal_types,
                "days": classification.affected_days,
                "query": classification.replacement_query or feedback_text,
                "_affected_slots": affected_slots,
                "_replacement_query": feedback_text,
            },
            context,
        )
        updates = _build_slot_updates(affected_slots, ranked.get("recipes", []))
        self._tool_update_meal_slots(state, {"updates": updates}, context)
        if state.locale != "en":
            self._tool_localize_meal_plan(state, {"locale": state.locale}, context)
        wa = self._tool_prepare_whatsapp_response(
            state,
            {"summary": f"Updated {len(updates)} meal(s) for {classification.intent}."},
            context,
        )
        return {"whatsapp_message": wa.get("message")}

    def _dispatch_tool(
        self,
        state: AgentState,
        name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        handler = self._tool_handlers.get(name)
        if not handler:
            state.log_tool(name, args, "Unknown tool", success=False)
            return {"error": f"Unknown tool: {name}"}
        return handler(state, args, context)

    # -------------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------------

    def _tool_add_blocked_ingredients(
        self,
        state: AgentState,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        start = time.monotonic()
        ingredients = args.get("ingredients", [])
        state.add_blocked_ingredients(ingredients)
        duration = (time.monotonic() - start) * 1000
        state.log_tool(
            "add_blocked_ingredients",
            {"ingredients": ingredients},
            f"Total blocked: {state.blocked_ingredients}",
            duration_ms=duration,
        )
        return {"blocked_ingredients": state.blocked_ingredients}

    def _tool_rerank_affected_meals(
        self,
        state: AgentState,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        start = time.monotonic()
        query = args.get("query") or args.get("_replacement_query", "low glycemic meal")
        affected_slots = args.get("_affected_slots") or context.get("affected_slots", [])

        meal_types = args.get("meal_types") or list({s["meal_type"] for s in affected_slots})
        query_parts = [query, *meal_types, *state.blocked_ingredients]
        full_query = " ".join(str(p) for p in query_parts if p)

        retrieved = retrieve_recipes(
            query=full_query,
            user_preferences=state.user_profile.get("preferences", {}),
            top_k=30,
        )
        filtered = _filter_blocked_recipes(retrieved, state.blocked_ingredients)
        ranked = rank_recipes(filtered, state.biomarkers)

        context["ranked_for_slots"] = ranked
        duration = (time.monotonic() - start) * 1000
        state.log_tool(
            "rerank_affected_meals",
            {"query": full_query, "slots": len(affected_slots)},
            f"Retrieved {len(retrieved)}, ranked {len(ranked)} after block filter",
            duration_ms=duration,
        )
        return {"recipes": ranked[:10], "count": len(ranked)}

    def _tool_update_meal_slots(
        self,
        state: AgentState,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        start = time.monotonic()
        updates = args.get("updates")
        if not updates:
            affected = args.get("_affected_slots") or context.get("affected_slots", [])
            ranked = context.get("ranked_for_slots", [])
            updates = _build_slot_updates(affected, ranked)

        patched = _patch_plan_slots(state, updates)
        duration = (time.monotonic() - start) * 1000
        state.log_tool(
            "update_meal_slots",
            {"updates": updates},
            f"Patched {patched} slot(s); preserved all others",
            duration_ms=duration,
        )
        return {"patched_slots": patched, "plan": state.current_meal_plan}

    def _tool_localize_meal_plan(
        self,
        state: AgentState,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        from tools.localization import localize_meal_plan_whatsapp

        start = time.monotonic()
        locale = args.get("locale", state.locale)
        plan = state.current_meal_plan or {}
        state.current_meal_plan = localize_meal_plan(plan, locale)
        state.locale = locale
        if locale.lower().startswith("hi"):
            state.current_meal_plan["whatsapp_text_hi"] = localize_meal_plan_whatsapp(
                state.current_meal_plan, locale=locale
            )
        duration = (time.monotonic() - start) * 1000
        state.log_tool(
            "localize_meal_plan",
            {"locale": locale},
            f"Localized plan to {locale} (WhatsApp-ready)",
            duration_ms=duration,
        )
        return {
            "locale": locale,
            "whatsapp_text_hi": state.current_meal_plan.get("whatsapp_text_hi"),
        }

    def _tool_prepare_whatsapp_response(
        self,
        state: AgentState,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        start = time.monotonic()
        summary = args.get("summary", "")
        plan = state.current_meal_plan or {}

        if self._openai_available() and not summary:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": WHATSAPP_SUMMARY_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps({
                                "intent": state.last_intent,
                                "affected_slots": state.last_affected_slots,
                                "blocked": state.blocked_ingredients,
                                "plan_excerpt": _plan_excerpt(plan),
                            }),
                        },
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )
                summary = response.choices[0].message.content or ""
            except Exception as exc:
                logger.warning("WhatsApp summary LLM failed: %s", exc)

        if not summary:
            summary = (
                f"Updated your plan ({state.last_intent or 'change'}). "
                f"Affected: {len(state.last_affected_slots)} meal(s). "
                f"Avoiding: {', '.join(state.blocked_ingredients[:5]) or 'none'}."
            )

        if state.locale.lower().startswith("hi") and plan.get("whatsapp_text_hi"):
            message = plan["whatsapp_text_hi"]
        elif state.locale.lower().startswith("hi"):
            from tools.localization import localize_meal_plan_whatsapp

            message = localize_meal_plan_whatsapp(plan, locale=state.locale)
        else:
            message = format_meal_plan_for_whatsapp({"content": summary, **plan})
        duration = (time.monotonic() - start) * 1000
        state.log_tool(
            "prepare_whatsapp_response",
            {"intent": state.last_intent},
            f"Prepared message ({len(message)} chars)",
            duration_ms=duration,
        )
        return {"message": message}

    def _handle_insufficient_query(
        self,
        state: AgentState,
        classification: FeedbackClassification,
        feedback_text: str,
    ) -> dict[str, Any]:
        message = (
            "I need a bit more detail. Which meal should change "
            "(breakfast/lunch/dinner), or which ingredient should I avoid? "
            'Examples: "No paneer tonight", "Replace breakfast".'
        )
        state.log_tool(
            "insufficient_query",
            {"feedback": feedback_text},
            "Returned clarification prompt",
        )
        return {
            "user_id": state.user_id,
            "intent": classification.intent,
            "affected_slots": [],
            "meal_plan": state.current_meal_plan,
            "whatsapp_message": message,
            "validation": validate_meal_plan(state.current_meal_plan),
            "tool_execution_log": [t.to_dict() for t in state.tool_execution_log],
            "blocked_ingredients": state.blocked_ingredients,
            "reasoning": classification.reasoning,
        }

    def _openai_available(self) -> bool:
        """True when OpenAI client is configured with a non-placeholder key."""
        key = get_settings().openai_api_key or ""
        if not self._client or not key:
            return False
        placeholders = ("sk-your", "your-openai", "changeme")
        return not any(p in key.lower() for p in placeholders)

    def _hydrate_state(self, user_id: str) -> AgentState:
        state = load_state(user_id)
        health = load_health_profile(user_id)
        if health:
            state.biomarkers = parse_biomarkers(health)
        elif state.user_profile.get("biomarkers"):
            state.biomarkers = parse_biomarkers(state.user_profile["biomarkers"])
        return state

    def _bootstrap_plan(self, state: AgentState) -> dict[str, Any]:
        ensure_meal_plan(state)
        ranked = state.ranked_recipes[:21]
        idx = 0
        for day_block in state.current_meal_plan.get("days", []):
            for meal_type in ("breakfast", "lunch", "dinner"):
                if idx < len(ranked):
                    day_block["meals"][meal_type] = {
                        "recipe_id": ranked[idx].get("recipe_id", ""),
                        "name": ranked[idx].get("name", ""),
                        "meal_type": meal_type,
                        "diabetes_score": ranked[idx].get("diabetes_score"),
                        "summary": ranked[idx].get("summary", ""),
                    }
                    idx += 1
        state.current_meal_plan["status"] = "active"
        state.current_meal_plan["updated_at"] = datetime.now(timezone.utc).isoformat()
        return state.current_meal_plan

    @staticmethod
    def _build_tool_definitions() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "add_blocked_ingredients",
                    "description": "Add ingredients or foods the user refuses to the block list.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ingredients": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["ingredients"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "rerank_affected_meals",
                    "description": "Retrieve and rerank recipes for affected meal slots only.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "meal_types": {"type": "array", "items": {"type": "string"}},
                            "days": {"type": "array", "items": {"type": "string"}},
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_meal_slots",
                    "description": "Patch specific day/meal slots; leave all other slots unchanged.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "updates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "day": {"type": "string"},
                                        "meal_type": {"type": "string"},
                                        "recipe_id": {"type": "string"},
                                        "name": {"type": "string"},
                                    },
                                    "required": ["day", "meal_type", "recipe_id"],
                                },
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "localize_meal_plan",
                    "description": "Localize meal plan text for user locale.",
                    "parameters": {
                        "type": "object",
                        "properties": {"locale": {"type": "string"}},
                        "required": ["locale"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "prepare_whatsapp_response",
                    "description": "Format a concise WhatsApp summary of plan changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                    },
                },
            },
        ]


# Alias for backward compatibility
MealPlanningAgent = WorkflowOrchestrator


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _classify_feedback_deterministic(feedback: str) -> FeedbackClassification:
    lower = feedback.lower()
    blocked: list[str] = []
    disliked: list[str] = []
    meal_types: list[str] = []
    days: list[str] = []

    mood_cue = "feel like" in lower or "don't want" in lower or "dont want" in lower

    for token in ("paneer", "idli", "dosa", "chicken", "fish", "egg", "rice", "roti"):
        if token in lower:
            if mood_cue:
                disliked.append(token)
            elif "no " in lower or "don't" in lower or "dont" in lower or "not" in lower:
                blocked.append(token)

    if "tonight" in lower or "today" in lower:
        days.append(_today_weekday())
    for mt in MEAL_TYPES:
        if mt in lower:
            meal_types.append(mt)

    if "replace" in lower:
        intent: IntentType = "meal_replacement"
        if not meal_types:
            meal_types = ["breakfast"]
    elif mood_cue:
        intent = "mood_change"
        if not meal_types:
            meal_types = list(MEAL_TYPES)
    elif blocked or ("no " in lower and any(
        ing in lower for ing in ("paneer", "chicken", "fish", "egg", "meat")
    )):
        intent = "ingredient_issue"
        if not meal_types:
            meal_types = ["dinner"] if "tonight" in lower else list(MEAL_TYPES)
        if "tonight" in lower:
            days.append(_today_weekday())
    elif len(lower.split()) < 4 or lower in ("help", "change", "update"):
        intent = "insufficient_query"
    else:
        intent = "mood_change"

    return FeedbackClassification(
        intent=intent,
        confidence=0.75,
        affected_meal_types=meal_types,
        affected_days=days or [_today_weekday()],
        blocked_ingredients=blocked,
        disliked_foods=disliked,
        replacement_query=feedback,
        reasoning="Deterministic fallback classification",
    )


def _tools_for_intent(intent: str) -> set[str]:
    if intent == "insufficient_query":
        return {"prepare_whatsapp_response"}
    tools = {
        "add_blocked_ingredients",
        "rerank_affected_meals",
        "update_meal_slots",
        "prepare_whatsapp_response",
    }
    if intent in ("ingredient_issue", "mood_change", "meal_replacement"):
        return tools
    return tools


def _normalize_meal_types(values: list[str]) -> list[str]:
    out = []
    for v in values:
        v = v.lower().strip()
        if v in MEAL_TYPES and v not in out:
            out.append(v)
    return out


def _resolve_days(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    days: list[str] = []
    for token in tokens:
        t = token.lower().strip()
        if t in WEEKDAYS:
            days.append(t)
        elif t in ("today", "tonight", "now"):
            days.append(_today_weekday())
    return days or [_today_weekday()]


def _today_weekday() -> str:
    return WEEKDAYS[datetime.now().weekday()]


def _looks_like_feedback(message: str) -> bool:
    lower = message.lower()
    triggers = (
        "no ", "don't", "dont", "not feel", "replace", "avoid", "without",
        "remove", "skip", "instead", "paneer", "idli", "tonight",
    )
    return any(t in lower for t in triggers)


def _filter_blocked_recipes(
    recipes: list[dict[str, Any]],
    blocked: list[str],
) -> list[dict[str, Any]]:
    if not blocked:
        return recipes
    filtered = []
    for recipe in recipes:
        blob = " ".join([
            recipe.get("name", ""),
            recipe.get("summary", ""),
            " ".join(recipe.get("ingredients", [])),
        ]).lower()
        if any(b in blob for b in blocked):
            continue
        filtered.append(recipe)
    return filtered or recipes


def _build_slot_updates(
    affected_slots: list[dict[str, str]],
    ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    if not ranked:
        return updates
    for index, slot in enumerate(affected_slots):
        recipe = ranked[index % len(ranked)]
        updates.append({
            "day": slot["day"],
            "meal_type": slot["meal_type"],
            "recipe_id": recipe.get("recipe_id", ""),
            "name": recipe.get("name", ""),
            "diabetes_score": recipe.get("diabetes_score"),
            "summary": recipe.get("summary", ""),
            "patched_reason": "user_feedback",
        })
    return updates


def _patch_plan_slots(state: AgentState, updates: list[dict[str, Any]]) -> int:
    """Patch only listed slots; preserve all others."""
    plan = state.current_meal_plan or {}
    update_map = {
        (u["day"].lower(), u["meal_type"].lower()): u for u in updates
    }
    patched = 0
    for day_block in plan.get("days", []):
        day = day_block.get("day", "").lower()
        meals = day_block.get("meals", {})
        for key, payload in update_map.items():
            if key[0] != day:
                continue
            if key[1] in meals:
                meals[key[1]] = {
                    "recipe_id": payload.get("recipe_id", ""),
                    "name": payload.get("name", ""),
                    "meal_type": key[1],
                    "diabetes_score": payload.get("diabetes_score"),
                    "summary": payload.get("summary", ""),
                    "patched_reason": payload.get("patched_reason", "user_feedback"),
                }
                patched += 1
    plan["updated_at"] = datetime.now(timezone.utc).isoformat()
    state.current_meal_plan = plan
    state.touch()
    return patched


def _narrow_slots_to_disliked(
    slots: list[dict[str, str]],
    foods: list[str],
) -> list[dict[str, str]]:
    """When possible, only target slots whose current meal mentions a disliked food."""
    if not foods:
        return slots
    matched = []
    for slot in slots:
        blob = slot.get("name", "").lower()
        if any(food in blob for food in foods):
            matched.append(slot)
    return matched if matched else slots


def _plan_excerpt(plan: dict[str, Any]) -> str:
    lines = []
    for day_block in plan.get("days", [])[:2]:
        day = day_block.get("day", "")
        for mt, meal in (day_block.get("meals") or {}).items():
            if isinstance(meal, dict):
                lines.append(f"{day} {mt}: {meal.get('name', '')}")
    return "; ".join(lines[:6])
