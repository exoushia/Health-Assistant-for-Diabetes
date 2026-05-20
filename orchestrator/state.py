"""
Per-user workflow state — meal plan, blocked ingredients, tool execution log.

Classes:
    ToolExecutionRecord — one tool call (name, args, result, timing)
    AgentState          — serializable session: plan, profile, blocked list, feedback history

Functions:
    load_state / save_state     — read/write data/sessions/{user_id}.json
    ensure_meal_plan            — create placeholder 7-day plan if missing
    _recipe_to_slot             — map enriched recipe dict into a meal slot
    _normalize_ingredient       — lowercase ingredient token for blocking
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
IntentType = Literal[
    "ingredient_issue",
    "mood_change",
    "meal_replacement",
    "insufficient_query",
]


@dataclass
class ToolExecutionRecord:
    """Log entry for a single tool invocation."""

    tool: str
    input: dict[str, Any]
    output_summary: str
    success: bool
    duration_ms: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "input": self.input,
            "output_summary": self.output_summary,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentState:
    """
    Mutable workflow state — not a chat session.

    Tracks profile, plan, constraints, feedback, and tool audit trail.
    """

    user_id: str
    user_profile: dict[str, Any] = field(default_factory=dict)
    current_meal_plan: dict[str, Any] | None = None
    health_constraints: dict[str, Any] = field(default_factory=dict)
    feedback_history: list[dict[str, Any]] = field(default_factory=list)
    blocked_ingredients: list[str] = field(default_factory=list)
    biomarkers: dict[str, Any] = field(default_factory=dict)
    retrieved_recipes: list[dict[str, Any]] = field(default_factory=list)
    ranked_recipes: list[dict[str, Any]] = field(default_factory=list)
    last_intent: str | None = None
    last_affected_slots: list[dict[str, str]] = field(default_factory=list)
    tool_execution_log: list[ToolExecutionRecord] = field(default_factory=list)
    locale: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    # Legacy aliases used by older API paths
    @property
    def meal_plan_draft(self) -> dict[str, Any] | None:
        return self.current_meal_plan

    @meal_plan_draft.setter
    def meal_plan_draft(self, value: dict[str, Any] | None) -> None:
        self.current_meal_plan = value

    @property
    def feedback(self) -> list[dict[str, Any]]:
        return self.feedback_history

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_blocked_ingredients(self, ingredients: list[str]) -> None:
        """Merge normalized ingredients into the block list."""
        for item in ingredients:
            normalized = _normalize_ingredient(item)
            if normalized and normalized not in self.blocked_ingredients:
                self.blocked_ingredients.append(normalized)
        self.touch()

    def record_feedback(self, entry: dict[str, Any]) -> None:
        """Append a feedback event with intent metadata."""
        self.feedback_history.append(entry)
        self.touch()

    def log_tool(
        self,
        tool: str,
        input_data: dict[str, Any],
        output_summary: str,
        *,
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        """Record tool execution for audit and debugging."""
        record = ToolExecutionRecord(
            tool=tool,
            input=input_data,
            output_summary=output_summary,
            success=success,
            duration_ms=duration_ms,
        )
        self.tool_execution_log.append(record)
        logger.info(
            "TOOL %s | success=%s | %.1fms | %s",
            tool,
            success,
            duration_ms,
            output_summary[:200],
        )

    def get_meal_slots(self) -> list[dict[str, str]]:
        """Flatten current plan into list of {day, meal_type, ...meal} dicts."""
        slots: list[dict[str, str]] = []
        plan = self.current_meal_plan or {}
        for day_block in plan.get("days", []):
            day = day_block.get("day", "")
            for meal_type, meal in (day_block.get("meals") or {}).items():
                if isinstance(meal, dict):
                    slots.append({"day": day, "meal_type": meal_type, **meal})
        return slots

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence or API responses."""
        return {
            "user_id": self.user_id,
            "user_profile": self.user_profile,
            "current_meal_plan": self.current_meal_plan,
            "health_constraints": self.health_constraints,
            "feedback_history": self.feedback_history,
            "blocked_ingredients": self.blocked_ingredients,
            "biomarkers": self.biomarkers,
            "retrieved_recipes": self.retrieved_recipes,
            "ranked_recipes": self.ranked_recipes,
            "last_intent": self.last_intent,
            "last_affected_slots": self.last_affected_slots,
            "tool_execution_log": [t.to_dict() for t in self.tool_execution_log],
            "locale": self.locale,
            "metadata": self.metadata,
            "updated_at": self.updated_at,
        }


def _state_path(user_id: str) -> Path:
    settings = get_settings()
    safe_id = re.sub(r"[^\w\-]", "_", user_id.strip()) or "unknown_user"
    return settings.data_dir / "sessions" / f"{safe_id}.json"


def _load_user_profile(user_id: str) -> dict[str, Any]:
    """Load user profile from data/users.json if present."""
    settings = get_settings()
    path = settings.users_json_path
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for user in data.get("users", []):
            if user.get("id") == user_id:
                return user
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load user profile: %s", exc)
    return {}


def load_state(user_id: str) -> AgentState:
    """
    Load workflow state from JSON session file.

    Hydrates user profile and health constraints from users.json on first load.
    """
    path = _state_path(user_id)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            log = [
                ToolExecutionRecord(**entry)
                for entry in raw.pop("tool_execution_log", [])
            ]
            raw.pop("user_id", None)
            fields = {
                k: v for k, v in raw.items()
                if k in AgentState.__dataclass_fields__
            }
            state = AgentState(user_id=user_id, **fields)
            state.tool_execution_log = log
            return state
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("Corrupt session %s: %s", path, exc)

    profile = _load_user_profile(user_id)
    constraints = {
        "diabetes_type": profile.get("diabetes_type"),
        "dietary_restrictions": profile.get("preferences", {}).get("dietary_restrictions", []),
        "allergies": profile.get("preferences", {}).get("allergies", []),
        "cuisine": profile.get("preferences", {}).get("cuisine", []),
    }
    return AgentState(
        user_id=user_id,
        user_profile=profile,
        health_constraints=constraints,
        biomarkers=profile.get("biomarkers", {}),
    )


def save_state(state: AgentState) -> Path:
    """Persist workflow state to data/sessions/{user_id}.json."""
    path = _state_path(state.user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.touch()
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    return path


def ensure_meal_plan(state: AgentState) -> dict[str, Any]:
    """
    Ensure state has a 7-day meal plan skeleton.

    Populates empty slots from ranked_recipes when available.
    """
    if state.current_meal_plan and state.current_meal_plan.get("days"):
        return state.current_meal_plan

    days = []
    recipes = state.ranked_recipes or state.retrieved_recipes or []
    idx = 0
    for day in WEEKDAYS:
        meals: dict[str, Any] = {}
        for meal_type in ("breakfast", "lunch", "dinner"):
            recipe = recipes[idx % len(recipes)] if recipes else _placeholder_recipe(meal_type)
            idx += 1
            meals[meal_type] = _recipe_to_slot(recipe)
        days.append({"day": day, "meals": meals})

    state.current_meal_plan = {
        "status": "active",
        "version": 1,
        "days": days,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state.touch()
    return state.current_meal_plan


def _placeholder_recipe(meal_type: str) -> dict[str, Any]:
    return {
        "recipe_id": f"placeholder_{meal_type}",
        "name": f"Balanced {meal_type}",
        "meal_type": meal_type,
        "ingredients": [],
    }


def _recipe_to_slot(recipe: dict[str, Any]) -> dict[str, Any]:
    return {
        "recipe_id": recipe.get("recipe_id", ""),
        "name": recipe.get("name", "Meal"),
        "meal_type": recipe.get("meal_type", ""),
        "diabetes_score": recipe.get("diabetes_score"),
        "summary": recipe.get("summary", ""),
    }


def _normalize_ingredient(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
