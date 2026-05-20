"""
Background jobs — medicine low-stock WhatsApp reminders and user profile I/O.

Classes:
    MedicineStockStatus — tablets left, days remaining, should_remind flag

Functions:
    compute_days_remaining / check_low_stock — refill alert when < MEDICINE_LOW_STOCK_DAYS
    trigger_reminder / run_medicine_adherence_checks — scan all users, send Twilio
    update_medicine_stock / get_user_medications — CRUD on user medication list
    daily_meal_reminder / refresh_stale_plans — placeholders for future cron work
    start_scheduler / stop_scheduler — APScheduler lifecycle (lazy import)
    load_all_users / load_user_profile / save_user_profile — data/users.json access
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from orchestrator.state import load_state, save_state
from tools.whatsapp import send_whatsapp_message
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

_scheduler: Any = None

# Default low-stock threshold (days of supply remaining)
_DEFAULT_LOW_STOCK_DAYS = 3
# Minimum hours between repeat reminders for the same medicine
_REMINDER_COOLDOWN_HOURS = 24


@dataclass
class MedicineStockStatus:
    """Deterministic stock evaluation for one medicine."""

    name: str
    tablets_left: float
    dosage_per_day: float
    days_remaining: float
    is_low_stock: bool
    medication_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "medication_id": self.medication_id,
            "name": self.name,
            "tablets_left": self.tablets_left,
            "dosage_per_day": self.dosage_per_day,
            "days_remaining": self.days_remaining,
            "is_low_stock": self.is_low_stock,
        }


def compute_days_remaining(tablets_left: float, dosage_per_day: float) -> float:
    """
    Estimate how many days the current tablet stock will last.

    Deterministic formula:
        days_remaining = tablets_left / dosage_per_day

    Args:
        tablets_left: Count of tablets/units on hand (>= 0).
        dosage_per_day: Tablets taken per day (> 0 for meaningful result).

    Returns:
        Days remaining as a float. Returns ``inf`` if dosage is zero;
        returns ``0.0`` if no tablets left.
    """
    tablets_left = max(0.0, float(tablets_left))
    dosage_per_day = float(dosage_per_day)

    if tablets_left == 0:
        return 0.0
    if dosage_per_day <= 0:
        return math.inf
    return round(tablets_left / dosage_per_day, 2)


def check_low_stock(
    medications: list[dict[str, Any]],
    *,
    threshold_days: float | None = None,
) -> list[MedicineStockStatus]:
    """
    Flag medicines with supply below the day threshold (default: 3 days).

    Args:
        medications: List of medicine dicts with ``tablets_left`` and ``dosage_per_day``.
        threshold_days: Remind when ``days_remaining`` < this value.

    Returns:
        List of MedicineStockStatus for low-stock items only.
    """
    settings = get_settings()
    threshold = (
        threshold_days
        if threshold_days is not None
        else settings.medicine_low_stock_days
    )

    low: list[MedicineStockStatus] = []
    for med in medications:
        if not med.get("active", True):
            continue

        name = str(med.get("name", "Medicine"))
        med_id = str(med.get("id") or name.lower().replace(" ", "_"))
        tablets = float(med.get("tablets_left", 0))
        dosage = float(med.get("dosage_per_day") or med.get("doses_per_day") or 0)
        days = compute_days_remaining(tablets, dosage)
        is_low = days < threshold

        status = MedicineStockStatus(
            medication_id=med_id,
            name=name,
            tablets_left=tablets,
            dosage_per_day=dosage,
            days_remaining=days,
            is_low_stock=is_low,
        )
        if is_low:
            low.append(status)
            logger.info(
                "Low stock: %s | tablets=%.1f dosage/day=%.1f days=%.2f",
                name,
                tablets,
                dosage,
                days,
            )

    return low


def trigger_reminder(
    user_id: str,
    medication: dict[str, Any] | MedicineStockStatus,
    *,
    user_profile: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Send a WhatsApp low-stock reminder for one medicine.

    Integrates user profile (phone, locale) and updates reminder timestamps
    in the user profile state.

    Args:
        user_id: Target user identifier.
        medication: Medicine dict or MedicineStockStatus.
        user_profile: Optional pre-loaded profile; loaded from users.json if omitted.
        force: Send even if cooldown has not elapsed.

    Returns:
        Dict with reminder_sent, message body, WhatsApp send result, and stock info.
    """
    profile = user_profile or load_user_profile(user_id)
    phone = profile.get("phone")
    if not phone:
        logger.warning("No phone for user %s; cannot send medicine reminder", user_id)
        return {
            "reminder_sent": False,
            "reason": "missing_phone",
            "user_id": user_id,
        }

    if isinstance(medication, MedicineStockStatus):
        status = medication
        med_dict = medication.to_dict()
    else:
        med_dict = medication
        status = MedicineStockStatus(
            medication_id=str(med_dict.get("id", "")),
            name=str(med_dict.get("name", "Medicine")),
            tablets_left=float(med_dict.get("tablets_left", 0)),
            dosage_per_day=float(med_dict.get("dosage_per_day") or med_dict.get("doses_per_day") or 0),
            days_remaining=compute_days_remaining(
                float(med_dict.get("tablets_left", 0)),
                float(med_dict.get("dosage_per_day") or med_dict.get("doses_per_day") or 0),
            ),
            is_low_stock=True,
        )

    if not force and _in_reminder_cooldown(med_dict):
        logger.info(
            "Reminder cooldown active for user=%s med=%s",
            user_id,
            status.name,
        )
        return {
            "reminder_sent": False,
            "reason": "cooldown",
            "user_id": user_id,
            "medication": status.to_dict(),
        }

    body = _build_reminder_message(profile, status)
    send_result = send_whatsapp_message(phone, body)

    sent = send_result.get("status") == "sent"
    if sent:
        _mark_reminder_sent(user_id, med_dict.get("id") or status.medication_id)
        _sync_session_medications(user_id, profile)

    logger.info(
        "Medicine reminder user=%s med=%s sent=%s status=%s",
        user_id,
        status.name,
        sent,
        send_result.get("status"),
    )

    return {
        "reminder_sent": sent,
        "user_id": user_id,
        "medication": status.to_dict(),
        "message": body,
        "whatsapp": send_result,
    }


def run_medicine_adherence_checks() -> dict[str, Any]:
    """
    Cron job: scan all users, check stock, trigger WhatsApp reminders.

    Returns:
        Summary with users_checked, reminders_sent, and per-user details.
    """
    users = load_all_users()
    summary: dict[str, Any] = {
        "users_checked": 0,
        "reminders_sent": 0,
        "details": [],
    }

    for user in users:
        user_id = user.get("id")
        if not user_id:
            continue
        summary["users_checked"] += 1
        medications = get_user_medications(user_id, user)
        low_stock = check_low_stock(medications)

        user_detail: dict[str, Any] = {
            "user_id": user_id,
            "low_stock_count": len(low_stock),
            "reminders": [],
        }

        for status in low_stock:
            med = _find_medication_by_id(medications, status.medication_id)
            result = trigger_reminder(user_id, med or status.to_dict(), user_profile=user)
            user_detail["reminders"].append(result)
            if result.get("reminder_sent"):
                summary["reminders_sent"] += 1

        summary["details"].append(user_detail)

    logger.info(
        "Medicine adherence run: users=%d reminders=%d",
        summary["users_checked"],
        summary["reminders_sent"],
    )
    return summary


def update_medicine_stock(
    user_id: str,
    medication_id: str,
    *,
    tablets_left: float | None = None,
    dosage_per_day: float | None = None,
) -> dict[str, Any]:
    """
    Update tablet count or dosage for a user's medicine (deterministic).

    Persists to users.json and syncs orchestrator session state.

    Returns:
        Updated medication dict with computed days_remaining.
    """
    profile = load_user_profile(user_id)
    medications = get_user_medications(user_id, profile)
    updated: dict[str, Any] | None = None

    for med in medications:
        if med.get("id") == medication_id or med.get("name") == medication_id:
            if tablets_left is not None:
                med["tablets_left"] = max(0.0, float(tablets_left))
            if dosage_per_day is not None:
                med["dosage_per_day"] = max(0.0, float(dosage_per_day))
            med["days_remaining"] = compute_days_remaining(
                float(med.get("tablets_left", 0)),
                float(med.get("dosage_per_day", 0)),
            )
            med["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated = med
            break

    if not updated:
        return {"error": "medication_not_found", "medication_id": medication_id}

    profile["medications"] = medications
    save_user_profile(user_id, profile)
    _sync_session_medications(user_id, profile)

    return {
        "user_id": user_id,
        "medication": updated,
        "is_low_stock": updated["days_remaining"] < get_settings().medicine_low_stock_days,
    }


def get_user_medications(
    user_id: str,
    profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Load medications from user profile, enriched with days_remaining.

    Merges orchestrator session metadata when present.
    """
    profile = profile or load_user_profile(user_id)
    medications = list(profile.get("medications") or [])

    try:
        session = load_state(user_id)
        session_meds = session.metadata.get("medications")
        if session_meds:
            medications = _merge_medications(medications, session_meds)
    except Exception:
        pass

    enriched = []
    for med in medications:
        m = dict(med)
        m["days_remaining"] = compute_days_remaining(
            float(m.get("tablets_left", 0)),
            float(m.get("dosage_per_day") or m.get("doses_per_day") or 0),
        )
        enriched.append(m)
    return enriched


# ---------------------------------------------------------------------------
# Existing / other scheduled jobs
# ---------------------------------------------------------------------------


def daily_meal_reminder(user_id: str, phone: str) -> dict[str, Any]:
    """Send a daily meal reminder via WhatsApp."""
    body = f"Good morning! Check today's meals in your plan (user: {user_id})."
    return send_whatsapp_message(phone, body)


def refresh_stale_plans() -> None:
    """Background job: regenerate plans older than N days."""
    logger.info("refresh_stale_plans: not implemented (MVP)")


def start_scheduler() -> Any:
    """
    Start APScheduler when enabled in settings.

    Registers:
        - Medicine adherence checks (configurable cron)
        - Stale plan refresh (06:00 UTC)
    """
    global _scheduler
    settings = get_settings()
    if not settings.scheduler_enabled:
        logger.info("Scheduler disabled (SCHEDULER_ENABLED=false)")
        return None

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError as exc:
        logger.error("APScheduler not installed: %s", exc)
        return None

    _scheduler = BackgroundScheduler(timezone=settings.scheduler_timezone)

    _scheduler.add_job(
        run_medicine_adherence_checks,
        "cron",
        hour=settings.medicine_check_hour,
        minute=settings.medicine_check_minute,
        id="medicine_adherence",
        replace_existing=True,
    )
    _scheduler.add_job(
        refresh_stale_plans,
        "cron",
        hour=6,
        minute=0,
        id="refresh_plans",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "Scheduler started (medicine check %02d:%02d %s)",
        settings.medicine_check_hour,
        settings.medicine_check_minute,
        settings.scheduler_timezone,
    )
    return _scheduler


def stop_scheduler() -> None:
    """Shut down the background scheduler if running."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
    _scheduler = None


# ---------------------------------------------------------------------------
# User profile persistence
# ---------------------------------------------------------------------------


def load_all_users() -> list[dict[str, Any]]:
    """Load all users from users.json."""
    settings = get_settings()
    path = settings.users_json_path
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("users", [])
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load users: %s", exc)
        return []


def load_user_profile(user_id: str) -> dict[str, Any]:
    """Load a single user profile by id."""
    for user in load_all_users():
        if user.get("id") == user_id:
            return dict(user)
    return {"id": user_id, "medications": []}


def save_user_profile(user_id: str, profile: dict[str, Any]) -> Path:
    """Persist user profile (including medications) to users.json."""
    settings = get_settings()
    path = settings.users_json_path
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {"users": []}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    users = data.get("users", [])
    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    found = False
    for index, user in enumerate(users):
        if user.get("id") == user_id:
            users[index] = {**user, **profile, "id": user_id}
            found = True
            break
    if not found:
        users.append({**profile, "id": user_id})

    data["users"] = users
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_reminder_message(profile: dict[str, Any], status: MedicineStockStatus) -> str:
    """Build deterministic WhatsApp reminder text."""
    name = profile.get("name", "there")
    days_display = (
        f"{status.days_remaining:.1f}"
        if math.isfinite(status.days_remaining)
        else "unknown"
    )
    return (
        f"💊 Medicine reminder for {name}\n\n"
        f"*{status.name}* is running low.\n"
        f"• Tablets left: {status.tablets_left:.0f}\n"
        f"• Dose per day: {status.dosage_per_day:.0f}\n"
        f"• Estimated days left: {days_display}\n\n"
        f"Please refill soon (less than {_DEFAULT_LOW_STOCK_DAYS} days supply)."
    )


def _in_reminder_cooldown(med: dict[str, Any]) -> bool:
    """True if a reminder was sent within the cooldown window."""
    last = med.get("last_reminder_at")
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        elapsed = datetime.now(timezone.utc) - last_dt
        return elapsed < timedelta(hours=_REMINDER_COOLDOWN_HOURS)
    except ValueError:
        return False


def _mark_reminder_sent(user_id: str, medication_id: str) -> None:
    """Record last_reminder_at on the medication in users.json."""
    profile = load_user_profile(user_id)
    for med in profile.get("medications") or []:
        if med.get("id") == medication_id or med.get("name") == medication_id:
            med["last_reminder_at"] = datetime.now(timezone.utc).isoformat()
            break
    save_user_profile(user_id, profile)


def _sync_session_medications(user_id: str, profile: dict[str, Any]) -> None:
    """Mirror medications into orchestrator session metadata."""
    try:
        state = load_state(user_id)
        state.user_profile = {**state.user_profile, **profile}
        state.metadata["medications"] = profile.get("medications", [])
        save_state(state)
    except Exception as exc:
        logger.debug("Session sync skipped for %s: %s", user_id, exc)


def _merge_medications(
    base: list[dict[str, Any]],
    session_meds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Session overrides profile for matching medication ids."""
    by_id = {m.get("id", m.get("name")): dict(m) for m in base}
    for med in session_meds:
        key = med.get("id", med.get("name"))
        if key in by_id:
            by_id[key].update(med)
        else:
            by_id[key] = dict(med)
    return list(by_id.values())


def _find_medication_by_id(
    medications: list[dict[str, Any]],
    medication_id: str,
) -> dict[str, Any] | None:
    for med in medications:
        if med.get("id") == medication_id or med.get("name") == medication_id:
            return med
    return None
