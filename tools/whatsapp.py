"""
Twilio WhatsApp — send plans, receive feedback, close the loop via orchestrator.

Classes:
    TwilioInboundMessage — parsed webhook fields
    WebhookHandleResult  — feedback processing outcome

Functions:
    parse_twilio_payload       — form body → structured message
    receive_webhook_message    — validate signature (optional), parse inbound
    send_whatsapp_message      — outbound with retries
    handle_feedback_loop       — inbound text → WorkflowOrchestrator → reply
    resolve_user_id_from_phone — map E.164 to user_id from users.json
    format_meal_plan_for_whatsapp — English compact plan text
    empty_twiml_response       — no-reply TwiML for webhook ACK
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Twilio transient error codes worth retrying
_RETRYABLE_TWILIO_CODES = {20429, 20500, 20503, 63016, 63018}


@dataclass
class TwilioInboundMessage:
    """Normalized inbound WhatsApp message from a Twilio webhook."""

    from_number: str
    to_number: str
    body: str
    message_sid: str
    num_media: int = 0
    profile_name: str | None = None
    raw_payload: dict[str, str] | None = None

    @property
    def phone_e164(self) -> str:
        """Sender phone without whatsapp: prefix."""
        return self.from_number.replace("whatsapp:", "").strip()


@dataclass
class WebhookHandleResult:
    """Result of processing an inbound webhook through the feedback loop."""

    success: bool
    user_id: str | None
    intent: str | None
    orchestrator_result: dict[str, Any] | None
    outbound: dict[str, Any] | None
    error: str | None = None


def parse_twilio_payload(payload: dict[str, Any]) -> TwilioInboundMessage:
    """
    Parse Twilio's application/x-www-form-urlencoded webhook body.

    Twilio sends fields such as From, To, Body, MessageSid, NumMedia, ProfileName.
    See: https://www.twilio.com/docs/messaging/guides/webhook-request

    Args:
        payload: Flat dict of form fields (str keys and values).

    Returns:
        TwilioInboundMessage with normalized addresses and text body.
    """
    normalized = {str(k): str(v) if v is not None else "" for k, v in payload.items()}

    from_number = normalize_whatsapp_address(normalized.get("From", ""))
    to_number = normalize_whatsapp_address(normalized.get("To", ""))
    body = normalized.get("Body", "").strip()

    try:
        num_media = int(normalized.get("NumMedia", "0") or "0")
    except ValueError:
        num_media = 0

    message = TwilioInboundMessage(
        from_number=from_number,
        to_number=to_number,
        body=body,
        message_sid=normalized.get("MessageSid", ""),
        num_media=num_media,
        profile_name=normalized.get("ProfileName") or None,
        raw_payload=normalized,
    )

    logger.debug(
        "Parsed Twilio payload: from=%s sid=%s body_len=%d",
        message.from_number,
        message.message_sid,
        len(body),
    )
    return message


def receive_webhook_message(
    payload: dict[str, Any],
    *,
    validate_signature: bool | None = None,
    request_url: str | None = None,
    signature: str | None = None,
) -> TwilioInboundMessage:
    """
    Receive and validate an inbound Twilio WhatsApp webhook message.

    Args:
        payload: Raw form payload from FastAPI ``await request.form()``.
        validate_signature: When True, verify X-Twilio-Signature (default from settings).
        request_url: Full public URL Twilio posted to (required for signature validation).
        signature: Value of X-Twilio-Signature header.

    Returns:
        Parsed TwilioInboundMessage.

    Raises:
        ValueError: If signature validation fails or Body is missing for user messages.
    """
    settings = get_settings()
    message = parse_twilio_payload(payload)

    should_validate = (
        validate_signature
        if validate_signature is not None
        else settings.twilio_validate_webhook_signature
    )

    if should_validate:
        if not request_url or not signature:
            raise ValueError("request_url and signature required for Twilio validation")
        if not _validate_twilio_signature(request_url, payload, signature):
            logger.warning("Invalid Twilio webhook signature for sid=%s", message.message_sid)
            raise ValueError("Invalid Twilio signature")

    if not message.from_number:
        raise ValueError("Missing From in Twilio payload")

    logger.info(
        "Inbound WhatsApp: from=%s sid=%s sandbox=%s",
        message.from_number,
        message.message_sid,
        settings.twilio_sandbox_mode,
    )
    return message


def send_whatsapp_message(
    to: str,
    body: str,
    *,
    media_url: str | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    """
    Send an outbound WhatsApp message via Twilio with graceful retry handling.

    Args:
        to: Recipient E.164 (e.g. +9198...) or whatsapp:+9198...
        body: Message text (keep under 1600 chars for best delivery).
        media_url: Optional media URL for MMS-style WhatsApp media.
        max_retries: Override default retry count from settings.

    Returns:
        Dict with status (sent|skipped|error), sid, and optional detail.
    """
    settings = get_settings()

    if not _twilio_configured():
        logger.warning("Twilio credentials not configured; message not sent")
        return {"status": "skipped", "reason": "missing_twilio_credentials"}

    if not body or not body.strip():
        return {"status": "skipped", "reason": "empty_body"}

    to_addr = normalize_whatsapp_address(to)
    from_addr = _resolve_from_address()

    retries = max_retries if max_retries is not None else settings.twilio_max_retries
    last_error: str | None = None

    for attempt in range(1, retries + 1):
        try:
            from twilio.rest import Client
            from twilio.base.exceptions import TwilioRestException

            client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
            kwargs: dict[str, Any] = {
                "from_": from_addr,
                "to": to_addr,
                "body": truncate_whatsapp_body(body),
            }
            if media_url:
                kwargs["media_url"] = [media_url]

            message = client.messages.create(**kwargs)
            logger.info(
                "WhatsApp sent: to=%s sid=%s sandbox=%s attempt=%d",
                to_addr,
                message.sid,
                settings.twilio_sandbox_mode,
                attempt,
            )
            return {
                "status": "sent",
                "sid": message.sid,
                "to": to_addr,
                "from": from_addr,
                "sandbox": settings.twilio_sandbox_mode,
            }

        except Exception as exc:
            last_error = str(exc)
            retryable = False
            try:
                from twilio.base.exceptions import TwilioRestException

                if isinstance(exc, TwilioRestException):
                    last_error = f"{exc.code}: {exc.msg}"
                    retryable = exc.code in _RETRYABLE_TWILIO_CODES
            except ImportError:
                pass

            if attempt < retries and retryable:
                delay = settings.twilio_retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Twilio send retry %d/%d in %.1fs: %s",
                    attempt,
                    retries,
                    delay,
                    last_error,
                )
                time.sleep(delay)
                continue

            logger.exception("WhatsApp send failed after %d attempts: %s", attempt, last_error)
            return {"status": "error", "detail": last_error, "attempts": attempt}

    return {"status": "error", "detail": last_error or "unknown"}


def handle_feedback_loop(
    inbound: TwilioInboundMessage,
    *,
    user_id: str | None = None,
    locale: str | None = None,
) -> WebhookHandleResult:
    """
    Run the conversational feedback loop for an inbound WhatsApp message.

    Pipeline:
        1. Resolve user_id from phone (or explicit override)
        2. Trigger WorkflowOrchestrator.process_feedback()
        3. Send updated meal plan (or clarification) back via WhatsApp

    Args:
        inbound: Parsed inbound message.
        user_id: Optional user_id override (skips phone lookup).
        locale: Locale for outbound formatting (default from settings).

    Returns:
        WebhookHandleResult with orchestrator and send metadata.
    """
    from orchestrator.agent import WorkflowOrchestrator

    settings = get_settings()
    locale = locale or settings.default_locale

    if not inbound.body:
        reply = send_whatsapp_message(
            inbound.from_number,
            "Please send feedback about your meal plan (e.g. 'No paneer tonight').",
        )
        return WebhookHandleResult(
            success=True,
            user_id=user_id,
            intent=None,
            orchestrator_result=None,
            outbound=reply,
        )

    resolved_user_id = user_id or resolve_user_id_from_phone(inbound.phone_e164)
    if not resolved_user_id:
        resolved_user_id = f"whatsapp_{_phone_to_slug(inbound.phone_e164)}"
        logger.info("Unknown phone %s → ephemeral user_id=%s", inbound.phone_e164, resolved_user_id)

    try:
        orchestrator = WorkflowOrchestrator()
        result = orchestrator.process_feedback(resolved_user_id, inbound.body)

        outbound_text = result.get("whatsapp_message")
        if not outbound_text:
            plan = result.get("meal_plan") or {}
            outbound_text = format_meal_plan_for_whatsapp(plan, locale=locale)

        if locale.lower().startswith("hi") and result.get("meal_plan", {}).get("whatsapp_text_hi"):
            outbound_text = result["meal_plan"]["whatsapp_text_hi"]

        send_result = send_whatsapp_message(inbound.from_number, outbound_text)

        return WebhookHandleResult(
            success=send_result.get("status") == "sent",
            user_id=resolved_user_id,
            intent=result.get("intent"),
            orchestrator_result=result,
            outbound=send_result,
        )

    except Exception as exc:
        logger.exception("Feedback loop failed for user=%s: %s", resolved_user_id, exc)
        err_reply = send_whatsapp_message(
            inbound.from_number,
            "Sorry, we could not update your meal plan right now. Please try again shortly.",
        )
        return WebhookHandleResult(
            success=False,
            user_id=resolved_user_id,
            intent=None,
            orchestrator_result=None,
            outbound=err_reply,
            error=str(exc),
        )


def resolve_user_id_from_phone(phone_e164: str) -> str | None:
    """
    Look up user id from data/users.json by phone number.

    Args:
        phone_e164: E.164 number without whatsapp: prefix.

    Returns:
        user id or None if not registered.
    """
    settings = get_settings()
    path = settings.users_json_path
    if not path.is_file():
        return None

    normalized = _normalize_phone_digits(phone_e164)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for user in data.get("users", []):
            user_phone = _normalize_phone_digits(str(user.get("phone", "")))
            if user_phone and user_phone == normalized:
                return user.get("id")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load users for phone lookup: %s", exc)
    return None


def format_meal_plan_for_whatsapp(
    meal_plan: dict[str, Any],
    locale: str | None = None,
) -> str:
    """
    Format meal plan dict as concise WhatsApp-friendly text.

    Uses Hindi localization when locale is hi/hi-IN.
    """
    from tools.localization import format_meal_plan_whatsapp_en, localize_meal_plan_whatsapp

    settings = get_settings()
    locale = locale or meal_plan.get("locale") or settings.default_locale

    if locale.lower().startswith("hi"):
        if meal_plan.get("whatsapp_text_hi"):
            return meal_plan["whatsapp_text_hi"]
        if meal_plan.get("whatsapp_text"):
            return meal_plan["whatsapp_text"]
        return localize_meal_plan_whatsapp(meal_plan, locale=locale)

    if meal_plan.get("whatsapp_text_en"):
        return meal_plan["whatsapp_text_en"]
    content = meal_plan.get("content") or meal_plan.get("message")
    if content:
        return f"🍽 Your meal plan update:\n\n{str(content)[:1500]}"
    return format_meal_plan_whatsapp_en(meal_plan)


def empty_twiml_response() -> str:
    """Minimal TwiML for webhook endpoints that reply via REST API."""
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response></Response>"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def normalize_whatsapp_address(number: str) -> str:
    """Ensure Twilio whatsapp: prefix on an E.164 number."""
    number = number.strip()
    if not number:
        return ""
    if number.startswith("whatsapp:"):
        return number
    if not number.startswith("+"):
        number = f"+{number}"
    return f"whatsapp:{number}"


def truncate_whatsapp_body(body: str, max_chars: int = 1600) -> str:
    """Truncate message to WhatsApp-friendly length."""
    if len(body) <= max_chars:
        return body
    return body[: max_chars - 15].rstrip() + "\n… (continued)"


def _twilio_configured() -> bool:
    settings = get_settings()
    sid = settings.twilio_account_sid or ""
    token = settings.twilio_auth_token or ""
    if not sid or not token:
        return False
    placeholders = ("your-twilio", "changeme", "account-sid")
    return not any(p in sid.lower() or p in token.lower() for p in placeholders)


def _resolve_from_address() -> str:
    """Return sandbox or production WhatsApp sender."""
    settings = get_settings()
    from_addr = settings.twilio_whatsapp_from
    if settings.twilio_sandbox_mode and settings.twilio_sandbox_from:
        from_addr = settings.twilio_sandbox_from
    return normalize_whatsapp_address(from_addr)


def _validate_twilio_signature(
    url: str,
    params: dict[str, Any],
    signature: str,
) -> bool:
    """Validate X-Twilio-Signature on inbound webhooks."""
    try:
        from twilio.request_validator import RequestValidator

        settings = get_settings()
        validator = RequestValidator(settings.twilio_auth_token)
        flat = {str(k): str(v) for k, v in params.items()}
        return validator.validate(url, flat, signature)
    except Exception as exc:
        logger.warning("Signature validation error: %s", exc)
        return False


def _normalize_phone_digits(phone: str) -> str:
    return "".join(c for c in phone if c.isdigit())


def _phone_to_slug(phone: str) -> str:
    digits = _normalize_phone_digits(phone)
    return digits[-12:] if len(digits) > 12 else digits
