"""
Twilio WhatsApp webhooks — inbound messages trigger the feedback loop.

Handlers:
    twilio_whatsapp_webhook — POST /webhook: parse body → handle_feedback_loop → TwiML
    twilio_status_callback  — delivery status logging (optional)
"""

from typing import Any

from fastapi import APIRouter, Form, Header, Request, Response

from tools.whatsapp import (
    empty_twiml_response,
    handle_feedback_loop,
    receive_webhook_message,
)
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Root-level router — mounted at /webhook (not under /api/v1)
webhook_router = APIRouter(tags=["whatsapp-webhook"])


@webhook_router.post("/webhook")
async def twilio_whatsapp_webhook(
    request: Request,
    x_twilio_signature: str | None = Header(default=None, alias="X-Twilio-Signature"),
) -> Response:
    """
    Twilio WhatsApp inbound webhook.

    Flow:
        1. Receive and parse Twilio form payload
        2. Run orchestrator feedback workflow for the sender
        3. Send updated meal plan (or clarification) via REST API
        4. Return empty TwiML (reply is sent asynchronously via API)

    Configure in Twilio Console:
        Messaging → WhatsApp Sandbox/Sender →
        \"When a message comes in\" → ``https://<host>/webhook``

    Sandbox:
        Set ``TWILIO_SANDBOX_MODE=true`` and join the sandbox from your device.
    """
    form = await request.form()
    payload: dict[str, Any] = dict(form)

    settings = get_settings()
    public_url = settings.twilio_webhook_url or str(request.url)

    try:
        inbound = receive_webhook_message(
            payload,
            request_url=public_url,
            signature=x_twilio_signature,
        )
    except ValueError as exc:
        logger.warning("Webhook rejected: %s", exc)
        return Response(content=empty_twiml_response(), media_type="application/xml", status_code=403)

    if not inbound.body:
        logger.info("Inbound WhatsApp with empty body from %s", inbound.from_number)
        return Response(content=empty_twiml_response(), media_type="application/xml")

    logger.info(
        "Webhook processing feedback from %s: %s",
        inbound.from_number,
        inbound.body[:80],
    )

    result = handle_feedback_loop(
        inbound,
        locale=settings.default_locale,
    )

    logger.info(
        "Webhook complete: user=%s intent=%s success=%s outbound=%s",
        result.user_id,
        result.intent,
        result.success,
        (result.outbound or {}).get("status"),
    )

    return Response(content=empty_twiml_response(), media_type="application/xml")


@webhook_router.post("/webhook/status")
async def twilio_status_callback(
    MessageSid: str = Form(default=""),
    MessageStatus: str = Form(default=""),
    ErrorCode: str = Form(default=""),
) -> dict[str, str]:
    """
    Optional Twilio delivery status callback.

    Logs sent/delivered/failed events for observability.
    """
    logger.info(
        "Twilio status: sid=%s status=%s error=%s",
        MessageSid,
        MessageStatus,
        ErrorCode or "none",
    )
    return {"status": "received"}
