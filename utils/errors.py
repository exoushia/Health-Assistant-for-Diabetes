"""
Custom exceptions — FastAPI maps CopilotError subclasses to HTTP 422.

Classes:
    CopilotError         — base
    ConfigurationError   — missing/invalid .env
    OCRError             — PDF/Tesseract failures
    BiomarkerError       — parse/profile failures
    OrchestrationError   — OpenAI workflow failures
    DeliveryError        — Twilio/WhatsApp failures
"""

from typing import Any


class CopilotError(Exception):
    """Base error for health copilot operations."""

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class ConfigurationError(CopilotError):
    """Missing or invalid environment configuration."""


class OCRError(CopilotError):
    """PDF OCR or text extraction failure."""


class BiomarkerError(CopilotError):
    """Biomarker parsing or validation failure."""


class OrchestrationError(CopilotError):
    """Workflow orchestrator failure."""


class DeliveryError(CopilotError):
    """WhatsApp or localization delivery failure."""
