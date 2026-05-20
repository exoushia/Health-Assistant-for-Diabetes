"""
Thin HTTP client so Gradio can call FastAPI without importing the whole app.

Classes:
    APIClient — register, ocr upload, pipeline, feedback, health profile, e2e

Functions:
    get_client — cached singleton with base URL from settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class APIClient:
    """Thin wrapper around the FastAPI meal planner API."""

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.api_base_url).rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(self._url("/api/v1/health"))
            r.raise_for_status()
            return r.json()

    def register(
        self,
        user_id: str,
        name: str,
        phone: str,
        diabetes_type: str = "type_2",
        locale: str = "en",
    ) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                self._url("/api/v1/register"),
                json={
                    "user_id": user_id,
                    "name": name,
                    "phone": phone,
                    "diabetes_type": diabetes_type,
                    "locale": locale,
                },
            )
            r.raise_for_status()
            return r.json()

    def upload_pdf(self, user_id: str, pdf_path: str | Path) -> dict[str, Any]:
        path = Path(pdf_path)
        with httpx.Client(timeout=self.timeout) as client:
            with path.open("rb") as handle:
                r = client.post(
                    self._url("/api/v1/ocr/upload"),
                    data={"user_id": user_id},
                    files={"file": (path.name, handle, "application/pdf")},
                )
            r.raise_for_status()
            return r.json()

    def health_profile(self, user_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self._url(f"/api/v1/users/{user_id}/health-profile"))
            r.raise_for_status()
            return r.json()

    def run_pipeline(
        self,
        user_id: str,
        *,
        message: str = "Create a 7-day diabetes-friendly meal plan",
        locale: str = "en",
        pdf_path: str | Path | None = None,
        send_whatsapp: bool = False,
        phone: str | None = None,
    ) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            if pdf_path:
                path = Path(pdf_path)
                with path.open("rb") as handle:
                    r = client.post(
                        self._url("/api/v1/pipeline/upload"),
                        data={
                            "user_id": user_id,
                            "message": message,
                            "locale": locale,
                            "send_whatsapp": str(send_whatsapp).lower(),
                            "phone": phone or "",
                        },
                        files={"file": (path.name, handle, "application/pdf")},
                    )
            else:
                r = client.post(
                    self._url("/api/v1/pipeline"),
                    json={
                        "user_id": user_id,
                        "message": message,
                        "locale": locale,
                        "send_whatsapp": send_whatsapp,
                        "phone": phone,
                    },
                )
            r.raise_for_status()
            return r.json()

    def feedback(
        self,
        user_id: str,
        feedback: str,
        locale: str = "en",
    ) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                self._url("/api/v1/feedback"),
                json={"user_id": user_id, "feedback": feedback, "locale": locale},
            )
            r.raise_for_status()
            return r.json()

    def send_whatsapp(
        self,
        user_id: str,
        phone: str,
        locale: str = "en",
    ) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                self._url("/api/v1/whatsapp/send"),
                json={
                    "user_id": user_id,
                    "phone": phone,
                    "locale": locale,
                    "send": True,
                },
            )
            r.raise_for_status()
            return r.json()


def get_client() -> APIClient:
    return APIClient()
