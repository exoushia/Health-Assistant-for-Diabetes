"""
Settings from .env — API keys, paths, Twilio, OCR, scheduler, enrichment tuning.

Classes:
    Settings — pydantic-settings model for all env vars

Functions:
    get_settings — cached Settings singleton (import this everywhere)
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: parent of utils/ (run app.py from here)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central settings for API, UI, AI, messaging, and data backends."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Health Meal Planner", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Server
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_base_url: str = Field(default="http://127.0.0.1:8000", alias="API_BASE_URL")
    gradio_host: str = Field(default="0.0.0.0", alias="GRADIO_HOST")
    gradio_port: int = Field(default=7860, alias="GRADIO_PORT")

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # Recipe enrichment pipeline
    enrich_max_retries: int = Field(default=5, alias="ENRICH_MAX_RETRIES")
    enrich_retry_base_delay: float = Field(default=2.0, alias="ENRICH_RETRY_BASE_DELAY")
    enrich_min_request_interval: float = Field(default=0.5, alias="ENRICH_MIN_REQUEST_INTERVAL")

    # Twilio WhatsApp
    twilio_account_sid: str = Field(default="", alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(default="", alias="TWILIO_AUTH_TOKEN")
    twilio_whatsapp_from: str = Field(default="", alias="TWILIO_WHATSAPP_FROM")
    twilio_sandbox_mode: bool = Field(default=True, alias="TWILIO_SANDBOX_MODE")
    twilio_sandbox_from: str = Field(
        default="whatsapp:+14155238886",
        alias="TWILIO_SANDBOX_FROM",
    )
    twilio_max_retries: int = Field(default=3, alias="TWILIO_MAX_RETRIES")
    twilio_retry_base_delay: float = Field(default=1.5, alias="TWILIO_RETRY_BASE_DELAY")
    twilio_validate_webhook_signature: bool = Field(
        default=False,
        alias="TWILIO_VALIDATE_WEBHOOK_SIGNATURE",
    )
    twilio_webhook_url: str = Field(
        default="",
        alias="TWILIO_WEBHOOK_URL",
        description="Public URL Twilio posts to (for signature validation)",
    )

    # Data backend
    data_backend: Literal["json", "mongodb"] = Field(default="json", alias="DATA_BACKEND")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    users_json_path: Path = Field(
        default=PROJECT_ROOT / "data" / "users.json",
        alias="USERS_JSON_PATH",
    )

    # MongoDB
    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_db_name: str = Field(default="health_meal_planner", alias="MONGODB_DB_NAME")

    # Recipes
    recipes_raw_dir: Path = Field(default=PROJECT_ROOT / "recipes" / "raw", alias="RECIPES_RAW_DIR")
    recipes_processed_dir: Path = Field(
        default=PROJECT_ROOT / "recipes" / "processed",
        alias="RECIPES_PROCESSED_DIR",
    )
    recipes_schemas_dir: Path = Field(
        default=PROJECT_ROOT / "recipes" / "schemas",
        alias="RECIPES_SCHEMAS_DIR",
    )

    # OCR
    ocr_pdf_dpi: int = Field(default=200, alias="OCR_PDF_DPI")
    ocr_tesseract_lang: str = Field(default="eng", alias="OCR_TESSERACT_LANG")
    tesseract_cmd: str | None = Field(default=None, alias="TESSERACT_CMD")

    # Localization (Sarvam + OpenAI fallback)
    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")
    sarvam_translate_url: str = Field(
        default="https://api.sarvam.ai/translate",
        alias="SARVAM_TRANSLATE_URL",
    )
    sarvam_translate_model: str = Field(default="mayura:v1", alias="SARVAM_TRANSLATE_MODEL")
    sarvam_max_chunk_size: int = Field(default=1000, alias="SARVAM_MAX_CHUNK_SIZE")
    default_locale: str = Field(default="en", alias="DEFAULT_LOCALE")

    # Scheduler
    scheduler_enabled: bool = Field(default=False, alias="SCHEDULER_ENABLED")
    scheduler_timezone: str = Field(default="UTC", alias="SCHEDULER_TIMEZONE")
    medicine_low_stock_days: float = Field(default=3.0, alias="MEDICINE_LOW_STOCK_DAYS")
    medicine_check_hour: int = Field(default=8, alias="MEDICINE_CHECK_HOUR")
    medicine_check_minute: int = Field(default=0, alias="MEDICINE_CHECK_MINUTE")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
