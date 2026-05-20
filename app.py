"""
FastAPI application entry — wires routes, webhooks, scheduler, and error handling.

Functions:
    lifespan          — start/stop APScheduler on app boot/shutdown
    create_app        — build FastAPI app with CORS, routers, CopilotError handler
    main              — run Uvicorn using settings from utils.config
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from api.webhook import webhook_router
from scheduler.jobs import start_scheduler, stop_scheduler
from utils.config import get_settings
from utils.errors import CopilotError
from utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: log config, start scheduler; shutdown: cleanup."""
    settings = get_settings()
    logger.info(
        "Health Copilot API starting | env=%s | data=%s | locale=%s",
        settings.app_env,
        settings.data_dir,
        settings.default_locale,
    )
    start_scheduler()
    yield
    stop_scheduler()
    logger.info("Health Copilot API shutdown")


def create_app() -> FastAPI:
    """Factory for the FastAPI application."""
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        description="Modular AI health copilot — OCR, biomarkers, meal plans, WhatsApp",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(CopilotError)
    async def copilot_error_handler(request: Request, exc: CopilotError) -> JSONResponse:
        logger.warning("CopilotError on %s: %s", request.url.path, exc.message)
        return JSONResponse(
            status_code=422,
            content={"error": exc.message, "detail": exc.detail, "type": "CopilotError"},
        )

    @application.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": type(exc).__name__},
        )

    application.include_router(router)
    application.include_router(webhook_router)
    return application


app = create_app()


def main() -> None:
    """Run uvicorn with settings from environment."""
    settings = get_settings()
    logger.info("Starting API on %s:%s", settings.api_host, settings.api_port)
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
