"""
FastAPI application factory and lifespan handler for the Dam Regulation System.

Usage
-----
Run via the installed entry point::

    uv run drs-api

Or directly with uvicorn::

    uv run uvicorn drs.api.app:app --reload --host 127.0.0.1 --port 8000

Interactive API docs available at:
    http://127.0.0.1:8000/docs   (Swagger UI)
    http://127.0.0.1:8000/redoc  (ReDoc)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from drs.api.routers.v1 import forecast, observations, recommendations, reservoirs, training
from drs.core.config import get_settings
from drs.core.logging import clear_reservoir_context, configure_uvicorn_logging, get_logger, setup_logging
from drs.db.engine import init_db

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from starlette.requests import Request

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Runs ``init_db()`` at startup so the database schema is always in sync
    without requiring a separate migration command for development.
    """
    cfg = get_settings()
    log.info(
        "Starting {name} v{version} (debug={debug})",
        name=cfg.app_name,
        version=cfg.app_version,
        debug=cfg.debug,
    )
    await init_db()
    log.success("Application ready.")
    yield
    log.info("Application shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    cfg = get_settings()

    # Reconfigure log sinks with the level from Settings.
    setup_logging(cfg.log_level)

    application = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description=(
            "**Dam Regulation System (DRS)** — ML-driven flood forecasting "
            "and controlled-release recommendation engine for Indian reservoirs.\n\n"
            "© 2026 DRS Engineering. All Rights Reserved. Patent Pending."
        ),
        contact={
            "name": "DRS Engineering",
            "email": "engineering@drs.local",
        },
        license_info={
            "name": "All Rights Reserved",
        },
        openapi_tags=[
            {"name": "reservoirs", "description": "Reservoir CRUD operations."},
            {"name": "observations", "description": "Daily water-balance data ingestion."},
            {"name": "training", "description": "ML model training for a specific reservoir."},
            {"name": "forecast", "description": "N-day water-level forecasting."},
            {"name": "recommendations", "description": "Graduated release recommendations."},
        ],
        lifespan=lifespan,
        debug=cfg.debug,
    )

    # Route uvicorn logs through loguru so all output shares the same format
    configure_uvicorn_logging()

    # Clear reservoir log context after every request so context from one
    # request never leaks into the next on the same worker.
    @application.middleware("http")
    async def _clear_log_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        try:
            return await call_next(request)
        finally:
            clear_reservoir_context()

    # CORS — restrict in production; open for local dev
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if cfg.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    _PREFIX = "/api/v1"
    application.include_router(reservoirs.router, prefix=_PREFIX)
    application.include_router(observations.router, prefix=_PREFIX)
    application.include_router(training.router, prefix=_PREFIX)
    application.include_router(forecast.router, prefix=_PREFIX)
    application.include_router(recommendations.router, prefix=_PREFIX)

    # Health check
    @application.get("/health", tags=["health"], summary="Health check")
    async def health() -> JSONResponse:
        """Return service liveness status."""
        return JSONResponse({"status": "ok", "version": cfg.app_version})

    return application


# Module-level app instance (referenced by uvicorn and the entry point)
app = create_app()


# ---------------------------------------------------------------------------
# Entry point (drs-api console script)
# ---------------------------------------------------------------------------


def start() -> None:
    """Launch the uvicorn server.  Called by the ``drs-api`` console script."""
    cfg = get_settings()
    uvicorn.run(
        "drs.api.app:app",
        host=cfg.host,
        port=cfg.port,
        reload=cfg.debug,
        log_config=None,  # prevent uvicorn from overriding our loguru handlers
        log_level="debug" if cfg.debug else "info",
    )


if __name__ == "__main__":
    start()
