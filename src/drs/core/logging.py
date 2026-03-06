"""
Structured application logging via Loguru.

All modules import the pre-configured ``logger`` from here — never from
``loguru`` directly — so the logging format and sinks can be changed in
a single place.

Usage
-----
>>> from drs.core.logging import get_logger
>>> log = get_logger(__name__)
>>> log.info("Reservoir {name} loaded.", name="Chembarambakkam")
"""

from __future__ import annotations

import logging as _stdlib
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

# Project root: src/drs/core/logging.py → up 4 levels
_LOG_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "storage" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Request-scoped context  (async-safe via contextvars)
# ---------------------------------------------------------------------------
# Each asyncio task (i.e. each HTTP request) gets its own isolated copy of
# this var, so concurrent requests never bleed context into one another.
# Identical behaviour to structlog.contextvars.bind_contextvars().

_reservoir_ctx: ContextVar[dict[str, str] | None] = ContextVar("reservoir_ctx", default=None)


def bind_reservoir_context(*, reservoir_id: str, reservoir_name: str, city: str) -> None:
    """
    Attach reservoir identity to all log lines emitted in the current async context.

    Call once at the entry point of any operation (service function or API handler).
    Every subsequent ``log.*`` call in this task — including nested helper functions —
    will automatically carry the bound values without any extra threading.
    """
    _reservoir_ctx.set({"reservoir_id": reservoir_id, "reservoir_name": reservoir_name, "city": city})


def clear_reservoir_context() -> None:
    """Reset reservoir context (called by API middleware after each request completes)."""
    _reservoir_ctx.set(None)


def _patch_context(record: dict) -> None:
    """Loguru patcher: injects current contextvar values into every log record."""
    ctx = _reservoir_ctx.get() or {}
    name = ctx.get("reservoir_name", "")
    record["extra"]["reservoir_id"] = ctx.get("reservoir_id", "")
    record["extra"]["reservoir_name"] = name
    record["extra"]["city"] = ctx.get("city", "")
    # ctx_col: shown at end of line when reservoir context is bound, empty otherwise.
    record["extra"]["ctx_col"] = f"  [reservoir={name}]" if name else ""


# ---------------------------------------------------------------------------
# Loguru bootstrap
# ---------------------------------------------------------------------------
logger.remove()

# Wire the context patcher — runs before every log record is emitted.
# extra= pre-seeds all format tokens so they never fail on non-reservoir logs.
logger.configure(
    patcher=_patch_context,
    extra={"reservoir_id": "", "reservoir_name": "", "city": "", "ctx_col": ""},
)

# ---------------------------------------------------------------------------
# Sink format & registration
# ---------------------------------------------------------------------------
_CONSOLE_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{module: <22}</cyan> — <level>{message}</level><dim>{extra[ctx_col]}</dim>"


def _setup_sinks(level: str) -> None:
    """Register console and file sinks at the given level."""
    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        _LOG_DIR / "drs_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="00:00",
        retention="30 days",
        serialize=True,
        backtrace=True,
        diagnose=False,
        encoding="utf-8",
    )


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure log sinks with the given level.

    Call once at application startup (API lifespan or CLI root command) so
    the level from Settings takes effect.  Any previously added sinks are
    removed and replaced.

    Parameters
    ----------
    log_level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.
        Matches the ``LOG_LEVEL`` env var / ``Settings.log_level``.
    """
    logger.remove()
    _setup_sinks(log_level.upper())


# Register default INFO sinks — reconfigured at startup via setup_logging().
_setup_sinks("INFO")


# ---------------------------------------------------------------------------
# Stdlib → Loguru bridge  (uvicorn, SQLAlchemy, etc.)
# ---------------------------------------------------------------------------


class _InterceptHandler(_stdlib.Handler):
    """
    Forwards any stdlib ``logging`` record into loguru.

    The ``module`` field in the loguru record is set to the stdlib logger
    name (e.g. ``uvicorn.access``) so it aligns cleanly with DRS log lines.
    """

    def emit(self, record: _stdlib.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Override {module} with the stdlib logger name for consistent display.
        logger.patch(lambda r: r.update(module=record.name)).opt(exception=record.exc_info).log(level, record.getMessage())


def configure_uvicorn_logging() -> None:
    """
    Replace uvicorn's default stdlib handlers with the loguru intercept handler.

    Call this once during application startup (before uvicorn begins serving)
    so that all HTTP access and error lines follow the same DRS log format.
    """
    handler = _InterceptHandler()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        log = _stdlib.getLogger(name)
        log.handlers = [handler]
        log.propagate = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> Logger:
    """
    Return a Loguru logger bound to the given module name.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    Logger
        A Loguru logger instance with the ``name`` field pre-bound.

    Examples
    --------
    >>> log = get_logger(__name__)
    >>> log.info("System starting.")
    """
    return logger.bind(name=name)
