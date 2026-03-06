"""
Application-wide configuration.

Loads all settings from environment variables (via a .env file at the
project root).  Pydantic-Settings is used so that every value is
type-validated and documented in one place.

Usage
-----
>>> from drs.core.config import get_settings
>>> cfg = get_settings()
>>> print(cfg.database_url)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root regardless of how the package is installed.
# src/drs/core/config.py  →  up 4 levels  →  project root
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent


class Settings(BaseSettings):
    """
    All runtime-configurable parameters for DRS.

    Every field maps 1-to-1 to an environment variable (case-insensitive).
    Defaults make the application runnable out-of-the-box with no .env file.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application metadata
    # ------------------------------------------------------------------
    app_name: str = Field(default="Dam Regulation System")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)

    # ------------------------------------------------------------------
    # HTTP server (uvicorn)
    # ------------------------------------------------------------------
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)

    # ------------------------------------------------------------------
    # Database
    # SQLite  (default): sqlite+aiosqlite:///./storage/drs.db
    # PostgreSQL (later): postgresql+asyncpg://user:pass@host/dbname
    # ------------------------------------------------------------------
    database_url: str = Field(default=f"sqlite+aiosqlite:///{(_PROJECT_ROOT / 'storage' / 'drs.db').as_posix()}")

    # ------------------------------------------------------------------
    # ML model artifact storage
    # ------------------------------------------------------------------
    models_dir: Path = Field(default=_PROJECT_ROOT / "storage" / "models")

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------
    default_forecast_days: int = Field(
        default=10,
        ge=1,
        le=365,
        description="Default number of days to forecast ahead.",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Minimum log level for console and file sinks (DEBUG / INFO / WARNING / ERROR).",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("models_dir", mode="before")
    @classmethod
    def _resolve_models_dir(cls, v: object) -> Path:
        """Expand relative paths relative to the project root."""
        path = Path(str(v))
        if not path.is_absolute():
            path = (_PROJECT_ROOT / path).resolve()
        return path

    @field_validator("database_url", mode="before")
    @classmethod
    def _resolve_sqlite_path(cls, v: object) -> str:
        """Expand relative SQLite paths (sqlite+aiosqlite:///./...) to absolute."""
        url = str(v)
        if url.startswith("sqlite") and ":///./" in url:
            relative = url.split("///./", 1)[1]
            absolute = (_PROJECT_ROOT / relative).resolve()
            # Preserve async driver prefix; use POSIX path to avoid backslashes
            driver = url.split("///")[0]
            return f"{driver}///{absolute.as_posix()}"
        return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a process-wide singleton Settings instance.

    The result is cached so the .env file is parsed exactly once per
    application lifetime.  Call ``get_settings.cache_clear()`` in tests
    to force re-evaluation with different env vars.
    """
    settings = Settings()
    # Ensure storage directories exist at startup.
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    # Ensure DB parent directory exists.
    _db_path = _extract_sqlite_path(settings.database_url)
    if _db_path:
        _db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


def _extract_sqlite_path(url: str) -> Path | None:
    """Return the filesystem Path embedded in a SQLite URL, or None."""
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if url.startswith(prefix):
            return Path(url.removeprefix(prefix))
    return None
