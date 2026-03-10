"""
Async SQLAlchemy engine and session factory.

This module is the single entry point for all database connectivity in DRS.
It creates:
- ``engine``   — an :class:`~sqlalchemy.ext.asyncio.AsyncEngine` instance.
- ``AsyncSessionLocal`` — a session factory for dependency injection.
- ``Base``     — the declarative base shared by all ORM models.

Migrate to PostgreSQL by changing ``DATABASE_URL`` in .env to::

    postgresql+asyncpg://user:password@host:5432/drs

No other code changes are required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from drs.core.config import get_settings
from drs.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

log = get_logger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base for all DRS ORM models."""


def _scrub_url(url: str) -> str:
    """Remove password from a DB URL before logging — safe for PostgreSQL."""
    parsed = urlparse(url)
    if parsed.password:
        safe_netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
        return urlunparse(parsed._replace(netloc=safe_netloc))
    return url


def _build_engine() -> AsyncEngine:
    """Build and return an ``AsyncEngine`` from current settings."""
    cfg = get_settings()
    connect_args: dict[str, object] = {}

    # SQLite requires check_same_thread=False for async use.
    if cfg.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    log.info("Creating database engine. url={url}", url=_scrub_url(cfg.database_url))
    return create_async_engine(
        cfg.database_url,
        echo=cfg.debug,
        connect_args=connect_args,
        pool_pre_ping=True,
    )


engine = _build_engine()

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """
    Create all database tables if they do not already exist.

    Call this once at application startup (inside the FastAPI lifespan handler
    or the CLI ``init`` command).
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database schema verified / created.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session and closes it afterward.

    Usage
    -----
    >>> async def my_route(db: AsyncSession = Depends(get_db)): ...
    """
    async with AsyncSessionLocal() as session:
        yield session
