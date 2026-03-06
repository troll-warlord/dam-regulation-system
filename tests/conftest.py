"""
Shared pytest fixtures for the DRS test suite.

Uses an in-memory SQLite database so tests are:
- Fully isolated (no shared state between runs)
- Fast (no disk I/O)
- Reproducible (same seed data every time)

Fixture hierarchy
-----------------
engine → tables → db_session → (test functions)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from drs.db.engine import Base
from drs.db.models import Observation, Reservoir

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# ---------------------------------------------------------------------------
# In-memory database
# ---------------------------------------------------------------------------

_IN_MEMORY_URL = "sqlite+aiosqlite://"  # Pure in-memory, not shared between connections


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:

    """
    Yield a fresh AsyncSession backed by an in-memory SQLite database.

    Every test function gets its own empty database to prevent state leakage.
    """
    engine = create_async_engine(_IN_MEMORY_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


# ---------------------------------------------------------------------------
# Sample data builders
# ---------------------------------------------------------------------------


def make_reservoir(**kwargs: Any) -> Reservoir:
    """Return an unsaved Reservoir ORM instance with sensible defaults."""
    defaults: dict[str, Any] = {
        "name": "Test Reservoir",
        "city": "Test City",
        "state": "Test State",
        "max_capacity_mcft": 3000.0,
        "alert_level_mcft": 2800.0,
        "dead_storage_mcft": 100.0,
        "coordinates_lat": 13.0,
        "coordinates_lon": 80.0,
    }
    defaults.update(kwargs)
    return Reservoir(**defaults)


def make_observations(
    reservoir_id: str,
    *,
    n: int = 60,
    base_level: float = 1500.0,
    rainfall_pattern: str = "flat",
) -> list[Observation]:
    """
    Generate a list of synthetic Observation ORM instances.

    Parameters
    ----------
    reservoir_id:
        UUID of the parent reservoir.
    n:
        Number of daily observations to generate.
    base_level:
        Starting water level (MCFt).
    rainfall_pattern:
        ``'flat'``     — constant 10 mm/day
        ``'monsoon'``  — 0 mm for first half, 80 mm for second half
        ``'overflow'`` — high rainfall that drives the level above 2900 MCFt
    """
    observations = []
    level = base_level
    start = date(2025, 1, 1)

    for i in range(n):
        d = start + timedelta(days=i)

        if rainfall_pattern == "flat":
            rain = 10.0
        elif rainfall_pattern == "monsoon":
            rain = 0.0 if i < n // 2 else 80.0
        elif rainfall_pattern == "overflow":
            rain = 150.0
        else:
            rain = 10.0

        inflow = rain * 0.6
        outflow = 8.0
        new_level = min(level + inflow - outflow, 2990.0)
        new_level = max(new_level, 100.0)

        observations.append(
            Observation(
                reservoir_id=reservoir_id,
                date=d,
                water_level_mcft=round(new_level, 2),
                rainfall_mm=rain,
                inflow_mcft=round(inflow, 2),
                outflow_mcft=outflow,
            )
        )
        level = new_level

    return observations


@pytest_asyncio.fixture
async def seeded_reservoir(db_session: AsyncSession) -> Reservoir:
    """A Reservoir with 60 days of flat observations — ready for training/forecast."""
    reservoir = make_reservoir()
    db_session.add(reservoir)
    await db_session.flush()

    obs = make_observations(reservoir.id, n=60, rainfall_pattern="flat")
    db_session.add_all(obs)
    await db_session.commit()
    await db_session.refresh(reservoir)
    return reservoir
