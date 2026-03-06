"""
Data access service — CRUD operations for Reservoir and Observation records.

All database interaction is centralised here so that routers and CLI
commands never touch SQLAlchemy directly.

Design notes
------------
- Every public function accepts an ``AsyncSession`` injected by the caller
  (FastAPI dependency or CLI context).
- Raises ``KeyError`` for "not found" and ``ValueError`` for constraint
  violations so callers can map them to appropriate HTTP status codes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from drs.core.logging import get_logger
from drs.db.models import Observation, Reservoir

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from sqlalchemy.ext.asyncio import AsyncSession

    from drs.schemas.observation import ObservationCreate
    from drs.schemas.reservoir import ReservoirCreate, ReservoirUpdate

log = get_logger(__name__)


# ===========================================================================
# Reservoir CRUD
# ===========================================================================


async def create_reservoir(session: AsyncSession, payload: ReservoirCreate) -> Reservoir:
    """
    Persist a new reservoir record.

    Parameters
    ----------
    session:
        Active async database session.
    payload:
        Validated :class:`~drs.schemas.reservoir.ReservoirCreate` data.

    Returns
    -------
    Reservoir
        The newly created ORM instance (with DB-generated ``id``).

    Raises
    ------
    ValueError
        If a reservoir with the same ``name`` already exists.
    """
    existing = await _get_reservoir_by_name(session, payload.name)
    if existing:
        raise ValueError(f"Reservoir with name '{payload.name}' already exists.")

    reservoir = Reservoir(**payload.model_dump())
    session.add(reservoir)
    await session.commit()
    await session.refresh(reservoir)
    log.info("Created reservoir. id={id} name={name}", id=reservoir.id, name=reservoir.name)
    return reservoir


async def get_reservoir(session: AsyncSession, reservoir_id: str) -> Reservoir:
    """
    Fetch a reservoir by UUID.

    Raises
    ------
    KeyError
        If no reservoir matches ``reservoir_id``.
    """
    result = await session.get(Reservoir, reservoir_id)
    if result is None:
        raise KeyError(f"Reservoir not found: {reservoir_id!r}")
    return result


async def list_reservoirs(session: AsyncSession) -> Sequence[Reservoir]:
    """Return all reservoirs ordered by name."""
    rows = await session.execute(select(Reservoir).order_by(Reservoir.name))
    return rows.scalars().all()


async def update_reservoir(session: AsyncSession, reservoir_id: str, payload: ReservoirUpdate) -> Reservoir:
    """
    Partially update a reservoir (PATCH semantics).

    Only non-``None`` fields from ``payload`` are applied.

    Raises
    ------
    KeyError
        If the reservoir does not exist.
    """
    reservoir = await get_reservoir(session, reservoir_id)
    update_data = payload.model_dump(exclude_none=True)
    for field, value in update_data.items():
        setattr(reservoir, field, value)
    await session.commit()
    await session.refresh(reservoir)
    log.info("Updated reservoir. id={id}", id=reservoir_id)
    return reservoir


async def delete_reservoir(session: AsyncSession, reservoir_id: str) -> None:
    """
    Delete a reservoir and all cascade-linked observations + model artifacts.

    Raises
    ------
    KeyError
        If the reservoir does not exist.
    """
    reservoir = await get_reservoir(session, reservoir_id)
    await session.delete(reservoir)
    await session.commit()
    log.info("Deleted reservoir. id={id}", id=reservoir_id)


# ===========================================================================
# Observation CRUD
# ===========================================================================


async def add_observation(session: AsyncSession, reservoir_id: str, payload: ObservationCreate) -> Observation:
    """
    Insert a single observation for a reservoir.

    Raises
    ------
    KeyError
        If the reservoir does not exist.
    ValueError
        If an observation for this ``(reservoir_id, date)`` already exists.
    """
    await get_reservoir(session, reservoir_id)  # Ensures reservoir exists
    obs = Observation(reservoir_id=reservoir_id, **payload.model_dump())
    session.add(obs)
    try:
        await session.commit()
    except Exception as exc:
        await session.rollback()
        raise ValueError(f"Observation for date {payload.date} already exists for reservoir {reservoir_id!r}. Detail: {exc}") from exc
    await session.refresh(obs)
    return obs


async def bulk_add_observations(
    session: AsyncSession,
    reservoir_id: str,
    payloads: list[ObservationCreate],
) -> list[Observation]:
    """
    Insert multiple observations for a reservoir in a single transaction.

    Existing ``(reservoir_id, date)`` pairs are silently skipped (upsert-lite).

    Parameters
    ----------
    session:
        Active async database session.
    reservoir_id:
        Target reservoir UUID.
    payloads:
        List of validated observation payloads.

    Returns
    -------
    list[Observation]
        Only the newly inserted observations.
    """
    await get_reservoir(session, reservoir_id)

    existing_dates: set[date] = set((await session.execute(select(Observation.date).where(Observation.reservoir_id == reservoir_id))).scalars().all())

    new_obs = [Observation(reservoir_id=reservoir_id, **p.model_dump()) for p in payloads if p.date not in existing_dates]
    session.add_all(new_obs)
    await session.commit()
    log.info(
        "Bulk-inserted {n} observations for reservoir {id} (skipped {s} duplicates).",
        n=len(new_obs),
        id=reservoir_id,
        s=len(payloads) - len(new_obs),
    )
    return new_obs


async def get_observations(
    session: AsyncSession,
    reservoir_id: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> Sequence[Observation]:
    """
    Query observations for a reservoir with optional date-range filtering.

    Parameters
    ----------
    session:
        Active async database session.
    reservoir_id:
        Target reservoir UUID.
    start_date:
        If provided, restrict to observations on or after this date.
    end_date:
        If provided, restrict to observations on or before this date.
    limit:
        Maximum number of records to return.
    offset:
        Number of records to skip (for pagination).
    """
    query = select(Observation).where(Observation.reservoir_id == reservoir_id).order_by(Observation.date)
    if start_date:
        query = query.where(Observation.date >= start_date)
    if end_date:
        query = query.where(Observation.date <= end_date)
    query = query.limit(limit).offset(offset)
    rows = await session.execute(query)
    return rows.scalars().all()


async def get_latest_observation(session: AsyncSession, reservoir_id: str) -> Observation | None:
    """Return the most recent observation for a reservoir, or None."""
    row = await session.execute(select(Observation).where(Observation.reservoir_id == reservoir_id).order_by(Observation.date.desc()).limit(1))
    return row.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _get_reservoir_by_name(session: AsyncSession, name: str) -> Reservoir | None:
    result = await session.execute(select(Reservoir).where(Reservoir.name == name))
    return result.scalar_one_or_none()
