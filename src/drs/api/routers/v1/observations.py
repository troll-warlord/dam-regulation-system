"""
Observations router — daily water-balance data ingestion.

Endpoints
---------
POST  /api/v1/reservoirs/{id}/observations        Add a single observation
POST  /api/v1/reservoirs/{id}/observations/bulk   Bulk-add observations
GET   /api/v1/reservoirs/{id}/observations        Query observations (filterable)
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from drs.db.engine import get_db
from drs.schemas.observation import (
    ObservationBulkCreate,
    ObservationCreate,
    ObservationRead,
)
from drs.services import data_service

router = APIRouter(prefix="/reservoirs", tags=["observations"])

_DB = Annotated[AsyncSession, Depends(get_db)]


@router.post(
    "/{reservoir_id}/observations",
    response_model=ObservationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add a single daily observation",
)
async def add_observation(reservoir_id: str, payload: ObservationCreate, db: _DB) -> ObservationRead:
    """Insert one daily water-balance reading for a reservoir."""
    try:
        obs = await data_service.add_observation(db, reservoir_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ObservationRead.model_validate(obs)


@router.post(
    "/{reservoir_id}/observations/bulk",
    response_model=list[ObservationRead],
    status_code=status.HTTP_201_CREATED,
    summary="Bulk-add daily observations",
)
async def bulk_add_observations(reservoir_id: str, payload: ObservationBulkCreate, db: _DB) -> list[ObservationRead]:
    """
    Insert multiple daily observations in a single transaction.
    Duplicate (date, reservoir) pairs are silently skipped.
    """
    try:
        obs_list = await data_service.bulk_add_observations(db, reservoir_id, payload.observations)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return [ObservationRead.model_validate(o) for o in obs_list]


@router.get(
    "/{reservoir_id}/observations",
    response_model=list[ObservationRead],
    summary="Query observations",
)
async def get_observations(
    reservoir_id: str,
    db: _DB,
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=10_000),
    offset: int = Query(default=0, ge=0),
) -> list[ObservationRead]:
    """Return observations for a reservoir with optional date-range filtering."""
    try:
        obs_list = await data_service.get_observations(
            db,
            reservoir_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return [ObservationRead.model_validate(o) for o in obs_list]
