"""
Reservoirs router — CRUD for reservoir resources.

Endpoints
---------
POST   /api/v1/reservoirs              Create a new reservoir
GET    /api/v1/reservoirs              List all reservoirs
GET    /api/v1/reservoirs/{id}         Fetch a specific reservoir
PATCH  /api/v1/reservoirs/{id}         Partially update a reservoir
DELETE /api/v1/reservoirs/{id}         Delete a reservoir (and all its data)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from drs.db.engine import get_db
from drs.schemas.reservoir import ReservoirCreate, ReservoirRead, ReservoirUpdate
from drs.services import data_service

router = APIRouter(prefix="/reservoirs", tags=["reservoirs"])

_DB = Annotated[AsyncSession, Depends(get_db)]


@router.post(
    "",
    response_model=ReservoirRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new reservoir",
)
async def create_reservoir(payload: ReservoirCreate, db: _DB) -> ReservoirRead:
    """Create and persist a new reservoir record."""
    try:
        reservoir = await data_service.create_reservoir(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ReservoirRead.model_validate(reservoir)


@router.get(
    "",
    response_model=list[ReservoirRead],
    summary="List all reservoirs",
)
async def list_reservoirs(db: _DB) -> list[ReservoirRead]:
    """Return all registered reservoirs ordered alphabetically by name."""
    reservoirs = await data_service.list_reservoirs(db)
    return [ReservoirRead.model_validate(r) for r in reservoirs]


@router.get(
    "/{reservoir_id}",
    response_model=ReservoirRead,
    summary="Fetch a reservoir",
)
async def get_reservoir(reservoir_id: str, db: _DB) -> ReservoirRead:
    """Fetch a single reservoir by its UUID."""
    try:
        reservoir = await data_service.get_reservoir(db, reservoir_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ReservoirRead.model_validate(reservoir)


@router.patch(
    "/{reservoir_id}",
    response_model=ReservoirRead,
    summary="Partially update a reservoir",
)
async def update_reservoir(reservoir_id: str, payload: ReservoirUpdate, db: _DB) -> ReservoirRead:
    """Update one or more fields on an existing reservoir (PATCH semantics)."""
    try:
        reservoir = await data_service.update_reservoir(db, reservoir_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ReservoirRead.model_validate(reservoir)


@router.delete(
    "/{reservoir_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a reservoir",
)
async def delete_reservoir(reservoir_id: str, db: _DB) -> None:
    """
    Permanently delete a reservoir and all its observations and model artifacts.
    This action is irreversible.
    """
    try:
        await data_service.delete_reservoir(db, reservoir_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
