"""
Training router — ML model training for a specific reservoir.

Endpoints
---------
POST  /api/v1/reservoirs/{id}/train    Trigger a full training run
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from drs.db.engine import get_db
from drs.schemas.forecast import TrainRequest, TrainResponse
from drs.services import data_service, training_service

router = APIRouter(prefix="/reservoirs", tags=["training"])

_DB = Annotated[AsyncSession, Depends(get_db)]


@router.post(
    "/{reservoir_id}/train",
    response_model=TrainResponse,
    status_code=status.HTTP_200_OK,
    summary="Train a water-level prediction model",
    description=(
        "Fetches all historical observations for the reservoir, builds a "
        "feature matrix (rainfall, inflow, outflow, previous level), trains a "
        "scikit-learn LinearRegression pipeline with StandardScaler preprocessing, "
        "evaluates on a held-out time-ordered test split, serialises the pipeline "
        "to disk (joblib), and writes artifact metadata to the database."
    ),
)
async def train_reservoir_model(
    reservoir_id: str,
    payload: TrainRequest,
    db: _DB,
) -> TrainResponse:
    """
    Trigger ML training for a reservoir.

    **Minimum requirement:** at least 30 daily observations must exist for
    the reservoir before training can proceed.
    """
    try:
        reservoir = await data_service.get_reservoir(db, reservoir_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    observations = await data_service.get_observations(db, reservoir_id, limit=999_999)

    try:
        result = await training_service.train_reservoir_model(
            db,
            reservoir,
            observations,
            test_split_ratio=payload.test_split_ratio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return result
