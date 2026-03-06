"""
Recommendations router — graduated release recommendation engine.

Endpoints
---------
POST  /api/v1/reservoirs/{id}/recommend    Generate a release recommendation
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from drs.db.engine import get_db
from drs.schemas.forecast import ForecastRequest
from drs.schemas.recommendation import RecommendationResponse
from drs.services import data_service, forecast_service, recommendation_service

router = APIRouter(prefix="/reservoirs", tags=["recommendations"])

_DB = Annotated[AsyncSession, Depends(get_db)]


@router.post(
    "/{reservoir_id}/recommend",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a graduated controlled-release recommendation",
    description=(
        "Runs the full DRS 7-step pipeline end-to-end:\n\n"
        "1. Load observations from DB\n"
        "2. Load trained model from disk\n"
        "3. Generate N-day forecast\n"
        "4. Predict per-day water levels\n"
        "5. Detect overflow conditions\n"
        "6. Compute graduated front-loaded release schedule\n"
        "7. Return structured recommendation report\n\n"
        "**A trained model must exist** (call ``/train`` first)."
    ),
)
async def generate_recommendation(
    reservoir_id: str,
    payload: ForecastRequest,
    db: _DB,
) -> RecommendationResponse:
    """
    Run the complete DRS pipeline and return a release recommendation.

    This is the primary operator-facing endpoint of the system.  It combines
    the forecast and recommendation steps into a single call for convenience.

    The ``release_schedule`` in the response lists the recommended daily
    release volumes (MCFt) using a descending linear-weight (front-loaded)
    algorithm to safely dissipate excess storage before overflow.
    """
    try:
        reservoir = await data_service.get_reservoir(db, reservoir_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    observations = await data_service.get_observations(db, reservoir_id, limit=999_999)

    try:
        forecast = forecast_service.generate_forecast(reservoir, observations, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_428_PRECONDITION_REQUIRED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    recommendation = recommendation_service.build_recommendation(reservoir, forecast)
    return recommendation
