"""
Forecast router — N-day water-level forecasting.

Endpoints
---------
POST  /api/v1/reservoirs/{id}/forecast    Generate an N-day forecast
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from drs.db.engine import get_db
from drs.schemas.forecast import ForecastRequest, ForecastResponse
from drs.services import data_service, forecast_service

router = APIRouter(prefix="/reservoirs", tags=["forecast"])

_DB = Annotated[AsyncSession, Depends(get_db)]


@router.post(
    "/{reservoir_id}/forecast",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate an N-day water-level forecast",
    description=(
        "Loads the trained model for the reservoir and rolls it forward "
        "``horizon_days`` days using a sinusoidal climatological rainfall proxy. "
        "Returns per-day predicted levels, capacity utilisation, and overflow "
        "detection. **A trained model must exist** (call ``/train`` first)."
    ),
)
async def generate_forecast(
    reservoir_id: str,
    payload: ForecastRequest,
    db: _DB,
) -> ForecastResponse:
    """
    Generate a deterministic N-day water-level forecast.

    The forecast uses:
    - The trained LinearRegression pipeline from disk.
    - Climatological expected rainfall (IMD envelope model).
    - Rolling mean inflow/outflow from the last 30 observations.
    """
    try:
        reservoir = await data_service.get_reservoir(db, reservoir_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    observations = await data_service.get_observations(db, reservoir_id, limit=999_999)

    try:
        result = forecast_service.generate_forecast(reservoir, observations, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_428_PRECONDITION_REQUIRED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return result
