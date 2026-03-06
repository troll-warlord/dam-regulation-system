"""
Pydantic V2 schemas for Training and Forecast resources.

Training
--------
- ``TrainRequest``   — body for ``POST /api/v1/reservoirs/{id}/train``
- ``TrainResponse``  — result of a training run (metrics + artifact ref)

Forecast
--------
- ``ForecastRequest``  — body for ``POST /api/v1/reservoirs/{id}/forecast``
- ``ForecastDay``      — a single day's predicted water balance
- ``ForecastResponse`` — full N-day forecast result
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Training schemas
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """
    Request body for training a reservoir-specific regression model.

    Parameters
    ----------
    test_split_ratio:
        Fraction of observations held out for evaluation (default 0.20).
    """

    test_split_ratio: float = Field(
        default=0.20,
        gt=0.05,
        lt=0.50,
        description="Fraction of observations held out for model evaluation.",
        examples=[0.20],
    )


class TrainResponse(BaseModel):
    """Response body returned after a successful model training run."""

    model_config = ConfigDict(from_attributes=True)

    artifact_id: str = Field(description="UUID of the saved ModelArtifact record.")
    reservoir_id: str
    reservoir_name: str
    algorithm: str
    feature_names: list[str]
    training_rows: int
    r2_score: float = Field(description="R² on the held-out test split.")
    mae_mcft: float = Field(description="Mean Absolute Error in MCFt on test split.")
    rmse_mcft: float = Field(description="Root Mean Squared Error in MCFt on test split.")
    trained_at: datetime
    artifact_path: str


# ---------------------------------------------------------------------------
# Forecast schemas
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    """
    Request body for generating an N-day water-level forecast.

    Parameters
    ----------
    horizon_days:
        Number of calendar days to forecast (1–365, default 10).
    from_date:
        Start date for the forecast window.  If omitted, defaults to
        tomorrow (today + 1 day).
    """

    horizon_days: int = Field(
        default=10,
        ge=1,
        le=365,
        description="Number of days to forecast ahead.",
        examples=[10],
    )
    from_date: date | None = Field(
        default=None,
        description=("First day of the forecast window.  Defaults to tomorrow."),
        examples=["2026-06-01"],
    )


class ForecastDay(BaseModel):
    """Predicted water balance for a single forecast day."""

    forecast_date: date
    predicted_level_mcft: float = Field(description="Predicted end-of-day water storage (MCFt).")
    predicted_delta_mcft: float = Field(description="Predicted change in storage vs. previous day (MCFt).")
    expected_rainfall_mm: float = Field(description="Climatological expected rainfall proxy used as input (mm).")
    capacity_utilisation_pct: float = Field(description="predicted_level / max_capacity × 100.")
    is_above_alert: bool = Field(description="True when predicted level exceeds the reservoir alert threshold.")


class ForecastResponse(BaseModel):
    """Full N-day forecast result for one reservoir."""

    reservoir_id: str
    reservoir_name: str
    max_capacity_mcft: float
    alert_level_mcft: float
    forecast_generated_at: datetime
    horizon_days: int
    from_date: date
    baseline_level_mcft: float = Field(description="Actual water level on the day before the forecast window.")
    days: list[ForecastDay]
    peak_predicted_level_mcft: float
    peak_date: date
    overflow_detected: bool
    excess_mcft: float = Field(
        description="max(peak_level - max_capacity, 0) in MCFt.",
        ge=0.0,
    )
