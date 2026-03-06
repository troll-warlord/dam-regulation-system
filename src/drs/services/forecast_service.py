"""
Forecast Service — Steps 3, 4 & 5 of the DRS 7-Step Process.

Step 3 — Forecast:    Roll the trained model forward N days.
Step 4 — Prediction:  Accumulate predicted deltas into absolute water levels.
Step 5 — Overflow detection: Flag if predicted peak exceeds max capacity.

Mathematical Overview
---------------------
Given:
- A fitted model  f: (rainfall, inflow, outflow, prev_level) → Δlevel
- A baseline water level  L₀  (last known observation)
- A climatological rainfall proxy  R̂_d  for each forecast day d

We compute::

    L_d = L_{d-1} + f(R̂_d, Î_d, Ô_d, L_{d-1})

where Î_d and Ô_d are estimated as rolling means of the last 30 days.

Overflow condition::

    overflow = max(peak(L_d) - max_capacity, 0) > 0

Rainfall proxy
--------------
We use the same sinusoidal climatological model as the seeder (see
:func:`drs.db.seed._rainfall_for_day`) rather than a second ML model.
This keeps the system self-contained and avoids a dependency on external
weather forecast data.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from drs.core.logging import bind_reservoir_context, get_logger
from drs.schemas.forecast import ForecastDay, ForecastRequest, ForecastResponse
from drs.services.training_service import load_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from drs.db.models import Observation, Reservoir

log = get_logger(__name__)

# Number of recent observations used to estimate baseline inflow/outflow.
_ROLLING_WINDOW: int = 30
# Seed for the rainfall proxy generator — fixed for reproducibility.
_PROXY_SEED: int = 99


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_forecast(
    reservoir: Reservoir,
    observations: Sequence[Observation],
    request: ForecastRequest,
) -> ForecastResponse:
    """
    Generate an N-day water-level forecast for a reservoir.

    This function implements Steps 3–5 of the DRS pipeline:
    - Loads the trained model from disk.
    - Bootstraps baseline inflow/outflow from recent observations.
    - Rolls the model forward day-by-day using a climatological rainfall proxy.
    - Detects overflow conditions.

    Parameters
    ----------
    reservoir:
        The :class:`~drs.db.models.Reservoir` being forecast.
    observations:
        All (or recent) historical observations, chronologically ordered.
        At least ``_ROLLING_WINDOW`` records are recommended for accurate
        baseline estimation.
    request:
        Validated :class:`~drs.schemas.forecast.ForecastRequest`.

    Returns
    -------
    ForecastResponse
        Full N-day forecast with per-day metrics and overflow detection.

    Raises
    ------
    FileNotFoundError
        If no trained model exists for this reservoir (train first).
    ValueError
        If there are no observations to establish a baseline.
    """
    if not observations:
        raise ValueError(f"No observations available for reservoir '{reservoir.name}'. Add observations before forecasting.")

    model = load_model(reservoir.name)
    bind_reservoir_context(
        reservoir_id=str(reservoir.id),
        reservoir_name=reservoir.name,
        city=reservoir.city,
    )
    log.info("Generating {n}-day forecast.", n=request.horizon_days)

    # Determine baseline: last known water level
    sorted_obs = sorted(observations, key=lambda o: o.date)
    latest = sorted_obs[-1]
    baseline_level = latest.water_level_mcft

    # Estimate mean inflow/outflow from recent window
    window = sorted_obs[-_ROLLING_WINDOW:]
    mean_inflow = sum(o.inflow_mcft for o in window) / len(window)
    mean_outflow = sum(o.outflow_mcft for o in window) / len(window)

    # Determine forecast start date
    from_date: date = request.from_date or (date.today() + timedelta(days=1))
    horizon = request.horizon_days

    # Generate forecast days
    forecast_days: list[ForecastDay] = []
    current_level = baseline_level
    rng = random.Random(_PROXY_SEED)  # noqa: S311
    max_raw_excess: float = 0.0  # tracks largest unclamped overflow before cap

    for i in range(horizon):
        forecast_date = from_date + timedelta(days=i)

        # Climatological rainfall proxy (deterministic for reproducibility)
        expected_rain = _climatological_rainfall(
            forecast_date,
            peak_mm=_infer_peak_rainfall(reservoir),
            rng=rng,
            use_mean=True,  # Use expected value (not stochastic) for forecasting
        )

        # Feature vector
        features = [
            [
                expected_rain,
                mean_inflow,
                mean_outflow,
                current_level,
            ]
        ]

        delta = float(model.predict(features)[0])

        raw_level = current_level + delta
        # Track unclamped excess for true overflow detection
        max_raw_excess = max(max_raw_excess, raw_level - reservoir.max_capacity_mcft)
        # Clamp to physically meaningful range for display and next-step baseline
        new_level = max(reservoir.dead_storage_mcft, min(raw_level, reservoir.max_capacity_mcft))

        utilisation = (new_level / reservoir.max_capacity_mcft) * 100.0

        forecast_days.append(
            ForecastDay(
                forecast_date=forecast_date,
                predicted_level_mcft=round(new_level, 3),
                predicted_delta_mcft=round(delta, 3),
                expected_rainfall_mm=round(expected_rain, 2),
                capacity_utilisation_pct=round(utilisation, 2),
                is_above_alert=new_level >= reservoir.alert_level_mcft,
            )
        )
        current_level = new_level

    # Overflow analysis
    peak_day = max(forecast_days, key=lambda d: d.predicted_level_mcft)
    excess = max(max_raw_excess, 0.0)

    log.info(
        "Forecast complete. peak={peak:.2f} MCFt on {date} | overflow={overflow}",
        peak=peak_day.predicted_level_mcft,
        date=peak_day.forecast_date,
        overflow=excess > 0,
    )

    return ForecastResponse(
        reservoir_id=reservoir.id,
        reservoir_name=reservoir.name,
        max_capacity_mcft=reservoir.max_capacity_mcft,
        alert_level_mcft=reservoir.alert_level_mcft,
        forecast_generated_at=datetime.now(UTC),
        horizon_days=horizon,
        from_date=from_date,
        baseline_level_mcft=round(baseline_level, 3),
        days=forecast_days,
        peak_predicted_level_mcft=round(peak_day.predicted_level_mcft, 3),
        peak_date=peak_day.forecast_date,
        overflow_detected=excess > 0,
        excess_mcft=round(excess, 3),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _climatological_rainfall(
    d: date,
    peak_mm: float,
    rng: random.Random,
    *,
    use_mean: bool = True,
) -> float:
    """
    Estimate expected daily rainfall using a sinusoidal monsoon envelope.

    Mathematical model::

        intensity(d) = max(0, cos(2π(DOY - 215) / 365))
        E[rain_d]    = peak_mm × 0.5 × intensity(d) + 2

    Parameters
    ----------
    d:
        The forecast date.
    peak_mm:
        Climatological peak daily rainfall for this reservoir's zone (mm).
    rng:
        Seeded RNG (unused when ``use_mean=True``).
    use_mean:
        If ``True`` (default for forecasting), return the expected value
        instead of a stochastic draw, giving a deterministic forecast.

    Returns
    -------
    float
        Expected (or sampled) daily rainfall in mm, >= 0.
    """
    day_of_year = d.timetuple().tm_yday
    phase = 2.0 * math.pi * (day_of_year - 215) / 365.0
    intensity = max(0.0, math.cos(phase))

    if use_mean:
        return round(peak_mm * 0.5 * intensity + 2.0, 2)

    # Stochastic draw (used in simulation / Monte Carlo mode)
    prob_rain = 0.10 + 0.70 * intensity
    if rng.random() > prob_rain:
        return 0.0
    mean_rain = peak_mm * 0.5 * intensity + 2.0
    rainfall = rng.lognormvariate(math.log(max(mean_rain, 1.0)), 0.8)
    return round(max(0.0, rainfall), 1)


def _infer_peak_rainfall(reservoir: Reservoir) -> float:
    """
    Return a climatological peak-rainfall estimate for a reservoir.

    In a production system this would come from a meteorological database.
    For now we use city-based constants aligned with IMD long-period averages.
    """
    city_peaks: dict[str, float] = {
        "Chennai": 180.0,
        "Bengaluru": 150.0,
        "Hyderabad": 140.0,
        "Mumbai": 200.0,
    }
    return city_peaks.get(reservoir.city, 120.0)
