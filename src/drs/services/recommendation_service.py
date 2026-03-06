"""
Release Recommendation Service — Steps 6 & 7 of the DRS 7-Step Process.

Step 6 — Release Recommendation:
    Generate a **graduated, front-loaded** daily release schedule that
    safely dissipates excess storage before the predicted overflow event.

Step 7 — Reporting:
    Build a structured :class:`~drs.schemas.recommendation.RecommendationResponse`
    that constitutes the primary operator-facing output of the DRS.

Algorithm — Graduated Front-Loaded Release
-------------------------------------------
Given:
- ``excess_mcft``   — total water above ``alert_level_mcft``
- ``N``             — number of days in the release window

The daily release volumes ``r_1, …, r_N`` are determined by descending
linear weights::

    w_i = (N + 1 - i)           for i = 1 … N
    r_i = excess × w_i / Σ w_j  for i = 1 … N

This produces the highest release on day 1, tapering linearly to the
lowest on day N.  The motivating principle is to release water
*before* the inflow peak (pre-positioning), which is consistent with
Indian Central Water Commission (CWC) emergency release protocols.

The final projected level after release is::

    proj_level_i = forecast_level_i - cumulative_released_i

clamped from below at ``dead_storage_mcft``.

Risk Classification
-------------------
+----------+-------------------------------+
| Level    | Condition                     |
+==========+===============================+
| SAFE     | peak utilisation < 85 %       |
+----------+-------------------------------+
| WATCH    | 85 % ≤ utilisation < 95 %     |
+----------+-------------------------------+
| WARNING  | 95 % ≤ utilisation < 100 %    |
+----------+-------------------------------+
| CRITICAL | utilisation ≥ 100 %           |
+----------+-------------------------------+
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from drs.core.logging import bind_reservoir_context, get_logger
from drs.schemas.recommendation import (
    RecommendationResponse,
    ReleaseDay,
    RiskLevel,
)

if TYPE_CHECKING:
    from drs.db.models import Reservoir
    from drs.schemas.forecast import ForecastResponse

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_recommendation(
    reservoir: Reservoir,
    forecast: ForecastResponse,
) -> RecommendationResponse:
    """
    Derive a graduated release recommendation from a completed forecast.

    Parameters
    ----------
    reservoir:
        The :class:`~drs.db.models.Reservoir` under evaluation.
    forecast:
        The :class:`~drs.schemas.forecast.ForecastResponse` computed by
        :func:`~drs.services.forecast_service.generate_forecast`.

    Returns
    -------
    RecommendationResponse
        Full structured recommendation including release schedule,
        risk level, and executive summary.
    """
    peak_utilisation = (forecast.peak_predicted_level_mcft / reservoir.max_capacity_mcft) * 100.0

    risk = _classify_risk(peak_utilisation)
    excess = max(
        forecast.peak_predicted_level_mcft - reservoir.alert_level_mcft,
        0.0,
    )

    bind_reservoir_context(
        reservoir_id=str(reservoir.id),
        reservoir_name=reservoir.name,
        city=reservoir.city,
    )
    log.info("Building recommendation. risk={risk} | excess={excess:.2f} MCFt", risk=risk.value, excess=excess)

    if excess == 0.0 or risk == RiskLevel.SAFE:
        schedule: list[ReleaseDay] = []
        total_release = 0.0
    else:
        schedule = _graduated_release_schedule(
            forecast=forecast,
            excess_mcft=excess,
            dead_storage_mcft=reservoir.dead_storage_mcft,
        )
        total_release = schedule[-1].cumulative_released_mcft if schedule else 0.0

    summary = _build_summary(reservoir.name, risk, excess, total_release, forecast.horizon_days)

    return RecommendationResponse(
        reservoir_id=reservoir.id,
        reservoir_name=reservoir.name,
        recommendation_generated_at=datetime.now(UTC),
        risk_level=risk,
        horizon_days=forecast.horizon_days,
        from_date=forecast.from_date,
        excess_mcft=round(excess, 3),
        total_recommended_release_mcft=round(total_release, 3),
        release_schedule=schedule,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_risk(peak_utilisation_pct: float) -> RiskLevel:
    """
    Map reservoir utilisation to a categorical risk level.

    Parameters
    ----------
    peak_utilisation_pct:
        ``peak_level / max_capacity × 100``.
    """
    if peak_utilisation_pct >= 100.0:
        return RiskLevel.CRITICAL
    if peak_utilisation_pct >= 95.0:
        return RiskLevel.WARNING
    if peak_utilisation_pct >= 85.0:
        return RiskLevel.WATCH
    return RiskLevel.SAFE


def _graduated_release_schedule(
    forecast: ForecastResponse,
    excess_mcft: float,
    dead_storage_mcft: float,
) -> list[ReleaseDay]:
    """
    Compute a graduated (front-loaded) daily release schedule.

    The release volumes follow descending linear weights so that the
    highest release occurs on day 1 and tapers to the lowest on day N::

        w_i = N + 1 - i    (i = 1 … N)
        r_i = excess × w_i / Σ w

    Parameters
    ----------
    forecast:
        The completed :class:`~drs.schemas.forecast.ForecastResponse`.
    excess_mcft:
        Total excess storage above the alert level (MCFt).
    dead_storage_mcft:
        Minimum operable storage — the projected level is clamped here.

    Returns
    -------
    list[ReleaseDay]
        One entry per forecast day, ordered chronologically.
    """
    n = len(forecast.days)
    if n == 0:
        return []

    # Descending weights: [N, N-1, …, 1]
    weights = [n + 1 - i for i in range(1, n + 1)]
    total_weight = sum(weights)

    schedule: list[ReleaseDay] = []
    cumulative = 0.0

    for day, weight in zip(forecast.days, weights, strict=False):
        daily_release = round(excess_mcft * weight / total_weight, 3)
        cumulative = round(cumulative + daily_release, 3)

        projected_level = max(
            day.predicted_level_mcft - cumulative,
            dead_storage_mcft,
        )

        schedule.append(
            ReleaseDay(
                release_date=day.forecast_date,
                recommended_release_mcft=daily_release,
                cumulative_released_mcft=cumulative,
                projected_level_after_release_mcft=round(projected_level, 3),
            )
        )

    return schedule


def _build_summary(
    name: str,
    risk: RiskLevel,
    excess_mcft: float,
    total_release: float,
    horizon_days: int,
) -> str:
    """Generate a human-readable executive summary string."""
    if risk == RiskLevel.SAFE:
        return f"Reservoir '{name}' is operating within safe limits for the next {horizon_days} days. No controlled release is required."
    if risk == RiskLevel.WATCH:
        return (
            f"Reservoir '{name}' is approaching its alert threshold (WATCH). "
            f"An estimated excess of {excess_mcft:.1f} MCFt is projected. "
            "Monitoring is advised; pre-emptive minor releases are recommended."
        )
    if risk == RiskLevel.WARNING:
        return (
            f"WARNING: Reservoir '{name}' is projected to reach near-capacity within "
            f"{horizon_days} days. A total controlled release of {total_release:.1f} MCFt "
            "is recommended using the graduated schedule provided."
        )
    # CRITICAL
    return (
        f"CRITICAL: Reservoir '{name}' is forecast to OVERFLOW. "
        f"Immediate action required. A graduated release of {total_release:.1f} MCFt "
        f"over {horizon_days} days is recommended to avert downstream flooding. "
        "Alert downstream authorities and activate emergency release protocols."
    )
