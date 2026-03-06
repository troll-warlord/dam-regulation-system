"""
Pydantic V2 schemas for Release Recommendation resources.

The recommendation engine takes a :class:`~drs.schemas.forecast.ForecastResponse`
and produces a graduated release schedule designed to bring the
reservoir back within safe operating limits.

Algorithm
---------
Given:
- ``excess_mcft``   — total water above the alert threshold
- ``horizon_days``  — days over which to spread the release

The graduated (front-loaded) schedule distributes the excess using
descending linear weights::

    weight_i = (N - i) / Σ(N - j for j in 0..N-1)

so that the release volume is highest on day 1 and tapers toward day N.
This matches standard dam-engineering practice of releasing heavily
before a rain event intensifies.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    """Categorical risk classification for the forecast window."""

    SAFE = "SAFE"
    WATCH = "WATCH"  # 85–95 % utilisation
    WARNING = "WARNING"  # 95–100 % utilisation
    CRITICAL = "CRITICAL"  # >100 % (overflow predicted)


class ReleaseDay(BaseModel):
    """Recommended controlled release for a single day."""

    release_date: date
    recommended_release_mcft: float = Field(
        ge=0.0,
        description="Volume of water to release on this day (MCFt).",
    )
    cumulative_released_mcft: float = Field(
        ge=0.0,
        description="Total released from day 1 through this day (MCFt).",
    )
    projected_level_after_release_mcft: float = Field(description=("Projected water level after applying this day's release, on top of the forecasted level."))


class RecommendationResponse(BaseModel):
    """
    Full release recommendation report for one reservoir + forecast window.

    This is the primary output of DRS and constitutes the core IP of the system.
    """

    reservoir_id: str
    reservoir_name: str
    recommendation_generated_at: datetime
    risk_level: RiskLevel
    horizon_days: int
    from_date: date
    excess_mcft: float = Field(
        ge=0.0,
        description="Total excess water above alert level (MCFt).",
    )
    total_recommended_release_mcft: float = Field(
        ge=0.0,
        description="Sum of all daily releases in the schedule (MCFt).",
    )
    release_schedule: list[ReleaseDay]
    summary: str = Field(
        description="Human-readable executive summary of the recommendation.",
    )
