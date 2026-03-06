"""
Unit tests for the Recommendation Service.

Covers:
- Risk classification logic (_classify_risk)
- Graduated release schedule properties
- Summary text correctness for all risk levels
- Edge cases: zero excess, exact capacity hit
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from drs.db.models import Reservoir
from drs.schemas.forecast import ForecastDay, ForecastResponse
from drs.schemas.recommendation import RiskLevel
from drs.services import recommendation_service
from drs.services.recommendation_service import _classify_risk, _graduated_release_schedule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reservoir(
    *,
    max_capacity: float = 3000.0,
    alert_level: float = 2800.0,
    dead_storage: float = 100.0,
) -> Reservoir:
    return Reservoir(
        id="test-res-id",
        name="Test Reservoir",
        city="Chennai",
        state="Tamil Nadu",
        max_capacity_mcft=max_capacity,
        alert_level_mcft=alert_level,
        dead_storage_mcft=dead_storage,
    )


def _make_forecast(
    peak_level: float,
    *,
    max_capacity: float = 3000.0,
    alert_level: float = 2800.0,
    horizon: int = 5,
) -> ForecastResponse:
    """Build a minimal ForecastResponse with ``horizon`` identical days at ``peak_level``."""
    start = date(2026, 8, 1)
    days = [
        ForecastDay(
            forecast_date=start + timedelta(days=i),
            predicted_level_mcft=peak_level,
            predicted_delta_mcft=0.0,
            expected_rainfall_mm=20.0,
            capacity_utilisation_pct=(peak_level / max_capacity) * 100,
            is_above_alert=peak_level > alert_level,
        )
        for i in range(horizon)
    ]
    excess = max(peak_level - max_capacity, 0.0)
    return ForecastResponse(
        reservoir_id="test-res-id",
        reservoir_name="Test Reservoir",
        max_capacity_mcft=max_capacity,
        alert_level_mcft=alert_level,
        forecast_generated_at=datetime.now(UTC),
        horizon_days=horizon,
        from_date=start,
        baseline_level_mcft=peak_level,
        days=days,
        peak_predicted_level_mcft=peak_level,
        peak_date=start,
        overflow_detected=excess > 0,
        excess_mcft=excess,
    )


# ---------------------------------------------------------------------------
# _classify_risk
# ---------------------------------------------------------------------------


class TestClassifyRisk:
    def test_safe_below_85_pct(self) -> None:
        assert _classify_risk(80.0) == RiskLevel.SAFE

    def test_watch_at_85_pct(self) -> None:
        assert _classify_risk(85.0) == RiskLevel.WATCH

    def test_watch_at_94_pct(self) -> None:
        assert _classify_risk(94.9) == RiskLevel.WATCH

    def test_warning_at_95_pct(self) -> None:
        assert _classify_risk(95.0) == RiskLevel.WARNING

    def test_warning_at_99_pct(self) -> None:
        assert _classify_risk(99.9) == RiskLevel.WARNING

    def test_critical_at_100_pct(self) -> None:
        assert _classify_risk(100.0) == RiskLevel.CRITICAL

    def test_critical_above_100_pct(self) -> None:
        assert _classify_risk(105.0) == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# _graduated_release_schedule
# ---------------------------------------------------------------------------


class TestGraduatedReleaseSchedule:
    def test_total_release_equals_excess(self) -> None:
        """The sum of all daily releases must exactly equal the excess."""
        forecast = _make_forecast(peak_level=3100.0, max_capacity=3000.0, horizon=10)
        excess = 3100.0 - 2800.0  # alert_level used for excess in service
        schedule = _graduated_release_schedule(forecast, excess_mcft=excess, dead_storage_mcft=100.0)
        total = sum(d.recommended_release_mcft for d in schedule)
        assert abs(total - excess) < 0.01

    def test_front_loaded_first_day_highest(self) -> None:
        """Day 1 must have the highest release volume."""
        forecast = _make_forecast(peak_level=3050.0, max_capacity=3000.0, horizon=7)
        schedule = _graduated_release_schedule(forecast, excess_mcft=250.0, dead_storage_mcft=100.0)
        releases = [d.recommended_release_mcft for d in schedule]
        assert releases[0] == max(releases), "Day 1 should have the highest release."

    def test_strictly_descending_releases(self) -> None:
        """Release volumes must be non-increasing (descending linear weights)."""
        forecast = _make_forecast(peak_level=3050.0, max_capacity=3000.0, horizon=5)
        schedule = _graduated_release_schedule(forecast, excess_mcft=300.0, dead_storage_mcft=100.0)
        releases = [d.recommended_release_mcft for d in schedule]
        for a, b in zip(releases, releases[1:], strict=False):
            assert a >= b, f"Release must be non-increasing, but got {a} < {b}"

    def test_cumulative_increases_monotonically(self) -> None:
        """Cumulative released must increase or stay flat (never decrease)."""
        forecast = _make_forecast(peak_level=3200.0, max_capacity=3000.0, horizon=10)
        schedule = _graduated_release_schedule(forecast, excess_mcft=400.0, dead_storage_mcft=100.0)
        for prev, curr in zip(schedule, schedule[1:], strict=False):
            assert curr.cumulative_released_mcft >= prev.cumulative_released_mcft

    def test_projected_level_clamped_at_dead_storage(self) -> None:
        """Projected level must never fall below dead_storage_mcft."""
        forecast = _make_forecast(peak_level=2500.0, max_capacity=3000.0, horizon=5)
        dead = 100.0
        # Use a very large excess to force clamping
        schedule = _graduated_release_schedule(forecast, excess_mcft=9000.0, dead_storage_mcft=dead)
        for day in schedule:
            assert day.projected_level_after_release_mcft >= dead

    def test_empty_forecast_returns_empty_schedule(self) -> None:
        """An empty forecast horizon should return an empty schedule."""
        forecast = _make_forecast(peak_level=3000.0, horizon=0)
        forecast.days = []
        schedule = _graduated_release_schedule(forecast, excess_mcft=100.0, dead_storage_mcft=100.0)
        assert schedule == []


# ---------------------------------------------------------------------------
# build_recommendation — integration
# ---------------------------------------------------------------------------


class TestBuildRecommendation:
    def test_safe_reservoir_no_schedule(self) -> None:
        """A safe reservoir should get an empty release schedule."""
        reservoir = _make_reservoir(max_capacity=3000.0, alert_level=2800.0)
        forecast = _make_forecast(peak_level=2400.0, max_capacity=3000.0, alert_level=2800.0)
        rec = recommendation_service.build_recommendation(reservoir, forecast)
        assert rec.risk_level == RiskLevel.SAFE
        assert rec.release_schedule == []
        assert rec.total_recommended_release_mcft == 0.0
        assert "safe" in rec.summary.lower()

    def test_critical_overflow_has_schedule(self) -> None:
        """An overflowing reservoir should produce a non-empty release schedule."""
        reservoir = _make_reservoir(max_capacity=3000.0, alert_level=2800.0)
        forecast = _make_forecast(peak_level=3200.0, max_capacity=3000.0, alert_level=2800.0)
        rec = recommendation_service.build_recommendation(reservoir, forecast)
        assert rec.risk_level == RiskLevel.CRITICAL
        assert len(rec.release_schedule) > 0
        assert rec.total_recommended_release_mcft > 0.0
        assert "CRITICAL" in rec.summary or "overflow" in rec.summary.lower()

    def test_recommendation_response_fields(self) -> None:
        """Verify all mandatory fields are populated on the response."""
        reservoir = _make_reservoir()
        forecast = _make_forecast(peak_level=2900.0, max_capacity=3000.0, alert_level=2800.0)
        rec = recommendation_service.build_recommendation(reservoir, forecast)

        assert rec.reservoir_id == reservoir.id
        assert rec.reservoir_name == reservoir.name
        assert isinstance(rec.recommendation_generated_at, datetime)
        assert rec.horizon_days == 5
        assert rec.excess_mcft >= 0.0
        assert rec.summary != ""
