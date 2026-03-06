"""
Integration tests for the Forecast Service + Training Service.

Tests the full train → forecast pipeline using an in-memory database
and synthetic observations from the conftest fixtures.

Coverage:
- Training succeeds with sufficient observations
- Training fails with insufficient observations
- Forecast produces correct number of days
- Forecast respects horizon_days parameter
- Overflow detection triggers at correct levels
- Baseline level is taken from the most recent observation
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from drs.schemas.forecast import ForecastRequest
from drs.services import forecast_service, training_service
from tests.conftest import make_observations

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from drs.db.models import Reservoir

# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestTrainingService:
    @pytest.mark.asyncio
    async def test_train_succeeds_with_60_observations(
        self,
        db_session: AsyncSession,
        seeded_reservoir: Reservoir,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Training should succeed and return valid metrics when data is sufficient."""
        # Redirect model storage to pytest's tmp dir
        from drs.core import config as cfg_module

        settings = cfg_module.get_settings()
        monkeypatch.setattr(settings, "models_dir", tmp_path)

        # Override artifact path via monkeypatching settings
        cfg_module.get_settings.cache_clear()

        obs = make_observations(seeded_reservoir.id, n=60, rainfall_pattern="flat")

        result = await training_service.train_reservoir_model(
            db_session,
            seeded_reservoir,
            obs,
        )

        assert result.training_rows > 0
        assert -1.0 <= result.r2_score <= 1.0
        assert result.mae_mcft >= 0.0
        assert result.rmse_mcft >= 0.0
        assert result.algorithm == "LinearRegression"
        assert len(result.feature_names) == 4

    @pytest.mark.asyncio
    async def test_train_fails_with_too_few_observations(
        self,
        db_session: AsyncSession,
        seeded_reservoir: Reservoir,
    ) -> None:
        """Training should raise ValueError when fewer than 30 observations exist."""
        obs = make_observations(seeded_reservoir.id, n=10, rainfall_pattern="flat")
        with pytest.raises(ValueError, match="Insufficient data"):
            await training_service.train_reservoir_model(
                db_session,
                seeded_reservoir,
                obs,
            )


# ---------------------------------------------------------------------------
# Forecast tests
# ---------------------------------------------------------------------------


class TestForecastService:
    def test_forecast_returns_correct_number_of_days(
        self,
        seeded_reservoir: Reservoir,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Forecast should return exactly horizon_days ForecastDay entries."""
        import joblib
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Create and save a dummy trained pipeline to tmp_path
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        # Fit on minimal dummy data
        X = np.array([[10, 5, 8, 1500], [20, 10, 8, 1502], [30, 15, 9, 1507]])
        y = np.array([2.0, 5.0, 6.0])
        pipeline.fit(X, y)

        safe_name = seeded_reservoir.name.replace(" ", "_").lower()
        model_path = tmp_path / f"{safe_name}.joblib"
        joblib.dump(pipeline, model_path)

        # Patch the models_dir setting
        from drs.core.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "models_dir", tmp_path)

        obs = make_observations(seeded_reservoir.id, n=60, rainfall_pattern="flat")
        request = ForecastRequest(horizon_days=7, from_date=date(2026, 9, 1))

        result = forecast_service.generate_forecast(seeded_reservoir, obs, request)

        assert len(result.days) == 7
        assert result.from_date == date(2026, 9, 1)
        assert result.horizon_days == 7

    def test_forecast_overflow_detected_for_high_rainfall(
        self,
        seeded_reservoir: Reservoir,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A model that always predicts a large positive delta should trigger overflow
        for a reservoir that starts near max capacity.
        """
        import joblib
        import numpy as np
        from sklearn.dummy import DummyRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Build a real picklable pipeline that always predicts +200 MCFt/day
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", DummyRegressor(strategy="constant", constant=200.0)),
            ]
        )
        pipeline.fit(np.zeros((2, 4)), np.array([200.0, 200.0]))

        safe_name = seeded_reservoir.name.replace(" ", "_").lower()
        model_path = tmp_path / f"{safe_name}.joblib"
        joblib.dump(pipeline, model_path)

        from drs.core.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "models_dir", tmp_path)

        # Start observations near max capacity
        obs = make_observations(
            seeded_reservoir.id,
            n=30,
            base_level=2950.0,
            rainfall_pattern="overflow",
        )
        request = ForecastRequest(horizon_days=5)

        result = forecast_service.generate_forecast(seeded_reservoir, obs, request)

        # With +200/day from near-max, overflow must be detected
        assert result.overflow_detected

    def test_forecast_baseline_uses_latest_observation(
        self,
        seeded_reservoir: Reservoir,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The baseline level must equal the most recent observation's water level."""
        import joblib
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        X = np.array([[10, 5, 8, 1500], [20, 10, 8, 1502]])
        y = np.array([2.0, 5.0])
        pipeline.fit(X, y)

        safe_name = seeded_reservoir.name.replace(" ", "_").lower()
        joblib.dump(pipeline, tmp_path / f"{safe_name}.joblib")

        from drs.core.config import get_settings

        monkeypatch.setattr(get_settings(), "models_dir", tmp_path)

        obs = make_observations(seeded_reservoir.id, n=30, base_level=1800.0)
        sorted_obs = sorted(obs, key=lambda o: o.date)
        expected_baseline = sorted_obs[-1].water_level_mcft

        result = forecast_service.generate_forecast(seeded_reservoir, obs, ForecastRequest(horizon_days=3))

        assert result.baseline_level_mcft == pytest.approx(expected_baseline, abs=0.01)

    def test_forecast_raises_if_no_observations(
        self,
        seeded_reservoir: Reservoir,
    ) -> None:
        """Forecast with zero observations must raise ValueError."""
        with pytest.raises(ValueError, match="No observations"):
            forecast_service.generate_forecast(seeded_reservoir, [], ForecastRequest(horizon_days=5))

    def test_forecast_raises_if_no_model(
        self,
        seeded_reservoir: Reservoir,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Forecast must raise FileNotFoundError when no model artifact exists."""
        from drs.core.config import get_settings

        monkeypatch.setattr(get_settings(), "models_dir", tmp_path)

        obs = make_observations(seeded_reservoir.id, n=30)
        with pytest.raises(FileNotFoundError):
            forecast_service.generate_forecast(seeded_reservoir, obs, ForecastRequest(horizon_days=5))
