"""
ML Training Service for the Dam Regulation System.

Step 1 — Dataset preparation
Step 2 — Model training

Algorithm
---------
The model learns to predict the **daily water-level delta** (MCFt) from
four observable features::

    Δlevel_t = f(rainfall_mm_t, inflow_mcft_t, outflow_mcft_t, level_{t-1})

where:
    Δlevel_t = level_t - level_{t-1}

A :class:`sklearn.linear_model.LinearRegression` (OLS) is used as the
baseline estimator because:
1. The relationship is approximately linear at daily granularity.
2. It is fast, interpretable, and fully serialisable with joblib.
3. The coefficients provide direct physical insight (β_rainfall tells us
   how much each mm of rain raises storage in MCFt).

The model artefact is persisted to ``{MODELS_DIR}/{reservoir_name}.joblib``
and the training metadata is written to the ``model_artifacts`` table so
the lineage is fully auditable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

from drs.core.config import get_settings
from drs.core.logging import bind_reservoir_context, get_logger
from drs.db.models import ModelArtifact, Observation, Reservoir
from drs.schemas.forecast import TrainResponse

log = get_logger(__name__)

# Feature column names — kept as a module-level constant so they are shared
# between training and inference without risk of divergence.
FEATURE_COLUMNS: list[str] = [
    "rainfall_mm",
    "inflow_mcft",
    "outflow_mcft",
    "prev_level_mcft",
]
TARGET_COLUMN: str = "delta_mcft"
_ALGORITHM: str = "LinearRegression"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def train_reservoir_model(
    session: AsyncSession,
    reservoir: Reservoir,
    observations: Sequence[Observation],
    *,
    test_split_ratio: float = 0.20,
) -> TrainResponse:
    """
    Train a water-level delta prediction model for a single reservoir.

    Parameters
    ----------
    session:
        Active async database session (used to persist the artefact record).
    reservoir:
        The :class:`~drs.db.models.Reservoir` being modelled.
    observations:
        All historical daily observations for that reservoir, chronologically
        ordered.  Minimum 30 records required.
    test_split_ratio:
        Fraction of observations held out for evaluation (default 0.20).

    Returns
    -------
    TrainResponse
        Training result including evaluation metrics and artefact path.

    Raises
    ------
    ValueError
        If there are insufficient observations to train (< 30 rows after
        feature engineering).
    """
    bind_reservoir_context(
        reservoir_id=str(reservoir.id),
        reservoir_name=reservoir.name,
        city=reservoir.city,
    )
    log.info("Starting training ({n} observations).", n=len(observations))

    # Step 1 -- Build feature matrix
    df = _build_feature_matrix(observations)
    if len(df) < 30:
        raise ValueError(f"Insufficient data: {len(df)} usable rows after feature engineering. At least 30 consecutive daily observations are required.")
    log.debug("Feature matrix shape: {shape}", shape=df.shape)

    # Step 2 — Split (time-aware: don't shuffle time-series data)
    split_idx = int(len(df) * (1 - test_split_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values

    # Step 3 — Build and fit the sklearn Pipeline
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Step 4 — Evaluate
    y_pred = pipeline.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    log.info(
        "Training complete. R2={r2:.4f}  MAE={mae:.3f} MCFt  RMSE={rmse:.3f} MCFt",
        r2=r2,
        mae=mae,
        rmse=rmse,
    )

    # Step 5 — Persist model artifact to disk
    artifact_path = _artifact_path(reservoir.name)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    log.info("Model artefact saved to {path}", path=artifact_path)

    # Step 6 — Write metadata to DB
    artifact = ModelArtifact(
        reservoir_id=reservoir.id,
        trained_at=datetime.now(UTC),
        algorithm=_ALGORITHM,
        feature_names=FEATURE_COLUMNS,
        r2_score=r2,
        mae_mcft=mae,
        rmse_mcft=rmse,
        training_rows=len(train_df),
        artifact_path=str(artifact_path),
    )
    session.add(artifact)
    await session.commit()
    await session.refresh(artifact)

    return TrainResponse(
        artifact_id=artifact.id,
        reservoir_id=reservoir.id,
        reservoir_name=reservoir.name,
        algorithm=_ALGORITHM,
        feature_names=FEATURE_COLUMNS,
        training_rows=len(train_df),
        r2_score=r2,
        mae_mcft=mae,
        rmse_mcft=rmse,
        trained_at=artifact.trained_at,
        artifact_path=str(artifact_path),
    )


def load_model(reservoir_name: str) -> Pipeline:
    """
    Load and return a trained scikit-learn Pipeline from disk.

    Parameters
    ----------
    reservoir_name:
        Exact name of the reservoir (matches the filename convention).

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline ready for ``.predict()``.

    Raises
    ------
    FileNotFoundError
        If no model artefact exists for this reservoir.
    """
    path = _artifact_path(reservoir_name)
    if not path.exists():
        raise FileNotFoundError(f"No trained model found for reservoir '{reservoir_name}'. Run POST /api/v1/reservoirs/{{id}}/train first.")
    return joblib.load(path)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_feature_matrix(observations: Sequence[Observation]) -> pd.DataFrame:
    """
    Convert a sequence of ORM observations into a pandas DataFrame of
    (features, target) rows.

    The target ``delta_mcft`` is computed as::

        delta_t = level_t - level_{t-1}

    The first observation in the sequence is dropped because it has no
    ``prev_level``.

    Parameters
    ----------
    observations:
        Chronologically ordered list of :class:`~drs.db.models.Observation`.

    Returns
    -------
    pd.DataFrame
        Columns: rainfall_mm, inflow_mcft, outflow_mcft, prev_level_mcft, delta_mcft
    """
    records = [
        {
            "date": obs.date,
            "water_level_mcft": obs.water_level_mcft,
            "rainfall_mm": obs.rainfall_mm,
            "inflow_mcft": obs.inflow_mcft,
            "outflow_mcft": obs.outflow_mcft,
        }
        for obs in observations
    ]
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["prev_level_mcft"] = df["water_level_mcft"].shift(1)
    df["delta_mcft"] = df["water_level_mcft"] - df["prev_level_mcft"]
    df = df.dropna(subset=["prev_level_mcft"]).reset_index(drop=True)
    return df


def _artifact_path(reservoir_name: str) -> Path:
    """Return the canonical ``.joblib`` path for a reservoir's model."""
    cfg = get_settings()
    safe_name = reservoir_name.replace(" ", "_").lower()
    return cfg.models_dir / f"{safe_name}.joblib"
