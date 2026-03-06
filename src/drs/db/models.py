"""
SQLAlchemy ORM models for the Dam Regulation System.

Schema
------
Reservoir (1) ──< Observation (many)
Reservoir (1) ──< ModelArtifact (many, one-per-train-run)

All primary keys are UUID strings for portability across SQLite and
PostgreSQL without schema changes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from drs.db.engine import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------
class Reservoir(Base):
    """
    Represents a physical reservoir / dam.

    Attributes
    ----------
    id:
        UUID primary key.
    name:
        Human-readable reservoir name (e.g. "Chembarambakkam").
    city:
        Nearest major city (e.g. "Chennai").
    state:
        Indian state (e.g. "Tamil Nadu").
    max_capacity_mcft:
        Full-reservoir-level storage in Million Cubic Feet (MCFt).
    alert_level_mcft:
        Water level at which overflow alerts are triggered (MCFt).
        Defaults to 95 % of max_capacity if not provided.
    dead_storage_mcft:
        Minimum operational storage — water below this cannot be released.
    coordinates_lat:
        Latitude of the reservoir centroid.
    coordinates_lon:
        Longitude of the reservoir centroid.
    created_at:
        Timestamp of record creation (UTC).
    """

    __tablename__ = "reservoirs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    city: Mapped[str] = mapped_column(String(120), nullable=False)
    state: Mapped[str] = mapped_column(String(120), nullable=False)
    max_capacity_mcft: Mapped[float] = mapped_column(Float, nullable=False)
    alert_level_mcft: Mapped[float] = mapped_column(Float, nullable=False)
    dead_storage_mcft: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    coordinates_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    coordinates_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    # Relationships
    observations: Mapped[list[Observation]] = relationship("Observation", back_populates="reservoir", cascade="all, delete-orphan")
    model_artifacts: Mapped[list[ModelArtifact]] = relationship("ModelArtifact", back_populates="reservoir", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Reservoir name={self.name!r} city={self.city!r} max={self.max_capacity_mcft} mcft>"


# ---------------------------------------------------------------------------
# Observation (daily water-balance reading)
# ---------------------------------------------------------------------------
class Observation(Base):
    """
    A single daily water-balance record for a reservoir.

    The four core hydrological fields capture the complete daily water budget::

        tomorrow_level ≈ today_level + inflow + rainfall_contribution - outflow

    The regression model is trained to predict::

        delta = today_level - yesterday_level

    from the features [rainfall_mm, inflow_mcft, outflow_mcft, prev_level_mcft].

    Attributes
    ----------
    id:
        UUID primary key.
    reservoir_id:
        Foreign key to :class:`Reservoir`.
    date:
        The calendar date of this reading.
    water_level_mcft:
        Measured water storage at end-of-day (MCFt).
    rainfall_mm:
        Total rainfall recorded at the reservoir that day (mm).
    inflow_mcft:
        Total inflow into the reservoir that day (MCFt).
    outflow_mcft:
        Total controlled outflow / release that day (MCFt).
    created_at:
        Timestamp of record insertion (UTC).
    """

    __tablename__ = "observations"
    __table_args__ = (UniqueConstraint("reservoir_id", "date", name="uq_obs_reservoir_date"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    reservoir_id: Mapped[str] = mapped_column(String(36), ForeignKey("reservoirs.id", ondelete="CASCADE"), nullable=False, index=True)
    date: Mapped[date] = mapped_column(nullable=False, index=True)
    water_level_mcft: Mapped[float] = mapped_column(Float, nullable=False)
    rainfall_mm: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    inflow_mcft: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    outflow_mcft: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    # Relationships
    reservoir: Mapped[Reservoir] = relationship("Reservoir", back_populates="observations")

    def __repr__(self) -> str:
        return f"<Observation reservoir={self.reservoir_id!r} date={self.date} level={self.water_level_mcft} mcft>"


# ---------------------------------------------------------------------------
# ModelArtifact (metadata for a trained ML model saved to disk)
# ---------------------------------------------------------------------------
class ModelArtifact(Base):
    """
    Metadata record for a trained scikit-learn model serialised to disk.

    The actual model binary (joblib file) lives in *storage/models/*.
    This table stores the evaluation metrics and lineage so you always know
    which model is active for each reservoir.

    Attributes
    ----------
    id:
        UUID primary key.
    reservoir_id:
        Foreign key to :class:`Reservoir`.
    trained_at:
        UTC timestamp when training completed.
    algorithm:
        Short name of the sklearn estimator (e.g. "LinearRegression").
    feature_names:
        JSON list of feature column names used during training.
    r2_score:
        Coefficient of determination on the test split.
    mae_mcft:
        Mean Absolute Error in MCFt on the test split.
    rmse_mcft:
        Root Mean Squared Error in MCFt on the test split.
    training_rows:
        Number of observations used for training.
    artifact_path:
        Absolute filesystem path to the serialised ``.joblib`` file.
    """

    __tablename__ = "model_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    reservoir_id: Mapped[str] = mapped_column(String(36), ForeignKey("reservoirs.id", ondelete="CASCADE"), nullable=False, index=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    algorithm: Mapped[str] = mapped_column(String(120), nullable=False, default="LinearRegression")
    feature_names: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    r2_score: Mapped[float] = mapped_column(Float, nullable=False)
    mae_mcft: Mapped[float] = mapped_column(Float, nullable=False)
    rmse_mcft: Mapped[float] = mapped_column(Float, nullable=False)
    training_rows: Mapped[int] = mapped_column(nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)

    # Relationships
    reservoir: Mapped[Reservoir] = relationship("Reservoir", back_populates="model_artifacts")

    def __repr__(self) -> str:
        return f"<ModelArtifact reservoir={self.reservoir_id!r} trained={self.trained_at} r2={self.r2_score:.4f}>"
