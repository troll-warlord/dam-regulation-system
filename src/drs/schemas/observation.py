"""
Pydantic V2 schemas for daily Observation resources.

An Observation represents one day's complete water-balance reading for a
specific reservoir:  date, water level, rainfall, inflow, and outflow.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ObservationBase(BaseModel):
    """Fields shared by create + read schemas."""

    date: Date = Field(
        ...,
        description="Calendar date of this observation (ISO 8601).",
        examples=["2025-08-15"],
    )
    water_level_mcft: float = Field(
        ...,
        ge=0.0,
        description="Measured reservoir storage at end-of-day (MCFt).",
        examples=[2450.0],
    )
    rainfall_mm: float = Field(
        default=0.0,
        ge=0.0,
        description="Total rainfall recorded at the reservoir that day (mm).",
        examples=[32.5],
    )
    inflow_mcft: float = Field(
        default=0.0,
        ge=0.0,
        description="Total inflow into the reservoir that day (MCFt).",
        examples=[18.4],
    )
    outflow_mcft: float = Field(
        default=0.0,
        ge=0.0,
        description="Total controlled outflow / release that day (MCFt).",
        examples=[10.0],
    )


class ObservationCreate(ObservationBase):
    """Request body for ``POST /api/v1/reservoirs/{reservoir_id}/observations``."""


class ObservationBulkCreate(BaseModel):
    """Request body for bulk-inserting multiple observations in one call."""

    observations: list[ObservationCreate] = Field(
        ...,
        min_length=1,
        description="List of daily observations to insert.",
    )


class ObservationRead(ObservationBase):
    """Response body — includes server-generated fields."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    reservoir_id: str
    created_at: datetime


class ObservationFilter(BaseModel):
    """Query-parameter schema for filtering observation lists."""

    start_date: Date | None = Field(default=None, description="Return observations on or after this date.")
    end_date: Date | None = Field(default=None, description="Return observations on or before this date.")
    limit: int = Field(default=100, ge=1, le=10_000)
    offset: int = Field(default=0, ge=0)
