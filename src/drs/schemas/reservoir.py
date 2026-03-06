"""
Pydantic V2 schemas for Reservoir resources.

Three-tier pattern:
- ``ReservoirCreate``  — request body for POST  /reservoirs
- ``ReservoirUpdate``  — request body for PATCH /reservoirs/{id}
- ``ReservoirRead``    — response body (includes DB-generated fields)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ReservoirBase(BaseModel):
    """Fields shared by create + read schemas."""

    name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Human-readable reservoir name.",
        examples=["Chembarambakkam"],
    )
    city: str = Field(..., min_length=2, max_length=120, examples=["Chennai"])
    state: str = Field(..., min_length=2, max_length=120, examples=["Tamil Nadu"])
    max_capacity_mcft: float = Field(
        ...,
        gt=0.0,
        description="Full-reservoir storage in Million Cubic Feet (MCFt).",
        examples=[3200.0],
    )
    alert_level_mcft: float = Field(
        ...,
        gt=0.0,
        description="Storage threshold (MCFt) at which overflow alerts fire.",
        examples=[3000.0],
    )
    dead_storage_mcft: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum operable storage — water below this cannot be released.",
        examples=[100.0],
    )
    coordinates_lat: float | None = Field(default=None, ge=-90.0, le=90.0, examples=[13.0067])
    coordinates_lon: float | None = Field(default=None, ge=-180.0, le=180.0, examples=[80.0706])

    @field_validator("alert_level_mcft")
    @classmethod
    def _alert_below_max(cls, v: float, info: object) -> float:
        """Ensure alert level does not exceed max capacity."""
        return v

    @model_validator(mode="after")
    def _validate_levels(self) -> ReservoirBase:
        if self.alert_level_mcft > self.max_capacity_mcft:
            raise ValueError(f"alert_level_mcft ({self.alert_level_mcft}) must not exceed max_capacity_mcft ({self.max_capacity_mcft}).")
        if self.dead_storage_mcft >= self.max_capacity_mcft:
            raise ValueError(f"dead_storage_mcft ({self.dead_storage_mcft}) must be less than max_capacity_mcft ({self.max_capacity_mcft}).")
        return self


class ReservoirCreate(ReservoirBase):
    """Request body for ``POST /api/v1/reservoirs``."""


class ReservoirUpdate(BaseModel):
    """
    Request body for ``PATCH /api/v1/reservoirs/{id}``.

    All fields are optional — only supplied fields are updated.
    """

    name: str | None = Field(default=None, min_length=2, max_length=120)
    city: str | None = Field(default=None, min_length=2, max_length=120)
    state: str | None = Field(default=None, min_length=2, max_length=120)
    max_capacity_mcft: float | None = Field(default=None, gt=0.0)
    alert_level_mcft: float | None = Field(default=None, gt=0.0)
    dead_storage_mcft: float | None = Field(default=None, ge=0.0)
    coordinates_lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    coordinates_lon: float | None = Field(default=None, ge=-180.0, le=180.0)


class ReservoirRead(ReservoirBase):
    """Response body — includes server-generated fields."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
