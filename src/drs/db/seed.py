"""
Database seeder for the Dam Regulation System.

Seeds the database with:
- 4 real Indian reservoirs (one per major city)
- 3 years of synthetic daily observations with realistic monsoon seasonality

The synthetic data follows Indian hydrological patterns:
- Monsoon months (Jun-Sep): high rainfall, rising levels
- Winter (Oct-Jan):         dry, slow drawdown for irrigation
- Summer (Feb-May):         minimal rainfall, reservoir at seasonal low

All numeric ranges are based on publicly available data for each reservoir.

Run via::

    uv run python -m drs.db.seed

or inside the CLI::

    drs init --seed
"""

from __future__ import annotations

import asyncio
import math
import random
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select

from drs.core.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
from drs.db.engine import AsyncSessionLocal, init_db
from drs.db.models import Observation, Reservoir

log = get_logger(__name__)

random.seed(42)  # Deterministic seed for reproducibility

# ---------------------------------------------------------------------------
# Master reservoir configuration
# ---------------------------------------------------------------------------
RESERVOIR_CONFIGS: list[dict[str, object]] = [
    {
        "name": "Chembarambakkam",
        "city": "Chennai",
        "state": "Tamil Nadu",
        "max_capacity_mcft": 3200.0,
        "alert_level_mcft": 3000.0,
        "dead_storage_mcft": 100.0,
        "coordinates_lat": 13.0067,
        "coordinates_lon": 80.0706,
        # Hydrological profile
        "_typical_low_mcft": 400.0,
        "_typical_high_mcft": 2800.0,
        "_monsoon_peak_rainfall_mm": 180.0,
    },
    {
        "name": "TG Halli",
        "city": "Bengaluru",
        "state": "Karnataka",
        "max_capacity_mcft": 3300.0,
        "alert_level_mcft": 3100.0,
        "dead_storage_mcft": 150.0,
        "coordinates_lat": 13.0285,
        "coordinates_lon": 77.4168,
        "_typical_low_mcft": 500.0,
        "_typical_high_mcft": 3000.0,
        "_monsoon_peak_rainfall_mm": 150.0,
    },
    {
        "name": "Osmansagar",
        "city": "Hyderabad",
        "state": "Telangana",
        "max_capacity_mcft": 2900.0,
        "alert_level_mcft": 2700.0,
        "dead_storage_mcft": 120.0,
        "coordinates_lat": 17.3850,
        "coordinates_lon": 78.2997,
        "_typical_low_mcft": 300.0,
        "_typical_high_mcft": 2600.0,
        "_monsoon_peak_rainfall_mm": 140.0,
    },
    {
        "name": "Powai",
        "city": "Mumbai",
        "state": "Maharashtra",
        "max_capacity_mcft": 5000.0,
        "alert_level_mcft": 4700.0,
        "dead_storage_mcft": 200.0,
        "coordinates_lat": 19.1176,
        "coordinates_lon": 72.9060,
        "_typical_low_mcft": 1000.0,
        "_typical_high_mcft": 4500.0,
        "_monsoon_peak_rainfall_mm": 200.0,
    },
]


# ---------------------------------------------------------------------------
# Synthetic time-series generation
# ---------------------------------------------------------------------------


def _rainfall_for_day(d: date, peak_mm: float) -> float:
    """
    Generate realistic daily rainfall (mm) based on Indian monsoon seasonality.

    The rainfall envelope follows a sinusoidal profile peaking in August
    (month 8), with a large stochastic component that produces ~60 % dry
    days outside the monsoon and ~20 % dry days during peak monsoon.

    Parameters
    ----------
    d:
        The calendar date.
    peak_mm:
        Peak daily rainfall (mm) specific to the reservoir's climatic zone.

    Returns
    -------
    float
        Daily rainfall >= 0 mm, rounded to 1 decimal place.
    """
    # Day-of-year normalised to [0, 2π] — peak at day ~215 (early August)
    day_of_year = d.timetuple().tm_yday
    phase = 2 * math.pi * (day_of_year - 215) / 365

    # Monsoon intensity: 0 (off-season) → 1.0 (peak August)
    monsoon_intensity = max(0.0, math.cos(phase))

    # Probability of rain: 10 % off-season, 80 % at monsoon peak
    prob_rain = 0.10 + 0.70 * monsoon_intensity
    if random.random() > prob_rain:
        return 0.0

    # Amount: log-normal centred around half the daily peak during monsoon
    mean_rain = peak_mm * 0.5 * monsoon_intensity + 2.0
    rainfall = random.lognormvariate(math.log(max(mean_rain, 1.0)), 0.8)
    return round(max(0.0, rainfall), 1)


def _simulate_water_balance(
    prev_level: float,
    rainfall_mm: float,
    max_capacity: float,
    dead_storage: float,
    low: float,
    high: float,
) -> tuple[float, float, float]:
    """
    Simulate one day of reservoir water balance.

    A simplified conceptual model::

        level_t+1 = level_t + inflow - outflow

    where inflow is rainfall-driven and outflow is a mix of controlled
    release and evaporation, targeting a seasonal equilibrium.

    Parameters
    ----------
    prev_level:
        Yesterday's water level (MCFt).
    rainfall_mm:
        Today's rainfall (mm).
    max_capacity:
        Reservoir full capacity (MCFt).
    dead_storage:
        Minimum operable storage (MCFt).
    low, high:
        Seasonal band for the reservoir's operating range.

    Returns
    -------
    tuple[float, float, float]
        ``(today_level, inflow_mcft, outflow_mcft)`` — all in MCFt.
    """
    # Rainfall-to-inflow conversion:
    # Approximate catchment area as a scaling factor (varies by reservoir)
    catchment_factor = (max_capacity / 3000.0) * 0.6
    inflow = round(max(0.0, rainfall_mm * catchment_factor + random.uniform(-5, 5)), 2)

    # Outflow: operational release to stay within safe range
    # If above high-water mark → accelerate release
    if prev_level > high:
        base_outflow = (prev_level - high) * 0.3 + random.uniform(5, 20)
    elif prev_level < low:
        base_outflow = random.uniform(1, 8)  # conserve water
    else:
        base_outflow = random.uniform(5, 15)  # routine release + evaporation

    outflow = round(max(0.0, base_outflow), 2)

    new_level = prev_level + inflow - outflow
    new_level = round(max(dead_storage, min(new_level, max_capacity * 0.98)), 2)
    return new_level, inflow, outflow


def _generate_observations(
    reservoir_id: str,
    config: dict[str, object],
    start: date,
    end: date,
) -> list[Observation]:
    """
    Generate a full synthetic daily time-series for one reservoir.

    Parameters
    ----------
    reservoir_id:
        UUID of the :class:`~drs.db.models.Reservoir` record in the DB.
    config:
        Reservoir configuration dict from ``RESERVOIR_CONFIGS``.
    start:
        First date of the series (inclusive).
    end:
        Last date of the series (inclusive).

    Returns
    -------
    list[Observation]
        One :class:`~drs.db.models.Observation` per calendar day in range.
    """
    peak_rain = float(config["_monsoon_peak_rainfall_mm"])  # type: ignore[arg-type]
    max_cap = float(config["max_capacity_mcft"])  # type: ignore[arg-type]
    dead = float(config["dead_storage_mcft"])  # type: ignore[arg-type]
    low = float(config["_typical_low_mcft"])  # type: ignore[arg-type]
    high = float(config["_typical_high_mcft"])  # type: ignore[arg-type]

    observations: list[Observation] = []
    current_level = (low + high) / 2.0  # Start at mid-range
    current_date = start

    while current_date <= end:
        rainfall = _rainfall_for_day(current_date, peak_rain)
        new_level, inflow, outflow = _simulate_water_balance(current_level, rainfall, max_cap, dead, low, high)
        observations.append(
            Observation(
                reservoir_id=reservoir_id,
                date=current_date,
                water_level_mcft=new_level,
                rainfall_mm=rainfall,
                inflow_mcft=inflow,
                outflow_mcft=outflow,
            )
        )
        current_level = new_level
        current_date += timedelta(days=1)

    return observations


# ---------------------------------------------------------------------------
# Main seeder
# ---------------------------------------------------------------------------


async def seed(
    session: AsyncSession,
    *,
    years: int = 3,
    force: bool = False,
) -> None:
    """
    Seed the database with reservoirs and synthetic historical observations.

    Parameters
    ----------
    session:
        An open :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
    years:
        Number of years of synthetic data to generate (default 3).
    force:
        If ``True``, delete and re-create all seed data even if it exists.
        Otherwise, skip reservoirs that are already present.
    """
    if force:
        log.warning("Force-seed requested — clearing existing seed data.")
        for res in (await session.execute(select(Reservoir))).scalars().all():
            await session.delete(res)
        await session.commit()

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years)

    for cfg in RESERVOIR_CONFIGS:
        # Skip if already seeded
        existing = (await session.execute(select(Reservoir).where(Reservoir.name == cfg["name"]))).scalar_one_or_none()

        if existing and not force:
            log.info("Reservoir already seeded — skipping. name={name}", name=cfg["name"])
            continue

        reservoir = Reservoir(
            name=str(cfg["name"]),
            city=str(cfg["city"]),
            state=str(cfg["state"]),
            max_capacity_mcft=float(cfg["max_capacity_mcft"]),  # type: ignore[arg-type]
            alert_level_mcft=float(cfg["alert_level_mcft"]),  # type: ignore[arg-type]
            dead_storage_mcft=float(cfg["dead_storage_mcft"]),  # type: ignore[arg-type]
            coordinates_lat=float(cfg["coordinates_lat"]),  # type: ignore[arg-type]
            coordinates_lon=float(cfg["coordinates_lon"]),  # type: ignore[arg-type]
        )
        session.add(reservoir)
        await session.flush()  # Populate reservoir.id before FK use

        log.info(
            "Seeding {years} years of observations for {name} ({city}) ...",
            years=years,
            name=reservoir.name,
            city=reservoir.city,
        )
        observations = _generate_observations(reservoir.id, cfg, start_date, end_date)
        session.add_all(observations)
        await session.commit()
        log.success(
            "Seeded {n} observations for {name}.",
            n=len(observations),
            name=reservoir.name,
        )


async def run_seed(years: int = 3, force: bool = False) -> None:
    """Initialise the DB schema and run the seeder as a standalone coroutine."""
    await init_db()
    async with AsyncSessionLocal() as session:
        await seed(session, years=years, force=force)
    log.success("Database seeding complete.")


if __name__ == "__main__":
    asyncio.run(run_seed())
