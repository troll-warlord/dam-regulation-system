"""
Command-Line Interface for the Dam Regulation System.

The ``drs`` command is the alternative entry point to the FastAPI server.
It provides direct access to every stage of the DRS pipeline without
needing an HTTP client.

Commands
--------
drs init        Initialise the database schema + (optionally) seed data.
drs reservoirs  List all registered reservoirs.
drs train       Train the ML model for a reservoir.
drs forecast    Generate an N-day forecast for a reservoir.
drs recommend   Run the full 7-step pipeline and get a release report.
drs serve       Start the FastAPI server.

Examples
--------
::

    # First-time setup
    uv run drs init --seed

    # Train model for Chembarambakkam reservoir
    uv run drs train --name "Chembarambakkam"

    # 10-day forecast
    uv run drs forecast --name "Chembarambakkam" --days 10

    # Full recommendation report
    uv run drs recommend --name "Chembarambakkam" --days 10

    # Serve the API
    uv run drs serve
"""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.table import Table

from drs.core.config import get_settings
from drs.core.logging import get_logger, setup_logging

log = get_logger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.version_option(package_name="dam-regulation-system")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    \b
    Dam Regulation System (DRS) — CLI
    ML-driven flood forecasting & controlled-release engine.
    © 2026 DRS Engineering. All Rights Reserved.
    """
    setup_logging(get_settings().log_level)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@cli.command("init")
@click.option(
    "--seed/--no-seed",
    default=False,
    help="Also seed the database with 4 cities and synthetic historical data.",
)
@click.option(
    "--years",
    default=3,
    show_default=True,
    type=click.IntRange(1, 10),
    help="Number of years of synthetic data to generate when --seed is used.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-seed even if data already exists (destructive!).",
)
def cmd_init(seed: bool, years: int, force: bool) -> None:
    """Initialise the database schema (and optionally seed sample data)."""
    from drs.db.engine import init_db
    from drs.db.seed import run_seed

    async def _run() -> None:
        await init_db()
        if seed:
            await run_seed(years=years, force=force)

    asyncio.run(_run())
    console.print("[bold green]✓ Database initialised.[/bold green]")
    if seed:
        console.print(f"[green]✓ Seeded with {years} years of synthetic data for 4 reservoirs.[/green]")


# ---------------------------------------------------------------------------
# reservoirs
# ---------------------------------------------------------------------------


@cli.command("reservoirs")
def cmd_reservoirs() -> None:
    """List all registered reservoirs."""
    from drs.db.engine import AsyncSessionLocal
    from drs.services import data_service

    async def _run() -> None:
        async with AsyncSessionLocal() as session:
            reservoirs = await data_service.list_reservoirs(session)

        table = Table(title="Registered Reservoirs", show_lines=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("City")
        table.add_column("State")
        table.add_column("Max Cap (MCFt)", justify="right")
        table.add_column("Alert (MCFt)", justify="right")
        table.add_column("ID", style="dim")

        for r in reservoirs:
            table.add_row(
                r.name,
                r.city,
                r.state,
                f"{r.max_capacity_mcft:,.1f}",
                f"{r.alert_level_mcft:,.1f}",
                r.id,
            )
        console.print(table)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@cli.command("train")
@click.option("--name", required=True, help="Reservoir name (exact match).")
@click.option(
    "--test-split",
    default=0.20,
    show_default=True,
    type=click.FloatRange(0.05, 0.49),
    help="Fraction of data held out for model evaluation.",
)
def cmd_train(name: str, test_split: float) -> None:
    """Train the ML model for a specific reservoir."""
    from drs.db.engine import AsyncSessionLocal
    from drs.services import data_service, training_service

    async def _run() -> None:
        async with AsyncSessionLocal() as session:
            reservoirs = await data_service.list_reservoirs(session)
            reservoir = next((r for r in reservoirs if r.name.lower() == name.lower()), None)
            if reservoir is None:
                console.print(f"[red]Reservoir '{name}' not found.[/red]")
                raise SystemExit(1)

            observations = await data_service.get_observations(session, reservoir.id, limit=999_999)
            try:
                result = await training_service.train_reservoir_model(
                    session,
                    reservoir,
                    observations,
                    test_split_ratio=test_split,
                )
            except ValueError as exc:
                console.print(f"[red]Training failed: {exc}[/red]")
                raise SystemExit(1) from exc

        console.print(f"\n[bold green]✓ Model trained for '{result.reservoir_name}'[/bold green]")
        console.print(f"  Algorithm     : {result.algorithm}")
        console.print(f"  Training rows : {result.training_rows}")
        console.print(f"  R²            : {result.r2_score:.4f}")
        console.print(f"  MAE           : {result.mae_mcft:.3f} MCFt")
        console.print(f"  RMSE          : {result.rmse_mcft:.3f} MCFt")
        console.print(f"  Artifact      : {result.artifact_path}")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------


@cli.command("forecast")
@click.option("--name", required=True, help="Reservoir name (exact match).")
@click.option(
    "--days",
    default=10,
    show_default=True,
    type=click.IntRange(1, 365),
    help="Number of days to forecast.",
)
@click.option(
    "--from-date",
    default=None,
    help="Start date in YYYY-MM-DD format.  Defaults to tomorrow.",
)
@click.option(
    "--json-output",
    is_flag=True,
    default=False,
    help="Print raw JSON instead of a table.",
)
def cmd_forecast(name: str, days: int, from_date: str | None, json_output: bool) -> None:
    """Generate an N-day water-level forecast for a reservoir."""
    from datetime import date

    from drs.db.engine import AsyncSessionLocal
    from drs.schemas.forecast import ForecastRequest
    from drs.services import data_service, forecast_service

    async def _run() -> None:
        async with AsyncSessionLocal() as session:
            reservoirs = await data_service.list_reservoirs(session)
            reservoir = next((r for r in reservoirs if r.name.lower() == name.lower()), None)
            if reservoir is None:
                console.print(f"[red]Reservoir '{name}' not found.[/red]")
                raise SystemExit(1)

            observations = await data_service.get_observations(session, reservoir.id, limit=999_999)

        parsed_date = date.fromisoformat(from_date) if from_date else None
        request = ForecastRequest(horizon_days=days, from_date=parsed_date)

        try:
            result = forecast_service.generate_forecast(reservoir, observations, request)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Forecast failed: {exc}[/red]")
            raise SystemExit(1) from exc

        if json_output:
            click.echo(result.model_dump_json(indent=2))
            return

        table = Table(title=f"Forecast — {result.reservoir_name} ({days} days)", show_lines=True)
        table.add_column("Date", style="cyan")
        table.add_column("Level (MCFt)", justify="right")
        table.add_column("Δ (MCFt)", justify="right")
        table.add_column("Rain (mm)", justify="right")
        table.add_column("Util %", justify="right")
        table.add_column("Alert", justify="center")

        for day in result.days:
            alert_str = "[bold red]YES[/bold red]" if day.is_above_alert else "[green]NO[/green]"
            table.add_row(
                str(day.forecast_date),
                f"{day.predicted_level_mcft:,.1f}",
                f"{day.predicted_delta_mcft:+.2f}",
                f"{day.expected_rainfall_mm:.1f}",
                f"{day.capacity_utilisation_pct:.1f}",
                alert_str,
            )
        console.print(table)
        overflow_str = "[bold red]YES[/bold red]" if result.overflow_detected else "[green]NO[/green]"
        console.print(f"\nOverflow detected: {overflow_str}  |  Peak: {result.peak_predicted_level_mcft:.1f} MCFt on {result.peak_date}")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------


@cli.command("recommend")
@click.option("--name", required=True, help="Reservoir name (exact match).")
@click.option(
    "--days",
    default=10,
    show_default=True,
    type=click.IntRange(1, 365),
    help="Forecast horizon for the recommendation.",
)
@click.option(
    "--from-date",
    default=None,
    help="Start date in YYYY-MM-DD format.  Defaults to tomorrow.",
)
@click.option(
    "--json-output",
    is_flag=True,
    default=False,
    help="Print raw JSON instead of formatted tables.",
)
def cmd_recommend(name: str, days: int, from_date: str | None, json_output: bool) -> None:
    """Run the full 7-step DRS pipeline and print a release recommendation."""
    from datetime import date

    from drs.db.engine import AsyncSessionLocal
    from drs.schemas.forecast import ForecastRequest
    from drs.services import data_service, forecast_service, recommendation_service

    async def _run() -> None:
        async with AsyncSessionLocal() as session:
            reservoirs = await data_service.list_reservoirs(session)
            reservoir = next((r for r in reservoirs if r.name.lower() == name.lower()), None)
            if reservoir is None:
                console.print(f"[red]Reservoir '{name}' not found.[/red]")
                raise SystemExit(1)

            observations = await data_service.get_observations(session, reservoir.id, limit=999_999)

        parsed_date = date.fromisoformat(from_date) if from_date else None
        request = ForecastRequest(horizon_days=days, from_date=parsed_date)

        try:
            forecast = forecast_service.generate_forecast(reservoir, observations, request)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Forecast failed: {exc}[/red]")
            raise SystemExit(1) from exc

        rec = recommendation_service.build_recommendation(reservoir, forecast)

        if json_output:
            click.echo(rec.model_dump_json(indent=2))
            return

        # Risk banner
        risk_colours = {
            "SAFE": "green",
            "WATCH": "yellow",
            "WARNING": "bold yellow",
            "CRITICAL": "bold red",
        }
        colour = risk_colours.get(rec.risk_level.value, "white")
        console.rule(f"[{colour}]Risk Level: {rec.risk_level.value}[/{colour}]")
        console.print(f"\n[bold]{rec.summary}[/bold]\n")

        if rec.release_schedule:
            table = Table(title="Graduated Release Schedule", show_lines=True)
            table.add_column("Date", style="cyan")
            table.add_column("Daily Release (MCFt)", justify="right")
            table.add_column("Cumulative (MCFt)", justify="right")
            table.add_column("Projected Level (MCFt)", justify="right")
            for rd in rec.release_schedule:
                table.add_row(
                    str(rd.release_date),
                    f"{rd.recommended_release_mcft:.3f}",
                    f"{rd.cumulative_released_mcft:.3f}",
                    f"{rd.projected_level_after_release_mcft:.1f}",
                )
            console.print(table)
            console.print(f"\nTotal recommended release: [bold]{rec.total_recommended_release_mcft:.2f} MCFt[/bold]")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command("serve")
@click.option("--host", default=None, help="Bind host (overrides .env).")
@click.option("--port", default=None, type=int, help="Bind port (overrides .env).")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (dev mode).")
def cmd_serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the FastAPI server with uvicorn."""
    import uvicorn

    from drs.core.config import get_settings

    cfg = get_settings()
    uvicorn.run(
        "drs.api.app:app",
        host=host or cfg.host,
        port=port or cfg.port,
        reload=reload,
        log_level="debug" if cfg.debug else "info",
    )
