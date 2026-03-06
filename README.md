# Dam Regulation System (DRS)

> **© 2026 DRS Engineering. All Rights Reserved. Patent Pending.**

An enterprise-grade, ML-driven flood forecasting and controlled-release recommendation engine for Indian reservoirs.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Mathematical Logic](#3-mathematical-logic)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [CLI Usage](#6-cli-usage)
7. [API Reference](#7-api-reference)
8. [Running Tests](#8-running-tests)
9. [Database Migration to PostgreSQL](#9-database-migration-to-postgresql)
10. [Legal Notice](#10-legal-notice)

---

## 1. Project Overview

The **Dam Regulation System (DRS)** addresses a critical gap in water-resource management: translating raw reservoir telemetry into actionable, quantitative release recommendations that prevent downstream flooding.

### Core Purpose

| Problem | Solution |
|---|---|
| Reservoir operators rely on manual judgement and legacy rules | ML-driven delta prediction from rainfall + water balance data |
| Ad-hoc release schedules cause downstream flooding | Graduated front-loaded release algorithm (mathematically derived) |
| No unified API for multi-reservoir management | FastAPI service with full OpenAPI documentation |
| No audit trail for model lineage | ModelArtifact table ties every prediction to a versioned trained model |

### Seeded Reservoirs

| Reservoir | City | State | Max Capacity (MCFt) | Alert Level (MCFt) |
|---|---|---|---|---|
| Chembarambakkam | Chennai | Tamil Nadu | 3,200 | 3,000 |
| TG Halli | Bengaluru | Karnataka | 3,300 | 3,100 |
| Osmansagar | Hyderabad | Telangana | 2,900 | 2,700 |
| Powai | Mumbai | Maharashtra | 5,000 | 4,700 |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Client Layer                      │
│          FastAPI (HTTP/REST)  │  Click (CLI)        │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│                  Service Layer                      │
│  DataService │ TrainingService │ ForecastService    │
│                  RecommendationService              │
└──────┬────────────────────┬─────────────────────────┘
       │                    │
┌──────▼──────┐    ┌────────▼─────────────────────────┐
│  SQLAlchemy │    │   scikit-learn Pipeline          │
│  Async ORM  │    │   (StandardScaler + LinReg)      │
│  SQLite /   │    │   Serialised via joblib          │
│  PostgreSQL │    └──────────────────────────────────┘
└─────────────┘
```

### Key Design Decisions

- **SQLite → PostgreSQL zero-change migration** via SQLAlchemy 2.0 (change one env var).
- **Async-first**: Every DB call uses `AsyncSession` and `aiosqlite` avoiding event loop blocking.
- **Strict typing**: `mypy --strict` is enforced on the entire `src/drs` package.
- **Pydantic V2**: All request/response boundaries are fully validated with Pydantic V2 models.
- **Loguru**: All modules use structured logging via `drs.core.logging.get_logger`.

---

## 3. Mathematical Logic

### 3.1 Feature Engineering

For each consecutive pair of daily observations, a training row is constructed:

$$\Delta L_t = L_t - L_{t-1}$$

**Feature vector:**

$$\mathbf{x}_t = \left[ R_t,\ I_t,\ O_t,\ L_{t-1} \right]$$

| Symbol | Description | Unit |
|---|---|---|
| $R_t$ | Daily rainfall | mm |
| $I_t$ | Daily inflow | MCFt |
| $O_t$ | Daily outflow / release | MCFt |
| $L_{t-1}$ | Previous day's water level | MCFt |

**Target:** $\Delta L_t$ — the daily change in storage (MCFt).

### 3.2 Regression Model

A scikit-learn `Pipeline` comprising `StandardScaler` + `LinearRegression` (OLS):

$$\hat{\Delta L}_t = \beta_0 + \beta_1 R_t + \beta_2 I_t + \beta_3 O_t + \beta_4 L_{t-1}$$

The scaler is included so coefficients are dimensionless and physically interpretable. A time-ordered train/test split (not random shuffling) is used to prevent look-ahead bias.

### 3.3 Forecast Roll-Forward

Given a baseline level $L_0$ and a horizon of $N$ days:

$$L_d = L_{d-1} + \hat{\Delta L}_d\left(\hat{R}_d,\ \bar{I},\ \bar{O},\ L_{d-1}\right)$$

where $\hat{R}_d$ is the climatological expected rainfall proxy (see §3.4) and $\bar{I}$, $\bar{O}$ are rolling 30-day means.

### 3.4 Climatological Rainfall Proxy

$$\hat{R}_d = P_{\max} \times 0.5 \times \max\!\left(0,\ \cos\!\left(\frac{2\pi\,(d_y - 215)}{365}\right)\right) + 2$$

where $d_y$ is the day-of-year (1–365) and $P_{\max}$ is the IMD long-period average peak daily rainfall for the reservoir's city.

### 3.5 Overflow Detection

$$\text{excess} = \max\!\left(\hat{L}_{\max} - C_{\max},\ 0\right)$$

where $C_{\max}$ is the reservoir's full-reservoir-level capacity.

### 3.6 Graduated Release Algorithm (Core IP)

Given total excess $E$ and horizon $N$, the daily recommended release volumes are assigned **descending linear weights**:

$$w_i = N + 1 - i \qquad (i = 1 \ldots N)$$

$$r_i = E \cdot \frac{w_i}{\displaystyle\sum_{j=1}^{N} w_j} = E \cdot \frac{2(N+1-i)}{N(N+1)}$$

**Properties:**
- $\sum_{i=1}^{N} r_i = E$ — conservation (total release equals excess)
- $r_1 > r_2 > \cdots > r_N$ — strictly decreasing (front-loaded)
- The ratio of first to last day release is exactly $N : 1$

Projected level after release on day $d$:

$$L_d^{\text{proj}} = \max\!\left(\hat{L}_d - \sum_{i=1}^{d} r_i,\ L_{\text{dead}}\right)$$

---

## 4. Installation

### Prerequisites

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) package manager

```bash
# Install uv (if not already installed)
pip install uv
```

### Setup

```bash
# Install all dependencies into a virtual environment
uv sync

# Copy env template
cp .env.example .env

# Initialise the database and seed 4 reservoirs with 3 years of synthetic data
uv run drs init --seed
```

---

## 5. Configuration

All configuration is read from the `.env` file in the project root.

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Dam Regulation System` | Display name |
| `DEBUG` | `false` | Enable debug logging + CORS |
| `HOST` | `127.0.0.1` | API server bind host |
| `PORT` | `8000` | API server bind port |
| `DATABASE_URL` | `sqlite+aiosqlite:///./storage/drs.db` | SQLAlchemy async connection URL |
| `MODELS_DIR` | `./storage/models` | Directory for `.joblib` model artifacts |
| `DEFAULT_FORECAST_DAYS` | `10` | Default forecast horizon |

---

## 6. CLI Usage

```bash
uv run drs --help

# Initialise DB + seed
uv run drs init --seed --years 3

# List all reservoirs
uv run drs reservoirs

# Train ML model
uv run drs train --name "Chembarambakkam"

# 10-day forecast (table)
uv run drs forecast --name "Chembarambakkam" --days 10

# 10-day forecast (JSON)
uv run drs forecast --name "Chembarambakkam" --days 10 --json-output

# Full recommendation report
uv run drs recommend --name "Chembarambakkam" --days 10

# Start API server
uv run drs serve --reload
```

---

## 7. API Reference

Start the server with `uv run drs serve`, then open:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

### Core Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/api/v1/reservoirs` | Register a new reservoir |
| `GET` | `/api/v1/reservoirs` | List all reservoirs |
| `POST` | `/api/v1/reservoirs/{id}/observations` | Add a daily observation |
| `POST` | `/api/v1/reservoirs/{id}/observations/bulk` | Bulk-add observations |
| `POST` | `/api/v1/reservoirs/{id}/train` | Train ML model |
| `POST` | `/api/v1/reservoirs/{id}/forecast` | Generate N-day forecast |
| `POST` | `/api/v1/reservoirs/{id}/recommend` | **Full 7-step pipeline → release recommendation** |

### Example: Full Workflow via cURL

```bash
# List reservoirs — copy an {id}
curl http://127.0.0.1:8000/api/v1/reservoirs

# Train
curl -X POST http://127.0.0.1:8000/api/v1/reservoirs/{id}/train \
  -H "Content-Type: application/json" -d '{"test_split_ratio": 0.20}'

# Recommend
curl -X POST http://127.0.0.1:8000/api/v1/reservoirs/{id}/recommend \
  -H "Content-Type: application/json" -d '{"horizon_days": 10}'
```

---

## 8. Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=src/drs --cov-report=term-missing

# Specific file
uv run pytest tests/test_recommendation.py -v
```

---

## 9. Database Migration to PostgreSQL

1. `uv add asyncpg`
2. Set in `.env`: `DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/drs`
3. `uv run drs init`

No Python code changes required.

---

## 10. Legal Notice

```
Copyright (C) 2026  DRS Engineering
All Rights Reserved. Patent Pending.

This software and the algorithms it embodies — including but not limited to
the Graduated Front-Loaded Release Schedule algorithm, the Rainfall-to-Delta
regression pipeline, and the 7-step DRS forecasting process — are the
exclusive intellectual property of DRS Engineering.

Unauthorised copying, distribution, modification, or use of this software,
in whole or in part, without the express written permission of DRS Engineering
is strictly prohibited.
```
