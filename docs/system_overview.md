# System overview

How the execution side of the repository fits together: universe construction, data
ingestion, walk-forward backtesting, LLM runtime controls and Alpaca paper/live trading.
The factor-mining experiments and their results are in
[research_notes.md](research_notes.md); none of the published results depend on the
trading path described here.

The repository is a US equities research and execution stack for daily S&P 500 alpha mining. It builds a point-in-time universe, maintains market data, evaluates candidate signals with walk-forward controls, and can route approved portfolios into Alpaca paper or live trading with explicit risk checks.

## Methodology

The system is organized as a staged research pipeline rather than a single notebook. It first constructs a point-in-time investable universe, ingests and validates daily OHLCV data, and computes baseline cross-sectional features. Those signals feed a constrained long-only portfolio construction step, which is then evaluated in walk-forward windows with explicit costs, turnover limits, and promotion gates. Frontier LLMs are used only for bounded factor ideation and are surrounded by validation, budget controls, and expression sanitization before any candidate reaches research or trading.

## Output

Typical runs produce:

- point-in-time universe and reference tables under `data/us_equities`
- processed bars and coverage artifacts for daily research
- signal files and walk-forward backtest outputs
- factor-mining candidate files and validation summaries
- broker-facing rebalance intents for paper or live deployment

## Notes

The repository is maintained at `Aroesler1/Alpha-Factor-Mining-Framework` (earlier name: `LLMStrat`) and keeps the `quantaalpha_us` module path for runtime compatibility with earlier internal tooling.

## Project Scope

This codebase covers the full lifecycle of a daily US large-cap systematic strategy:

- historical universe construction
- daily OHLCV ingestion and quality control
- baseline portfolio construction
- walk-forward backtesting
- LLM-assisted factor ideation
- paper and live trading orchestration
- pre-trade and post-trade risk checks

It is not a high-frequency system and does not attempt to model intraday microstructure beyond what is realistic for a daily rebalance process.

## Architecture

### Data

- `quantaalpha_us/data`
  - CRSP/WRDS client
  - membership builders
  - data quality checks

### Research

- `quantaalpha_us/pipeline`
  - baseline feature generation
  - signal generation
- `quantaalpha_us/backtest`
  - point-in-time universe handling
  - walk-forward runner
  - transaction cost model
  - validation gates

### LLM Runtime

- `quantaalpha_us/llm`
  - request budgeting
  - fallback handling
  - factor extraction
  - expression sanitization

### Trading

- `quantaalpha_us/trading`
  - Alpaca REST adapter
  - operational risk controls
  - post-trade reconciliation checks

### Entry Points

- `scripts/`
  - membership build
  - backfill and daily ingest
  - research
  - factor mining
  - reporting
  - orchestration
  - trade submission

### Validation

- `tests/`
  - unit coverage over the current stack

## Technical Implementation

The system is implemented as a Python-based research and execution stack with explicit separation between data, research, LLM runtime, and trading concerns.

Key technical characteristics:

- configuration-driven workflows through YAML
- parquet and CSV artifacts for reproducible intermediate datasets
- point-in-time universe handling rather than static ticker lists
- explicit CLI entrypoints for each stage of the pipeline
- deterministic walk-forward evaluation instead of a single in-sample backtest
- automated checks around data quality, turnover, concentration, and research promotion
- broker integration through a REST execution layer rather than notebook-driven manual trading

The code is structured so that research artifacts, signals, and trade actions can be produced from the same underlying pipeline rather than from disconnected scripts.

## Design Priorities

Several design choices define the project:

- data realism before model complexity
- explicit point-in-time universe handling
- research outputs that can fail promotion rather than silently pass
- operational controls around factor mining instead of unconstrained prompting
- retail-aware execution assumptions instead of idealized frictionless backtests

The result is a system that aims to be honest about what is known, what is approximated, and what still needs empirical validation.

## Tools And Technologies

Beyond LLM usage, this project uses and demonstrates familiarity with:

- Python for the full research and execution stack
- `pandas` for feature engineering, panel manipulation, and research outputs
- parquet-based data artifacts for reproducible local research datasets
- CRSP through WRDS for research-grade historical membership and price data
- Alpaca REST APIs for paper and live execution
- YAML-based configuration for research, mining, and trading profiles
- CLI-oriented orchestration through standalone Python entrypoints
- `pytest` for automated test coverage
- Git and GitHub for versioned development and deployment of research code

From a systems perspective, the project required work across:

- external data integration
- schema normalization
- data quality validation
- portfolio construction logic
- transaction cost modeling
- broker API integration
- runtime fault handling
- test-driven refactoring

## Data Modes

The project supports two operating modes.

### Research-Grade Mode

This mode uses a proper historical membership history sourced from CRSP
through WRDS.

This is the intended mode for serious walk-forward research.

### Approximate Mode

This mode builds a constant-membership S&P 500 approximation from a current constituent snapshot and is retained for:

- bring-up
- pipeline validation
- engineering checks
- low-cost research scaffolding

It is useful operationally, but it is not treated as a substitute for true historical membership data.

## Current Capabilities

At the current stage, the repo supports:

- CRSP market data via WRDS
- historical or approximate S&P 500 membership construction
- daily bar backfill and coverage reporting
- baseline signal generation
- walk-forward research with validation gates
- LLM factor mining with exact model enforcement
- retail-oriented execution simulation
- paper and live trading entrypoints through Alpaca

The main open problem is not infrastructure completeness. The remaining challenge is improving strategy quality enough to satisfy the research gates under realistic assumptions.

## Research Philosophy

The research loop is deliberately conservative.

The backtest includes:

- next-day open execution alignment
- retained cash buffer
- minimum trade-size filtering
- fractional-share handling
- participation limits relative to ADV
- liquidity-aware transaction cost modeling
- sector caps in portfolio construction
- split-adjusted entry and exit prices
- non-overlapping test windows across folds
- positions whose stock has no exit bar (delisted or halted) marked at the last close,
  because the panel has no delisting returns

The trading path refuses stale signal files, an empty universe and a missing membership
file. `--allow-risk-fail` cannot override the kill-switch or universe checks, and a
level 2 or 3 kill switch also writes the pause flag. These were corrected in 2026-10;
see [audit_history.md](audit_history.md).

The goal is not to create a perfect live execution simulator. The goal is to avoid the far more common failure mode of producing a backtest that is materially cleaner than a retail-accessible implementation could ever achieve.

## LLM Factor Mining

The factor-mining subsystem is treated as a constrained research tool, not a source of unchecked strategy logic.

This section describes `scripts/sp500_run_factor_mining.py --live-call`, the OpenAI-compatible path configured in `configs/llm_sp500.yaml`. The published Claude Fable 5 and Sonnet 5 candidate sets and the five model-driven evolutionary arms were generated with Anthropic Claude models through the backends in `quantaalpha_us/factors/llm_client.py` and `quantaalpha_us/evo/backends.py`; see [research_notes.md](research_notes.md).

Runtime behavior of the `--live-call` path:

- exact primary model: `gpt-5.4`
- exact fallback model: `gemini-3.1-pro`
- OpenAI-compatible endpoint requirement
- strict model-catalog enforcement
- JSON-only output expectation
- sanitizer pass before any expression is accepted
- request and token budgets
- early halt on low-quality output

This is intended to keep model experimentation productive without allowing generated expressions to quietly degrade research quality.

## Setup

```bash
cd Alpha-Factor-Mining-Framework
./bootstrap.sh
source .venv/bin/activate
cp .env.example .env
set -a && source .env && set +a
```

Typical environment configuration includes:

- `CRSP_USERNAME`
- `CRSP_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `ALPACA_PAPER_API_KEY`
- `ALPACA_PAPER_API_SECRET`
- `ALPACA_LIVE_API_KEY`
- `ALPACA_LIVE_API_SECRET`

## Core Artifacts

The central pipeline artifacts are:

- `data/us_equities/reference/sp500_membership_daily.parquet`
- `data/us_equities/reference/gics_sectors.csv`
- `data/us_equities/reference/ticker_mapping.csv`
- `data/us_equities/processed/daily_bars.parquet`

## External Interfaces

The project interacts with several real external systems:

- WRDS / CRSP for research-grade historical data
- OpenAI-compatible chat-completions endpoints for factor mining
- Alpaca for execution and account state

That matters because the repository is not only a modeling exercise. It includes API integration, operational failure handling, and the practical engineering needed to move from research outputs to broker-facing actions.

## Representative Workflow

### 1. Build Membership

```bash
python scripts/sp500_build_membership.py
```

Source selection can also be made explicit:

```bash
python scripts/sp500_build_membership.py --source crsp
python scripts/sp500_build_membership_approx.py
```

### 2. Backfill Daily Bars

```bash
python scripts/sp500_backfill_history.py
python scripts/sp500_data_coverage_report.py
```

### 3. Generate Signals

```bash
python scripts/sp500_generate_signals.py --config configs/backtest_sp500_research.yaml
```

### 4. Run Walk-Forward Research

```bash
python scripts/sp500_run_research.py --config configs/backtest_sp500_research.yaml
```

### 5. Run LLM Factor Mining

```bash
python scripts/sp500_run_factor_mining.py --config configs/llm_sp500.yaml --live-call
```

### 6. Submit a Paper Rebalance

```bash
python scripts/trade_once.py --config configs/paper_sp500.yaml --dry-run
python scripts/trade_once.py --config configs/paper_sp500.yaml
```

### 7. Orchestrate Scheduled Modes

```bash
python scripts/sp500_orchestrator.py --mode research --run-date 2026-03-10
python scripts/sp500_orchestrator.py --mode paper --run-date 2026-03-10 --dry-run
python scripts/sp500_orchestrator.py --mode live --run-date 2026-03-10 --dry-run
```

## Script-Level Flow

The main CLI entrypoints correspond to specific parts of the pipeline:

- `scripts/sp500_build_membership.py`
  - builds historical membership from CRSP
- `scripts/sp500_build_membership_approx.py`
  - builds a constant-membership approximation from a current snapshot
- `scripts/sp500_backfill_history.py`
  - backfills historical daily bars
- `scripts/sp500_data_coverage_report.py`
  - measures member-day coverage of the dataset
- `scripts/sp500_ingest_daily.py`
  - performs daily incremental updates
- `scripts/sp500_generate_signals.py`
  - converts market data into target portfolio weights
- `scripts/sp500_run_research.py`
  - runs walk-forward research and validation
- `scripts/sp500_run_factor_mining.py`
  - generates and filters candidate factor expressions
- `scripts/trade_once.py`
  - submits one rebalance cycle
- `scripts/sp500_orchestrator.py`
  - coordinates multi-step scheduled runs

That division is deliberate. Each script has a narrow responsibility, which keeps the project easier to test, inspect, and operate.

## Research Outputs

A research run writes artifacts under `data/results/research/...`, including:

- walk-forward returns
- fold metadata
- research summary
- validation gate results

The validation layer is a first-class part of the project. A run completing successfully is not treated as equivalent to a strategy passing research review.

## Operational Tooling

Additional operational scripts include:

Daily report:

```bash
python scripts/sp500_daily_report.py
```

Kill switch:

```bash
python scripts/kill_switch.py --config configs/paper_sp500.yaml --level 2 --reason "manual test" --yes
```

## Why This Project Exists

The point of this repository is not to present a polished research result. It is to show a full-stack, research-to-execution implementation that takes data integrity, execution realism, and model governance seriously.

In practical terms, that means:

- a real historical universe path exists
- a lower-fidelity approximation path exists when needed
- research and trading share the same operational assumptions where possible
- LLM usage is constrained and reviewable
- promotion depends on validation, not narrative

## Tests

```bash
pytest
```

## Summary

This repository is best understood as a serious systematic trading workbench rather than a toy backtest or a generic LLM wrapper. It combines data engineering, research discipline, execution awareness, and model-safety controls in a single repo.

The remaining work is primarily on strategy quality, not scaffolding. That is the right stage for a project of this kind to be in.
