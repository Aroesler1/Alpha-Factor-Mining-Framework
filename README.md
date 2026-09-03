# LLMStrat

LLMStrat is a US equities research and execution stack for daily S&P 500 alpha mining. It builds a point-in-time universe, maintains market data, evaluates candidate signals with walk-forward controls, and can route approved portfolios into Alpaca paper or live trading with explicit risk checks.

## Provenance

This project descends from **QuantaAlpha** — [QuantaAlpha/QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha), MIT License, paper: *QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining* ([arXiv:2602.07085](https://arxiv.org/abs/2602.07085)). QuantaAlpha targets the China A-share market; this is a rewrite for US equities.

**Kept from QuantaAlpha:**

- The **staged research pipeline** concept: ideate, express, sanitise, evaluate, gate — with the LLM confined to the ideation stage and every stage after it mechanical and auditable.
- The **formulaic-alpha DSL** as the LLM's output contract, rather than free-form generated code. 22 of this repo's 32 operator names (`TS_MEAN`, `RANK`, `DELAY`, `TS_CORR`, `ZSCORE`, …) also appear in upstream's function library, though most are the common Alpha101/Qlib vocabulary predating both projects.
- The **LLM-ideation framing**: a frontier model is a hypothesis generator whose output is worthless until it survives an evaluation harness it cannot influence.

**Rebuilt here, with no upstream counterpart:** the data layer (CRSP via WRDS, replacing Qlib/A-share); the point-in-time S&P 500 membership filter joined on PERMNO rather than ticker; rank-space deduplication of candidate signals; the out-of-sample holdout with frozen in-sample sign; the sanitizer's identifier and arity checks; the Claude Code LLM backend; and the append-only, content-addressed experiment trace.

**Not carried over:** upstream's evolutionary search itself — the trajectory-level mutation and crossover operators that are the paper's actual contribution. This repo does single-shot ideation plus selection. It inherits QuantaAlpha's scaffolding, not its algorithm.

At the code level the two share nothing: a file-by-file comparison of all 27 modules in `quantaalpha_us/` against all 159 upstream modules found no file above 0.17 similarity, and the four files that share a basename with an upstream file share only imports and `@dataclass` decorators. The full table, the method, and its caveats are in **[docs/PROVENANCE.md](docs/PROVENANCE.md)**.

### On comparing IC against the paper

Upstream's headline result is an **IC of 0.1501 on CSI 300** (GPT-5.2 backbone, ARR 27.75%, MDD 7.98%). This repo measures mean ICs of **0.003–0.011** on the S&P 500. Those numbers are not comparable, and the gap should not be read as either a validation or a failure of this port:

- **Different market.** Upstream's headline is CSI 300. US large-cap is the most heavily arbitraged equity universe in the world; daily cross-sectional ICs there are structurally smaller than in A-shares.
- **The paper reports no US IC.** QuantaAlpha does test zero-shot transfer of CSI-300-mined factors onto the S&P 500, and reports it as successful — roughly 137% cumulative excess return over the 2022–2025 test window (Figure 1). But that is a portfolio-level cumulative excess return under a TopkDropout strategy with China-calibrated transaction costs, not an information coefficient. There is no published upstream S&P 500 IC to benchmark against.
- **Different search.** The evolutionary mutation/crossover loop that produces upstream's headline number is not implemented here.

The honest statement is the one already made under [Known limits](#known-limits): ICs of 0.003–0.011 are weak against the 0.02–0.05 of a decent published factor, and the large t-statistics come from roughly 6,500 trading days rather than from effect size. Factor quality, not tooling, is this repo's binding constraint.

### Citation

If you use this work, please cite the upstream paper:

```bibtex
@misc{han2026quantaalphaevolutionaryframeworkllmdriven,
      title={QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining},
      author={Jun Han and Shuo Zhang and Wei Li and Zhi Yang and Yifan Dong and Tu Hu and Jialuo Yuan and Xiaomin Yu and Yumo Zhu and Fangqi Lou and Xin Guo and Zhaowei Liu and Tianyi Jiang and Ruichuan An and Jingping Liu and Biao Wu and Rongze Chen and Kunyi Wang and Yifan Wang and Sen Hu and Xinbing Kong and Liwen Zhang and Ronghao Chen and Huacan Wang},
      year={2026},
      eprint={2602.07085},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2602.07085},
}
```

## Repository layout

- `configs/`: research, paper, live, and LLM runtime configuration
- `quantaalpha_us/`: reusable package code for data, factors, backtests, and execution
- `scripts/`: entry points for universe construction, ingestion, research, signal generation, and trading
- `tests/`: regression coverage for data quality, factor mining, risk controls, and walk-forward validation
- `bootstrap.sh`: one-command environment setup for a fresh clone

## Setup

```bash
./bootstrap.sh
source .venv/bin/activate
```

## CLI usage

Build or refresh the S&P 500 membership table:

```bash
python scripts/sp500_build_membership.py --help
```

Backfill or refresh daily market data:

```bash
python scripts/sp500_ingest_daily.py --help
```

Run the core walk-forward research loop:

```bash
python scripts/sp500_run_research.py --config configs/backtest_sp500_research.yaml
```

## Methodology

The system is organized as a staged research pipeline rather than a single notebook. It first constructs a point-in-time investable universe, ingests and validates daily OHLCV data, and computes baseline cross-sectional features. Those signals feed a constrained long-only portfolio construction step, which is then evaluated in walk-forward windows with explicit costs, turnover limits, and promotion gates. Frontier LLMs are used only for bounded factor ideation and are surrounded by validation, budget controls, and expression sanitization before any candidate reaches research or trading.

## Output

Typical runs produce:

- point-in-time universe and reference tables under `data/us_equities`
- processed bars and coverage artifacts for daily research
- signal files and walk-forward backtest outputs
- factor-mining candidate files and validation summaries
- broker-facing rebalance intents for paper or live deployment

## Mined-factor research loop (2026-08)

The loop from mining to backtest is now closed end to end:

1. **Evaluator** (`quantaalpha_us/factors/expression_evaluator.py`): sanitized expression strings such as `TS_MEAN($close, 21) / (TS_STD($close, 21) + 1e-8)` are parsed with a strict AST whitelist (second guard behind the sanitizer) and evaluated into date x symbol signal panels. All time-series operators use strict windows, so warm-up periods are NaN rather than biased.
2. **Research scoring** (`quantaalpha_us/factors/factor_research.py` and `scripts/sp500_score_mined_factors.py`): candidates are scored by daily cross-sectional Spearman IC against forward open-to-open returns under the repo's execution convention, with an IC t-statistic across days and a signal-autocorrelation turnover proxy. Selection is greedy by |t-stat| with a pairwise signal-correlation cap, because string-level dedup cannot catch an LLM restating one idea five ways. That cap is applied in **rank space**, matching the rank IC selection ranks on: pooled Pearson on raw values is not invariant to a monotone cross-sectional transform, so `CS_RANK(X)` and `X` — identical orderings, identical IC — measured 0.21 against each other and both entered a set billed as uncorrelated. In rank space that pair is 1.00.
3. **Backtest integration**: `signals.mined_expressions` in the research config feeds selected expressions into `build_features`, where they are cross-sectionally ranked and averaged into the score next to the baseline factors, and the gate-6 stability measurement covers them automatically.

`configs/mined_factors_claude_2026-08.txt` ships a candidate set generated by Claude Fable 5 (Anthropic). These are candidates, not validated factors: every one must clear the IC scoring script on research-grade data and then the walk-forward promotion gates. A unit test guarantees the whole file sanitizes and evaluates.

## Auditable experiment trace

A mining loop that keeps only its winners cannot be audited. The rejected hypotheses are what show whether a search was disciplined or simply ran until something passed, and they are exactly what a leaderboard discards.

`quantaalpha_us/factors/experiment_trace.py` records every hypothesis considered — the expression, the reasoning that produced it, the scores it earned, and the verdict, failures included. It follows the auditable-trace design argued for in [arXiv 2604.26747](https://arxiv.org/abs/2604.26747), with two properties that make it evidence rather than decoration:

- **Append-only.** Records are JSON Lines, flushed and fsynced on write, never rewritten. There is no update or delete operation, because a trace that can be edited afterwards proves nothing about what was tried.
- **Content-addressed.** Each record carries a hash of the normalised expression, so the same idea proposed twice is detected across runs even when spelled differently. That is what lets the trace act as search memory rather than a log.

The trace is also **the multiple-testing denominator**. Running the fundamental candidate set:

```
verdicts: {'selected': 10, 'rejected_correlated': 1, 'rejected_score': 1, 'distinct_expressions': 12}
distinct hypotheses ever tried: 12 (use this as the DSR trial count, not 10)
```

A Deflated Sharpe computed against the survivors understates the search by exactly the number of rejections. Computed against the trace, it reflects what was actually tried — and across many runs the trace accumulates, so the denominator keeps growing the way an honest one should.

```bash
python scripts/sp500_score_mined_factors.py --bars <panel> --trace data/trace.jsonl
```

## Out-of-sample holdout

Factors are selected on **2000-2017** and reported on **2018-2025**, with the
in-sample choices frozen: which expressions, and which sign each was oriented
to. Refitting the sign out of sample would be the same overfitting one layer
down.

| candidate set | keeps sign OOS | median IC retention |
|---|---|---|
| Claude Fable 5 (20 candidates) | 7 / 8 | **29.5%** |
| Claude Sonnet 5 (54 candidates) | 10 / 10 | **52.8%** |
| fundamental (12 candidates) | 8 / 9 | **93.3%** |

The price-only sets lose most of their edge. On the Fable set, **one of eight
factors clears |t| > 2 out of sample**, against eight of eight in sample by
construction, and one flips sign outright. That is the factor-zoo result
measured on this repo's own factors rather than cited from a paper.

Signals are computed over the full panel and only the *IC dates* are
restricted. Every operator is backward-looking, so this leaks nothing, and
unlike slicing the bars it does not blank the holdout's first 252 days to
rolling-window warm-up.

`--train-end none` selects on the full sample and says so.

## Universe

Scoring is restricted to **point-in-time S&P 500 membership**, joined on
`(date, permno)` before any factor is computed.

This is not cosmetic. `daily_bars.parquet` carries every name that was ever a
constituent over 2000-2025, which is ~745 names per date against the index's
~502. On 2005-06-15, 315 of the 814 names in the panel (39%) were not in the
S&P 500 that day, and they are future members as often as former ones. Ranking
a factor cross-sectionally over that panel scores it on a universe no strategy
could have held, and it inflates significance: the best candidate measured
|t| = 8.21 unfiltered against 7.56 filtered.

`--membership none` scores the raw panel and says so loudly; a missing
membership file is a hard error rather than a silent skip.

## Data source

Market data comes from **CRSP via WRDS**. Every call site this repo actually
uses — daily bars, bulk daily bars, and the ticker mapping — is served by
`quantaalpha_us/data/crsp_client.py`, which also carries PERMNO identity,
delisting returns and point-in-time membership that a plain vendor EOD feed
does not.

`build_market_data_client()` returns that client and raises if
`CRSP_USERNAME`/`CRSP_API_KEY` are absent, rather than silently degrading to a
weaker source. An EODHD fallback was removed in 2026-09: it was never the source
of any published result here, and a second vendor path that nothing exercises is
a maintenance and credential liability rather than resilience.

Running this repo therefore requires a WRDS entitlement.

CRSP data is licensed and is deliberately not committed: the repo ships code,
derived factor scores and figures, not raw vendor data.

## Known limits

- Factor provenance is not recorded: `factor_scores.csv` carries no column
  identifying which model generated a candidate, so two mining runs cannot be
  compared after the fact.
- Factor quality, not tooling, is the binding limit: mean ICs of 0.003-0.011 are weak against 0.02-0.05 for a decent published factor, and the large t-statistics (up to 7.6) come from ~6,500 trading days rather than effect size.
- Research quality still depends on the quality and timeliness of external data providers
- Daily signals and retail-oriented execution assumptions are intentionally conservative and do not represent intraday HFT infrastructure
- LLM factor generation is bounded and audited, but it still needs human judgment before production use

## Validation notes (2026-08 revision)

- Gate-1 now uses the actual Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014): the probability that the strategy's true Sharpe exceeds the expected maximum Sharpe of `n_trials` zero-skill strategies, adjusted for skewness and kurtosis. The threshold is a probability (default 0.95). The previous implementation was a heuristic penalty and overstated significance.
- Gate-6 (factor stability) is now measured: per-fold Spearman rank ICs of each baseline factor against realized entry-to-exit returns, top-3 factor sets per fold, mean Jaccard overlap across consecutive folds. It was previously hard-coded to pass.
- Holdings that drop out of the tradable context (index exit, missing bar) are now explicitly liquidated at the entry open with transaction costs; they previously converted to cash silently with no cost or turnover.

## Notes

The repository is maintained at `Aroesler1/LLMStrat` and keeps the `quantaalpha_us` module path for runtime compatibility with earlier internal tooling

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

The goal is not to create a perfect live execution simulator. The goal is to avoid the far more common failure mode of producing a backtest that is materially cleaner than a retail-accessible implementation could ever achieve.

## LLM Factor Mining

The factor-mining subsystem is treated as a constrained research tool, not a source of unchecked strategy logic.

Current runtime behavior:

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
cd /absolute/path/to/QuantaAlpha_US
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

`QuantaAlpha_US` is best understood as a serious systematic trading workbench rather than a toy backtest or a generic LLM wrapper. It combines data engineering, research discipline, execution awareness, and model-safety controls in a single repo.

The remaining work is primarily on strategy quality, not scaffolding. That is the right stage for a project of this kind to be in.
