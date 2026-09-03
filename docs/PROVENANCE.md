# Provenance

This repository is a US-equities research stack whose design descends from
**QuantaAlpha** ([QuantaAlpha/QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha),
[arXiv:2602.07085](https://arxiv.org/abs/2602.07085)). This document records, at file
level, what is shared code and what was written from scratch.

The short answer: **no file in `quantaalpha_us/` is copied or adapted from upstream at
the code level.** What was inherited is architectural and conceptual, and is itemised in
[Concept-level inheritance](#concept-level-inheritance) below.

## Method

Comparison was mechanical, not from memory:

1. Every `.py` file in `quantaalpha_us/` (27 files) was compared against every `.py`
   file in the upstream package tree (159 files) — 4,293 pairwise comparisons.
2. Both sides were normalised before comparison: blank lines and comments stripped,
   whitespace collapsed, and package-name differences (`quantaalpha_us` vs
   `quantaalpha`) rewritten so that naming alone could not depress the score.
3. Similarity is `difflib.SequenceMatcher` ratio over the normalised line sequences.
   Each file below is reported against its **best match anywhere in the upstream tree**,
   not merely against a file of the same name.
4. Files sharing a basename with upstream were additionally diffed directly, and the
   exact intersecting lines were enumerated (reproduced below).

### Baseline caveat

The upstream tree used for this comparison was a local working copy at
`../QuantaAlpha_CN`. That copy is **not pristine**: it carries an earlier local rebrand
(39 files contain substituted project names) and has no git history to diff against.
The name substitutions were normalised away in step 2, and the comparison's conclusion
is one of *absence* of overlap, which a rebrand of this kind cannot manufacture — string
substitution can hide a matching identifier, not 500 matching lines of logic. The
numbers below should still be treated as measured against that working copy rather than
against an untouched upstream checkout. To re-verify against pristine upstream:

```bash
git clone --depth 1 https://github.com/QuantaAlpha/QuantaAlpha /tmp/qa_pristine
```

then re-run the comparison with `/tmp/qa_pristine/quantaalpha` as the upstream root.

## File-level table

`similarity` is the best normalised match against the entire upstream tree.
`nloc` is normalised (comment- and blank-stripped) line count.

| File | nloc | Closest upstream file | Similarity | Classification |
|---|---:|---|---:|---|
| `quantaalpha_us/__init__.py` | 9 | `quantaalpha/trading/__init__.py` | 0.160 | original |
| `quantaalpha_us/paths.py` | 17 | `quantaalpha/factors/coder/test.py` | 0.071 | original |
| `quantaalpha_us/backtest/__init__.py` | 14 | `quantaalpha/trading/__init__.py` | 0.133 | original |
| `quantaalpha_us/backtest/costs.py` | 39 | `quantaalpha/backtest/cost_model.py` | 0.173 | original |
| `quantaalpha_us/backtest/universe.py` | 103 | `quantaalpha/trading/portfolio_state.py` | 0.084 | original |
| `quantaalpha_us/backtest/validation.py` | 237 | `quantaalpha/contrib/model/coder/evolving_strategy.py` | 0.066 | original |
| `quantaalpha_us/backtest/walk_forward.py` | 511 | `quantaalpha/backtest/walk_forward.py` | 0.069 | original |
| `quantaalpha_us/data/__init__.py` | 11 | `quantaalpha/trading/__init__.py` | 0.148 | original |
| `quantaalpha_us/data/crsp_client.py` | 520 | `quantaalpha/contrib/model/coder/eva_utils.py` | 0.050 | original |
| `quantaalpha_us/data/market_data.py` | 24 | `quantaalpha/utils/warning_policy.py` | 0.100 | original |
| `quantaalpha_us/data/membership.py` | 187 | `quantaalpha/trading/reconciler.py` | 0.070 | original |
| `quantaalpha_us/data/quality.py` | 285 | `quantaalpha/coder/costeer/__init__.py` | 0.063 | original |
| `quantaalpha_us/factors/__init__.py` | 3 | (no match) | 0.000 | original |
| `quantaalpha_us/factors/experiment_trace.py` | 115 | `quantaalpha/trading/reconciler.py` | 0.065 | original |
| `quantaalpha_us/factors/expression_evaluator.py` | 289 | `quantaalpha/backtest/metrics.py` | 0.038 | original |
| `quantaalpha_us/factors/expression_sanitizer.py` | 223 | `quantaalpha/coder/costeer/__init__.py` | 0.063 | original |
| `quantaalpha_us/factors/factor_research.py` | 185 | `quantaalpha/backtest/benchmark.py` | 0.062 | original |
| `quantaalpha_us/factors/fundamentals.py` | 110 | `quantaalpha/trading/market_data.py` | 0.075 | original |
| `quantaalpha_us/factors/llm_client.py` | 173 | `quantaalpha/factors/experiment.py` | 0.064 | original |
| `quantaalpha_us/llm/__init__.py` | 4 | (no match) | 0.000 | original |
| `quantaalpha_us/llm/budget.py` | 85 | `quantaalpha/trading/session.py` | 0.089 | original |
| `quantaalpha_us/llm/mining.py` | 185 | `quantaalpha/trading/reconciler.py` | 0.062 | original |
| `quantaalpha_us/pipeline/__init__.py` | 3 | (no match) | 0.000 | original |
| `quantaalpha_us/pipeline/signal_generator.py` | 257 | `quantaalpha/backtest/walk_forward.py` | 0.054 | original |
| `quantaalpha_us/trading/__init__.py` | 22 | `quantaalpha/factors/regulator/__init__.py` | 0.109 | original |
| `quantaalpha_us/trading/alpaca_rest.py` | 84 | `quantaalpha/storage/event_log.py` | 0.078 | original |
| `quantaalpha_us/trading/risk.py` | 251 | `quantaalpha/contrib/model/coder/eva_utils.py` | 0.053 | original |

**Totals: 27 original, 0 adapted, 0 shared.** No file exceeds 0.173 similarity against
any upstream file. For calibration, two independent implementations of the same idea in
the same language typically land in this range purely on shared imports and idiom.

### The shared-basename files, examined directly

Four non-`__init__` files share a basename with an upstream file. In every case the
basename is the only thing shared — note that for three of the four, the upstream file
of the same name is not even the closest match in the tree.

| Pair | Lines US / upstream | Intersecting non-blank lines |
|---|---|---|
| `backtest/universe.py` vs `backtest/universe.py` | 123 / 163 | 7 |
| `backtest/walk_forward.py` vs `backtest/walk_forward.py` | 591 / 181 | 23 |
| `data/market_data.py` vs `trading/market_data.py` | 30 / 60 | 1 |
| `trading/risk.py` vs `trading/risk.py` | 286 / 133 | 6 |

The intersecting lines are language boilerplate, not logic. In full:

- `universe.py` — `from __future__ import annotations`, `from dataclasses import dataclass`,
  `from pathlib import Path`, `import pandas as pd`, `@dataclass`, `if not path.exists():`, `)`
- `walk_forward.py` — the same imports and `@dataclass`, plus the four window-boundary
  field names `train_start` / `train_end` / `test_start` / `test_end` and their
  constructor keywords. These are the standard vocabulary for a walk-forward split, and
  the surrounding 500+ lines share nothing.
- `market_data.py` — `from __future__ import annotations`, and nothing else.
- `trading/risk.py` — `from __future__ import annotations`,
  `from dataclasses import dataclass, field`, `@dataclass`, and three risk-config fields
  that coincide in both name and default: `min_positions: int = 5`,
  `min_cash_pct: float = 0.02`, `flatten_on_kill: bool = True`. This is the single
  closest thing to a borrowing found anywhere in the comparison. It is three
  configuration defaults, not an implementation.

## Concept-level inheritance

Not code, but genuinely owed to upstream and to the wider formulaic-alpha literature it
sits in:

**Kept from QuantaAlpha:**

- **The staged research pipeline.** Treating alpha discovery as
  ideate → express → sanitise → evaluate → gate, with the LLM confined to the ideation
  stage and every later stage mechanical and auditable.
- **The formulaic-alpha DSL as the LLM's output contract.** Constraining the model to
  emit a bounded expression grammar rather than free-form code is upstream's central
  design decision, and this repo adopts it.
- **The operator naming convention.** 22 of this repo's 32 operator names also appear in
  upstream's function library: `ABS`, `COUNT`, `DELAY`, `DELTA`, `EMA`, `LOG`, `MAX`,
  `MIN`, `RANK`, `SIGN`, `SQRT`, `TS_ARGMAX`, `TS_ARGMIN`, `TS_CORR`, `TS_COVARIANCE`,
  `TS_MAX`, `TS_MEAN`, `TS_MIN`, `TS_RANK`, `TS_STD`, `TS_SUM`, `ZSCORE`. These names are
  largely the common Alpha101/Qlib vocabulary that predates both projects; the
  implementations behind them here are independent, and the semantics differ (all
  time-series operators in this repo use strict windows, so warm-up periods are NaN).
- **The LLM-ideation framing** — that a frontier model is a hypothesis generator whose
  output is worthless until it survives an evaluation harness it cannot influence.

**Rebuilt from scratch, with no upstream counterpart:**

- **Data layer.** CRSP via WRDS (`data/crsp_client.py`, 520 nloc) against upstream's
  Qlib/China A-share stack. Different vendor, different identifiers, different schema.
- **Point-in-time membership filter** (`data/membership.py`). S&P 500 constituent
  history joined on PERMNO rather than ticker, so survivorship and symbol reuse are
  handled correctly. Upstream has no equivalent.
- **Rank-space deduplication** (`factors/factor_research.py`). The pairwise
  correlation cap is applied in rank space, because pooled Pearson on raw values is not
  invariant to a monotone cross-sectional transform.
- **Out-of-sample holdout** with frozen in-sample choices, including factor sign.
- **Sanitizer identifier and arity checks** (`factors/expression_sanitizer.py`), backed
  by a strict AST whitelist in the evaluator as a second guard.
- **Append-only, content-addressed experiment trace** (`factors/experiment_trace.py`),
  used as the multiple-testing denominator.
- **Claude Code LLM backend** (`factors/llm_client.py`) and run-budget controls
  (`llm/budget.py`).
- **Alpaca execution path and risk controls** (`trading/`).

**Not carried over:** upstream's evolutionary search itself — the trajectory-level
mutation and crossover operators that are the paper's actual contribution — has no
counterpart here. This repo does single-shot ideation plus selection. It is a
descendant of QuantaAlpha's scaffolding, not a reimplementation of its algorithm.

## Prompts

Upstream's prompt library (`factors/prompts/prompts.yaml` and siblings) is Jinja2-templated
and multi-round, carrying hypothesis-feedback history between iterations. This repo's
`configs/prompts/factor_gen_us.txt` is a single-shot prompt with a JSON schema contract.
No prompt text is shared between the two.

## Reproducing this comparison

The scan is a short script; the normalisation rules in [Method](#method) are the whole of
it. Re-running it against a pristine upstream clone is the recommended way to audit this
document.
