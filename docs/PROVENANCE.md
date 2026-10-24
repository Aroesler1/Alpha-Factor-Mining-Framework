> Updated scope, 2026-09-08: the original comparison below predates this repository's completed Part C evolutionary loop. Statements that evolution is absent describe the earlier one-shot study only. Current loop code is independently implemented; historical results have unpurged selection labels. See [integrity_audit.md](integrity_audit.md).

# Provenance

This repository is a US-equities research stack whose design descends from
**QuantaAlpha** ([QuantaAlpha/QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha),
[arXiv:2602.07085](https://arxiv.org/abs/2602.07085)). This document records, at file
level, what is shared code and what was written from scratch.

The short answer: **no file in `quantaalpha_us/` is copied or adapted from upstream at
the code level.** What was inherited is architectural and conceptual, and is itemised in
[Concept-level inheritance](#concept-level-inheritance) below.

## Re-check against the public upstream repository (2026-10-23)

The file-level comparison below used a local, partly rebranded copy of upstream and
covered the 27 modules that existed at the time. It was repeated against a fresh
shallow clone of the public repository (commit `b7ceb27`, 135 Python files) with
[`scripts/provenance_upstream_compare.py`](../scripts/provenance_upstream_compare.py),
this time covering all 118 Python files in `quantaalpha_us/` (including the later
`evo/` package), `scripts/` and `tests/`, plus the 7 prompt templates in
`configs/prompts/`. No local file exceeded 0.19 line-sequence similarity to any
upstream file (the highest scores are short `__init__.py` files). The longest run of
consecutive shared normalised lines is three, and those three are
`from __future__ import annotations`, `import json` and `import re`. Every other shared
line is an import, a method signature such as `def to_dict(self) -> dict[str, Any]:`,
or boilerplate such as `if __name__ == "__main__":`. No prompt text is shared.

To repeat it:

```bash
git clone --depth 1 https://github.com/QuantaAlpha/QuantaAlpha /tmp/qa_upstream
python scripts/provenance_upstream_compare.py --upstream /tmp/qa_upstream
```

Upstream declares the MIT license in its `pyproject.toml` classifiers and README
badge; that commit has no LICENSE file. No upstream code or prompt text is included
here, so this repository does not reproduce upstream's license notice. The summary
and citation at the end of this document credit the design.

## Method

Comparison was mechanical, not from memory:

1. Every `.py` file in `quantaalpha_us/` (27 files) was compared against every `.py`
   file in the upstream package tree (159 files), 4,293 pairwise comparisons.
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
is one of *absence* of overlap, which a rebrand of this kind cannot manufacture: string
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
basename is the only thing shared. Note that for three of the four, the upstream file
of the same name is not even the closest match in the tree.

| Pair | Lines US / upstream | Intersecting non-blank lines |
|---|---|---|
| `backtest/universe.py` vs `backtest/universe.py` | 123 / 163 | 7 |
| `backtest/walk_forward.py` vs `backtest/walk_forward.py` | 591 / 181 | 23 |
| `data/market_data.py` vs `trading/market_data.py` | 30 / 60 | 1 |
| `trading/risk.py` vs `trading/risk.py` | 286 / 133 | 6 |

The intersecting lines are language boilerplate, not logic. In full:

- `universe.py`: `from __future__ import annotations`, `from dataclasses import dataclass`,
  `from pathlib import Path`, `import pandas as pd`, `@dataclass`, `if not path.exists():`, `)`
- `walk_forward.py`: the same imports and `@dataclass`, plus the four window-boundary
  field names `train_start` / `train_end` / `test_start` / `test_end` and their
  constructor keywords. These are the standard vocabulary for a walk-forward split, and
  the surrounding 500+ lines share nothing.
- `market_data.py`: `from __future__ import annotations`, and nothing else.
- `trading/risk.py`: `from __future__ import annotations`,
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
- **The LLM-ideation framing**: that a frontier model is a hypothesis generator whose
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

**Not carried over:** upstream's evolutionary search itself (the trajectory-level
mutation and crossover operators that are the paper's actual contribution) has no
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

## Summary previously kept in the README

This project descends from **QuantaAlpha**: [QuantaAlpha/QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha), MIT License, paper: *QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining* ([arXiv:2602.07085](https://arxiv.org/abs/2602.07085)). QuantaAlpha targets the China A-share market; this is a rewrite for US equities.

**Kept from QuantaAlpha:**

- The **staged research pipeline** concept: ideate, express, sanitise, evaluate, gate, with the LLM confined to the ideation stage and every stage after it mechanical and auditable.
- The **formulaic-alpha DSL** as the LLM's output contract, rather than free-form generated code. 22 of this repo's 32 operator names (`TS_MEAN`, `RANK`, `DELAY`, `TS_CORR`, `ZSCORE`, …) also appear in upstream's function library, though most are the common Alpha101/Qlib vocabulary predating both projects.
- The **LLM-ideation framing**: a frontier model is a hypothesis generator whose output is worthless until it survives an evaluation harness it cannot influence.

**Rebuilt here, with no upstream counterpart:** the data layer (CRSP via WRDS, replacing Qlib/A-share); the point-in-time S&P 500 membership filter joined on PERMNO rather than ticker; rank-space deduplication of candidate signals; the out-of-sample holdout with frozen in-sample sign; the sanitizer's identifier and arity checks; the Claude Code LLM backend; and the append-only, content-addressed experiment trace.

**Search implementation:** the original study used single-shot ideation. The completed Part C adds this repository's own evolutionary loop, with typed formula edits, archive selection and GP controls. It is not a code-level reproduction of upstream's trajectory search.

At the code level the two share nothing: a file-by-file comparison of all 27 modules in `quantaalpha_us/` against all 159 upstream modules found no file above 0.17 similarity, and the four files that share a basename with an upstream file share only imports and `@dataclass` decorators. The full table, the method, and its caveats are in this document.

### Comparing IC against the paper

Upstream v1 reported CSI 300 IC 0.1501, ARR 27.75% and MDD 7.98%. Its [v3 dated 2026-05-18](https://arxiv.org/html/2602.07085v3) reports IC 0.0472, ARR 4.68% and MDD 11.8%; S&P 500 transfer cumulative excess return changes from about 137% to 19.1%. These versioned literature values are recorded in [reports/literature_versions.csv](../reports/literature_versions.csv).

Different universes, costs, IC conventions and search procedures prevent a direct performance ranking. There is no universal IC cutoff for a useful factor independent of its horizon, turnover and exposures. The relevant comparisons here are the published and random controls under the same local protocol.

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
