# Evolutionary Alpha Factor Mining

A research framework that uses large language models and genetic programming to search
for formulaic stock-selection signals in S&P 500 stocks, then tests whether that search
beats simple baselines.

Recent papers report large gains from letting an LLM propose and evolve alpha formulas.
Those gains are hard to trust. Mined factors face heavy multiple testing, the model may
be recalling published alphas, and look-ahead can leak in at the window boundaries. I
built an evolutionary mining loop and ran it next to controls that share every gate,
fitness rule and archive rule but swap the proposer: a genetic-programming search,
random formulas from the same grammar, and the published Alpha101 set. That way any
difference in the results comes from the proposer.

## Results

Sample: CRSP daily data for S&P 500 members on each date, 2000-01-03 to 2025-12-31:
4,874,082 stock-days for 1,107 stocks (1,734 ticker symbols) before the point-in-time
membership filter ([panel_summary.csv](reports/panel_summary.csv)). The evolutionary
arms fit on 2000-2013, use 2014-2017 for validation and report 2018-2025. The one-shot
sets were selected on 2000-2017.

The searches ran in 2026-09 on a panel with a price-scale bug. In 2026-10 I fixed the
bug and re-scored every frozen archive and factor set on the corrected panel with purged
labels. Every number below comes from that rescore; the search itself was not re-run.
The [audit history](docs/audit_history.md) lists what changed.

- **Evolutionary arms against controls.** I combined each arm's top 20 factors with
  equal weights. Mean daily rank IC on the 2018-2025 holdout was 0.0101 for the Claude
  Sonnet 5 loop (Newey-West t 3.42), 0.0081 for the GP loop, 0.0088 for Alpha101 and
  0.0005 for random formulas. The other LLM arms scored 0.0058 to 0.0100.
  Source: [evo_combined_table.csv](data/factor_zoo/evo_combined_table.csv).
- **No significant search advantage after familywise correction.** Sonnet loop vs GP
  loop is +0.0020 IC with raw p = 0.462. Loop vs one-shot gives raw p = 0.961 for
  Sonnet and 0.793 for Opus 5. The smallest raw p of the five comparisons is 0.026
  (Sonnet loop with full vs fitness-only feedback), which becomes 0.13 after Holm
  correction. Source: [evo_comparison_audit.csv](reports/evo_comparison_audit.csv).
- **One-shot LLM factors are no stronger than random formulas in sample.** The best
  in-sample |t| on 2000-2017 was 6.74 for Sonnet 5 and 7.03 for Fable 5, against 8.47
  for Alpha101 and 6.49 to 11.35 for five same-sized random-grammar sets. They held up
  better out of sample: on 2018-2025 the selected factors kept a median 37.3% (Sonnet 5)
  and 28.1% (Fable 5) of their in-sample IC, against at most 21.7% for the random sets
  and 60.9% for Alpha101. Source: [baseline_comparison.csv](reports/baseline_comparison.csv).
- **Factor-zoo correction.** I pooled 401 factors: 85 from LLMs, 266 random and 50
  from Alpha101. A Romano-Wolf stepdown keeps 24% of the LLM factors, 30% of the
  random ones and 52% of Alpha101. Source: [rw_survivors.csv](reports/rw_survivors.csv).
- **Little outright copying of Alpha101.** One of 53 scorable Sonnet 5 factors and none
  of 20 Fable 5 factors correlate above 0.9 with one of the 50 transcribed Alpha101
  formulas. Source: [memorization_test.csv](data/baseline_comparison/memorization_test.csv).

## How it works

1. **Data.** The CRSP daily stock file (`dsf_v2`) comes through WRDS for every stock
   that was ever an S&P 500 member. Opens, highs and lows are put on the same
   split-adjusted scale as the close, and every series is keyed by PERMNO. Before any
   factor is scored, the panel is filtered to point-in-time index membership. Optional
   Compustat quarterly fields are stamped from the report date, not the period end.
2. **Formula language.** Each factor is a short expression such as
   `TS_MEAN($close, 21) / (TS_STD($close, 21) + 1e-8)`. A sanitizer enforces the
   grammar, field names, arity and depth. The evaluator only runs whitelisted syntax
   and leaves rolling-window warm-up periods empty.
3. **Proposers.** Claude Sonnet 5 or Opus 5 runs through the Claude Code CLI, or a
   genetic-programming control makes random typed tree edits. Four operators
   (explore, mutate, crossover, simplify) and a short reflection memory run on three
   islands: price trend, liquidity and volume, and fundamentals. The six search arms
   are a loop and a one-shot set for each of Sonnet 5 and Opus 5, a Sonnet 5 loop that
   sees only its fitness score, and the GP loop. The one-shot baseline comparison uses
   two further candidate sets written by Claude Sonnet 5 and Claude Fable 5.
4. **Gates and fitness.** Cheap checks run first: syntax, complexity, shared
   sub-expressions with Alpha101 or the archive, coverage, and rank correlation with
   archive members. Fitness is the block-bootstrap |t| of the daily rank IC on
   2000-2013, scaled by year-to-year stability, minus complexity and turnover
   penalties. The label is the next-open to following-open return, and labels that
   end after a window's last day are dropped.
5. **Archive.** A MAP-Elites archive has 18 niches: IC half-life (3 buckets) by data
   family (3) by turnover (2), with up to 8 factors per niche. Loop arms run up to 8
   rounds and stop early when the validation IC stops improving. One-shot arms make a
   single call per island.
6. **Evaluation.** Each arm's top 20 factors are combined with equal weights and
   with ridge weights, then scored on 2018-2025. Arms are compared with a paired
   block bootstrap (21-day blocks, 2,000 draws) and Holm correction across the five
   comparisons.
7. **Audit trail.** Every prompt and raw reply is written to disk before it is parsed.
   [docs/lineage.md](docs/lineage.md) traces every top-20 factor to its parents and
   the model's stated rationale, and the rescore records a hash of every input in
   [evo_rescore_manifest.json](data/factor_zoo/evo_rescore_manifest.json).

## Limitations

- The archives, niches and gate thresholds come from searches run on the old panel,
  whose selection labels also crossed window cutoffs. The rescore fixes the scores, not
  the search path, so the holdout figures are not clean out-of-sample evidence. Network
  failures cut the Sonnet 5 loop short in its last round, so it has 7 complete rounds
  of 8.
- Split-adjusted price levels carry information about later splits. Ratios and returns
  are unaffected, but a formula that ranks a raw price level (for example
  `RANK($close)`) can still look ahead.
- The Claude models were most likely trained on text covering 2018-2025, so this
  holdout cannot tell reasoning apart from recall.
- The signals are weak (combined holdout ICs of 0.0005 to 0.0101), and I report no
  cost-adjusted portfolio for the mined factors. The paper-trading code is
  infrastructure, not a validated strategy.
- The Compustat cache matches restated data for 32 of 32 sampled keys but unrevised
  data for only 26 of 32, so fundamental factors may contain restatement look-ahead.
  See [fundamentals_provenance_audit.csv](reports/fundamentals_provenance_audit.csv).

## Repository layout

| Path | Contents |
|---|---|
| `quantaalpha_us/evo/` | Evolutionary loop: operators, islands, gates, fitness, archive, scoring, LLM and GP backends |
| `quantaalpha_us/factors/` | Formula sanitizer and evaluator, IC panels, Alpha101 transcription, random-formula generator, multiple-testing and expression-tree tools |
| `quantaalpha_us/data/`, `backtest/`, `trading/` | CRSP/WRDS client and membership, walk-forward backtest with validation gates, Alpaca execution and risk checks |
| `scripts/` | Command-line entry points, one per experiment or pipeline step |
| `configs/` | Candidate factor files, Alpha101 transcription, prompts, frozen thresholds, backtest and trading configs |
| `data/evo_runs/` | Every prompt and raw model reply, plus the archives and rejections for the six arms |
| `data/factor_zoo/`, `data/baseline_comparison/`, `reports/` | Derived result tables quoted in this README and the docs |
| `docs/` | Research notes, audit history, factor lineage, provenance |
| `tests/` | Offline tests on synthetic panels, including an end-to-end mock evolutionary run |

## Running it

These run without licensed data. They use the tests, the committed result tables, and
an offline expression-tree comparison:

```bash
./bootstrap.sh && source .venv/bin/activate   # creates .venv and installs requirements.txt
python -m pytest tests -q
python scripts/sp500_evo_audit.py --check                  # re-derives the Holm-adjusted comparisons
python scripts/sp500_build_reports.py --check              # rebuilds reports/ summaries from data/
python scripts/sp500_fundamentals_provenance.py --check    # rebuilds the provenance report from hashes
python scripts/sp500_structure_report.py --out-dir /tmp/structure_report
```

Everything else needs a WRDS subscription with CRSP access, plus Compustat for the
fundamental fields. Copy `.env.example` to `.env` and set `CRSP_USERNAME` and
`CRSP_API_KEY`. After you approve the Duo prompt, set `WRDS_DUO_READY=1`. Then:

```bash
python scripts/sp500_build_membership.py --source crsp
python scripts/sp500_backfill_history.py
python scripts/sp500_build_fundamentals_panel.py       # needs local comp.fundq and CCM link files
python scripts/sp500_build_factor_pool.py              # IC cache for the factor-zoo scripts
python scripts/sp500_run_baseline_comparison.py --reuse-random
python scripts/sp500_evo_final_table.py --rescore --out-dir /tmp/evo_rescore
python scripts/sp500_run_evolution.py --arm my-gp-run --backend gp
```

Pulling `comp.fundq` and the CRSP/Compustat link table from WRDS is not scripted.
Model-driven arms also need the Claude Code CLI. Licensed CRSP and Compustat data are
not committed; see [DATA.md](DATA.md).

## More detail

- [docs/research_notes.md](docs/research_notes.md): the full write-up covering one-shot
  mining, baselines, the memorization and paraphrase checks, the factor-zoo
  correction, Alpha101 publication decay, signal horizons, and the evolutionary loop
  design with per-arm results and costs.
- [docs/audit_history.md](docs/audit_history.md) and
  [docs/integrity_audit.md](docs/integrity_audit.md): the 2026-10 panel correction with
  old and new numbers, the label-leakage repair, and earlier corrections. The pre-fix
  write-up is kept in [docs/archive/](docs/archive/research_notes_2026-09.md).
- [docs/frozen_new_experiment_protocol.md](docs/frozen_new_experiment_protocol.md):
  the rules any new test must follow.
- [docs/lineage.md](docs/lineage.md): prompt-to-factor lineage for every top-20
  factor.
- [docs/system_overview.md](docs/system_overview.md): universe construction, the
  walk-forward backtest, and the Alpaca paper and live trading path.
- [DATA.md](DATA.md): data sources, which CRSP tape is used, price adjustment, and what
  is and is not committed.
- [docs/PROVENANCE.md](docs/PROVENANCE.md): relationship to
  [QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha) (MIT,
  [arXiv:2602.07085](https://arxiv.org/abs/2602.07085)), whose staged LLM-mining
  design this project follows. The code and prompts here were written independently;
  that file also has the citation.
