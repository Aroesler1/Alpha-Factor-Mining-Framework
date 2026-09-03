# Handoff

## Current goal

Answer "does the LLM add anything?" with a measured baseline, and record the
repo's relationship to its upstream.

## Verified state (2026-09-03)

### Baselines (this session)

The answer is **no**: the LLM sets do not beat expressions drawn at random from
the same grammar, and both are beaten by Alpha101. Reproduce with one command:

```
python scripts/sp500_run_baseline_comparison.py --bars data/us_equities/processed/daily_bars.parquet
```

Runtime ~25 min on the full panel. Artifacts land in `data/baseline_comparison/`.

| set | best \|t\| IS | OOS median retention | sign held |
|---|---:|---:|---:|
| Claude Fable 5 (20) | 9.82 | 29.5% | 7/8 |
| Claude Sonnet 5 (54) | 11.64 | 52.8% | 10/10 |
| random-grammar x5 seeds (54 each) | 9.11 - 12.41 | 10.7% - 66.6% | 5-8/10 |
| Alpha101 (49) | 12.11 | 58.8% | 10/10 |

- E[max \|t\|] under an iid null at N=54 is 2.54. Every set including pure noise
  clears 9, because daily ICs are autocorrelated and the t-stat's standard error
  assumes independence it does not have. The random-grammar row, not the iid
  figure, is the real bar.
- Memorization: 2 of 53 Sonnet candidates (4%) exceed 0.9 rank-space correlation
  to a published alpha; 0 of 20 Fable. Median closest match ~0.41 for both.

### Files added

- `quantaalpha_us/factors/random_expressions.py` - typed grammar sampler. Reads
  the function set, arities and fields off `ExpressionSanitizer` rather than
  restating them; `test_signature_map_covers_the_sanitizer_exactly` fails if the
  two drift.
- `configs/alpha101_us.txt` - 49 of 101 transcribed, 52 dropped with a reason
  each in place. A test asserts 49 + 52 = 101.
- `scripts/sp500_run_baseline_comparison.py` - the one command.
- `tests/test_random_expressions.py`, `tests/test_alpha101_transcription.py` - 16 tests.

### Files changed

- `expression_evaluator.build_field_panels` now reindexes every field panel onto
  a common (dates x symbols) union. Field coverage differed (`open`/`high`/`low`
  had 1,275 columns against `close`'s 1,734 pre-filter), which hard-crashed
  `MIN`/`MAX` and every comparison, and meant a cross-sectional `RANK` ranked
  over a different universe depending on which fields an expression touched.
- `factor_research` gained `ranked_flat`, `_corr_flat`, `rank_space_correlation`
  and `holdout_frame`; `select_uncorrelated` now caches the kept set's ranks.
- `scripts/sp500_score_mined_factors.py` uses the shared `holdout_frame`.

### Validation

- `pytest tests/ -q` -> **97 passed** (81 before this session, 16 new).
- Regression check on the evaluator and refactor: re-running the Fable set
  through the scoring CLI reproduces the stored numbers exactly (7/8 sign held,
  29.5% median retention, per-factor ICs identical to 6 dp).

### Provenance (earlier session)

`README.md` `## Provenance` + `docs/PROVENANCE.md`: file-level diff of all 27
`quantaalpha_us/` modules against all 159 upstream modules, max similarity 0.173,
27 original / 0 adapted / 0 shared. Commits `f1def2c`, `49ff805`.

## Known risks / caveats

- **Alpha101 transcription was not checked against the paper text.** Structure
  and operator mapping are sound and every line sanitizes and evaluates, but the
  fractional constants in alphas above #60 (0.876703, 0.518371) were written from
  the formula list, not re-read from arXiv 1601.00991. Most of that range is
  dropped for `vwap` anyway. Spot-check before the memorization number is used
  anywhere load-bearing.
- **`adv{d}` is an interpretation in five alphas.** The paper defines it as
  average daily *dollar* volume, but alphas 7, 17, 21, 39 and 43 compare or
  divide it against share `volume`; under the literal reading alpha007 is -1 on
  99% of observations. Those five use `TS_MEAN($volume, d)`, flagged in the file
  header.
- **No post-training-cutoff window exists.** The panel ends 2025-12-31 and both
  generating models were trained through 2025+, so the 2018-2025 holdout is
  inside their training window. A real test needs 2026 bars appended.
- The random-grammar top-up replaces the ~25% of draws that evaluate to a
  constant cross-section. It reads the feature panel only, never forward returns.
- Upstream baseline for `docs/PROVENANCE.md` was a local rebranded copy, not a
  pristine clone (see that file).

## Next action

Append 2026 bars and run the post-cutoff test: score the already-frozen
expressions on 2026 alone against the same five random-grammar seeds. Everything
else for that test is already built.
