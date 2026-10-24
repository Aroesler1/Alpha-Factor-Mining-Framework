# Audit history

Integrity notes and corrections, kept in one place so the README can stay short.
The detailed record of the 2026-09 integrity repair is in
[integrity_audit.md](integrity_audit.md); the protocol any new experiment must follow
is in [frozen_new_experiment_protocol.md](frozen_new_experiment_protocol.md).

## 2026-10: panel correction and rescore

**What was wrong.**

- CRSP opens, highs and lows are raw prices, but `$close` was the split-adjusted close.
  On 1,397,052 of 4,874,082 rows the two scales differ, so intraday ratios such as
  `($close - $open) / $open` and every open-to-open label jumped by the cumulative
  split factor. Because that factor depends on splits that happen later, the mismatch
  also leaked future information into any formula that mixed the two scales.
- Factor series were keyed by ticker. A rename restarted every rolling window and a
  reused ticker joined two companies into one series. Series are now keyed by PERMNO.
- The combined top-20 signal averaged signed ranks over only the factors present, which
  gave a name missing a negative-sign factor a level boost. Ridge weights were fitted on
  demeaned ranks but applied to raw ones. Both now use demeaned signed ranks, with a
  missing factor counted as neutral.
- The horizon-curve, factor-zoo, publication-decay and quarterly scripts sliced the
  cached ICs by signal date, keeping labels that ended after the cut. They now purge by
  outcome end date. The zoo holdout Newey-West t also counted missing days as zero IC.
- The crossover fusion gate was skipped when only one parent resolved (one `opus5-loop`
  archive member entered that way; it is not in the rescored top 20). The threshold
  sweep replayed under a different arm name, so crossover pairings did not match the
  saved run.
- The walk-forward backtest used raw opens for P&L, dropped names without an exit bar
  at entry time, renamed point-in-time members with an undated ticker map, and let fold
  test windows overlap. None of its output is in the published results.
- Trading path: a stale signal file could be traded, `--allow-risk-fail` also bypassed
  the kill-switch and universe checks, a level 2 kill switch flattened without pausing,
  an empty membership file switched the universe filter off, and the daily-loss monitor
  measured run-to-run rather than day-to-day loss. The default factor-mining path sent
  the endpoint model names (`gpt-5.4`) to the Claude backend and exited 0 with no
  factors. Each is fixed with a regression test.

**What was re-run.** The factor pool IC cache, the factor-zoo hurdles, publication
decay, signal horizons, quarterly sampling, the one-shot baseline comparison and its
per-set scoring outputs, the memorization test, and the final evolutionary table. For
the evolutionary arms the archives are frozen: every member's fitness was recomputed on
the corrected panel, the top 20 were re-chosen by that fitness, and the holdout and
paired comparisons were recomputed (`scripts/sp500_evo_final_table.py --rescore`, input
hashes in `data/factor_zoo/evo_rescore_manifest.json`). The random-grammar candidate
files were reused unchanged. Rescore records were appended to the trace
(`run_id` `rescore-2026-10-split-adjusted`).

**What was not re-run.** The searches themselves (no model calls), the threshold
sweep, the turnover calibration (`configs/evo_calibration.json`,
`data/factor_zoo/alpha101_turnover.csv`), and the replay check
(`data/factor_zoo/evo_replay_verification.csv`). Those files, the per-run archive
tables and round curves under `data/evo_runs/`, and `data/factor_zoo/evo_threshold_sweep.csv`
are search-time records from the old panel. The structure report reads expressions only
and did not change.

**Headline numbers, old to new.**

| result | before (2026-09) | after (2026-10) | file |
|---|---|---|---|
| combined holdout IC, sonnet5-loop (NW t) | 0.0117 (3.73) | 0.0101 (3.42) | `data/factor_zoo/evo_combined_table.csv` |
| combined holdout IC, gp-loop | 0.0070 | 0.0081 | same |
| combined holdout IC, opus5-loop / opus5-oneshot | 0.0093 / 0.0078 | 0.0088 / 0.0095 | same |
| combined holdout IC, sonnet5-oneshot / sonnet5-scalar | 0.0092 / 0.0089 | 0.0100 / 0.0058 | same |
| combined holdout IC, Alpha101 | 0.0095 | 0.0088 | same |
| combined holdout IC, random grammar (NW t) | 0.0055 (1.94) | 0.0005 (0.18) | same |
| sonnet5 loop vs GP: difference, raw p, Holm p | +0.0047, 0.047, 0.235 | +0.0020, 0.462, 1.000 | `reports/evo_comparison_audit.csv` |
| loop vs one-shot raw p, Sonnet / Opus | 0.399 / 0.582 | 0.961 / 0.793 | same |
| full vs scalar feedback: raw p, Holm p | 0.122, 0.488 | 0.026, 0.130 | same |
| best in-sample \|t\|, Sonnet 5 / Fable 5 / Alpha101 | 11.64 / 9.82 / 12.11 | 6.74 / 7.03 / 8.47 | `reports/baseline_comparison.csv` |
| best in-sample \|t\|, five random seeds | 9.11 to 12.41 | 6.49 to 11.35 | same |
| OOS median retention, Sonnet 5 / Fable 5 / Alpha101 | 52.8% / 29.5% / 58.8% | 37.3% / 28.1% / 60.9% | same |
| OOS median retention, random seeds | 10.7% to 66.6% | -9.1% to 21.7% | same |
| Sonnet 5 factors above 0.9 correlation with Alpha101 | 2 of 53 | 1 of 53 | `data/baseline_comparison/memorization_test.csv` |
| zoo pool (LLM / random / Alpha101) | 397 (85 / 262 / 50) | 401 (85 / 266 / 50) | `data/factor_zoo/zoo_hurdle.csv` |
| Romano-Wolf survival, LLM / random / Alpha101 | 28% / 41% / 54% | 24% / 30% / 52% | `reports/rw_survivors.csv` |
| survivors of both bars using `$dollar_volume` | 66 of 158; retention -29% vs +59% | 39 of 127; retention 36% vs 57% | `data/factor_zoo/zoo_hurdle.csv` |
| post/pre-publication mean IC, Alpha101 (bootstrap p) | 77.8% (0.32) | 81.6% (0.48) | `reports/publication_decay.csv` |
| post/pre mean IC, LLM / random controls | 62.4% / 51.0% | 48.5% / 9.7% | same |

The biggest change is in the random formulas. Many of them combine `$open`, `$high` or
`$low` with `$close`, so under the old scales they partly measured the future split
factor. With consistent scales their holdout IC largely disappears, while Alpha101 and
the model sets change much less. The conclusion of the comparison family does not
change: no paired comparison survives Holm correction.

**Not fixed.** Split-adjusted price levels still carry information about later splits,
so an expression that ranks a raw price level (for example `RANK($close)`) can look
ahead; ratios and returns are unaffected. `$volume` is not split-adjusted because the
panel has no share-adjustment factor. The panel carries no delisting returns; the
backtest marks a position without an exit bar at its last close.

## 2026-10: documentation corrections

- The README was split: the experiment write-up moved to
  [research_notes.md](research_notes.md) (rewritten with the rescored numbers; the
  pre-fix version is in [archive/research_notes_2026-09.md](archive/research_notes_2026-09.md)),
  the execution-stack description to [system_overview.md](system_overview.md), and the
  QuantaAlpha provenance summary and citation to [PROVENANCE.md](PROVENANCE.md).
- The data-source text claimed the CRSP client carries delisting returns. No delisting
  return is queried; the text now says so.
- The statement that both generating models were trained on 2018-2025 data was made
  consistent with the separate note that their training-data vintages are unverified.
- The fundamental row of the one-shot holdout table (8 of 9 signs held, 93.3% median
  retention) had no committed per-factor output; it is not carried into the rewritten
  notes.
- The LLM runtime section (primary `gpt-5.4`, fallback `gemini-3.1-pro`) now says it
  describes only the `--live-call` mining path; the published LLM factor sets and
  evolutionary arms came from Claude models.
- Stale project names (`LLMStrat`, `QuantaAlpha_US`) in setup and summary text were
  replaced with the current repository name.
- Local machine paths were removed from `configs/evo_thresholds.json` and the
  `selected_factors.yaml` outputs.

## 2026-09: integrity repair of the completed evolutionary run

These paragraphs opened the README after the 2026-09 audit.

**Research scope:** point-in-time S&P 500 selection on the cached 2000-2025 US panel. The completed evolutionary experiment fitted on 2000-2013, used 2014-2017 for validation, and reported 2018-2025. Its historical selection labels cross window boundaries; these archived results are not clean out-of-sample confirmation. New scoring code purges labels by outcome end date. No model was called and no historical holdout was rescored for this repair.

**Completed-run audit:** Sonnet's loop has higher historical equal-weight IC than the GP loop, but none of the five reported paired comparisons survives Holm correction. The smallest raw p-value, 0.047, becomes 0.235. Loop versus one-shot is not established for either model. Sonnet's loop stopped during its last round, and the saved partial-round admissions remain in the historical record.

```bash
python scripts/sp500_evo_audit.py --check
```

This offline command verifies the five published IC differences and [adjusted comparison table](../reports/evo_comparison_audit.csv), using committed aggregates only. The underlying bootstrap p-values and confidence intervals are historical inputs, not independently rebuilt from daily IC paths. Holm covers these five contrasts, not every earlier search choice. See [integrity and completion notes](integrity_audit.md).

**Fundamentals provenance:** an approved offline fingerprint identifies the cache as Compustat North America quarterly fundamentals and finds 32 of 32 sampled keys in current/restated data, versus 26 of 32 in the current unrevised view. It cannot recover the exact source alias or historical extract timestamp because no original sidecar exists. The aggregate report is rebuilt from committed per-sample hashes and match booleans with:

```bash
python scripts/sp500_fundamentals_provenance.py --check
```

This command is fully offline and reads only sanitized derived evidence in `reports/`; it does not read the licensed fundamentals cache, make a WRDS/network/model call, or touch the consumed holdout. Any genuinely new test must follow the [separate frozen protocol](frozen_new_experiment_protocol.md); it must not resume the interrupted arm or reuse the consumed holdout.

## 2026-09: corrected explanation of large t-statistics

An earlier version of the baseline section attributed the gap between the iid null
and the observed maximum |t| to daily IC autocorrelation. Newey-West standard errors
showed that explanation was wrong; see "Corrected 2026-09" under
[Does the LLM add anything?](archive/research_notes_2026-09.md#does-the-llm-add-anything) in the pre-fix notes.

## 2026-08: walk-forward validation fixes

- Gate-1 now uses the actual Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014): the probability that the strategy's true Sharpe exceeds the expected maximum Sharpe of `n_trials` zero-skill strategies, adjusted for skewness and kurtosis. The threshold is a probability (default 0.95). The previous implementation was a heuristic penalty and overstated significance.
- Gate-6 (factor stability) is now measured: per-fold Spearman rank ICs of each baseline factor against realized entry-to-exit returns, top-3 factor sets per fold, mean Jaccard overlap across consecutive folds. It was previously hard-coded to pass.
- Holdings that drop out of the tradable context (index exit, missing bar) are now explicitly liquidated at the entry open with transaction costs; they previously converted to cash silently with no cost or turnover.
