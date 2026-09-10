# Completed evolutionary run: integrity repair

Reviewed 2026-09-08 at original revision 6bc4939. Original archives, prompts, saved replies and final tables are unchanged. The old replay table says all six archives reproduced under the old scoring code; that is historical recorded evidence, not a new replay under the corrected label policy.

## Measured comparison

The five published equal-weight IC contrasts reproduce from the committed combined table. Their minimum raw p=.047 becomes Holm p=.235. No contrast clears 5% familywise. Loop against one-shot is not established for either model. This adjusts the reported p-values without reading the price panel, changing selected formulas or rescoring the holdout. The bootstrap interval and p-value use different historical constructions; underlying daily combined IC paths are not committed, so their bootstrap itself was not independently regenerated here. This five-test family also does not account for every upstream search and weighting choice.

## Boundary correction

For signal T, both return conventions finish at T+h+1. The scorer now masks any label ending after its window cutoff, retaining earlier feature history. Half-life comparisons share dates valid at the longest horizon. One-shot scoring and cached IC window access use the same outcome-end rule. A cached IC calendar with missing sessions can conservatively over-purge, which is disclosed instead of pretending its rows are the full vendor calendar. Synthetic tests perturb future prices and require training labels to remain identical.

Existing scores, reflection prompts, threshold selection and archive membership already used historical labels. Re-evaluating saved candidates cannot erase that path. New runs record outcome-end-v1 and cannot resume a directory with the old policy. No new model call or search occurred.

## Evaluation protocol

The historical run directory contains a retrospective consumed record with hashes of its published outputs. It does not claim an original first-access timestamp. The final-table command refuses to rescore that directory and directs readers to the offline audit. For a future compatible experiment, --freeze-evaluation records hashes of all arms, selected settings, code, reference candidates, input data and IC cache before scoring. A reviewed manifest is checked and an exclusive consumption record is written before outcome reads. Failed evaluations remain consumed. These are audit controls, not protection against manual deletion or an independently written evaluator. Historical outputs cannot be overwritten by this entry point.

The current final-table defaults still refer to the historical window. A genuinely new evaluation requires a separately reviewed configuration and suitable unseen data; freezing the same old dates does not create a new holdout. The original ridge penalty selection used validation in addition to early stopping and threshold selection. That protocol deviation is retained explicitly.

## Literature and provenance

- QuantaAlpha v1 versus [v3](https://arxiv.org/html/2602.07085v3): CSI 300 headline IC .1501 versus .0472, author-reported under distinct versions. The versioned table is reports/literature_versions.csv; these are not this repository's measurements.
- [AutoAlpha](https://arxiv.org/html/2002.08245v2) uses expression trees, evolutionary search and randomly generated formula controls on Chinese equities, with one- and five-day targets. This supports the control design, not the specific performance here.
- [Romano and Wolf](https://www.econ.uzh.ch/apps/workingpapers/wp/econwp219.pdf) and Holm family corrections concern defined statistical families, not economic novelty. Random formulas can contain genuine characteristic exposure.
- [S&P/WRDS February 2026 methodology](https://wrds-www.wharton.upenn.edu/documents/2180/WRDS_SP_Webinar_Feb_002.pdf) distinguishes ordinary restated quarterly data and historical Snapshot. One approved 2026-09-08 WRDS session fingerprinted 32 deterministic local keys. All 32 match current `comp.fundq`, the current restated Snapshot view, and at least one revision in the point-in-time history; only 26 match the current unrevised view. This identifies Compustat North America quarterly fundamentals and is consistent with revised data, but aliases and revision histories prevent recovery of the exact source table. The historical extract vintage is still unrecoverable because its parquet contains no query manifest or vintage metadata. The committed aggregate report is reconstructed from sanitized per-sample hashes and match booleans by `python scripts/sp500_fundamentals_provenance.py --check`; that checker does not read vendor values or the consumed holdout. See `reports/fundamentals_provenance_audit.csv`.
- Quarterly subsampling reduces daily outcome observations from 4,528 to 72. Its loss of significance does not prove repeated observations or independence of quarterly samples. The daily HAC result still has 10 of 12 above two.

## What remains before stronger claims

The authorized fingerprint could not recover the missing historical extract vintage. Preserve that limitation unless an original query log or sidecar is found. Freeze completed candidates and full search accounting. A fresh test needs an outcome window not already used for research decisions and a documented model version; do not finish the interrupted arm after inspecting its holdout and describe it as the original experiment. The separate protocol is `docs/frozen_new_experiment_protocol.md`. Comparing candidate quality, portfolio diversification and search cost under matched budgets is a useful next design, not an improvement measured here.

No profitable strategy, post-model-cutoff success, clean re-run of the original search, or causal publication effect is established by these repairs.
