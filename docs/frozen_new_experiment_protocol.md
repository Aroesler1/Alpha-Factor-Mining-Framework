# Frozen protocol for any new experiment

This protocol is separate from the consumed 2018-2025 evaluation. It does not
resume the interrupted Sonnet loop, rescore the historical holdout, or relabel
retrospective repairs as independent evidence.

## Freeze before any new outcome is inspected

1. Use outcomes strictly after the consumed historical window. Purge every
   label whose outcome end crosses a fit, validation, or evaluation boundary.
2. Record the exact CRSP and Compustat product, table, extract timestamp,
   Snapshot or restated status, query text, row count, and file hash in a
   sidecar. Report-date alignment alone is insufficient.
3. Freeze the complete factor list, expression hashes, random seeds, model
   provider and version, prompts, call budget, random-grammar budget, and
   stopping rule. No paid call starts without separate approval.
4. Predeclare fit, validation, and one-shot evaluation dates. Claim the
   evaluation as consumed before reading it, including on a failed run.
5. Predeclare implementation assumptions: next-open execution, commissions,
   spread and slippage model, borrow and capacity rules, turnover limits, and
   portfolio construction.
6. Predeclare the five-comparison family or a replacement family before
   scoring. Apply Holm to the full family, including failed or unfavorable
   comparisons.

## Required reporting

Preserve every candidate and failure in the append-only trace. Report
explanatory fit, predictive IC, and executable net performance separately.
Show the random-search control under a matched budget, all rejected
hypotheses, sensitivity to costs, and the frozen evaluation result exactly
once. A favorable outcome is not assumed.

The original Sonnet loop remains a 7-of-8-round interrupted historical arm.
Any future run receives a new experiment identifier and output directory.
