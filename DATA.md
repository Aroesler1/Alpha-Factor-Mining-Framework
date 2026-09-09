# Data provenance

**Primary source:** CRSP daily stock file via WRDS (Berkeley/Haas subscription), processed into a daily bar panel.

**Which tape:** `crsp.dsf_v2`, CRSP's Flat File Format 2.0 ("CIZ"). The loader in
`quantaalpha_us/data/crsp_client.py` resolves `crsp.dsf_v2` first and falls back to the
legacy `crsp.dsf` only if the entitlement lacks it. CRSP stopped updating the legacy SIZ
tape after January 2025 and the two tapes disagree: Schwarz, Walter and Weiss, "Rewriting
CRSP's History: Impact of Altered Monthly Returns on Asset Pricing", *Journal of Financial
and Quantitative Analysis* (2026), find 9.62% of monthly returns altered by more than 1 bp
(mainly a changed dividend-reinvestment assumption) and 11.43% of monthly long-short
returns altered by more than 10 bp, while average premia and their significance hold up.
A reproduction on `crsp.dsf` should expect small differences for that reason alone. The
sibling trend-following repo (`VOO_Backtest`) is on the legacy tape, so figures do not
cross-check between the two.

Sample as used: 4,874,082 rows, 1,734 symbols, 2000-01-03 to 2025-12-31, with point-in-time S&P 500 membership and GICS sector reference data.

## What is committed

- Source code, the factor DSL and evaluator, tests
- Candidate factor expressions (`configs/`)
- Derived results: IC tables, factor scores, walk-forward summaries

## What is not committed

- `data/us_equities/` (gitignored): raw and processed CRSP bars, membership, and reference tables. These are licensed vendor data and are not redistributed.

## Reproducing

With a WRDS entitlement, rebuild the bar panel through `quantaalpha_us/data/`, then:

```bash
python scripts/sp500_score_mined_factors.py --bars data/us_equities/processed/daily_bars.parquet
```

## Licence and retention

CRSP is licensed through the university subscription. Only derived outputs are published. Raw extracts are deleted at the end of the associated academic affiliation.

## Integrity audit and authorized access

The completed evolutionary artifacts are preserved under `data/evo_runs/` and `data/factor_zoo/`. New `reports/` files contain aggregates, literature comparisons and source hashes only. Their historical values were not rescored on vendor observations. Historical selection labels cross window cutoffs.

An approved 2026-09-08 WRDS fingerprint compared 32 deterministic keys from the local fundamentals parquet. All 32 match current `comp.fundq`, current restated Snapshot, and revisions present in Snapshot PIT; 26 match current unrevised Snapshot. The product family is therefore Compustat North America quarterly fundamentals and the cache is not established as unrevised history. The exact source alias and historical extraction timestamp remain unrecoverable because the parquet has no source or vintage sidecar.

The repository commits only deterministic selection positions, per-sample hashes, table match booleans, cached-input hashes, and those explicit limitations. It contains no fundamental values, GVKEYs, sample dates, query text, credentials, or source paths. Rebuild and validate the aggregate report using only that sanitized evidence:

```bash
python scripts/sp500_fundamentals_provenance.py --check
```

To derive a fresh sanitized bundle from authorized files already present on local disk, supply an external parquet and an external approved candidate-cache directory. This command is also offline:

```bash
python scripts/sp500_fundamentals_provenance.py \
  --local-parquet /external/local-fundamentals.parquet \
  --candidate-cache /external/approved-cache \
  --output-dir /tmp/fundamentals-provenance
```

An approved cache that contains only the previously selected local rows can be processed with `--cached-local-sample` plus `--preserved-report`; the latter preserves the full-file date profile that cannot be recovered from a 32-row cache. Generation never stores the supplied absolute paths. Licensed input rows stay outside the repository, and neither mode reads or rescores the consumed holdout.

WRDS access now refuses to connect unless `WRDS_DUO_READY=1` follows current-session approval. A client attempts at most one connection, without vendor authentication retries; batch queries on that connection and cache results. The audit and test suite need no WRDS login.
