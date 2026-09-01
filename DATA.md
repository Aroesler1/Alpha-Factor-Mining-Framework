# Data provenance

**Primary source:** CRSP daily stock file via WRDS (Berkeley/Haas subscription), processed into a daily bar panel.

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
