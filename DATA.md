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
