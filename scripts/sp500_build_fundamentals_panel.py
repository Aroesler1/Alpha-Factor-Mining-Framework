#!/usr/bin/env python3
"""Attach point-in-time Compustat fields to the daily bar panel.

Builds `daily_bars_fundamentals.parquet`, the input the evolutionary scripts
and the factor pool read by default, from three local files:

    daily bars       data/us_equities/processed/daily_bars.parquet
    Compustat fundq  data/us_equities/reference/compustat_fundq.parquet
    CCM link table   data/us_equities/reference/ccm_link.parquet

    python scripts/sp500_build_fundamentals_panel.py

Pulling `comp.fundq` and the CRSP/Compustat link table from WRDS is not scripted
here; both are licensed and stay outside the repository. Each field becomes
usable one day after its report date (`rdq`), never at the period end; see
`quantaalpha_us/factors/fundamentals.py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantaalpha_us.factors.fundamentals import attach_to_bars, build_pit_panel  # noqa: E402

DATA = ROOT / "data" / "us_equities"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bars", default=str(DATA / "processed" / "daily_bars.parquet"))
    parser.add_argument("--fundq", default=str(DATA / "reference" / "compustat_fundq.parquet"))
    parser.add_argument("--link", default=str(DATA / "reference" / "ccm_link.parquet"))
    parser.add_argument("--out", default=str(DATA / "processed" / "daily_bars_fundamentals.parquet"))
    args = parser.parse_args()

    bars = pd.read_parquet(args.bars)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(bars["date"]).unique()))
    panels = build_pit_panel(pd.read_parquet(args.fundq), pd.read_parquet(args.link), dates,
                             permnos=pd.Index(bars["permno"].dropna().unique()))
    out = attach_to_bars(bars, panels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    filled = {field: float(out[field].notna().mean()) for field in panels}
    print(f"{len(out):,} rows -> {args.out}")
    print("share of rows with a value: " + ", ".join(f"{k} {v:.2f}" for k, v in filled.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
