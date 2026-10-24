#!/usr/bin/env python3
"""Aggregate counts that describe the daily panel, written to reports/panel_summary.csv.

Reads the licensed bar panel and membership file in place and writes only
counts and dates, so the sample description in the README and DATA.md has a
committed source.

    python scripts/sp500_panel_summary.py \
        --bars data/us_equities/processed/daily_bars.parquet \
        --membership data/us_equities/reference/sp500_membership_daily.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summarise(bars: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    bars = bars.assign(date=pd.to_datetime(bars["date"]), permno=bars["permno"].astype("int64"))
    members = membership.assign(date=pd.to_datetime(membership["date"]),
                                permno=membership["permno"].astype("int64"))
    if "active" in members.columns:
        members = members[members["active"].astype(bool)]
    members = members[["date", "permno"]].drop_duplicates()
    filtered = bars.merge(members, on=["date", "permno"], how="inner")
    rows = [
        ("panel_rows", len(bars)),
        ("panel_permnos", bars["permno"].nunique()),
        ("panel_ticker_symbols", bars["symbol"].nunique()),
        ("panel_trading_days", bars["date"].nunique()),
        ("panel_first_date", str(bars["date"].min().date())),
        ("panel_last_date", str(bars["date"].max().date())),
        ("panel_mean_names_per_date", round(bars.groupby("date")["permno"].nunique().mean(), 1)),
        ("member_mean_names_per_date", round(members.groupby("date")["permno"].nunique().mean(), 1)),
        ("member_filtered_rows", len(filtered)),
        ("member_filtered_permnos", filtered["permno"].nunique()),
        ("rows_with_split_adjustment", int((bars["adj_close"] != bars["close"]).sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bars", default=str(ROOT / "data" / "us_equities" / "processed"
                                              / "daily_bars.parquet"))
    parser.add_argument("--membership", default=str(ROOT / "data" / "us_equities" / "reference"
                                                    / "sp500_membership_daily.parquet"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "panel_summary.csv"))
    args = parser.parse_args()
    bars = pd.read_parquet(args.bars, columns=["date", "symbol", "permno", "close", "adj_close"])
    table = summarise(bars, pd.read_parquet(args.membership))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
