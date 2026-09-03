#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.paths import resolve_from_us_root

from quantaalpha_us.data.market_data import build_market_data_client
from quantaalpha_us.data.membership import (  # noqa: E402
    build_membership_daily,
    default_ticker_mapping,
    extract_sector_table,
    get_trading_days,
    save_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time S&P 500 membership artifacts.")
    parser.add_argument("--source", choices=["auto", "crsp"], default="auto")
    parser.add_argument("--start-date", default="2000-01-03")
    parser.add_argument("--end-date", default=str(date.today()))
    parser.add_argument("--membership-out", default="data/us_equities/reference/sp500_membership_daily.parquet")
    parser.add_argument("--sectors-out", default="data/us_equities/reference/gics_sectors.csv")
    parser.add_argument("--ticker-map-out", default="data/us_equities/reference/ticker_mapping.csv")
    parser.add_argument("--no-cache", action="store_true", help="Disable client-level response cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    membership_out = resolve_from_us_root(args.membership_out, US_ROOT)
    sectors_out = resolve_from_us_root(args.sectors_out, US_ROOT)
    ticker_out = resolve_from_us_root(args.ticker_map_out, US_ROOT)

    client = build_market_data_client(
        source=args.source,
    )

    constituents = client.get_sp500_constituents_historical(use_cache=not args.no_cache)
    if constituents.empty:
        raise RuntimeError("No constituents returned from the configured primary/fallback data source")

    trading_days = get_trading_days(args.start_date, args.end_date)
    membership = build_membership_daily(
        constituents,
        start_date=args.start_date,
        end_date=args.end_date,
        trading_days=trading_days,
    )

    sectors = extract_sector_table(constituents)
    ticker_map = client.get_ticker_mapping(use_cache=not args.no_cache)
    if ticker_map.empty:
        ticker_map = default_ticker_mapping()

    membership_path = save_dataframe(membership, membership_out)
    sectors_path = save_dataframe(sectors, sectors_out)
    ticker_path = save_dataframe(ticker_map, ticker_out)

    counts = membership.groupby("date")["symbol"].nunique() if not membership.empty else pd.Series(dtype=float)
    summary = {
        "constituent_rows": int(len(constituents)),
        "unique_symbols": int(constituents["Code"].nunique()) if "Code" in constituents.columns else 0,
        "membership_rows": int(len(membership)),
        "date_min": str(membership["date"].min().date()) if not membership.empty else None,
        "date_max": str(membership["date"].max().date()) if not membership.empty else None,
        "min_members": int(counts.min()) if not counts.empty else 0,
        "max_members": int(counts.max()) if not counts.empty else 0,
        "membership_path": str(membership_path),
        "sectors_path": str(sectors_path),
        "ticker_map_path": str(ticker_path),
        "source": client.source_name,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
