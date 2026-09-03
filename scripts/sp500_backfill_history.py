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

from quantaalpha_us.data.market_data import build_market_data_client  # noqa: E402


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
    df.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical bars for S&P 500 historical constituents.")
    parser.add_argument("--source", choices=["auto", "crsp"], default="auto")
    parser.add_argument(
        "--membership-file",
        default="data/us_equities/reference/sp500_membership_daily.parquet",
        help="Membership artifact produced by sp500_build_membership.py",
    )
    parser.add_argument("--output", default="data/us_equities/processed/daily_bars.parquet")
    parser.add_argument("--from-date", default="2000-01-03")
    parser.add_argument("--to-date", default=str(date.today()))
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap for dry runs")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    membership_path = resolve_from_us_root(args.membership_file, US_ROOT)
    output_path = resolve_from_us_root(args.output, US_ROOT)

    membership = _load_table(membership_path)
    if membership.empty or "symbol" not in membership.columns:
        raise RuntimeError("Membership file is empty or missing symbol column")

    membership = membership.copy().assign(symbol=membership["symbol"].astype(str).str.upper())
    if "permno" in membership.columns:
        membership = membership.assign(permno=pd.to_numeric(membership["permno"], errors="coerce").astype("Int64"))
        requested = (
            membership[["symbol", "permno"]]
            .drop_duplicates()
            .sort_values(["symbol", "permno"], na_position="last")
            .to_dict("records")
        )
    else:
        requested = [{"symbol": s, "permno": None} for s in sorted(set(membership["symbol"].tolist()))]
    if args.max_symbols and args.max_symbols > 0:
        requested = requested[: args.max_symbols]

    client = build_market_data_client(
        source=args.source,
    )

    rows: list[pd.DataFrame] = []
    no_data: list[str] = []
    if getattr(client, "source_name", "") == "crsp":
        bars_batch = client.get_eod_history_batch(
            requested,
            from_date=args.from_date,
            to_date=args.to_date,
        )
        if bars_batch.empty:
            raise RuntimeError("No CRSP price data fetched for any requested symbol")
        rows.append(bars_batch)
        fetched = set(bars_batch["symbol"].astype(str).str.upper().unique().tolist())
        for item in requested:
            symbol = str(item["symbol"]).upper()
            permno = item.get("permno")
            if symbol not in fetched:
                no_data.append(f"{symbol}:{int(permno)}" if pd.notna(permno) else symbol)
        print(f"progress={len(requested)}/{len(requested)}")
    else:
        for idx, item in enumerate(requested, start=1):
            symbol = str(item["symbol"]).upper()
            permno = item.get("permno")
            df = client.get_eod_history(
                symbol=symbol,
                exchange="US",
                from_date=args.from_date,
                to_date=args.to_date,
                use_cache=not args.no_cache,
                permno=int(permno) if pd.notna(permno) else None,
            )
            if df.empty:
                no_data.append(f"{symbol}:{int(permno)}" if pd.notna(permno) else symbol)
            else:
                rows.append(df)

            if idx % 25 == 0 or idx == len(requested):
                print(f"progress={idx}/{len(requested)}")

    if not rows:
        raise RuntimeError("No price data fetched for any symbol")

    bars = pd.concat(rows, ignore_index=True).copy()
    bars = bars.assign(date=pd.to_datetime(bars["date"], errors="coerce").dt.normalize())
    bars = bars.dropna(subset=["date", "symbol", "close"]).copy()
    bars = bars.assign(
        symbol=bars["symbol"].astype(str).str.upper(),
        dollar_volume=pd.to_numeric(bars["close"], errors="coerce") * pd.to_numeric(bars["volume"], errors="coerce"),
    )

    bars = bars.drop_duplicates(subset=["date", "symbol"], keep="last")
    bars = bars.sort_values(["date", "symbol"]).reset_index(drop=True)

    saved_path = _save_dataframe(bars, output_path)
    summary = {
        "symbols_requested": len(requested),
        "symbols_with_data": int(bars["symbol"].nunique()),
        "rows": int(len(bars)),
        "date_min": str(bars["date"].min().date()),
        "date_max": str(bars["date"].max().date()),
        "output": str(saved_path),
        "symbols_without_data_sample": no_data[:25],
        "source": client.source_name,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
