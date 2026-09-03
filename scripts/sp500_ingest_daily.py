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
from quantaalpha_us.data.quality import DataQualityGate  # noqa: E402


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _save_table(df: pd.DataFrame, path: Path) -> Path:
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


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    for col in ["open", "high", "low", "close", "adj_close", "volume", "dollar_volume"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        else:
            work[col] = pd.NA
    work = work.dropna(subset=["date", "symbol"])
    work = work.drop_duplicates(subset=["date", "symbol"], keep="last")
    return work.sort_values(["date", "symbol"]).reset_index(drop=True)


def _get_active_symbols_for_dates(membership: pd.DataFrame, dates: list[pd.Timestamp], fallback: set[str]) -> list[str]:
    if membership.empty or "symbol" not in membership.columns or not dates:
        return sorted(fallback)
    m = membership.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
    m["symbol"] = m["symbol"].astype(str).str.upper()
    if "active" in m.columns:
        m = m[m["active"] == True]  # noqa: E712
    wanted = set(pd.DatetimeIndex(dates).normalize())
    symbols = sorted(set(m[m["date"].isin(wanted)]["symbol"].tolist()))
    if symbols:
        return symbols
    return sorted(fallback)


def _backfill_missing_days(
    *,
    client,
    symbols: list[str],
    missing_dates: list[pd.Timestamp],
    use_cache: bool,
) -> pd.DataFrame:
    if not symbols or not missing_dates:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "dollar_volume"])

    from_date = pd.Timestamp(min(missing_dates)).strftime("%Y-%m-%d")
    to_date = pd.Timestamp(max(missing_dates)).strftime("%Y-%m-%d")
    missing_set = set(pd.DatetimeIndex(missing_dates).normalize())

    rows: list[pd.DataFrame] = []
    for idx, sym in enumerate(symbols, start=1):
        hist = client.get_eod_history(
            symbol=sym,
            exchange="US",
            from_date=from_date,
            to_date=to_date,
            use_cache=use_cache,
        )
        if hist.empty:
            continue
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()
        hist = hist[hist["date"].isin(missing_set)].copy()
        if hist.empty:
            continue
        hist["dollar_volume"] = pd.to_numeric(hist["close"], errors="coerce") * pd.to_numeric(hist["volume"], errors="coerce")
        rows.append(hist[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "dollar_volume"]])
        if idx % 50 == 0 or idx == len(symbols):
            print(f"gap_backfill_progress={idx}/{len(symbols)}")

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "dollar_volume"])
    return _normalize_bars(pd.concat(rows, ignore_index=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily ingest of US bars from CRSP.")
    parser.add_argument("--source", choices=["auto", "crsp"], default="auto")
    parser.add_argument("--date", default=str(date.today()), help="Trading date to ingest (YYYY-MM-DD)")
    parser.add_argument("--bars-file", default="data/us_equities/processed/daily_bars.parquet")
    parser.add_argument("--membership-file", default="data/us_equities/reference/sp500_membership_daily.parquet")
    parser.add_argument("--mode", choices=["strict", "lenient"], default="strict")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = pd.Timestamp(args.date).normalize()

    bars_path = resolve_from_us_root(args.bars_file, US_ROOT)
    membership_path = resolve_from_us_root(args.membership_file, US_ROOT)

    client = build_market_data_client(
        source=args.source,
    )
    quality = DataQualityGate()
    use_cache = not args.no_cache

    existing = _normalize_bars(_load_table(bars_path))
    membership = _load_table(membership_path)
    if not membership.empty:
        membership["date"] = pd.to_datetime(membership["date"], errors="coerce").dt.normalize()
        membership["symbol"] = membership["symbol"].astype(str).str.upper()

    gap_check = None
    backfill_rows = pd.DataFrame()
    missing_mid_dates: list[pd.Timestamp] = []
    if not existing.empty:
        gap_check = quality.check_gap_backfill(existing, target_date)
        if not gap_check.passed:
            severity = gap_check.severity
            gap_business_days = int(gap_check.metrics.get("gap_business_days", 0) or 0)
            if severity == "critical":
                raise RuntimeError(f"Gap backfill check failed: {gap_check.details}")
            latest_date = pd.Timestamp(gap_check.metrics["latest_date"]).normalize()
            # Backfill intermediate missing days; target date itself is filled by bulk endpoint.
            if target_date > latest_date:
                missing_mid_dates = list(
                    pd.bdate_range(latest_date + pd.Timedelta(days=1), target_date - pd.Timedelta(days=1), freq="B")
                )
            if missing_mid_dates:
                fallback_symbols = set(existing["symbol"].astype(str).str.upper().tolist())
                symbols = _get_active_symbols_for_dates(membership, missing_mid_dates, fallback=fallback_symbols)
                print(
                    f"gap_backfill_start days={len(missing_mid_dates)} symbols={len(symbols)} "
                    f"reported_gap_business_days={gap_business_days}"
                )
                backfill_rows = _backfill_missing_days(
                    client=client,
                    symbols=symbols,
                    missing_dates=missing_mid_dates,
                    use_cache=use_cache,
                )

    bulk = client.get_bulk_eod(exchange="US", date=target_date.strftime("%Y-%m-%d"), use_cache=use_cache)
    if bulk.empty:
        raise RuntimeError(f"No bulk data returned for date={target_date.date()}")

    bulk = bulk.rename(columns={"code": "symbol", "adjusted_close": "adj_close", "adjustedClose": "adj_close"})
    if "symbol" not in bulk.columns:
        raise RuntimeError("Bulk payload missing symbol/code column")

    bulk["symbol"] = bulk["symbol"].astype(str).str.upper()
    bulk["date"] = pd.to_datetime(bulk.get("date", target_date), errors="coerce").dt.normalize().fillna(target_date)
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in bulk.columns:
            bulk[col] = pd.to_numeric(bulk[col], errors="coerce")
        else:
            bulk[col] = pd.NA
    bulk["dollar_volume"] = pd.to_numeric(bulk["close"], errors="coerce") * pd.to_numeric(bulk["volume"], errors="coerce")
    new_rows = bulk[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "dollar_volume"]]

    if not membership.empty:
        if "active" in membership.columns:
            active_symbols = set(membership[(membership["date"] == target_date) & (membership["active"] == True)]["symbol"])  # noqa: E712
        else:
            active_symbols = set(membership[membership["date"] == target_date]["symbol"])
        if not active_symbols:
            raise RuntimeError(
                f"No active membership symbols found for date={target_date.date()} in {membership_path}"
            )
        new_rows = new_rows[new_rows["symbol"].isin(active_symbols)]

    all_parts = [df for df in [existing, backfill_rows, new_rows] if not df.empty]
    all_rows = _normalize_bars(pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame())
    report = quality.run_all_checks(
        all_rows,
        date=target_date,
        membership_df=membership if not membership.empty else None,
        mode=args.mode,
    )
    if not report.passed:
        raise RuntimeError(f"Data quality checks failed: {report.halt_reason}")

    saved_path = _save_table(all_rows, bars_path)
    summary = {
        "date": str(target_date.date()),
        "new_symbols": int(new_rows["symbol"].nunique()),
        "new_rows": int(len(new_rows)),
        "backfilled_rows": int(len(backfill_rows)),
        "backfilled_days": len(missing_mid_dates),
        "total_rows": int(len(all_rows)),
        "total_symbols": int(all_rows["symbol"].nunique()),
        "output": str(saved_path),
        "quality": report.to_dict(),
        "source": client.source_name,
    }
    if gap_check is not None:
        summary["gap_check"] = {
            "passed": bool(gap_check.passed),
            "severity": gap_check.severity,
            "details": gap_check.details,
            "metrics": gap_check.metrics,
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
