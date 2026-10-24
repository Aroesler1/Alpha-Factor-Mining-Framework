#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.paths import resolve_from_us_root

from quantaalpha_us.backtest.universe import SP500Universe  # noqa: E402
from quantaalpha_us.data.quality import DataQualityGate  # noqa: E402
from quantaalpha_us.pipeline.signal_generator import SignalConfig, generate_signals  # noqa: E402


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _signal_config_from_yaml(cfg: dict) -> SignalConfig:
    portfolio_cfg = cfg.get("portfolio", {}) if isinstance(cfg.get("portfolio"), dict) else {}
    signals_cfg = cfg.get("signals", {}) if isinstance(cfg.get("signals"), dict) else {}

    top_k = int(portfolio_cfg.get("top_k", signals_cfg.get("topk", 30)))
    max_weight = float(portfolio_cfg.get("max_weight_per_name", signals_cfg.get("max_weight", 0.05)))
    max_turnover = float(portfolio_cfg.get("max_daily_turnover", cfg.get("risk", {}).get("max_turnover_daily", 0.20)))
    min_adv = float(portfolio_cfg.get("min_avg_daily_volume_usd", 5_000_000))
    long_only = bool(portfolio_cfg.get("long_only", signals_cfg.get("long_only", True)))

    return SignalConfig(
        top_k=top_k,
        max_weight=max_weight,
        max_sector_weight=float(portfolio_cfg.get("max_sector_weight", 1.0)),
        long_only=long_only,
        max_turnover_daily=max_turnover,
        min_avg_dollar_volume=min_adv,
    )


def _load_sector_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = _load_table(path)
    if df.empty or "symbol" not in df.columns:
        return {}
    sector_col = "sector" if "sector" in df.columns else None
    if sector_col is None:
        return {}
    work = df[["symbol", sector_col]].dropna(subset=["symbol"]).copy()
    work = work.assign(
        symbol=work["symbol"].astype(str).str.upper(),
        **{sector_col: work[sector_col].fillna("Unknown").astype(str)},
    )
    return dict(work.drop_duplicates(subset=["symbol"]).itertuples(index=False, name=None))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate US S&P500 signal CSV for trading runtime.")
    parser.add_argument("--config", default="configs/backtest_sp500_research.yaml")
    parser.add_argument("--bars-file", default="data/us_equities/processed/daily_bars.parquet")
    parser.add_argument("--membership-file", default="data/us_equities/reference/sp500_membership_daily.parquet")
    parser.add_argument("--ticker-mapping-file", default="data/us_equities/reference/ticker_mapping.csv")
    parser.add_argument("--as-of", default=None, help="Optional as-of date (YYYY-MM-DD). Defaults to latest bar date.")
    parser.add_argument("--output", default="data/results/trading/latest_signals.csv")
    parser.add_argument("--metadata-output", default="data/results/trading/latest_signals_meta.json")
    parser.add_argument(
        "--history-window-days",
        type=int,
        default=320,
        help="Days of bar history to load for feature computation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_from_us_root(args.config, US_ROOT)
    cfg = _load_config(config_path)
    bars_path = resolve_from_us_root(args.bars_file, US_ROOT)
    membership_path = resolve_from_us_root(args.membership_file, US_ROOT)
    ticker_map_path = resolve_from_us_root(args.ticker_mapping_file, US_ROOT)
    sector_path = resolve_from_us_root(
        str((cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}).get("sector_file", "data/us_equities/reference/gics_sectors.csv")),
        US_ROOT,
    )
    output_path = resolve_from_us_root(args.output, US_ROOT)
    metadata_path = resolve_from_us_root(args.metadata_output, US_ROOT)

    signal_cfg = _signal_config_from_yaml(cfg)

    bars = _load_table(bars_path)
    if bars.empty:
        raise RuntimeError(f"Bars file is empty or missing: {bars_path}")

    bars = bars.copy().assign(date=pd.to_datetime(bars["date"], errors="coerce").dt.normalize())
    bars = bars.dropna(subset=["date", "symbol"])

    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else bars["date"].max()
    quality = DataQualityGate().run_all_checks(bars, date=as_of, membership_df=_load_table(membership_path), mode="strict")
    if not quality.passed:
        raise RuntimeError(f"Signal generation blocked by data quality checks: {quality.halt_reason}")

    start_cutoff = as_of - pd.Timedelta(days=max(args.history_window_days, 60))
    bars = bars[bars["date"] >= start_cutoff].copy()

    universe = SP500Universe(str(membership_path), ticker_mapping_file=str(ticker_map_path))
    active = universe.get_members(as_of)
    if not active:
        raise RuntimeError(f"No active index members for {as_of.date()} in {membership_path}; "
                           "refusing to generate signals without a universe")

    prev_signals = _load_table(output_path)
    previous_weights = None
    if not prev_signals.empty and {"symbol", "weight"}.issubset(prev_signals.columns):
        prev_signals = prev_signals.copy().assign(
            date=pd.to_datetime(prev_signals.get("date"), errors="coerce").dt.normalize()
        )
        prev_date = prev_signals["date"].max()
        if pd.notna(prev_date) and pd.Timestamp(prev_date).normalize() < as_of:
            prev_slice = prev_signals[prev_signals["date"] == prev_date]
            previous_weights = {
                str(r.symbol).upper(): float(r.weight)
                for r in prev_slice.itertuples(index=False)
                if pd.notna(r.weight)
            }

    signals = generate_signals(
        bars,
        config=signal_cfg,
        as_of=as_of,
        active_universe=active,
        previous_weights=previous_weights,
        sector_map=_load_sector_map(sector_path),
    )
    if signals.empty:
        raise RuntimeError("Signal generation produced no tradable symbols")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output_path, index=False)

    cfg_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    model_hash = hashlib.sha256((cfg_hash + "|baseline_rank_model_v1").encode("utf-8")).hexdigest()
    metadata = {
        "as_of": str(as_of.date()),
        "model_name": "baseline_rank_model_v1",
        "model_version_hash": model_hash,
        "config_hash": cfg_hash,
        "signals_file": str(output_path),
        "quality": quality.to_dict(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = {
        "as_of": str(as_of.date()),
        "symbols_active": len(active),
        "signals": int(len(signals)),
        "weights_sum": float(signals["weight"].sum()),
        "max_weight": float(signals["weight"].max()),
        "output": str(output_path),
        "metadata_output": str(metadata_path),
        "model_version_hash": model_hash,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
