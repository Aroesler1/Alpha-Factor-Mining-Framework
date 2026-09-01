from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


BASELINE_FACTOR_COLUMNS = [
    "r_mom_5d",
    "r_mom_21d",
    "r_rev_5d",
    "r_vol_21d",
    "r_vol_ratio",
    "r_volume_accel",
    "r_range",
]


@dataclass
class SignalConfig:
    top_k: int = 30
    max_weight: float = 0.05
    max_sector_weight: float = 1.0
    long_only: bool = True
    max_turnover_daily: float = 0.20
    min_avg_dollar_volume: float = 5_000_000.0


def _rank_by_date(series: pd.Series, dates: pd.Series, ascending: bool = True) -> pd.Series:
    out = series.groupby(dates).rank(pct=True, method="average", ascending=ascending)
    return out


def active_factor_columns(n_mined: int) -> list[str]:
    """Factor columns entering the score: baseline ranks plus mined ranks."""
    return list(BASELINE_FACTOR_COLUMNS) + [f"r_mined_{i:02d}" for i in range(int(n_mined))]


def build_features(bars: pd.DataFrame, mined_expressions: Optional[list[str]] = None) -> pd.DataFrame:
    required_cols = {"date", "symbol", "close"}
    missing = required_cols - set(bars.columns)
    if missing:
        raise ValueError(f"Missing required bar columns: {sorted(missing)}")

    df = bars.copy().assign(
        date=pd.to_datetime(bars["date"], errors="coerce").dt.normalize(),
        symbol=bars["symbol"].astype(str).str.upper(),
    )
    if "adj_close" not in df.columns:
        df = df.assign(adj_close=df["close"])
    if "dollar_volume" not in df.columns:
        volume = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
        close = pd.to_numeric(df["close"], errors="coerce")
        df = df.assign(dollar_volume=close * volume)

    for col in ["close", "adj_close", "open", "high", "low", "volume", "dollar_volume"]:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    grp = df.groupby("symbol", group_keys=False)
    ret1 = grp["adj_close"].pct_change()
    df["mom_5d"] = grp["adj_close"].pct_change(5)
    df["mom_21d"] = grp["adj_close"].pct_change(21)
    df["rev_5d"] = -df["mom_5d"]
    df["vol_21d"] = ret1.groupby(df["symbol"]).rolling(21, min_periods=10).std().reset_index(level=0, drop=True)
    df["vol_63d"] = ret1.groupby(df["symbol"]).rolling(63, min_periods=20).std().reset_index(level=0, drop=True)

    adv5 = grp["dollar_volume"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    adv20 = grp["dollar_volume"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    df["volume_accel"] = adv5 / adv20.replace(0, pd.NA)
    df["adv20"] = adv20

    if {"high", "low", "close"}.issubset(df.columns):
        df["range"] = (df["high"] - df["low"]) / df["close"].replace(0, pd.NA)
    else:
        df["range"] = pd.NA

    # Cross-sectional ranks per date.
    df["r_mom_5d"] = _rank_by_date(df["mom_5d"], df["date"], ascending=True)
    df["r_mom_21d"] = _rank_by_date(df["mom_21d"], df["date"], ascending=True)
    df["r_rev_5d"] = _rank_by_date(df["rev_5d"], df["date"], ascending=True)
    df["r_vol_21d"] = _rank_by_date(df["vol_21d"], df["date"], ascending=False)  # prefer lower vol
    df["r_vol_ratio"] = _rank_by_date(df["vol_21d"] / df["vol_63d"].replace(0, pd.NA), df["date"], ascending=False)
    df["r_volume_accel"] = _rank_by_date(df["volume_accel"], df["date"], ascending=True)
    df["r_range"] = _rank_by_date(df["range"], df["date"], ascending=False)

    factor_cols = list(BASELINE_FACTOR_COLUMNS)

    # Mined factors: sanitized expressions evaluated over the field panels,
    # cross-sectionally ranked per date, and averaged into the score on the
    # same footing as the baseline ranks. Signals here use data through the
    # feature date only (the evaluator's operators are all backward-looking),
    # matching the T-close -> T+1-open execution convention.
    if mined_expressions:
        from quantaalpha_us.factors.expression_evaluator import (
            ExpressionEvaluator,
            build_field_panels,
        )

        panels = build_field_panels(df)
        evaluator = ExpressionEvaluator(panels)
        pair_index = pd.MultiIndex.from_frame(df[["date", "symbol"]])
        for i, expr in enumerate(mined_expressions):
            col = f"r_mined_{i:02d}"
            signal = evaluator.evaluate(str(expr))
            ranked = signal.rank(axis=1, pct=True)
            df[col] = ranked.stack().reindex(pair_index).to_numpy()
            factor_cols.append(col)

    df["score"] = df[factor_cols].mean(axis=1, skipna=True)
    return df


def _select_signals_from_snapshot(
    today: pd.DataFrame,
    *,
    config: SignalConfig,
    active_universe: Optional[list[str]] = None,
    previous_weights: Optional[dict[str, float]] = None,
    sector_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    if today.empty:
        return pd.DataFrame(columns=["date", "symbol", "score", "weight"])

    work = today.copy()

    if active_universe:
        allowed = {s.upper() for s in active_universe}
        work = work[work["symbol"].isin(allowed)]

    work = work[work["adv20"].fillna(0.0) >= float(config.min_avg_dollar_volume)]
    work = work.dropna(subset=["score"]).sort_values("score", ascending=False)
    if work.empty:
        return pd.DataFrame(columns=["date", "symbol", "score", "weight"])

    desired = _build_desired_weights(
        work,
        top_k=int(config.top_k),
        max_weight=float(config.max_weight),
        max_sector_weight=float(config.max_sector_weight),
        sector_map=sector_map,
    )
    if not desired:
        return pd.DataFrame(columns=["date", "symbol", "score", "weight"])

    # Keep residual cash when top_k * cap < 1.0.
    weights = _apply_turnover_cap(
        desired,
        previous_weights=previous_weights,
        max_turnover_daily=float(config.max_turnover_daily),
        max_weight=float(config.max_weight),
        target_total_weight=1.0,
    )

    selected = work[work["symbol"].isin(weights)].copy()
    selected = selected.assign(weight=selected["symbol"].map(weights).fillna(0.0))
    selected = selected[selected["weight"] > 0]

    out = selected[["date", "symbol", "score", "weight"]].sort_values("symbol")
    return out.reset_index(drop=True)


def select_signals(
    features: pd.DataFrame,
    *,
    config: SignalConfig,
    as_of: Optional[str | pd.Timestamp] = None,
    active_universe: Optional[list[str]] = None,
    previous_weights: Optional[dict[str, float]] = None,
    sector_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["date", "symbol", "score", "weight"])

    as_of_date = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp(features["date"].max()).normalize()
    today = features[features["date"] == as_of_date].copy()
    return _select_signals_from_snapshot(
        today,
        config=config,
        active_universe=active_universe,
        previous_weights=previous_weights,
        sector_map=sector_map,
    )


def select_signals_from_snapshot(
    today: pd.DataFrame,
    *,
    config: SignalConfig,
    active_universe: Optional[list[str]] = None,
    previous_weights: Optional[dict[str, float]] = None,
    sector_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    return _select_signals_from_snapshot(
        today,
        config=config,
        active_universe=active_universe,
        previous_weights=previous_weights,
        sector_map=sector_map,
    )


def _apply_turnover_cap(
    desired_weights: dict[str, float],
    previous_weights: Optional[dict[str, float]],
    max_turnover_daily: float,
    max_weight: float,
    target_total_weight: float = 1.0,
) -> dict[str, float]:
    if not previous_weights or max_turnover_daily <= 0:
        return desired_weights

    symbols = sorted(set(desired_weights) | set(previous_weights))
    turnover = 0.5 * sum(abs(desired_weights.get(s, 0.0) - previous_weights.get(s, 0.0)) for s in symbols)
    if turnover <= max_turnover_daily:
        return desired_weights

    scale = max_turnover_daily / turnover if turnover > 0 else 0.0
    blended = {}
    for s in symbols:
        prev = previous_weights.get(s, 0.0)
        target = desired_weights.get(s, 0.0)
        w = prev + scale * (target - prev)
        blended[s] = max(0.0, min(max_weight, w))

    total = sum(blended.values())
    if total > target_total_weight and total > 0:
        scale_down = target_total_weight / total
        blended = {s: w * scale_down for s, w in blended.items()}

    return {s: w for s, w in blended.items() if w > 1e-8}


def _build_desired_weights(
    ranked: pd.DataFrame,
    *,
    top_k: int,
    max_weight: float,
    max_sector_weight: float,
    sector_map: Optional[dict[str, str]],
) -> dict[str, float]:
    if ranked.empty or top_k <= 0 or max_weight <= 0:
        return {}

    target_weight = min(1.0 / max(top_k, 1), max_weight)
    sector_cap = max(min(float(max_sector_weight), 1.0), 0.0)
    use_sector_cap = sector_map is not None and sector_cap < 0.999999

    desired: dict[str, float] = {}
    sector_weights: dict[str, float] = {}
    candidates = ranked.copy()

    for row in candidates.itertuples(index=False):
        sym = str(row.symbol).upper()
        if sym in desired:
            continue
        if len(desired) >= top_k:
            break
        sector = str(sector_map.get(sym, "Unknown")) if use_sector_cap and sector_map else "__ALL__"
        room_sector = sector_cap - sector_weights.get(sector, 0.0) if use_sector_cap else 1.0
        alloc = min(target_weight, max_weight, room_sector)
        if alloc <= 1e-10:
            continue
        desired[sym] = alloc
        sector_weights[sector] = sector_weights.get(sector, 0.0) + alloc

    remaining = max(1.0 - sum(desired.values()), 0.0)
    if remaining <= 1e-10:
        return desired

    for row in candidates.itertuples(index=False):
        sym = str(row.symbol).upper()
        sector = str(sector_map.get(sym, "Unknown")) if use_sector_cap and sector_map else "__ALL__"
        room_name = max_weight - desired.get(sym, 0.0)
        room_sector = sector_cap - sector_weights.get(sector, 0.0) if use_sector_cap else 1.0
        if sym not in desired and len(desired) >= top_k:
            continue
        alloc = min(remaining, room_name, room_sector)
        if alloc <= 1e-10:
            continue
        desired[sym] = desired.get(sym, 0.0) + alloc
        sector_weights[sector] = sector_weights.get(sector, 0.0) + alloc
        remaining -= alloc
        if remaining <= 1e-10:
            break

    total = sum(desired.values())
    if total > 1.0 + 1e-10:
        scale = 1.0 / total
        desired = {sym: weight * scale for sym, weight in desired.items()}
    return {sym: weight for sym, weight in desired.items() if weight > 1e-8}


def generate_signals(
    bars: pd.DataFrame,
    *,
    config: SignalConfig,
    as_of: Optional[str | pd.Timestamp] = None,
    active_universe: Optional[list[str]] = None,
    previous_weights: Optional[dict[str, float]] = None,
    sector_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Generate daily long-only TopK weights from baseline factor ranks."""
    feats = build_features(bars)
    return select_signals(
        feats,
        config=config,
        as_of=as_of,
        active_universe=active_universe,
        previous_weights=previous_weights,
        sector_map=sector_map,
    )


def baseline_factor_names() -> list[str]:
    return list(BASELINE_FACTOR_COLUMNS)
