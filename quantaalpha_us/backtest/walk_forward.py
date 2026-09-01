from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from quantaalpha_us.backtest.costs import TransactionCostModel
from quantaalpha_us.backtest.universe import SP500Universe
from quantaalpha_us.pipeline.signal_generator import (
    SignalConfig,
    active_factor_columns,
    baseline_factor_names,
    build_features,
    select_signals_from_snapshot,
)


@dataclass
class FoldWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for k in out:
            out[k] = str(pd.Timestamp(out[k]).date())
        return out


@dataclass
class WalkForwardSummary:
    num_folds: int
    num_observations: int
    start_date: str
    end_date: str
    gross_sharpe: float
    net_sharpe: float
    avg_turnover: float
    max_drawdown: float
    cumulative_net_return: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WalkForwardResult:
    returns: pd.DataFrame
    folds: list[FoldWindow]
    summary: WalkForwardSummary
    sector_pnl_share: Optional[dict[str, float]] = None
    factor_overlap_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "folds": [fold.to_dict() for fold in self.folds],
            "rows": int(len(self.returns)),
            "sector_pnl_share": self.sector_pnl_share,
            "factor_overlap_score": self.factor_overlap_score,
        }


def _annualized_sharpe(returns: pd.Series) -> float:
    s = pd.to_numeric(returns, errors="coerce").dropna()
    if s.empty:
        return 0.0
    std = float(s.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(s.mean()) / std * math.sqrt(252.0)


def _max_drawdown(returns: pd.Series) -> float:
    s = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if s.empty:
        return 0.0
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    dd = 1.0 - equity / peak.replace(0, pd.NA)
    return float(dd.max()) if len(dd) else 0.0


class WalkForwardRunner:
    """Deterministic walk-forward runner for the US baseline strategy."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        wf = config.get("walk_forward", {}) if isinstance(config.get("walk_forward"), dict) else {}
        portfolio = config.get("portfolio", {}) if isinstance(config.get("portfolio"), dict) else {}
        costs = config.get("costs", {}) if isinstance(config.get("costs"), dict) else {}
        exec_cfg = config.get("execution_alignment", {}) if isinstance(config.get("execution_alignment"), dict) else {}

        self.initial_train_months = int(wf.get("initial_train_months", 60))
        self.validation_months = int(wf.get("validation_months", 12))
        self.test_months = int(wf.get("test_months", 3))
        self.embargo_days = int(wf.get("embargo_days", 5))
        self.expanding_window = bool(wf.get("expanding_window", True))
        self.warm_up_days = int(wf.get("warm_up_trading_days", 252))
        self.history_window_days = int(max(wf.get("history_window_days", 400), 120))
        self.signal_lag_days = int(exec_cfg.get("signal_lag_days", 1))
        self.label_horizon_days = int(exec_cfg.get("rebalance_frequency_days", 1))

        self.signal_config = SignalConfig(
            top_k=int(portfolio.get("top_k", 30)),
            max_weight=float(portfolio.get("max_weight_per_name", 0.05)),
            max_sector_weight=float(portfolio.get("max_sector_weight", 1.0)),
            long_only=bool(portfolio.get("long_only", True)),
            max_turnover_daily=float(portfolio.get("max_daily_turnover", 0.20)),
            min_avg_dollar_volume=float(portfolio.get("min_avg_daily_volume_usd", 5_000_000)),
        )
        signals_cfg = config.get("signals", {}) if isinstance(config.get("signals"), dict) else {}
        # mined factor expressions (already sanitized upstream) evaluated into
        # the score alongside the baseline ranks; see signal_generator
        self.mined_expressions = [str(e) for e in (signals_cfg.get("mined_expressions") or [])]
        retail = config.get("retail_execution", {}) if isinstance(config.get("retail_execution"), dict) else {}
        self.starting_equity = float(retail.get("starting_equity", 250_000.0))
        self.cash_buffer_pct = float(retail.get("cash_buffer_pct", 0.02))
        self.min_trade_dollars = float(retail.get("min_trade_dollars", 25.0))
        self.fractional_shares = bool(retail.get("fractional_shares", True))
        self.max_participation_rate = float(retail.get("max_participation_rate", 0.05))
        self.cost_model = TransactionCostModel(
            commission_per_share=float(costs.get("commission_per_share", 0.0)),
            half_spread_bps=float(costs.get("spread_bps", 1.5)),
            slippage_bps=float(costs.get("slippage_bps", 1.0)),
            impact_coefficient=float(costs.get("impact_coefficient", 0.1)),
        )

    @staticmethod
    def _nearest_on_or_before(dates: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
        idx = dates.searchsorted(target, side="right") - 1
        if idx < 0 or idx >= len(dates):
            return None
        return pd.Timestamp(dates[idx]).normalize()

    @staticmethod
    def _nearest_on_or_after(dates: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
        idx = dates.searchsorted(target, side="left")
        if idx < 0 or idx >= len(dates):
            return None
        return pd.Timestamp(dates[idx]).normalize()

    @staticmethod
    def _shift_trading_days(dates: pd.DatetimeIndex, date_value: pd.Timestamp, n: int) -> Optional[pd.Timestamp]:
        idx = dates.searchsorted(pd.Timestamp(date_value).normalize(), side="left")
        out_idx = idx + int(n)
        if out_idx < 0 or out_idx >= len(dates):
            return None
        return pd.Timestamp(dates[out_idx]).normalize()

    def build_folds(self, trading_dates: pd.DatetimeIndex) -> list[FoldWindow]:
        dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize().unique().sort_values()
        if len(dates) < self.warm_up_days + 252:
            return []

        folds: list[FoldWindow] = []
        train_start = pd.Timestamp(dates[0]).normalize()
        train_end_target = train_start + pd.DateOffset(months=self.initial_train_months)
        fold_id = 0

        while True:
            train_end = self._nearest_on_or_before(dates, pd.Timestamp(train_end_target))
            if train_end is None:
                break

            valid_start = self._nearest_on_or_after(dates, train_end + pd.offsets.BDay(self.embargo_days + 1))
            if valid_start is None:
                break
            valid_end = self._nearest_on_or_before(dates, valid_start + pd.DateOffset(months=self.validation_months))
            if valid_end is None:
                break
            test_start = self._nearest_on_or_after(dates, valid_end + pd.offsets.BDay(1))
            if test_start is None:
                break
            test_end = self._nearest_on_or_before(dates, test_start + pd.DateOffset(months=self.test_months))
            if test_end is None or test_end <= test_start:
                break

            folds.append(
                FoldWindow(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            fold_id += 1

            shift = pd.DateOffset(months=self.test_months)
            if self.expanding_window:
                train_end_target = train_end + shift
            else:
                train_start = self._nearest_on_or_after(dates, train_start + shift) or train_start
                train_end_target = train_end + shift

            if test_end >= pd.Timestamp(dates[-1]).normalize():
                break
            if fold_id > 200:
                break

        return folds

    @staticmethod
    def _overlap_score(factor_sets: list[set[str]]) -> Optional[float]:
        """
        Gate-6 stability score: mean Jaccard overlap of consecutive folds'
        top-factor sets. Needs at least two folds with measurable ICs;
        otherwise returns None so the gate fails honestly instead of passing
        on a constant.
        """
        if len(factor_sets) < 2:
            return None
        overlaps = [
            len(a & b) / len(a | b)
            for a, b in zip(factor_sets, factor_sets[1:])
            if len(a | b) > 0
        ]
        if not overlaps:
            return None
        return float(sum(overlaps) / len(overlaps))

    @staticmethod
    def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
        work = bars.copy().assign(
            date=pd.to_datetime(bars["date"], errors="coerce").dt.normalize(),
            symbol=bars["symbol"].astype(str).str.upper(),
        )
        for col in ["open", "high", "low", "close", "adj_close", "volume", "dollar_volume"]:
            if col in work.columns:
                work.loc[:, col] = pd.to_numeric(work[col], errors="coerce")
        if "adj_close" not in work.columns:
            work = work.assign(adj_close=work["close"])
        work = work.dropna(subset=["date", "symbol", "open", "close"])
        work = work.sort_values(["date", "symbol"]).reset_index(drop=True)
        return work

    @staticmethod
    def _open_return_for_symbols(
        bars_entry: pd.DataFrame,
        bars_exit: pd.DataFrame,
        symbols: list[str],
    ) -> pd.DataFrame:
        e = bars_entry[bars_entry["symbol"].isin(symbols)][["symbol", "open"]].rename(columns={"open": "open_entry"})
        x = bars_exit[bars_exit["symbol"].isin(symbols)][["symbol", "open"]].rename(columns={"open": "open_exit"})
        merged = e.merge(x, on="symbol", how="inner")
        merged = merged[(merged["open_entry"] > 0) & (merged["open_exit"] > 0)]
        merged = merged.assign(ret=merged["open_exit"] / merged["open_entry"] - 1.0)
        return merged

    def _simulate_retail_rebalance(
        self,
        *,
        signal_df: pd.DataFrame,
        feature_snapshot: pd.DataFrame,
        bars_entry: pd.DataFrame,
        bars_exit: pd.DataFrame,
        previous_shares: dict[str, float],
        current_equity: float,
    ) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], float, float, float]:
        if signal_df.empty:
            return pd.DataFrame(), {}, {}, 0.0, 0.0, 0.0

        snapshot = feature_snapshot.copy()
        if snapshot.empty:
            return pd.DataFrame(), {}, {}, 0.0, 0.0, 0.0
        snapshot = snapshot.drop_duplicates(subset=["symbol"]).copy()
        entry = bars_entry[["symbol", "open"]].rename(columns={"open": "entry_open"}).drop_duplicates(subset=["symbol"])
        exit_ = bars_exit[["symbol", "open"]].rename(columns={"open": "exit_open"}).drop_duplicates(subset=["symbol"])
        context = snapshot[["symbol", "adv20", "vol_21d"]].merge(entry, on="symbol", how="inner").merge(exit_, on="symbol", how="inner")
        context = context[(context["entry_open"] > 0) & (context["exit_open"] > 0)].copy()
        if context.empty:
            return pd.DataFrame(), {}, {}, 0.0, 0.0, 0.0

        target_map = {
            str(row.symbol).upper(): float(row.weight)
            for row in signal_df.itertuples(index=False)
            if pd.notna(row.weight) and float(row.weight) > 0
        }
        deployable_notional = max(current_equity * (1.0 - self.cash_buffer_pct), 0.0)
        context = context.assign(
            symbol=context["symbol"].astype(str).str.upper(),
            target_weight=context["symbol"].map(target_map).fillna(0.0),
        )

        # Explicitly liquidate holdings that dropped out of today's context
        # (index exit, missing features). They previously became cash with no
        # sell trade, no cost, and no turnover. Sell at the entry open when a
        # bar exists; with no bar at all the position still converts at its
        # last mark, which is the best available under daily data.
        context_syms = set(context["symbol"])
        forced_liquidations: set[str] = set()
        if previous_shares and not bars_entry.empty:
            entry_px_map = {
                str(r.symbol).upper(): float(r.open)
                for r in bars_entry[["symbol", "open"]].itertuples(index=False)
                if pd.notna(r.open) and float(r.open) > 0
            }
            extra_rows = []
            for held_sym, held_shares in previous_shares.items():
                s = str(held_sym).upper()
                if s in context_syms or held_shares <= 0:
                    continue
                px = entry_px_map.get(s)
                if px is None:
                    continue
                extra_rows.append(
                    {
                        "symbol": s,
                        "adv20": float("nan"),
                        "vol_21d": float("nan"),
                        "entry_open": px,
                        "exit_open": px,
                        "target_weight": 0.0,
                    }
                )
                forced_liquidations.add(s)
            if extra_rows:
                context = pd.concat([context, pd.DataFrame(extra_rows)], ignore_index=True)

        current_weights: dict[str, float] = {}
        target_weights: dict[str, float] = {}
        new_shares: dict[str, float] = {}
        turnover = 0.0
        cost_return = 0.0
        rows: list[dict[str, Any]] = []

        for row in context.itertuples(index=False):
            sym = str(row.symbol).upper()
            entry_px = float(row.entry_open)
            exit_px = float(row.exit_open)
            adv20 = float(row.adv20) if pd.notna(row.adv20) else None
            vol21 = float(row.vol_21d) if pd.notna(row.vol_21d) else None

            current_shares = float(previous_shares.get(sym, 0.0))
            current_value = current_shares * entry_px
            raw_target_value = max(deployable_notional * float(row.target_weight), 0.0)
            delta_value = raw_target_value - current_value

            trade_cap = None
            if adv20 is not None and adv20 > 0 and self.max_participation_rate > 0:
                trade_cap = adv20 * self.max_participation_rate
            if abs(delta_value) < self.min_trade_dollars and sym not in forced_liquidations:
                adjusted_target_value = current_value
            elif trade_cap is not None and trade_cap > 0 and abs(delta_value) > trade_cap:
                adjusted_target_value = max(0.0, current_value + math.copysign(trade_cap, delta_value))
            else:
                adjusted_target_value = raw_target_value

            if self.fractional_shares:
                adjusted_target_shares = adjusted_target_value / entry_px if entry_px > 0 else 0.0
            else:
                adjusted_target_shares = math.floor(max(adjusted_target_value, 0.0) / entry_px + 1e-12) if entry_px > 0 else 0.0
                adjusted_target_value = adjusted_target_shares * entry_px

            trade_notional = abs(adjusted_target_value - current_value)
            current_weight = current_value / current_equity if current_equity > 0 else 0.0
            target_weight_actual = adjusted_target_value / current_equity if current_equity > 0 else 0.0

            current_weights[sym] = current_weight
            if adjusted_target_shares > 1e-10:
                new_shares[sym] = adjusted_target_shares
                target_weights[sym] = target_weight_actual

            turnover += 0.5 * abs(target_weight_actual - current_weight)
            if trade_notional > 0 and current_equity > 0:
                cost_return += (trade_notional / current_equity) * self.cost_model.estimate_cost_fraction(
                    trade_notional=trade_notional,
                    adv_20d=adv20,
                    daily_vol=vol21,
                )

            rows.append(
                {
                    "symbol": sym,
                    "entry_open": entry_px,
                    "exit_open": exit_px,
                    "ret": exit_px / entry_px - 1.0,
                    "weight": target_weight_actual,
                    "current_weight": current_weight,
                    "trade_notional": trade_notional,
                }
            )

        merged = pd.DataFrame(rows)
        merged = merged[merged["weight"] > 0].copy()
        gross_return = float((merged["weight"] * merged["ret"]).sum()) if not merged.empty else 0.0
        return merged, new_shares, target_weights, turnover, cost_return, gross_return

    def run(
        self,
        *,
        bars: pd.DataFrame,
        universe: SP500Universe,
        output_dir: Optional[Path] = None,
        sector_map: Optional[dict[str, str]] = None,
    ) -> WalkForwardResult:
        data = self._normalize_bars(bars)
        data = data.assign(mom_252=data.groupby("symbol")["adj_close"].pct_change(252))
        features = build_features(data, mined_expressions=self.mined_expressions)
        features_by_date = {
            pd.Timestamp(day).normalize(): group.reset_index(drop=True)
            for day, group in features.groupby("date", sort=False)
        }
        bars_by_date = {
            pd.Timestamp(day).normalize(): group.reset_index(drop=True)
            for day, group in data.groupby("date", sort=False)
        }
        dates = pd.DatetimeIndex(sorted(data["date"].dropna().unique())).normalize()
        folds = self.build_folds(dates)
        if not folds:
            raise RuntimeError("Unable to build walk-forward folds from available data")

        records: list[dict[str, Any]] = []
        sector_contrib_abs: dict[str, float] = {}
        factor_sets: list[set[str]] = []
        for fold in folds:
            fold_test_dates = dates[(dates >= fold.test_start) & (dates <= fold.test_end)]
            previous_weights: Optional[dict[str, float]] = None
            previous_shares: dict[str, float] = {}
            current_equity = self.starting_equity
            # per-fold daily rank ICs per factor, for the gate-6 stability score
            fold_factor_ics: dict[str, list[float]] = {}

            for as_of in fold_test_dates:
                active = universe.get_members(as_of)
                if not active:
                    continue

                feature_snapshot = features_by_date.get(pd.Timestamp(as_of).normalize(), pd.DataFrame())
                signal_df = select_signals_from_snapshot(
                    feature_snapshot,
                    config=self.signal_config,
                    active_universe=active,
                    previous_weights=previous_weights,
                    sector_map=sector_map,
                )
                if signal_df.empty:
                    continue

                entry_date = self._shift_trading_days(dates, as_of, self.signal_lag_days)
                exit_date = self._shift_trading_days(dates, entry_date, self.label_horizon_days) if entry_date is not None else None
                if entry_date is None or exit_date is None:
                    continue

                bars_entry = bars_by_date.get(pd.Timestamp(entry_date).normalize(), pd.DataFrame())
                bars_exit = bars_by_date.get(pd.Timestamp(exit_date).normalize(), pd.DataFrame())
                merged, current_shares, current_weights_actual, turnover, cost_return, gross_return = self._simulate_retail_rebalance(
                    signal_df=signal_df,
                    feature_snapshot=feature_snapshot,
                    bars_entry=bars_entry,
                    bars_exit=bars_exit,
                    previous_shares=previous_shares,
                    current_equity=current_equity,
                )
                if merged.empty:
                    continue
                net_return = gross_return - cost_return
                current_equity *= 1.0 + net_return
                curr_weights = dict(current_weights_actual)
                gross_exposure = float(sum(curr_weights.values()))

                if sector_map:
                    merged = merged.assign(
                        sector=merged["symbol"].astype(str).str.upper().map(lambda s: sector_map.get(s, "Unknown")),
                        abs_contrib=(
                            pd.to_numeric(merged["weight"], errors="coerce")
                            * pd.to_numeric(merged["ret"], errors="coerce")
                        ).abs(),
                    )
                    sector_daily = merged.groupby("sector", dropna=False)["abs_contrib"].sum()
                    for sector, contrib in sector_daily.items():
                        key = str(sector or "Unknown")
                        sector_contrib_abs[key] = sector_contrib_abs.get(key, 0.0) + float(contrib)

                # Baselines for gate-7.
                spy_ret = float("nan")
                spy_df = self._open_return_for_symbols(bars_entry, bars_exit, ["SPY"])
                if not spy_df.empty:
                    spy_ret = float(spy_df["ret"].iloc[0]) * (1.0 - self.cash_buffer_pct)

                active_rets = self._open_return_for_symbols(bars_entry, bars_exit, active)
                eqw_ret = float(active_rets["ret"].mean()) * (1.0 - self.cash_buffer_pct) if not active_rets.empty else float("nan")

                # Daily rank IC of each baseline factor against realized
                # entry-open -> exit-open returns over the active universe
                # (feeds the gate-6 factor-stability score).
                if not active_rets.empty and not feature_snapshot.empty:
                    ic_frame = feature_snapshot.merge(
                        active_rets[["symbol", "ret"]], on="symbol", how="inner"
                    )
                    if len(ic_frame) >= 30:
                        for factor_col in active_factor_columns(len(self.mined_expressions)):
                            if factor_col in ic_frame.columns:
                                ic = ic_frame[factor_col].corr(ic_frame["ret"], method="spearman")
                                if pd.notna(ic):
                                    fold_factor_ics.setdefault(factor_col, []).append(float(ic))

                mom_ret = float("nan")
                latest = bars_by_date.get(pd.Timestamp(as_of).normalize(), pd.DataFrame())
                if not latest.empty and "mom_252" in latest.columns:
                    latest = latest[["symbol", "mom_252"]].dropna()
                    if not latest.empty:
                        top = latest.nlargest(self.signal_config.top_k, "mom_252")["symbol"].astype(str).tolist()
                        top_rets = self._open_return_for_symbols(bars_entry, bars_exit, top)
                        if not top_rets.empty:
                            mom_ret = float(top_rets["ret"].mean()) * (1.0 - self.cash_buffer_pct)

                records.append(
                    {
                        "fold_id": fold.fold_id,
                        "as_of_date": str(pd.Timestamp(as_of).date()),
                        "entry_date": str(pd.Timestamp(entry_date).date()),
                        "exit_date": str(pd.Timestamp(exit_date).date()),
                        "positions": int(len(curr_weights)),
                        "turnover": turnover,
                        "gross_exposure": gross_exposure,
                        "cash_weight": max(0.0, 1.0 - gross_exposure),
                        "portfolio_equity": current_equity,
                        "gross_return": gross_return,
                        "cost_return": cost_return,
                        "net_return": net_return,
                        "baseline_spy": spy_ret,
                        "baseline_equal_weight": eqw_ret,
                        "baseline_momentum": mom_ret,
                    }
                )
                previous_weights = curr_weights
                previous_shares = current_shares

            # Fold-level factor set: top-3 factors by |mean daily IC|
            if fold_factor_ics:
                mean_ics = {f: sum(v) / len(v) for f, v in fold_factor_ics.items()}
                top = set(sorted(mean_ics, key=lambda f: abs(mean_ics[f]), reverse=True)[:3])
                factor_sets.append(top)

        returns = pd.DataFrame(records)
        if returns.empty:
            raise RuntimeError("Walk-forward run produced no test observations")

        gross_sharpe = _annualized_sharpe(returns["gross_return"])
        net_sharpe = _annualized_sharpe(returns["net_return"])
        cumulative = float((1.0 + pd.to_numeric(returns["net_return"], errors="coerce").fillna(0.0)).prod() - 1.0)
        summary = WalkForwardSummary(
            num_folds=len(folds),
            num_observations=int(len(returns)),
            start_date=str(returns["entry_date"].min()),
            end_date=str(returns["exit_date"].max()),
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            avg_turnover=float(pd.to_numeric(returns["turnover"], errors="coerce").mean()),
            max_drawdown=_max_drawdown(returns["net_return"]),
            cumulative_net_return=cumulative,
        )
        total_sector_abs = sum(sector_contrib_abs.values())
        sector_pnl_share = (
            {k: float(v) / total_sector_abs for k, v in sector_contrib_abs.items() if total_sector_abs > 0}
            if total_sector_abs > 0
            else None
        )
        factor_overlap_score = self._overlap_score(factor_sets)
        result = WalkForwardResult(
            returns=returns,
            folds=folds,
            summary=summary,
            sector_pnl_share=sector_pnl_share,
            factor_overlap_score=factor_overlap_score,
        )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            returns_path = output_dir / "walk_forward_returns.csv"
            folds_path = output_dir / "walk_forward_folds.json"
            summary_path = output_dir / "walk_forward_summary.json"

            returns.to_csv(returns_path, index=False)
            folds_path.write_text(json.dumps([f.to_dict() for f in folds], indent=2), encoding="utf-8")
            summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

        return result
