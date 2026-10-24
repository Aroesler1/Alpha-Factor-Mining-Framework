from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class RiskConfig:
    max_position_pct: float = 0.05
    max_drawdown_pct: float = 0.10
    max_daily_loss_pct: float = 0.03
    max_turnover_daily: float = 0.20
    max_positions: int = 30
    min_positions: int = 5
    min_cash_pct: float = 0.02
    max_portfolio_value: float = 250000.0
    max_single_order_value: float = 12500.0
    flatten_on_kill: bool = True


@dataclass
class CheckOutcome:
    passed: bool
    details: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "details": self.details,
            "metrics": self.metrics,
        }


@dataclass
class PreTradeReport:
    passed: bool
    checks: dict[str, CheckOutcome]

    # Checks that no operator flag may override: an engaged kill switch and
    # trading names outside the index universe.
    NON_BYPASSABLE = ("PT-03_universe", "PT-07_kill_switch")

    def hard_failures(self) -> list[str]:
        """Failed checks that `--allow-risk-fail` must not wave through."""
        return [name for name in self.NON_BYPASSABLE
                if name in self.checks and not self.checks[name].passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
        }


@dataclass
class PostTradeReport:
    passed: bool
    checks: dict[str, CheckOutcome]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
        }


def load_risk_config(cfg: dict[str, Any]) -> RiskConfig:
    risk = cfg.get("risk", {}) if isinstance(cfg.get("risk"), dict) else {}
    return RiskConfig(
        max_position_pct=float(risk.get("max_position_pct", 0.05)),
        max_drawdown_pct=float(risk.get("max_drawdown_pct", 0.10)),
        max_daily_loss_pct=float(risk.get("max_daily_loss_pct", 0.03)),
        max_turnover_daily=float(risk.get("max_turnover_daily", 0.20)),
        max_positions=int(risk.get("max_positions", 30)),
        min_positions=int(risk.get("min_positions", 5)),
        min_cash_pct=float(risk.get("min_cash_pct", 0.02)),
        max_portfolio_value=float(risk.get("max_portfolio_value", 250000.0)),
        max_single_order_value=float(risk.get("max_single_order_value", 12500.0)),
        flatten_on_kill=bool(risk.get("flatten_on_kill", True)),
    )


def _intent_notional(intent: dict[str, Any]) -> float:
    delta_value = intent.get("delta_value")
    if isinstance(delta_value, (int, float)):
        return abs(float(delta_value))
    notional = intent.get("notional")
    if isinstance(notional, (int, float)):
        return abs(float(notional))
    qty = intent.get("qty")
    px = intent.get("price")
    if isinstance(qty, (int, float)) and isinstance(px, (int, float)):
        return abs(float(qty) * float(px))
    return 0.0


def evaluate_pre_trade(
    *,
    signals: pd.DataFrame,
    order_intents: list[dict[str, Any]],
    equity: float,
    buying_power: float,
    risk_config: RiskConfig,
    active_universe: Optional[set[str]] = None,
    latest_data_date: Optional[pd.Timestamp] = None,
    as_of_date: Optional[pd.Timestamp] = None,
    kill_switch_engaged: bool = False,
    is_trading_day: bool = True,
) -> PreTradeReport:
    checks: dict[str, CheckOutcome] = {}
    safe_equity = max(float(equity), 0.0)

    total_notional = sum(_intent_notional(intent) for intent in order_intents)
    checks["PT-01_total_notional"] = CheckOutcome(
        passed=total_notional <= 0.40 * safe_equity + 1e-9,
        details="Total order notional must be <= 40% of portfolio equity",
        metrics={"total_notional": total_notional, "limit": 0.40 * safe_equity},
    )

    max_single = max((_intent_notional(intent) for intent in order_intents), default=0.0)
    single_limit = min(risk_config.max_single_order_value, risk_config.max_position_pct * safe_equity)
    checks["PT-02_single_order"] = CheckOutcome(
        passed=max_single <= single_limit + 1e-9,
        details="No single order may exceed configured max",
        metrics={"max_single_order": max_single, "limit": single_limit},
    )

    if signals.empty:
        symbol_list: list[str] = []
    else:
        symbol_list = sorted(set(signals["symbol"].astype(str).str.upper().tolist()))
    if active_universe is None:
        checks["PT-03_universe"] = CheckOutcome(
            passed=True,
            details="Universe membership check skipped (active universe not provided)",
            metrics={"symbols": len(symbol_list)},
        )
    else:
        missing = sorted(set(symbol_list) - set(active_universe))
        checks["PT-03_universe"] = CheckOutcome(
            passed=len(missing) == 0,
            details="All symbols must be in active S&P 500 universe",
            metrics={"missing_symbols": missing[:25], "missing_count": len(missing)},
        )

    if as_of_date is None or latest_data_date is None:
        checks["PT-04_data_freshness"] = CheckOutcome(
            passed=True,
            details="Data freshness check skipped (date inputs not provided)",
            metrics={},
        )
    else:
        latest = pd.Timestamp(latest_data_date).normalize()
        target = pd.Timestamp(as_of_date).normalize()
        checks["PT-04_data_freshness"] = CheckOutcome(
            passed=latest >= target,
            details="Latest data date must be current signal date",
            metrics={"latest_data_date": str(latest.date()), "target_date": str(target.date())},
        )

    weights_sum = float(pd.to_numeric(signals.get("weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    max_weight = float(pd.to_numeric(signals.get("weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0).max()) if not signals.empty else 0.0
    n_positions = int(len(signals)) if not signals.empty else 0
    checks["PT-05_portfolio_constraints"] = CheckOutcome(
        passed=(
            n_positions <= risk_config.max_positions
            and n_positions >= risk_config.min_positions
            and max_weight <= risk_config.max_position_pct + 1e-9
            and weights_sum <= 1.0 + 1e-9
        ),
        details="Target portfolio must satisfy count, weight, and sum constraints",
        metrics={
            "positions": n_positions,
            "max_positions": risk_config.max_positions,
            "max_weight": max_weight,
            "max_weight_limit": risk_config.max_position_pct,
            "weights_sum": weights_sum,
        },
    )

    buy_notional = sum(
        _intent_notional(intent)
        for intent in order_intents
        if str(intent.get("side", "")).lower() == "buy"
    )
    checks["PT-06_buying_power"] = CheckOutcome(
        passed=float(buying_power) + 1e-9 >= buy_notional,
        details="Buying power must cover buy-side order notional",
        metrics={"buying_power": float(buying_power), "required_notional": buy_notional},
    )

    checks["PT-07_kill_switch"] = CheckOutcome(
        passed=not kill_switch_engaged,
        details="Kill switch must not be engaged",
        metrics={"kill_switch_engaged": bool(kill_switch_engaged)},
    )

    checks["PT-08_session"] = CheckOutcome(
        passed=bool(is_trading_day),
        details="Submission must happen on a valid trading day/window",
        metrics={"is_trading_day": bool(is_trading_day)},
    )

    passed = all(item.passed for item in checks.values())
    return PreTradeReport(passed=passed, checks=checks)


def evaluate_post_trade(
    *,
    target_weights: dict[str, float],
    positions_after: list[dict[str, Any]],
    order_responses: list[dict[str, Any]],
    equity: float,
    risk_config: RiskConfig,
    expected_open_prices: Optional[dict[str, float]] = None,
    account_cash: Optional[float] = None,
) -> PostTradeReport:
    checks: dict[str, CheckOutcome] = {}

    statuses: list[str] = []
    for wrapper in order_responses:
        resp = wrapper.get("response", {})
        status = str(resp.get("status", "")).lower() if isinstance(resp, dict) else ""
        if status:
            statuses.append(status)
    fatal_status = {"rejected", "canceled", "expired", "suspended"}
    fatal = sorted({s for s in statuses if s in fatal_status})
    checks["PT-09_fill_status"] = CheckOutcome(
        passed=len(fatal) == 0,
        details="Orders must not end in fatal status",
        metrics={"fatal_statuses": fatal, "all_statuses": statuses[:200]},
    )

    slippages: list[float] = []
    if expected_open_prices:
        for wrapper in order_responses:
            resp = wrapper.get("response", {})
            intent = wrapper.get("intent", {})
            if not isinstance(resp, dict):
                continue
            symbol = str(intent.get("symbol", "")).upper()
            expected = expected_open_prices.get(symbol)
            fill = resp.get("filled_avg_price")
            if expected and isinstance(fill, (int, float, str)):
                try:
                    fill_px = float(fill)
                    expected_px = float(expected)
                    if expected_px > 0:
                        bps = abs(fill_px / expected_px - 1.0) * 10000.0
                        slippages.append(bps)
                except Exception:
                    continue
    max_slippage = max(slippages) if slippages else 0.0
    checks["PT-10_slippage"] = CheckOutcome(
        passed=max_slippage <= 100.0 + 1e-9,
        details="Fill slippage must be <= 100 bps against expected open",
        metrics={"max_slippage_bps": max_slippage, "samples": len(slippages)},
    )

    safe_equity = max(float(equity), 0.0)
    actual_weights: dict[str, float] = {}
    for pos in positions_after:
        sym = str(pos.get("symbol", "")).upper()
        if not sym:
            continue
        mv = float(pos.get("market_value", 0.0) or 0.0)
        actual_weights[sym] = mv / safe_equity if safe_equity > 0 else 0.0
    symbols = sorted(set(target_weights) | set(actual_weights))
    diffs = [abs(actual_weights.get(sym, 0.0) - target_weights.get(sym, 0.0)) for sym in symbols]
    max_diff = max(diffs) if diffs else 0.0
    checks["PT-11_position_reconcile"] = CheckOutcome(
        passed=max_diff <= 0.01 + 1e-9,
        details="Actual portfolio weights should match target within +/- 1%",
        metrics={"max_abs_weight_diff": max_diff, "symbols_compared": len(symbols)},
    )

    if account_cash is None:
        checks["PT-12_cash_sanity"] = CheckOutcome(
            passed=True,
            details="Cash sanity skipped (account cash not provided)",
            metrics={},
        )
    else:
        min_cash = risk_config.min_cash_pct * safe_equity
        checks["PT-12_cash_sanity"] = CheckOutcome(
            passed=float(account_cash) + 1e-9 >= min_cash,
            details="Cash must remain above configured minimum fraction of equity",
            metrics={"cash": float(account_cash), "min_required_cash": min_cash},
        )

    passed = all(item.passed for item in checks.values())
    return PostTradeReport(passed=passed, checks=checks)
