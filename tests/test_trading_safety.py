"""Trading-path safety: stale signals, non-bypassable checks, kill switch, empty universe."""

from pathlib import Path

import pandas as pd
import pytest

from quantaalpha_us.pipeline.signal_generator import SignalConfig, select_signals_from_snapshot
from quantaalpha_us.trading.risk import evaluate_pre_trade, load_risk_config


def _report(*, kill_switch=False, notional=1_000.0, universe=frozenset({"AAPL"})):
    return evaluate_pre_trade(
        signals=pd.DataFrame({"symbol": ["AAPL"], "weight": [0.03]}),
        order_intents=[{"symbol": "AAPL", "side": "buy", "notional": notional}],
        equity=100_000.0, buying_power=100_000.0,
        risk_config=load_risk_config({"risk": {"min_positions": 1,
                                               "max_single_order_value": 1e9,
                                               "max_position_pct": 1.0}}),
        active_universe=set(universe),
        latest_data_date=pd.Timestamp("2026-03-10"), as_of_date=pd.Timestamp("2026-03-10"),
        kill_switch_engaged=kill_switch, is_trading_day=True,
    )


def test_stale_signal_file_is_refused():
    from scripts.trade_once import resolve_signal_date

    signals = pd.DataFrame({"date": pd.to_datetime(["2026-03-09"]), "symbol": ["AAPL"],
                            "weight": [0.1]})
    with pytest.raises(RuntimeError, match="stale"):
        resolve_signal_date(signals, "2026-03-10")
    assert resolve_signal_date(signals, "2026-03-09") == pd.Timestamp("2026-03-09")
    assert resolve_signal_date(signals, None) == pd.Timestamp("2026-03-09")
    undated = resolve_signal_date(signals.drop(columns=["date"]), None)
    assert undated.tzinfo is None


def test_allow_risk_fail_never_bypasses_the_kill_switch_or_universe():
    from scripts.trade_once import enforce_pre_trade

    with pytest.raises(RuntimeError, match="PT-07"):
        enforce_pre_trade(_report(kill_switch=True), allow_risk_fail=True)
    with pytest.raises(RuntimeError, match="PT-03"):
        enforce_pre_trade(_report(universe={"MSFT"}), allow_risk_fail=True)
    # an ordinary sizing failure is still the operator's call
    oversized = _report(notional=90_000.0)
    assert not oversized.passed
    enforce_pre_trade(oversized, allow_risk_fail=True)
    with pytest.raises(RuntimeError):
        enforce_pre_trade(oversized, allow_risk_fail=False)


def test_empty_universe_fails_the_universe_check():
    assert not _report(universe=set()).checks["PT-03_universe"].passed


def test_empty_universe_selects_nothing():
    snapshot = pd.DataFrame({"date": pd.Timestamp("2026-03-10"), "symbol": ["AAPL", "MSFT"],
                             "score": [0.9, 0.8], "adv20": [1e9, 1e9]})
    out = select_signals_from_snapshot(snapshot, config=SignalConfig(top_k=2, max_weight=0.5),
                                       active_universe=[])
    assert out.empty


def test_level_two_kill_switch_pauses_trading(tmp_path, monkeypatch):
    from scripts import kill_switch

    class _Client:
        def cancel_all_orders(self):
            return []

        def close_all_positions(self):
            return []

    monkeypatch.setattr(kill_switch, "_build_client", lambda cfg: _Client())
    monkeypatch.setattr(kill_switch, "resolve_from_us_root", lambda rel, root: tmp_path / rel)
    payload = kill_switch._apply_level(2, "test", {}, Path("cfg.yaml"))
    assert (tmp_path / "data/results/trading/PAUSE_US.flag").exists()
    assert payload["action"] == "cancel_all_and_close_all"


def test_daily_loss_is_measured_against_the_previous_day(tmp_path, monkeypatch):
    from scripts import kill_switch

    equities = iter([100_000.0, 99_000.0, 97_500.0, 97_000.0])

    class _Client:
        def get_account(self):
            return {"equity": next(equities)}

    monkeypatch.setattr(kill_switch, "_build_client", lambda cfg: _Client())
    monkeypatch.setattr(kill_switch, "_apply_level",
                        lambda level, reason, cfg, path: {"level": level})
    state = tmp_path / "state.json"
    cfg = {"kill_switch": {"daily_loss": {"threshold": 0.02}}}
    kill_switch._monitor(cfg, Path("c.yaml"), state, today="2026-03-09")
    # two runs on the next day: 1% then 2.5% below the previous day's last equity
    first = kill_switch._monitor(cfg, Path("c.yaml"), state, today="2026-03-10")
    assert not first["triggered"] and first["days_seen"] == 2
    second = kill_switch._monitor(cfg, Path("c.yaml"), state, today="2026-03-10")
    assert second["trigger"] == "daily_loss"
    assert second["days_seen"] == 2
    assert second["daily_loss"] == pytest.approx(0.025)
