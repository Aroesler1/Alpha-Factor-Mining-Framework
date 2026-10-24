"""Walk-forward regressions: splits, exit-bar look-ahead, fold overlap, tickers."""

import numpy as np
import pandas as pd
import pytest

from quantaalpha_us.backtest.universe import SP500Universe
from quantaalpha_us.backtest.walk_forward import WalkForwardRunner

CFG = {
    "walk_forward": {"initial_train_months": 18, "validation_months": 6, "test_months": 2,
                     "embargo_days": 2, "expanding_window": True,
                     "warm_up_trading_days": 60, "history_window_days": 260},
    "execution_alignment": {"signal_lag_days": 1, "rebalance_frequency_days": 1},
    "portfolio": {"top_k": 2, "max_weight_per_name": 0.5, "max_sector_weight": 1.0,
                  "long_only": True, "max_daily_turnover": 1.0,
                  "min_avg_daily_volume_usd": 1000},
    "retail_execution": {"starting_equity": 100000.0, "cash_buffer_pct": 0.0,
                         "min_trade_dollars": 0.0, "fractional_shares": True,
                         "max_participation_rate": 1.0},
    "costs": {"spread_bps": 0.0, "slippage_bps": 0.0, "commission_per_share": 0.0,
              "impact_coefficient": 0.0},
}


def test_a_split_does_not_book_a_fake_return(tmp_path):
    # two names whose economic price never moves; B splits 2:1 mid-sample
    dates = pd.bdate_range("2020-01-01", "2023-06-30")
    split = dates[-60]  # inside the last test window
    rows = []
    for d in dates:
        for sym, base in (("A", 100.0), ("B", 80.0)):
            factor = 2.0 if (sym == "B" and d < split) else 1.0
            raw = base * factor if sym == "B" else base
            rows.append({"date": d, "symbol": sym, "open": raw, "high": raw, "low": raw,
                         "close": raw, "adj_close": raw / factor, "volume": 1e6,
                         "dollar_volume": raw * 1e6})
    bars = pd.DataFrame(rows)
    membership = pd.DataFrame([{"date": d, "symbol": s, "active": True}
                               for d in dates for s in ("A", "B")])
    path = tmp_path / "m.csv"
    membership.to_csv(path, index=False)
    result = WalkForwardRunner(CFG).run(bars=bars, universe=SP500Universe(str(path)))
    assert result.returns["gross_return"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_a_name_without_an_exit_bar_is_still_held():
    runner = WalkForwardRunner(CFG)
    snapshot = pd.DataFrame({"symbol": ["AAPL", "XYZ"], "adv20": [1e9, 1e9],
                             "vol_21d": [0.01, 0.01]})
    entry = pd.DataFrame({"symbol": ["AAPL", "XYZ"], "trade_open": [100.0, 50.0],
                          "adj_close": [101.0, 40.0]})
    exit_ = pd.DataFrame({"symbol": ["AAPL"], "trade_open": [102.0]})
    signals = pd.DataFrame({"symbol": ["AAPL", "XYZ"], "weight": [0.5, 0.5]})
    merged, *_ , gross = runner._simulate_retail_rebalance(
        signal_df=signals, feature_snapshot=snapshot, bars_entry=entry, bars_exit=exit_,
        previous_shares={}, current_equity=100_000.0)
    held = merged.set_index("symbol")
    assert set(held.index) == {"AAPL", "XYZ"}
    # XYZ is marked at its last close: 50 -> 40
    assert held.loc["XYZ", "ret"] == pytest.approx(-0.2)
    assert gross == pytest.approx(0.5 * 0.02 + 0.5 * -0.2)


def test_fold_test_windows_do_not_overlap():
    dates = pd.bdate_range("2000-01-03", "2025-12-31")
    runner = WalkForwardRunner({"walk_forward": {"initial_train_months": 60,
                                                 "validation_months": 12, "test_months": 3,
                                                 "embargo_days": 5}})
    folds = runner.build_folds(dates)
    assert len(folds) > 50
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.test_start > prev.test_end


def test_dated_membership_symbols_are_not_renamed(tmp_path):
    membership = pd.DataFrame({"date": ["2015-06-01"] * 3, "symbol": ["FB", "GOOG", "GOOGL"],
                               "permno": [13407, 14542, 90319], "active": True})
    mpath = tmp_path / "m.csv"
    membership.to_csv(mpath, index=False)
    tmap = tmp_path / "t.csv"
    pd.DataFrame({"old_symbol": ["FB", "GOOG"], "new_symbol": ["META", "GOOGL"]}).to_csv(
        tmap, index=False)
    universe = SP500Universe(str(mpath), ticker_mapping_file=str(tmap))
    assert universe.get_members("2015-06-01") == ["FB", "GOOG", "GOOGL"]
