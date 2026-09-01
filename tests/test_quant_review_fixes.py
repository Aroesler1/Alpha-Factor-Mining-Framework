"""
Regression tests for the 2026-08 review fixes:
- real Bailey / Lopez de Prado deflated Sharpe (gate-1)
- measured factor-overlap stability instead of a constant (gate-6)
- explicit liquidation of holdings that drop out of the daily context
"""
import numpy as np
import pandas as pd

from quantaalpha_us.backtest.validation import (
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
)
from quantaalpha_us.backtest.walk_forward import WalkForwardRunner


def _runner() -> WalkForwardRunner:
    cfg = {
        "walk_forward": {"initial_train_months": 12},
        "execution_alignment": {"signal_lag_days": 1, "rebalance_frequency_days": 1},
        "portfolio": {"top_k": 2, "max_weight_per_name": 0.5},
        "retail_execution": {
            "starting_equity": 100000.0,
            "cash_buffer_pct": 0.0,
            "min_trade_dollars": 25.0,
            "fractional_shares": True,
            "max_participation_rate": 0.0,
        },
        "costs": {"spread_bps": 2.0, "slippage_bps": 1.0, "impact_coefficient": 0.0},
    }
    return WalkForwardRunner(cfg)


def test_psr_of_noise_is_near_half() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, size=5000))
    r = r - r.mean()  # exactly zero Sharpe; PSR is sharp in n so demean explicitly
    assert abs(probabilistic_sharpe(r) - 0.5) < 0.02


def test_expected_max_sharpe_monotone_in_trials() -> None:
    var = 0.02**2
    vals = [expected_max_sharpe(n, var) for n in (1, 5, 25, 100)]
    assert vals[0] == 0.0
    assert vals == sorted(vals)


def test_dsr_below_psr_under_multiple_trials() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, size=500))
    psr = probabilistic_sharpe(r)
    dsr = deflated_sharpe(r, n_trials=50, trial_sharpes=[0.0, 0.05, -0.03, 0.02, 0.01])
    assert dsr < psr


def test_overlap_score_measures_jaccard() -> None:
    sets = [{"a", "b", "c"}, {"a", "b", "c"}, {"a", "b", "d"}]
    # consecutive overlaps: 3/3 and 2/4 -> mean 0.75
    assert abs(WalkForwardRunner._overlap_score(sets) - 0.75) < 1e-12
    assert WalkForwardRunner._overlap_score([{"a"}]) is None
    assert WalkForwardRunner._overlap_score([]) is None


def test_dropped_holding_is_liquidated_with_cost() -> None:
    runner = _runner()

    # OLD is held but absent from today's features (e.g. index exit).
    # It must be sold at the entry open with a cost, not silently zeroed.
    feature_snapshot = pd.DataFrame(
        [{"symbol": "NEW", "adv20": 5e7, "vol_21d": 0.02}]
    )
    signal_df = pd.DataFrame([{"date": "2024-01-02", "symbol": "NEW", "score": 1.0, "weight": 0.5}])
    bars_entry = pd.DataFrame(
        [
            {"symbol": "NEW", "open": 100.0},
            {"symbol": "OLD", "open": 50.0},
        ]
    )
    bars_exit = pd.DataFrame(
        [
            {"symbol": "NEW", "open": 101.0},
            {"symbol": "OLD", "open": 49.0},
        ]
    )
    previous_shares = {"OLD": 100.0}  # $5,000 position

    merged, new_shares, target_weights, turnover, cost_return, gross_return = (
        runner._simulate_retail_rebalance(
            signal_df=signal_df,
            feature_snapshot=feature_snapshot,
            bars_entry=bars_entry,
            bars_exit=bars_exit,
            previous_shares=previous_shares,
            current_equity=100000.0,
        )
    )

    assert "OLD" not in new_shares
    # sale of $5,000 at 3bps (half-spread 2/2 + slippage 1... spread 2bps + slippage 1bps = 3bps here)
    assert cost_return > 0.0
    # turnover includes both the OLD sale and the NEW buy
    assert turnover > 0.5 * (5000.0 / 100000.0)


def test_tiny_dropped_holding_still_liquidated() -> None:
    runner = _runner()
    feature_snapshot = pd.DataFrame([{"symbol": "NEW", "adv20": 5e7, "vol_21d": 0.02}])
    signal_df = pd.DataFrame([{"date": "2024-01-02", "symbol": "NEW", "score": 1.0, "weight": 0.5}])
    bars_entry = pd.DataFrame(
        [{"symbol": "NEW", "open": 100.0}, {"symbol": "OLD", "open": 5.0}]
    )
    bars_exit = pd.DataFrame(
        [{"symbol": "NEW", "open": 101.0}, {"symbol": "OLD", "open": 5.0}]
    )
    # $10 position, below min_trade_dollars: must still be force-liquidated
    _, new_shares, _, _, _, _ = runner._simulate_retail_rebalance(
        signal_df=signal_df,
        feature_snapshot=feature_snapshot,
        bars_entry=bars_entry,
        bars_exit=bars_exit,
        previous_shares={"OLD": 2.0},
        current_equity=100000.0,
    )
    assert "OLD" not in new_shares
