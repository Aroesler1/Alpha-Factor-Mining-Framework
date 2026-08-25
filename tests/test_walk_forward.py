import numpy as np
import pandas as pd

from quantaalpha_us.backtest.universe import SP500Universe
from quantaalpha_us.backtest.walk_forward import WalkForwardRunner


def test_walk_forward_runs_with_synthetic_data(tmp_path) -> None:
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    symbols = ["AAPL", "MSFT", "SPY"]
    rows = []
    rng = np.random.default_rng(7)
    for i, d in enumerate(dates):
        for j, s in enumerate(symbols):
            base = 100 + 0.02 * i + j
            open_px = base * (1 + rng.normal(0, 0.002))
            close_px = open_px * (1 + rng.normal(0.0002, 0.01))
            high_px = max(open_px, close_px) * 1.01
            low_px = min(open_px, close_px) * 0.99
            rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "open": open_px,
                    "high": high_px,
                    "low": low_px,
                    "close": close_px,
                    "adj_close": close_px,
                    "volume": 1_000_000 + 50_000 * j,
                    "dollar_volume": close_px * (1_000_000 + 50_000 * j),
                }
            )
    bars = pd.DataFrame(rows)

    membership_rows = []
    for d in dates:
        for s in symbols:
            membership_rows.append({"date": d, "symbol": s, "active": True})
    membership = pd.DataFrame(membership_rows)
    membership_path = tmp_path / "membership.csv"
    membership.to_csv(membership_path, index=False)
    universe = SP500Universe(str(membership_path))
    sector_map = {s: "Information Technology" for s in symbols}

    cfg = {
        "walk_forward": {
            "initial_train_months": 18,
            "validation_months": 6,
            "test_months": 2,
            "embargo_days": 2,
            "expanding_window": True,
            "warm_up_trading_days": 60,
            "history_window_days": 260,
        },
        "execution_alignment": {"signal_lag_days": 1, "rebalance_frequency_days": 1},
        "portfolio": {
            "top_k": 2,
            "max_weight_per_name": 0.5,
            "max_sector_weight": 1.0,
            "long_only": True,
            "max_daily_turnover": 0.20,
            "min_avg_daily_volume_usd": 1000,
        },
        "retail_execution": {
            "starting_equity": 100000.0,
            "cash_buffer_pct": 0.10,
            "min_trade_dollars": 50.0,
            "fractional_shares": True,
            "max_participation_rate": 0.05,
        },
        "costs": {"spread_bps": 1.5, "slippage_bps": 1.0, "commission_per_share": 0.0, "impact_coefficient": 0.1},
    }

    runner = WalkForwardRunner(cfg)
    result = runner.run(bars=bars, universe=universe, output_dir=tmp_path, sector_map=sector_map)
    assert not result.returns.empty
    assert len(result.folds) > 0
    assert "net_return" in result.returns.columns
    # 3 symbols is below the minimum cross-section for meaningful rank ICs,
    # so the factor-stability score must be honestly unmeasurable (None),
    # not the old hard-coded 1.0
    assert result.factor_overlap_score is None
    assert result.sector_pnl_share is not None
    assert abs(sum(result.sector_pnl_share.values()) - 1.0) < 1e-9
    assert float(result.returns["turnover"].max()) <= 1.0
    assert float(result.returns["gross_exposure"].max()) <= 0.91
