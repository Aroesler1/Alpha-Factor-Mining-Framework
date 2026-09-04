"""A synthetic bar panel with a known signal planted in it.

Shared by the IC-panel tests and the evolutionary loop's end-to-end mock run.
Both need the same thing: a panel small enough to score in a second, on which
exactly one factor is genuinely predictive, so a test can assert that the
machinery finds it rather than asserting that the machinery ran.

Construction. Prices are built forward one day at a time. The return credited
to day t+1 is driven by the cross-sectional rank of a 5-day reversal feature
measured at day t, plus noise:

    feature_t = -(close_t / close_{t-5} - 1)
    return_{t+1} = strength * (rank(feature_t) - 0.5) * sigma + noise

Opens are set to the previous close, so the repo's label at date T -- open(T+2)
over open(T+1) -- is exactly return_{T+1}. That makes the planted relationship
land on the same alignment the scoring pipeline uses, rather than one day away
from it, which is the mistake that makes a planted-signal fixture quietly test
nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The factor the panel is built to reward. A test that plants a signal and then
# looks for a DIFFERENT expression is testing luck.
PLANTED_EXPRESSION = "-TS_DELTA($close, 5) / ($close + 1e-8)"
DECOY_EXPRESSIONS = (
    "TS_MEAN($volume, 21)",
    "TS_STD($return, 21)",
    "RANK($high - $low)",
)


def planted_bars(
    *,
    n_days: int = 500,
    n_symbols: int = 50,
    strength: float = 6.0,
    sigma: float = 0.012,
    seed: int = 20260904,
    start: str = "2010-01-04",
) -> pd.DataFrame:
    """Long-format daily bars with a 5-day reversal planted in the returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    close = np.empty((n_days, n_symbols))
    close[0] = 100.0
    rets = np.zeros((n_days, n_symbols))

    for t in range(1, n_days):
        if t > 5:
            feature = -(close[t - 1] / close[t - 6] - 1.0)
            order = feature.argsort().argsort() / max(n_symbols - 1, 1)
            tilt = strength * (order - 0.5) * sigma
        else:
            tilt = np.zeros(n_symbols)
        rets[t] = tilt + rng.normal(0.0, sigma, n_symbols)
        close[t] = close[t - 1] * (1.0 + rets[t])

    open_ = np.vstack([close[0][None, :], close[:-1]])
    high = close * (1.0 + np.abs(rng.normal(0, 0.004, close.shape)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.004, close.shape)))
    volume = rng.lognormal(12.0, 0.45, close.shape)

    frames = []
    for i in range(n_symbols):
        frames.append(pd.DataFrame({
            "date": dates,
            "symbol": f"S{i:03d}",
            "permno": 90000 + i,
            "open": open_[:, i],
            "high": high[:, i],
            "low": low[:, i],
            "close": close[:, i],
            "adj_close": close[:, i],
            "volume": volume[:, i],
            "dollar_volume": close[:, i] * volume[:, i],
        }))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def planted_bars_with_fundamentals(**kwargs) -> pd.DataFrame:
    """`planted_bars` plus slow-moving quarterly fundamental columns.

    The fundamentals carry no signal; they exist so an expression touching them
    is evaluable, which is what the archive's data-family niche needs to be
    non-degenerate in a mock run.
    """
    bars = planted_bars(**kwargs)
    rng = np.random.default_rng(7)
    quarters = bars["date"].dt.to_period("Q").astype(str) + "_" + bars["symbol"]
    codes = pd.factorize(quarters)[0]
    columns = {}
    for name, scale in (("roa", 0.05), ("roe", 0.12), ("operating_margin", 0.15),
                        ("leverage", 1.5), ("asset_turnover", 0.8),
                        ("book_per_share", 20.0), ("earnings_per_share", 3.0),
                        ("accrual_gap", 0.02)):
        draws = rng.normal(scale, abs(scale) * 0.4, codes.max() + 1)
        columns[name] = draws[codes]
    return pd.concat([bars, pd.DataFrame(columns, index=bars.index)], axis=1)
