"""Selection must not see the holdout, and the holdout must not be re-fit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantaalpha_us.factors.factor_research import score_expressions


def _bars(n_days=600, n_syms=40):
    dates = pd.date_range("2015-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(5)
    rows = []
    for i in range(n_syms):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        rows.append(pd.DataFrame({
            "date": dates, "symbol": f"S{i}", "permno": 1000 + i,
            "open": px * 1.001, "high": px * 1.01, "low": px * 0.99,
            "close": px, "adj_close": px,
            "volume": rng.lognormal(12, 0.4, n_days),
        }))
    return pd.concat(rows, ignore_index=True)


def test_ic_dates_restricts_scoring_without_truncating_warmup():
    """A 252-day window must still be computable on day one of the holdout.

    Slicing the bars before evaluation would blank the holdout's first year to
    rolling warm-up; restricting only the IC dates does not.
    """
    bars = _bars()
    dates = pd.to_datetime(bars["date"]).drop_duplicates().sort_values()
    cut = dates[len(dates) // 2]
    test_dates = pd.DatetimeIndex(dates[dates > cut])

    report, _ = score_expressions(bars, ["TS_MEAN($close, 252)"], ic_dates=test_dates)
    score = report.scores[0]
    assert score.error is None, score.error
    # every holdout date should contribute, not just those 252 days past the cut
    assert score.ic_days >= len(test_dates) - 5, (
        f"only {score.ic_days} of {len(test_dates)} holdout days scored; "
        "the window was truncated by warm-up"
    )


def test_selection_window_and_holdout_score_different_periods():
    bars = _bars()
    dates = pd.to_datetime(bars["date"]).drop_duplicates().sort_values()
    cut = dates[len(dates) // 2]
    train = pd.DatetimeIndex(dates[dates <= cut])
    test = pd.DatetimeIndex(dates[dates > cut])

    expr = ["-RANK(TS_DELTA($close, 3))"]
    in_sample, _ = score_expressions(bars, expr, ic_dates=train)
    out_sample, _ = score_expressions(bars, expr, ic_dates=test)
    assert in_sample.scores[0].ic_days > 0 and out_sample.scores[0].ic_days > 0
    assert in_sample.scores[0].ic_days + out_sample.scores[0].ic_days <= len(dates)
    # disjoint windows on random data should not produce identical statistics
    assert in_sample.scores[0].mean_ic != out_sample.scores[0].mean_ic


def test_ic_dates_none_scores_everything():
    bars = _bars()
    full, _ = score_expressions(bars, ["-RANK(TS_DELTA($close, 3))"])
    dates = pd.to_datetime(bars["date"]).drop_duplicates()
    half = pd.DatetimeIndex(sorted(dates)[: len(dates) // 2])
    part, _ = score_expressions(bars, ["-RANK(TS_DELTA($close, 3))"], ic_dates=half)
    assert full.scores[0].ic_days > part.scores[0].ic_days
