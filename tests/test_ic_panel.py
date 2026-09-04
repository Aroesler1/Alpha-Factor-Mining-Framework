"""Tests for the shared IC-series builder.

The zoo statistics, the decay curve and the evolutionary loop's fitness all
read these series, so the two things that must not drift are the alignment of
the forward-return conventions and the agreement with the scoring pipeline the
rest of the repo already reports numbers from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantaalpha_us.factors.expression_evaluator import build_field_panels
from quantaalpha_us.factors.factor_research import (
    _daily_spearman_ic,
    forward_open_returns,
    score_expressions,
)
from quantaalpha_us.factors.ic_panel import (
    CUMULATIVE,
    HORIZONS,
    LAGGED,
    ICTable,
    build_ic_table,
    daily_turnover,
    forward_returns_at,
    ic_half_life,
)
from tests.synthetic_panel import DECOY_EXPRESSIONS, PLANTED_EXPRESSION, planted_bars


@pytest.fixture(scope="module")
def bars():
    return planted_bars(n_days=500, n_symbols=50)


@pytest.fixture(scope="module")
def panels(bars):
    return build_field_panels(bars)


# ---- forward-return conventions -----------------------------------------


def test_both_conventions_reduce_to_the_repo_label_at_horizon_one(panels):
    base = forward_open_returns(panels)
    for kind in (LAGGED, CUMULATIVE):
        got = forward_returns_at(panels, 1, kind=kind)
        pd.testing.assert_frame_equal(got, base)


def test_cumulative_compounds_and_lagged_does_not(panels):
    op = panels["open"] * (panels["adj_close"] / panels["close"])
    cum5 = forward_returns_at(panels, 5, kind=CUMULATIVE)
    expected = op.shift(-6) / op.shift(-1) - 1.0
    pd.testing.assert_frame_equal(cum5, expected)

    lag5 = forward_returns_at(panels, 5, kind=LAGGED)
    expected = op.shift(-6) / op.shift(-5) - 1.0
    pd.testing.assert_frame_equal(lag5, expected)


def test_no_convention_looks_back_before_the_execution_lag(panels):
    """Every label must start at the open of T+1 or later. A convention that
    accidentally starts at T is a one-day lookahead worth several IC points."""
    op = panels["open"] * (panels["adj_close"] / panels["close"])
    for kind in (LAGGED, CUMULATIVE):
        for h in HORIZONS:
            fwd = forward_returns_at(panels, h, kind=kind)
            # the last (h + 1) rows must be unusable: their label reaches past
            # the end of the sample
            assert fwd.iloc[-(h + 1):].notna().sum().sum() == 0


def test_horizon_must_be_at_least_one(panels):
    with pytest.raises(ValueError):
        forward_returns_at(panels, 0)
    with pytest.raises(ValueError):
        forward_returns_at(panels, 1, kind="whatever")


# ---- turnover ------------------------------------------------------------


def test_turnover_of_a_never_changing_signal_is_zero(panels):
    static = pd.DataFrame(
        np.tile(np.arange(panels["close"].shape[1], dtype=float), (panels["close"].shape[0], 1)),
        index=panels["close"].index, columns=panels["close"].columns,
    )
    assert daily_turnover(static) == pytest.approx(0.0, abs=1e-12)


def test_turnover_of_an_independently_reshuffled_signal_is_two_thirds(panels):
    """Closed form: unit-gross rank weights are w = 4(u - 1/2)/n, so the daily
    traded fraction is 2 * E|U1 - U2| = 2/3 when consecutive days are
    independent. Pinning it stops the normalisation drifting unnoticed."""
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.standard_normal(panels["close"].shape),
                         index=panels["close"].index, columns=panels["close"].columns)
    assert daily_turnover(noise) == pytest.approx(2 / 3, abs=0.02)


def test_turnover_of_a_perfectly_reversing_signal_is_one(panels):
    shape = panels["close"].shape
    base = np.tile(np.arange(shape[1], dtype=float), (shape[0], 1))
    flip = np.where((np.arange(shape[0]) % 2)[:, None] == 0, base, base[:, ::-1])
    signal = pd.DataFrame(flip, index=panels["close"].index, columns=panels["close"].columns)
    assert daily_turnover(signal) == pytest.approx(1.0, abs=1e-9)


def test_turnover_orders_a_slow_signal_below_a_fast_one(panels):
    slow = panels["close"].rolling(252, min_periods=252).mean()
    fast = panels["close"].pct_change(fill_method=None)
    assert daily_turnover(slow) < daily_turnover(fast)


# ---- half-life -----------------------------------------------------------


def test_half_life_interpolates_between_grid_points():
    # 0.10 at h=1, 0.04 at h=5: crosses 0.05 five sixths of the way across
    curve = {1: 0.10, 5: 0.04, 10: 0.01, 21: 0.0, 63: 0.0}
    assert ic_half_life(curve) == pytest.approx(1 + (0.10 - 0.05) / (0.10 - 0.04) * 4)


def test_half_life_is_infinite_when_the_curve_never_halves():
    assert ic_half_life({1: 0.05, 5: 0.05, 10: 0.05, 21: 0.05, 63: 0.05}) == float("inf")


def test_half_life_uses_absolute_ic_so_a_negative_factor_is_not_special():
    positive = ic_half_life({1: 0.10, 5: 0.04, 10: 0.01, 21: 0.0, 63: 0.0})
    negative = ic_half_life({1: -0.10, 5: -0.04, 10: -0.01, 21: 0.0, 63: 0.0})
    assert positive == pytest.approx(negative)


def test_half_life_is_nan_without_a_usable_base():
    assert np.isnan(ic_half_life({1: 0.0, 5: 0.0}))
    assert np.isnan(ic_half_life({5: 0.1, 10: 0.02}))


# ---- the table -----------------------------------------------------------


def test_ic_at_horizon_one_matches_the_scoring_pipeline(bars):
    """The number this module reports must be the number the rest of the repo
    already publishes, or the zoo hurdle is measured on a different statistic
    than the README's IC table."""
    exprs = [PLANTED_EXPRESSION, *DECOY_EXPRESSIONS]
    table = build_ic_table(bars, exprs, horizons=(1,), kinds=(LAGGED,), verbose=False)
    report, _ = score_expressions(bars, exprs)
    for score in report.scores:
        got = table.series(score.expression, horizon=1, kind=LAGGED)
        assert got.mean() == pytest.approx(score.mean_ic, rel=1e-10)
        assert len(got) == score.ic_days


def test_the_planted_factor_is_the_strongest_in_the_panel(bars):
    table = build_ic_table(bars, [PLANTED_EXPRESSION, *DECOY_EXPRESSIONS],
                           horizons=(1,), kinds=(LAGGED,), verbose=False)
    means = table.ic[(LAGGED, 1)].mean().abs()
    assert means.idxmax() == PLANTED_EXPRESSION
    assert means.max() > 0.05


def test_the_planted_signal_decays_with_the_lagged_convention(bars):
    table = build_ic_table(bars, [PLANTED_EXPRESSION], horizons=HORIZONS,
                           kinds=(LAGGED,), verbose=False)
    curve = {h: table.ic[(LAGGED, h)][PLANTED_EXPRESSION].mean() for h in HORIZONS}
    assert abs(curve[1]) > abs(curve[21])
    assert np.isfinite(ic_half_life(curve))


def test_failures_are_recorded_with_a_reason_not_dropped(bars):
    table = build_ic_table(bars, [PLANTED_EXPRESSION, "TS_MEAN(close, 21)",
                                  "NOPE($close)", "TS_MEAN($close)"],
                           horizons=(1,), kinds=(LAGGED,), verbose=False)
    assert len(table.meta) == 4
    failed = table.meta[table.meta["error"].notna()]
    assert len(failed) == 3
    assert any("must be written as" in e for e in failed["error"])
    assert any("Unknown function" in e for e in failed["error"])
    assert any("expects 2 argument" in e for e in failed["error"])


def test_horizon_one_is_computed_once_under_both_conventions(bars):
    table = build_ic_table(bars, [PLANTED_EXPRESSION], horizons=(1, 5),
                           kinds=(LAGGED, CUMULATIVE), verbose=False)
    assert (CUMULATIVE, 1) not in table.ic
    assert (LAGGED, 1) in table.ic and (CUMULATIVE, 5) in table.ic


def test_complete_case_intersects_the_pool_and_costs_the_slowest_warmup(bars):
    exprs = ["TS_MEAN($close, 5)", "TS_MEAN($close, 252)"]
    table = build_ic_table(bars, exprs, horizons=(1,), kinds=(LAGGED,), verbose=False)
    both = table.complete_case(exprs)
    fast_only = table.complete_case(exprs[:1])
    assert len(both) < len(fast_only)
    assert both.notna().all().all()


def test_window_slices_by_date_without_rescoring(bars):
    table = build_ic_table(bars, [PLANTED_EXPRESSION], horizons=(1,), kinds=(LAGGED,),
                           verbose=False)
    full = table.window(None, None)
    early = table.window(None, "2011-01-01")
    assert len(early) < len(full)
    assert early.index.max() <= pd.Timestamp("2011-01-01")


def test_coverage_and_turnover_land_in_meta(bars):
    table = build_ic_table(bars, [PLANTED_EXPRESSION], horizons=(1,), kinds=(LAGGED,),
                           verbose=False)
    row = table.meta.iloc[0]
    assert 0.9 < row["coverage"] <= 1.0
    assert 0.0 < row["turnover"] < 1.0
