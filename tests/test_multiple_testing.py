"""Tests for the factor-zoo corrections.

These decide which factors the README calls survivors, so each hurdle is
checked against a case where the right answer is known independently: a HAC
t-stat against a hand-computed Bartlett sum, BH against the textbook worked
example, Romano-Wolf against its own definition at step 1, and the whole set
against pure-noise input, where the false-positive rate is the specification.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantaalpha_us.factors.multiple_testing import (
    DEFAULT_BLOCK,
    HLZ_TSTAT_HURDLE,
    benjamini_hochberg,
    bootstrap_null_tstats,
    circular_block_indices,
    evaluate_pool,
    newey_west_tstat,
    newey_west_tstats,
    normal_sf,
    romano_wolf,
)


# ---- Newey-West ----------------------------------------------------------


def test_newey_west_at_lag_zero_is_the_iid_tstat():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    iid = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_tstat(x, lag=0) == pytest.approx(iid, rel=1e-12)


def test_newey_west_matches_a_hand_computed_bartlett_sum():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    lag = 5
    n = len(x)
    xc = x - x.mean()
    var = (xc * xc).sum() / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (xc[:-k] * xc[k:]).sum() / n
    expected = x.mean() / np.sqrt(var / n)
    assert newey_west_tstat(x, lag=lag) == pytest.approx(expected, rel=1e-12)


def test_newey_west_shrinks_the_tstat_of_a_positively_autocorrelated_series():
    """The whole reason this module exists: an iid standard error on an
    autocorrelated IC series overstates significance."""
    rng = np.random.default_rng(2)
    n = 4000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + e[t]
    x = x + 0.5
    iid = newey_west_tstat(x, lag=0)
    hac = newey_west_tstat(x, lag=21)
    assert abs(hac) < abs(iid)
    # AR(1) at rho = 0.9 inflates the iid t by roughly sqrt((1+rho)/(1-rho)) = 4.4
    assert abs(iid) / abs(hac) > 2.0


def test_newey_west_is_vectorised_consistently_across_columns():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 7))
    got = newey_west_tstats(X, lag=4)
    for j in range(X.shape[1]):
        assert got[j] == pytest.approx(newey_west_tstat(X[:, j], lag=4), rel=1e-12)


def test_newey_west_returns_nan_for_a_constant_column():
    assert np.isnan(newey_west_tstats(np.ones((100, 1)), lag=3)[0])


def test_newey_west_returns_nan_when_the_sample_is_shorter_than_the_lag():
    assert np.isnan(newey_west_tstats(np.zeros((5, 2)), lag=21)).all()


# ---- block bootstrap -----------------------------------------------------


def test_circular_block_indices_returns_exactly_n_positions_in_range():
    rng = np.random.default_rng(4)
    idx = circular_block_indices(1000, DEFAULT_BLOCK, rng)
    assert idx.shape == (1000,)
    assert idx.min() >= 0 and idx.max() < 1000


def test_circular_blocks_are_contiguous_modulo_wraparound():
    rng = np.random.default_rng(5)
    n, block = 50, 10
    idx = circular_block_indices(n, block, rng)
    for start in range(0, n, block):
        chunk = idx[start:start + block]
        expected = (chunk[0] + np.arange(len(chunk))) % n
        assert np.array_equal(chunk, expected)


def test_circular_bootstrap_weights_every_date_equally_in_expectation():
    """A plain moving block would under-sample the first and last block-1
    dates; the wrap is what removes that."""
    rng = np.random.default_rng(6)
    n, block = 40, 8
    counts = np.zeros(n)
    for _ in range(4000):
        idx = circular_block_indices(n, block, rng)
        counts += np.bincount(idx, minlength=n)
    share = counts / counts.sum()
    assert share.max() / share.min() < 1.25


def test_bootstrap_is_seeded():
    X = np.random.default_rng(7).standard_normal((300, 4))
    a = bootstrap_null_tstats(X, draws=25, seed=11)
    b = bootstrap_null_tstats(X, draws=25, seed=11)
    c = bootstrap_null_tstats(X, draws=25, seed=12)
    assert np.array_equal(a.t_boot, b.t_boot)
    assert not np.array_equal(a.t_boot, c.t_boot)


def test_bootstrap_is_centred_so_a_strong_signal_does_not_move_the_null():
    """The columns are demeaned before resampling, so a column with a huge mean
    contributes a null distribution centred on zero, not on its own t."""
    rng = np.random.default_rng(8)
    X = rng.standard_normal((600, 3))
    X[:, 0] += 5.0
    boot = bootstrap_null_tstats(X, draws=200, seed=1)
    assert abs(np.median(boot.t_boot[:, 0])) < 0.5


def test_bootstrap_rejects_missing_values_rather_than_silently_dropping_them():
    X = np.zeros((100, 2))
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        bootstrap_null_tstats(X, draws=2)


# ---- Benjamini-Hochberg --------------------------------------------------


def test_benjamini_hochberg_worked_example():
    # m = 10, alpha = 0.05, so the step-up thresholds are 0.005, 0.010, ... 0.050.
    # Sorted p against its own threshold:
    #   0.0001<=0.005 0.0004<=0.010 0.0019<=0.015 0.0095<=0.020 0.0201<=0.025
    #   0.0278<=0.030 0.0298<=0.035 0.0344<=0.040 | 0.0459>0.045  0.3240>0.050
    # The largest passing rank is 8, so all eight smallest are rejected.
    p = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201,
                  0.0278, 0.0298, 0.0344, 0.0459, 0.3240])
    rejected, thresh = benjamini_hochberg(p, alpha=0.05)
    assert rejected.sum() == 8
    assert rejected[:8].all() and not rejected[8:].any()
    assert thresh == pytest.approx(0.0344)


def test_benjamini_hochberg_is_step_up_not_step_down():
    """A p-value above its own threshold is still rejected when a smaller-ranked
    one clears; a step-down rule would stop at the first failure."""
    p = np.array([0.001, 0.04, 0.045])
    rejected, _ = benjamini_hochberg(p, alpha=0.05)
    assert rejected.tolist() == [True, True, True]


def test_benjamini_hochberg_rejects_nothing_when_nothing_is_significant():
    rejected, thresh = benjamini_hochberg(np.full(20, 0.6), alpha=0.05)
    assert not rejected.any()
    assert thresh == 0.0


def test_benjamini_hochberg_ignores_missing_pvalues():
    p = np.array([0.001, np.nan, 0.9])
    rejected, _ = benjamini_hochberg(p, alpha=0.05)
    assert rejected.tolist() == [True, False, False]


def test_normal_sf_is_two_sided():
    assert normal_sf(1.959963985) == pytest.approx(0.05, abs=1e-6)
    assert normal_sf(0.0) == pytest.approx(1.0)


# ---- Romano-Wolf ---------------------------------------------------------


def test_romano_wolf_first_step_uses_the_pool_wide_bootstrap_quantile():
    rng = np.random.default_rng(9)
    boot = rng.standard_normal((2000, 5))
    t_obs = np.array([10.0, 0.1, 0.2, 0.3, 0.4])
    rejected, crit = romano_wolf(t_obs, boot, alpha=0.05)
    expected = np.quantile(np.abs(boot).max(axis=1), 0.95)
    assert rejected.tolist() == [True, False, False, False, False]
    assert crit[0] == pytest.approx(expected)


def test_romano_wolf_steps_down_and_the_bar_falls():
    """After the biggest factor leaves the family the maximum is taken over a
    smaller set, so the critical value can only fall -- that is the whole point
    of the stepdown, and it is what makes it beat Bonferroni."""
    rng = np.random.default_rng(10)
    boot = rng.standard_normal((4000, 6))
    boot[:, 0] *= 4.0  # one very volatile column dominates the pool maximum
    t_obs = np.array([20.0, 3.4, 0.1, 0.1, 0.1, 0.1])
    rejected, crit = romano_wolf(t_obs, boot, alpha=0.05)
    assert rejected[0]
    assert crit[1] < crit[0]


def test_romano_wolf_rejects_nothing_under_the_null():
    rng = np.random.default_rng(11)
    boot = rng.standard_normal((2000, 8))
    t_obs = rng.standard_normal(8)
    rejected, _ = romano_wolf(t_obs, boot, alpha=0.05)
    assert not rejected.any()


# ---- the whole pool ------------------------------------------------------


def test_evaluate_pool_finds_nothing_in_pure_noise():
    """Specification test: 60 independent noise columns, 20 of them labelled
    'random grammar'. Every hurdle should come back close to empty."""
    rng = np.random.default_rng(12)
    ic = rng.standard_normal((1500, 60)) * 0.02
    is_random = np.zeros(60, dtype=bool)
    is_random[:20] = True
    verdict = evaluate_pool(ic, random_columns=is_random, draws=300, seed=3)
    assert verdict.passes_romano_wolf.sum() == 0
    assert verdict.passes_bh.sum() <= 3
    assert verdict.passes_random_max.sum() <= 6


def test_evaluate_pool_finds_a_planted_signal():
    rng = np.random.default_rng(13)
    ic = rng.standard_normal((1500, 40)) * 0.02
    ic[:, 39] += 0.02  # a genuine, large mean IC
    is_random = np.zeros(40, dtype=bool)
    is_random[:20] = True
    verdict = evaluate_pool(ic, random_columns=is_random, draws=300, seed=4)
    assert verdict.passes_romano_wolf[39]
    assert verdict.passes_bh[39]
    assert verdict.passes_random_max[39]
    assert verdict.passes_hlz[39]
    assert abs(verdict.nw_tstat[39]) > HLZ_TSTAT_HURDLE


def test_evaluate_pool_hurdles_are_ordered_as_expected_in_strength():
    """BH controls FDR and Romano-Wolf controls FWER, so on the same pool the
    FDR rule can never reject strictly fewer -- if it does, one of them is
    wired up wrong."""
    rng = np.random.default_rng(14)
    ic = rng.standard_normal((1200, 30)) * 0.02
    ic[:, 25:] += 0.015
    is_random = np.zeros(30, dtype=bool)
    is_random[:20] = True
    verdict = evaluate_pool(ic, random_columns=is_random, draws=300, seed=5)
    assert verdict.passes_bh.sum() >= verdict.passes_romano_wolf.sum()
