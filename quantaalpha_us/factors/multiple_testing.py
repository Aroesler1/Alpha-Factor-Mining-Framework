"""Multiple-testing corrections for a factor zoo, and the HAC t-stat they act on.

The repo already knows that "the best of 54 candidates has |t| = 11.6" is not
evidence: `sp500_run_baseline_comparison.py` shows random expressions clearing
the same bar. This module is the next question -- *given* the whole pool, which
individual factors survive a correction that accounts for how many were tried?

Four corrections, deliberately not one:

- **Newey-West t** (Bartlett kernel, 21 lags) replaces the iid t the scoring
  pipeline reports. Daily cross-sectional ICs can be autocorrelated -- a 21-day
  momentum factor's IC today shares 20 of its 21 input days with yesterday's --
  and an iid standard error treats those as independent draws. MEASURED ON THIS
  POOL the correction turns out to be small: the median |iid t| / |NW t| is
  1.03 and the pool maximum moves from 10.88 to 10.27. That is worth stating
  plainly, because it was not the expected answer; the inflation in these
  t-statistics is multiplicity, not autocorrelation.
- **Harvey, Liu and Zhu (2016)**: a flat |t| > 3.0 hurdle. Crude, but it is the
  number the cross-sectional literature actually adopted, so it is the one a
  reader will compare against.
- **Benjamini-Hochberg** at 5%: controls the false discovery rate across the
  pool. More powerful than a family-wise method, and the right target when the
  question is "how many of these are real", not "is any of them real".
- **Romano-Wolf stepdown**: family-wise error, bootstrapped, so it accounts for
  the pool's cross-correlation instead of assuming independence the way a
  Bonferroni-style bound does. Two factors that are the same idea should not
  cost the same as two independent tests, and here they do not.

Plus the empirical null the brief actually asks for: the distribution of the
MAXIMUM |t| over the random-grammar pool under a circular block bootstrap. That
is the honest bar. It is not a correction applied to a factor; it is the answer
to "how large a |t| does a search of this size and this autocorrelation produce
when nothing is there".

Everything is numpy. No scipy: the normal CDF is `math.erf`, the quantiles are
`np.quantile`, and the bootstrap is a gather plus a dot product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Harvey, Liu and Zhu (2016), "... and the Cross-Section of Expected Returns",
# Review of Financial Studies 29(1), 5-68. Their recommended hurdle for a newly
# proposed factor, once the ~300 published candidates are counted as the search.
HLZ_TSTAT_HURDLE = 3.0

# Bartlett-kernel lag. 21 trading days is one month, which is the longest
# lookback the bulk of these expressions use and therefore the horizon over
# which two consecutive daily ICs share most of their inputs.
DEFAULT_NW_LAG = 21

# Block length for the bootstrap, matched to the same 21 days.
DEFAULT_BLOCK = 21


def normal_sf(x: float | np.ndarray) -> np.ndarray:
    """Two-sided tail probability of a standard normal at |x|."""
    z = np.abs(np.asarray(x, dtype=float))
    return np.array([math.erfc(v / math.sqrt(2.0)) for v in np.atleast_1d(z)]).reshape(z.shape)


def newey_west_tstats(x: np.ndarray, lag: int = DEFAULT_NW_LAG) -> np.ndarray:
    """HAC t-statistic of the column means of `x` (dates x series), Bartlett kernel.

    var = gamma_0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * gamma_k, divided by n.

    Columns must be complete (no NaN): the caller restricts the pool to a common
    date index first, because a max-|t| statistic over columns measured on
    different date sets is not a maximum over anything.
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    if n <= lag + 1:
        return np.full(X.shape[1], np.nan)
    mean = X.mean(axis=0)
    Xc = X - mean
    var = (Xc * Xc).sum(axis=0) / n
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        var += 2.0 * weight * (Xc[:-k] * Xc[k:]).sum(axis=0) / n
    # The Bartlett kernel makes the estimator positive semi-definite, so a
    # non-positive variance means a degenerate (constant) column, not a bug.
    var = np.where(var > 0, var, np.nan)
    return mean / np.sqrt(var / n)


def newey_west_tstat(series: np.ndarray, lag: int = DEFAULT_NW_LAG) -> float:
    """Scalar convenience wrapper for one series."""
    out = newey_west_tstats(np.asarray(series, dtype=float)[:, None], lag=lag)
    return float(out[0])


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One circular-block-bootstrap resample of `n` date positions.

    Circular (Politis and Romano, 1992) rather than a plain moving block: with a
    moving block the first and last `block - 1` observations can only appear in
    fewer blocks than the interior ones, so the resample quietly under-weights
    the ends of the sample. Wrapping removes that without changing the block
    length, which is the thing that has to match the dependence being preserved.
    """
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block)
    idx = (starts[:, None] + offsets[None, :]).reshape(-1) % n
    return idx[:n]


@dataclass
class BootstrapResult:
    """Bootstrapped t-statistics under the null that every column has zero mean."""

    t_boot: np.ndarray        # (draws, n_series), centred at the sample estimate
    block: int
    draws: int
    lag: int

    def max_abs_t(self, columns: np.ndarray | slice | None = None) -> np.ndarray:
        sub = self.t_boot if columns is None else self.t_boot[:, columns]
        return np.nanmax(np.abs(sub), axis=1)


def bootstrap_null_tstats(
    ic: np.ndarray,
    *,
    block: int = DEFAULT_BLOCK,
    draws: int = 2000,
    lag: int = DEFAULT_NW_LAG,
    seed: int = 0,
) -> BootstrapResult:
    """Block-bootstrap the HAC t of every column under a zero-mean null.

    `ic` is (dates x factors) with no missing values. Each column is demeaned
    ONCE before resampling, which is what makes the resulting t-statistics both
    (a) draws from the null for the max-|t| hurdle and (b) the centred,
    studentised statistics Romano-Wolf's stepdown needs. The same block indices
    are applied to every column in a draw, so the pool's cross-correlation --
    the whole reason a family-wise correction can beat Bonferroni here -- is
    carried through rather than assumed away.
    """
    X = np.asarray(ic, dtype=float)
    if np.isnan(X).any():
        raise ValueError("bootstrap_null_tstats needs a complete (dates x factors) matrix")
    centred = X - X.mean(axis=0)
    n = centred.shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((draws, centred.shape[1]), dtype=float)
    for b in range(draws):
        idx = circular_block_indices(n, block, rng)
        out[b] = newey_west_tstats(centred[idx], lag=lag)
    return BootstrapResult(t_boot=out, block=block, draws=draws, lag=lag)


def block_bootstrap_se(
    series: np.ndarray,
    *,
    block: int = DEFAULT_BLOCK,
    draws: int = 200,
    seed: int = 0,
) -> float:
    """Standard error of a mean under a circular block bootstrap.

    The evolutionary loop's fitness divides by this rather than by the Bartlett
    HAC standard error. Both correct for the same thing -- daily ICs are not
    independent draws -- and on these series they agree closely; the bootstrap
    is preferred inside the loop because it is the same resampling scheme the
    zoo hurdle uses, so a fitness number and a hurdle number are on one scale.

    Seeded, so a candidate's fitness is a function of the candidate and nothing
    else. That is what makes a replay reproduce an archive exactly.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= block:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = x[circular_block_indices(n, block, rng)].mean()
    return float(means.std(ddof=1))


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, float]:
    """BH step-up. Returns (rejected mask, the p-value threshold actually used).

    Controls the false discovery rate at `alpha` under positive dependence,
    which daily IC series across a factor pool plausibly have (the pool is full
    of momentum restatements that co-move). Returns a threshold of 0.0 when
    nothing is rejected, so a caller can report the bar even when it was
    cleared by nobody.
    """
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    m = int(finite.sum())
    rejected = np.zeros(p.shape, dtype=bool)
    if m == 0:
        return rejected, 0.0
    order = np.argsort(np.where(finite, p, np.inf))[:m]
    ranked = p[order]
    thresholds = alpha * np.arange(1, m + 1) / m
    passing = np.nonzero(ranked <= thresholds)[0]
    if len(passing) == 0:
        return rejected, 0.0
    k = int(passing[-1])
    rejected[order[: k + 1]] = True
    return rejected, float(ranked[k])


def romano_wolf(
    t_obs: np.ndarray,
    t_boot: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Romano-Wolf (2005) stepdown on |t|. Returns (rejected mask, critical value used).

    Step 1 takes the (1 - alpha) quantile of the bootstrap max |t| over the
    whole pool and rejects everything above it. Each later step recomputes that
    maximum over only the factors NOT yet rejected, which lowers the bar as the
    obvious winners leave the family. The result is family-wise error control
    that is strictly more powerful than Bonferroni and, unlike Bonferroni, gets
    cheaper the more correlated the pool is.

    The per-factor critical value is recorded so the report can show which step
    each survivor was rejected at, rather than only that it survived.
    """
    t = np.abs(np.asarray(t_obs, dtype=float))
    boot = np.abs(np.asarray(t_boot, dtype=float))
    n = t.shape[0]
    rejected = np.zeros(n, dtype=bool)
    critical = np.full(n, np.nan)
    active = np.isfinite(t)
    while active.any():
        maxima = np.nanmax(boot[:, active], axis=1)
        crit = float(np.quantile(maxima, 1.0 - alpha))
        newly = active & (t > crit)
        if not newly.any():
            critical[active] = crit
            break
        rejected |= newly
        critical[newly] = crit
        active = active & ~newly
    return rejected, critical


@dataclass
class PoolVerdict:
    """Per-factor outcome of every correction, in one place."""

    nw_tstat: np.ndarray
    p_value: np.ndarray
    passes_hlz: np.ndarray
    passes_bh: np.ndarray
    passes_romano_wolf: np.ndarray
    passes_random_max: np.ndarray
    bh_threshold: float
    romano_wolf_critical: np.ndarray
    random_max_critical: float


def evaluate_pool(
    ic: np.ndarray,
    *,
    random_columns: np.ndarray,
    block: int = DEFAULT_BLOCK,
    draws: int = 2000,
    lag: int = DEFAULT_NW_LAG,
    alpha: float = 0.05,
    seed: int = 0,
) -> PoolVerdict:
    """Run all four hurdles over one (dates x factors) IC matrix.

    `random_columns` is a boolean mask marking the random-grammar members of the
    pool. Their bootstrapped maximum |t| is the empirical null: a factor clears
    it only by beating what a same-sized search over the same grammar produces
    from nothing. One bootstrap serves both that hurdle and Romano-Wolf, so the
    two are measured on the same resamples rather than on two runs that differ
    by their seed.
    """
    t_obs = newey_west_tstats(ic, lag=lag)
    p = normal_sf(t_obs)
    boot = bootstrap_null_tstats(ic, block=block, draws=draws, lag=lag, seed=seed)

    random_max = boot.max_abs_t(columns=np.asarray(random_columns, dtype=bool))
    random_crit = float(np.quantile(random_max, 1.0 - alpha))

    rejected_bh, bh_thresh = benjamini_hochberg(p, alpha=alpha)
    rejected_rw, rw_crit = romano_wolf(t_obs, boot.t_boot, alpha=alpha)

    return PoolVerdict(
        nw_tstat=t_obs,
        p_value=p,
        passes_hlz=np.abs(t_obs) > HLZ_TSTAT_HURDLE,
        passes_bh=rejected_bh,
        passes_romano_wolf=rejected_rw,
        passes_random_max=np.abs(t_obs) > random_crit,
        bh_threshold=bh_thresh,
        romano_wolf_critical=rw_crit,
        random_max_critical=random_crit,
    )
