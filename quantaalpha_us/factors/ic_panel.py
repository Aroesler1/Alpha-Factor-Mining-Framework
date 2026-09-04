"""Build the per-factor daily IC series that every factor-zoo statistic reads.

`factor_research.score_expressions` answers "what is this factor's mean IC and
t-stat". Almost every question in the zoo work needs the underlying *series*
instead: a HAC t-stat needs the autocovariances, a block bootstrap needs to
resample days, an IC-decay curve needs the same signal scored against several
forward horizons, and a train/holdout split of any of those is a slice of one
series rather than a second scoring pass.

So this module evaluates each expression once and returns the daily IC series
per horizon, plus the two panel-level numbers the evolutionary loop's gates
need (coverage and turnover). One pass over the panel, everything downstream is
arithmetic on small frames.

Two forward-return conventions, because "score it at horizon h" has two
readings and they answer different questions:

- ``cumulative``: open T+1 to open T+1+h. "How much of the next h days' move
  does this factor capture?" This curve usually RISES with h for a slow signal,
  because the return accumulates faster than the signal decays.
- ``lagged``: the single day from open T+h to open T+h+1. "Is the factor still
  informative h days later?" This is the curve that decays, so it is the one a
  half-life can be read off.

They coincide at h = 1, where both reduce to the repo's standard label
(`factor_research.forward_open_returns`); a test asserts that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quantaalpha_us.factors.expression_evaluator import (
    ExpressionError,
    ExpressionEvaluator,
    build_field_panels,
)
from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer
from quantaalpha_us.factors.factor_research import _daily_spearman_ic

# The horizon grid the IC-decay curve is measured on.
HORIZONS: tuple[int, ...] = (1, 5, 10, 21, 63)

CUMULATIVE = "cumulative"
LAGGED = "lagged"


def _adjusted_open(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The tradable open, split-adjusted the same way `forward_open_returns` does."""
    if "open" in panels and "adj_close" in panels and "close" in panels:
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = panels["adj_close"] / panels["close"]
        return panels["open"] * factor
    if "open" in panels:
        return panels["open"]
    return panels["close"]


def forward_returns_at(
    panels: Mapping[str, pd.DataFrame], horizon: int, kind: str = LAGGED
) -> pd.DataFrame:
    """Forward return at `horizon`, aligned to the signal date.

    Both conventions keep the repo's one-day execution lag: nothing is ever
    labelled with a return that starts before the open of T+1.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    op = _adjusted_open(panels)
    if kind == CUMULATIVE:
        return op.shift(-(horizon + 1)) / op.shift(-1) - 1.0
    if kind == LAGGED:
        return op.shift(-(horizon + 1)) / op.shift(-horizon) - 1.0
    raise ValueError(f"unknown forward-return convention: {kind!r}")


def daily_turnover(signal: pd.DataFrame) -> float:
    """Mean one-way daily turnover of the portfolio this signal implies.

    Convention, stated because a turnover number means nothing without one: the
    signal is ranked cross-sectionally to [0, 1], demeaned so the book is
    dollar-neutral, and scaled so gross exposure is 1. Turnover on a day is then
    half the sum of absolute weight changes, i.e. the one-way traded fraction of
    the book. A name that enters or leaves the cross-section is a full trade,
    which is why missing weights are filled with zero rather than skipped.

    Scale check, worth stating because the number is otherwise unanchored: a
    signal whose cross-section is reshuffled independently every day lands at
    2 * E|U1 - U2| = 2/3, a signal that exactly reverses its book every day at
    1, and a signal that never changes its ordering at 0. So 1 is the ceiling
    and 2/3, not 1, is what "no persistence at all" costs.
    """
    ranks = signal.rank(axis=1, pct=True)
    weights = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = weights.abs().sum(axis=1).replace(0, np.nan)
    weights = weights.div(gross, axis=0).fillna(0.0)
    traded = (weights - weights.shift(1)).abs().sum(axis=1)
    return float(0.5 * traded.iloc[1:].mean())


def universe_coverage(signal: pd.DataFrame, reference: pd.DataFrame) -> float:
    """Share of the INVESTABLE cross-section the signal is defined on.

    The denominator matters more than it looks. After the point-in-time
    membership filter the panel is 1,185 columns wide but only ~500 names are
    index members on any given day, so a factor defined on every tradable name
    still covers only ~42% of the panel. Measuring coverage against the panel
    would make a gate of "coverage >= 0.9" unsatisfiable by construction, which
    is exactly what the first calibration run of the evolutionary loop found:
    every candidate rejected, for a reason that was arithmetic rather than
    financial.

    So the denominator is the number of names with a usable price that day.
    Dates on which nothing is available contribute nothing rather than a zero.
    """
    common_dates = signal.index.intersection(reference.index)
    common_names = signal.columns.intersection(reference.columns)
    if len(common_dates) == 0 or len(common_names) == 0:
        return 0.0
    available = reference.loc[common_dates, common_names].notna()
    present = signal.loc[common_dates, common_names].notna() & available
    denominator = available.sum(axis=1)
    ratio = present.sum(axis=1) / denominator.replace(0, np.nan)
    return float(ratio.mean()) if ratio.notna().any() else 0.0


def ic_half_life(ic_by_horizon: Mapping[int, float], horizons: Sequence[int] = HORIZONS) -> float:
    """Horizon at which |IC| first falls to half its one-day value.

    Linear interpolation between the two bracketing grid points. Returns inf
    when the curve never halves inside the grid, which is a real answer for a
    slow factor and must not be silently coerced to the largest horizon: the
    difference between "halves at 60 days" and "has not halved by 63" is the
    difference between a medium and a slow niche.
    """
    grid = [h for h in horizons if h in ic_by_horizon and np.isfinite(ic_by_horizon[h])]
    if not grid or grid[0] != min(horizons):
        return float("nan")
    base = abs(ic_by_horizon[grid[0]])
    if not base > 0:
        return float("nan")
    target = base / 2.0
    prev_h, prev_v = grid[0], base
    for h in grid[1:]:
        v = abs(ic_by_horizon[h])
        if v <= target:
            if prev_v == v:
                return float(h)
            frac = (prev_v - target) / (prev_v - v)
            return float(prev_h + frac * (h - prev_h))
        prev_h, prev_v = h, v
    return float("inf")


# The archive's horizon axis. Stated once, next to the half-life that feeds it,
# because `scripts/sp500_ic_horizon_curve.py` reports the buckets and
# `quantaalpha_us.evo.archive` sorts candidates into them, and two copies of a
# boundary is one copy too many.
FAST_MAX_HALF_LIFE = 5.0
MEDIUM_MAX_HALF_LIFE = 21.0


def horizon_bucket(half_life: float) -> str:
    """"fast", "medium" or "slow" from a half-life in trading days.

    A factor that never halves inside the 63-day grid comes back as `inf` and
    lands in "slow", which is the honest reading: the grid cannot distinguish
    "halves at 80 days" from "does not decay", and both are slow.
    """
    if not np.isfinite(half_life):
        return "slow"
    if half_life < FAST_MAX_HALF_LIFE:
        return "fast"
    if half_life <= MEDIUM_MAX_HALF_LIFE:
        return "medium"
    return "slow"


@dataclass
class ICTable:
    """Daily IC series per (convention, horizon), plus per-factor panel facts."""

    ic: dict[tuple[str, int], pd.DataFrame] = field(default_factory=dict)
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def series(self, expression: str, horizon: int = 1, kind: str = LAGGED) -> pd.Series:
        return self.ic[(kind, horizon)][expression].dropna()

    def window(self, start: str | None, end: str | None, horizon: int = 1,
               kind: str = LAGGED) -> pd.DataFrame:
        """The (dates x factors) IC frame restricted to a date window."""
        frame = self.ic[(kind, horizon)]
        idx = frame.index
        if start is not None:
            idx = idx[idx >= pd.Timestamp(start)]
        if end is not None:
            idx = idx[idx <= pd.Timestamp(end)]
        return frame.loc[idx]

    def complete_case(self, expressions: Sequence[str], start: str | None = None,
                      end: str | None = None, horizon: int = 1,
                      kind: str = LAGGED) -> pd.DataFrame:
        """Rows where EVERY listed factor has a finite IC.

        A maximum-|t| statistic across a pool has to be a maximum over one
        sample. Factors here warm up at different speeds -- a 252-day lookback
        produces nothing for its first year -- so the pool is intersected before
        anything is maximised over it, and the caller reports how many days that
        costs.
        """
        frame = self.window(start, end, horizon=horizon, kind=kind)[list(expressions)]
        return frame.dropna(axis=0, how="any")


def build_ic_table(
    bars: pd.DataFrame,
    expressions: Sequence[str],
    *,
    groups: Mapping[str, str] | None = None,
    horizons: Sequence[int] = HORIZONS,
    kinds: Sequence[str] = (LAGGED, CUMULATIVE),
    min_cross_section: int = 30,
    sanitizer: ExpressionSanitizer | None = None,
    panels: Mapping[str, pd.DataFrame] | None = None,
    stat_dates: pd.DatetimeIndex | None = None,
    verbose: bool = True,
) -> ICTable:
    """Evaluate each expression once and score it at every (convention, horizon).

    Expressions that fail the sanitizer or the evaluator are recorded in `meta`
    with the reason and contribute an all-NaN column, so a caller counting the
    pool sees the same denominator the mining trace does rather than silently
    losing the failures.

    `stat_dates` restricts which dates the coverage and turnover figures are
    measured over, leaving evaluation on the full panel. The evolutionary loop
    needs both numbers on its fit window alone -- a turnover measured partly on
    the holdout is a holdout number leaking into a gate -- while the zoo report
    wants them over the whole sample.
    """
    sanitizer = sanitizer or ExpressionSanitizer()
    panels = panels if panels is not None else build_field_panels(bars)
    evaluator = ExpressionEvaluator(panels)

    keys: list[tuple[str, int]] = []
    fwd: dict[tuple[str, int], pd.DataFrame] = {}
    for kind in kinds:
        for h in horizons:
            # h = 1 is the same panel under both conventions; computing it twice
            # would double the cost of the most-used horizon for nothing.
            if kind == CUMULATIVE and h == min(horizons) and LAGGED in kinds:
                continue
            keys.append((kind, h))
            fwd[(kind, h)] = forward_returns_at(panels, h, kind=kind)

    dates = panels["close"].index
    columns: dict[tuple[str, int], dict[str, pd.Series]] = {k: {} for k in keys}
    rows: list[dict] = []
    started = time.time()

    for i, raw in enumerate(expressions):
        checked = sanitizer.sanitize(raw)
        expr = checked.cleaned
        if not checked.valid:
            rows.append({"expression": expr, "group": (groups or {}).get(raw, ""),
                         "coverage": 0.0, "turnover": np.nan,
                         "error": "; ".join(checked.errors)})
            continue
        try:
            signal = evaluator.evaluate(expr)
        except ExpressionError as exc:
            rows.append({"expression": expr, "group": (groups or {}).get(raw, ""),
                         "coverage": 0.0, "turnover": np.nan, "error": str(exc)})
            continue

        for key in keys:
            columns[key][expr] = _daily_spearman_ic(signal, fwd[key], min_cross_section)
        stat_signal = signal if stat_dates is None else signal.loc[signal.index.isin(stat_dates)]
        rows.append({
            "expression": expr,
            "group": (groups or {}).get(raw, ""),
            # share of the FULL panel, not of the investable universe. Under
            # point-in-time membership this tops out near 0.42, because ~500 of
            # the panel's 1,185 names are members on any date. Use
            # `universe_coverage` for anything that gates on it.
            "coverage": float(stat_signal.notna().mean().mean()),
            "turnover": daily_turnover(stat_signal),
            "error": None,
        })
        if verbose and (i + 1) % 25 == 0:
            done = i + 1
            rate = (time.time() - started) / done
            print(f"  scored {done}/{len(expressions)}  "
                  f"{rate:.2f}s each, ~{rate * (len(expressions) - done) / 60:.1f} min left",
                  flush=True)

    table = ICTable(meta=pd.DataFrame(rows))
    for key in keys:
        table.ic[key] = pd.DataFrame(columns[key], index=dates).sort_index()
    return table


def load_expression_file(path) -> list[str]:
    """Expressions from a candidate file, comments and blank lines dropped."""
    from pathlib import Path

    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def apply_membership_filter(bars: pd.DataFrame, membership) -> pd.DataFrame:
    """Point-in-time S&P 500 membership, joined on (date, permno).

    Same join the scoring CLI and the baseline runner do; factored out here so a
    third caller cannot introduce a fourth slightly different universe.
    """
    from pathlib import Path

    members = pd.read_parquet(Path(membership))
    if "active" in members.columns:
        members = members.loc[members["active"]]
    members = members[["date", "permno"]].dropna().drop_duplicates()
    members = members.assign(
        date=pd.to_datetime(members["date"]),
        permno=members["permno"].astype("int64"),
    )
    out = bars.assign(
        date=pd.to_datetime(bars["date"]),
        permno=bars["permno"].astype("int64"),
    ).merge(members, on=["date", "permno"], how="inner")
    return out


def save_ic_table(table: ICTable, out_dir) -> None:
    """Write the table to a directory: one parquet per (convention, horizon) plus meta."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.meta.to_parquet(out / "meta.parquet", index=False)
    for (kind, h), frame in table.ic.items():
        frame.to_parquet(out / f"ic_{kind}_{h}.parquet")


def load_ic_table(out_dir) -> ICTable:
    from pathlib import Path

    out = Path(out_dir)
    table = ICTable(meta=pd.read_parquet(out / "meta.parquet"))
    for path in sorted(out.glob("ic_*.parquet")):
        _, kind, h = path.stem.split("_")
        table.ic[(kind, int(h))] = pd.read_parquet(path)
    return table
