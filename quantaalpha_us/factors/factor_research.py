"""Research scoring for mined factor expressions.

Bridges mining and the backtest: sanitized expressions are evaluated into
signal panels (expression_evaluator), scored against forward returns under
the repo's execution convention (signal at close T, enter open T+1, exit
open T+2), then de-duplicated by signal correlation so the surviving set is
not the same idea rented out under five names.

Outputs are deliberately conservative: daily cross-sectional Spearman ICs
with a t-statistic across days, plus a signal-autocorrelation turnover
proxy. High mean IC with a low t-stat or low autocorrelation is flagged by
the caller's thresholds, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quantaalpha_us.factors.expression_evaluator import (
    ExpressionError,
    ExpressionEvaluator,
    build_field_panels,
)
from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer


@dataclass
class FactorScore:
    expression: str
    mean_ic: float
    ic_tstat: float
    ic_days: int
    signal_autocorr: float
    coverage: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "mean_ic": self.mean_ic,
            "ic_tstat": self.ic_tstat,
            "ic_days": self.ic_days,
            "signal_autocorr": self.signal_autocorr,
            "coverage": self.coverage,
            "error": self.error,
        }


@dataclass
class ResearchReport:
    scores: list[FactorScore] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self.scores])


def forward_open_returns(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Forward return aligned to the signal date under the repo convention.

    Signal computed after close of day T; position entered at open T+1 and
    exited at open T+2, so the label at row T is adj_open(T+2)/adj_open(T+1)-1.
    Raw opens are scaled by adj_close/close so splits do not create fake
    returns.
    """
    if "open" in panels and "adj_close" in panels and "close" in panels:
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = panels["adj_close"] / panels["close"]
        adj_open = panels["open"] * factor
    elif "open" in panels:
        adj_open = panels["open"]
    else:
        # fall back to close-to-close when no opens exist
        adj_open = panels["close"]
    return adj_open.shift(-2) / adj_open.shift(-1) - 1.0


def _daily_spearman_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_cross_section: int) -> pd.Series:
    common_cols = signal.columns.intersection(fwd.columns)
    common_idx = signal.index.intersection(fwd.index)
    s = signal.loc[common_idx, common_cols]
    f = fwd.loc[common_idx, common_cols]

    valid = s.notna() & f.notna()
    n_valid = valid.sum(axis=1)

    s_rank = s.where(valid).rank(axis=1)
    f_rank = f.where(valid).rank(axis=1)

    s_c = s_rank.sub(s_rank.mean(axis=1), axis=0)
    f_c = f_rank.sub(f_rank.mean(axis=1), axis=0)
    cov = (s_c * f_c).sum(axis=1)
    denom = np.sqrt((s_c**2).sum(axis=1) * (f_c**2).sum(axis=1))
    ic = cov / denom.replace(0, np.nan)
    return ic[n_valid >= min_cross_section].dropna()


def _signal_autocorr(signal: pd.DataFrame) -> float:
    """Mean per-symbol lag-1 autocorrelation; a fast-turnover red flag when low."""
    corr = signal.corrwith(signal.shift(1), axis=0)
    return float(corr.mean()) if len(corr.dropna()) else float("nan")


def score_expressions(
    bars: pd.DataFrame,
    expressions: Sequence[str],
    *,
    min_cross_section: int = 30,
    sanitizer: Optional[ExpressionSanitizer] = None,
) -> tuple[ResearchReport, dict[str, pd.DataFrame]]:
    """Sanitize, evaluate, and score expressions. Returns (report, signals)."""
    sanitizer = sanitizer or ExpressionSanitizer()
    panels = build_field_panels(bars)
    evaluator = ExpressionEvaluator(panels)
    fwd = forward_open_returns(panels)

    report = ResearchReport()
    signals: dict[str, pd.DataFrame] = {}
    for raw_expr in expressions:
        result = sanitizer.sanitize(raw_expr)
        if not result.valid:
            report.scores.append(FactorScore(raw_expr, np.nan, np.nan, 0, np.nan, 0.0,
                                             error="; ".join(result.errors)))
            continue
        expr = result.cleaned
        try:
            signal = evaluator.evaluate(expr)
        except ExpressionError as exc:
            report.scores.append(FactorScore(expr, np.nan, np.nan, 0, np.nan, 0.0, error=str(exc)))
            continue

        ic = _daily_spearman_ic(signal, fwd, min_cross_section)
        n = len(ic)
        mean_ic = float(ic.mean()) if n else np.nan
        ic_std = float(ic.std(ddof=1)) if n > 1 else np.nan
        tstat = mean_ic / ic_std * np.sqrt(n) if n > 1 and ic_std and ic_std > 0 else np.nan
        coverage = float(signal.notna().mean().mean())

        report.scores.append(FactorScore(expr, mean_ic, tstat, n, _signal_autocorr(signal), coverage))
        signals[expr] = signal
    return report, signals


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank each date's cross-section to [0, 1], leaving missing values missing."""
    return panel.rank(axis=1, pct=True)


def select_uncorrelated(
    report: ResearchReport,
    signals: dict[str, pd.DataFrame],
    *,
    min_abs_tstat: float = 2.0,
    max_abs_corr: float = 0.7,
    max_factors: int = 10,
    min_autocorr: float = 0.2,
) -> list[str]:
    """Greedy selection by |IC t-stat| with a pairwise signal-correlation cap.

    String-level dedup upstream cannot catch semantic duplicates (an LLM
    happily restates one idea five ways); correlation of the realized signal
    panels can.

    Two guards that matter once this runs on real data:

    - Selection ranks on |IC| but the backtest averages mined factors into a
      score where higher means better, so a negative-IC factor would enter
      backwards and actively degrade the composite. Emitted expressions are
      sign-corrected: a factor with mean IC < 0 is emitted negated.
    - `min_autocorr` drops signals that barely persist day to day. A lag-1
      signal autocorrelation near zero means near-total daily turnover, which
      transaction costs destroy regardless of how significant the IC looks.
    """
    candidates = [
        s for s in report.scores
        if s.error is None
        and np.isfinite(s.ic_tstat)
        and abs(s.ic_tstat) >= min_abs_tstat
        and not (np.isfinite(s.signal_autocorr) and s.signal_autocorr < min_autocorr)
    ]
    candidates.sort(key=lambda s: abs(s.ic_tstat), reverse=True)

    selected: list[str] = []
    # correlation is compared on the ORIGINAL signals (negation cannot change
    # |corr|), while `selected` carries the sign-corrected expressions we emit
    kept_originals: list[str] = []
    for cand in candidates:
        if len(selected) >= max_factors:
            break
        sig = signals[cand.expression]
        redundant = False
        for kept in kept_originals:
            other = signals[kept]
            common = sig.columns.intersection(other.columns)
            # Compare in RANK space, per date, because the score these factors
            # are ranked by is a daily cross-sectional rank IC. Pooled Pearson
            # on raw values is not invariant to a monotone cross-sectional
            # transform, so CS_RANK(X) and X -- the same ordering, hence the
            # same IC to four decimals -- measured 0.21 and both entered the
            # final set as "uncorrelated". In rank space that pair is 1.00.
            # pct=True normalises for a cross-section whose width changes daily.
            a = _cross_sectional_ranks(sig[common])
            b = _cross_sectional_ranks(other.reindex(sig.index)[common])
            a_flat = a.to_numpy(dtype=float).ravel()
            b_flat = b.to_numpy(dtype=float).ravel()
            mask = np.isfinite(a_flat) & np.isfinite(b_flat)
            if mask.sum() >= 100:
                corr = np.corrcoef(a_flat[mask], b_flat[mask])[0, 1]
                if np.isfinite(corr) and abs(corr) > max_abs_corr:
                    redundant = True
                    break
        if not redundant:
            # orient the emitted factor so higher rank = higher expected return
            oriented = cand.expression if cand.mean_ic >= 0 else f"-({cand.expression})"
            selected.append(oriented)
            kept_originals.append(cand.expression)

    report.selected = selected
    return selected
