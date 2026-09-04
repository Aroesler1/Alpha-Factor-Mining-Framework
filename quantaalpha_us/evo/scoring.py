"""Panel scoring for the loop: one signal evaluation, every number the gates need.

Everything the model is allowed to see is computed on the FIT window
(2000-2013). Validation and holdout numbers are computed by different methods
on this same object, are never returned in the same structure, and are never
passed to anything that builds a prompt. That separation is structural rather
than a convention: `CandidateMetrics` has no field that could hold one.

Cost discipline matters here because the loop scores hundreds of candidates per
arm. The order is: evaluate the panel once, gate on coverage, gate on
correlation against the archive, score three cheap years, and only then score
the full fit window and the four extra horizons the half-life needs. A
candidate that dies at the cheap screen never pays for the horizon grid.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from quantaalpha_us.evo.config import EvoConfig
from quantaalpha_us.factors import ast_tools as at
from quantaalpha_us.factors.expression_evaluator import (
    ExpressionError,
    ExpressionEvaluator,
    build_field_panels,
)
from quantaalpha_us.factors.factor_research import _daily_spearman_ic
from quantaalpha_us.factors.ic_panel import (
    HORIZONS,
    LAGGED,
    daily_turnover,
    forward_returns_at,
    horizon_bucket,
    ic_half_life,
    universe_coverage,
)
from quantaalpha_us.factors.multiple_testing import block_bootstrap_se


def candidate_id(expression: str) -> str:
    """Stable identity for a factor, insensitive to how it is spelled.

    Two expressions with the same canonical tree are the same factor and must
    not occupy two archive slots, so identity is the canonical-tree hash rather
    than the string. Falls back to hashing the raw text for anything that will
    not parse, which only happens for candidates that are about to be rejected
    anyway but still need an id to be logged under.
    """
    try:
        return at.subtree_hash(at.parse(expression))
    except at.ParseError:
        return hashlib.blake2b(expression.encode("utf-8"), digest_size=8).hexdigest()


@dataclass
class CandidateMetrics:
    """Fit-window facts about one candidate. Nothing here comes from after 2013."""

    expression: str
    mean_ic: float
    ic_se: float
    tstat: float
    icir: float
    coverage: float
    turnover: float
    half_life: float
    stability: float
    sign: int
    ic_by_year: dict[int, float] = field(default_factory=dict)
    complexity: dict[str, int] = field(default_factory=dict)
    fitness: float = float("nan")
    complexity_penalty: float = 0.0
    turnover_penalty: float = 0.0
    horizon_bucket: str = ""
    data_family: str = ""
    turnover_bucket: str = ""

    @property
    def niche(self) -> tuple[str, str, str]:
        return (self.horizon_bucket, self.data_family, self.turnover_bucket)

    @classmethod
    def from_dict(cls, payload: dict) -> "CandidateMetrics":
        """Rebuild from an archive record, for --resume and --replay.

        `half_life` is stored as null-plus-a-flag rather than as `Infinity`,
        because JSON has no infinity and every encoder spells it differently.
        """
        half_life = payload.get("half_life")
        if half_life is None:
            half_life = (float("inf") if payload.get("half_life_is_beyond_grid")
                         else float("nan"))
        niche = tuple(payload.get("niche", ("", "", "")))
        return cls(
            expression=payload["expression"],
            mean_ic=float(payload.get("mean_ic", float("nan"))),
            ic_se=float(payload.get("ic_se", float("nan"))),
            tstat=float(payload.get("tstat", float("nan"))),
            icir=float(payload.get("icir", float("nan"))),
            coverage=float(payload.get("coverage", float("nan"))),
            turnover=float(payload.get("turnover", float("nan"))),
            half_life=float(half_life),
            stability=float(payload.get("stability", float("nan"))),
            sign=int(payload.get("sign", 1)),
            ic_by_year={int(k): float(v) for k, v in payload.get("ic_by_year", {}).items()},
            complexity={k: int(v) for k, v in payload.get("complexity", {}).items()},
            fitness=float(payload.get("fitness", float("nan"))),
            complexity_penalty=float(payload.get("complexity_penalty", 0.0)),
            turnover_penalty=float(payload.get("turnover_penalty", 0.0)),
            horizon_bucket=niche[0] if len(niche) > 0 else "",
            data_family=niche[1] if len(niche) > 1 else "",
            turnover_bucket=niche[2] if len(niche) > 2 else "",
        )

    def to_dict(self) -> dict:
        out = {
            "expression": self.expression,
            "mean_ic": self.mean_ic,
            "ic_se": self.ic_se,
            "tstat": self.tstat,
            "icir": self.icir,
            "coverage": self.coverage,
            "turnover": self.turnover,
            "half_life": (None if not np.isfinite(self.half_life) else self.half_life),
            "half_life_is_beyond_grid": bool(np.isinf(self.half_life)),
            "stability": self.stability,
            "sign": self.sign,
            "fitness": self.fitness,
            "complexity_penalty": self.complexity_penalty,
            "turnover_penalty": self.turnover_penalty,
            "ic_by_year": {str(k): v for k, v in sorted(self.ic_by_year.items())},
            "complexity": dict(sorted(self.complexity.items())),
            "niche": list(self.niche),
        }
        return out


class PanelScorer:
    """Evaluates expressions and scores them on the loop's windows."""

    def __init__(
        self,
        bars: pd.DataFrame,
        config: EvoConfig,
        *,
        min_cross_section: int = 30,
        panels: Mapping[str, pd.DataFrame] | None = None,
        memoize: bool = False,
    ) -> None:
        self.config = config
        self.min_cross_section = min_cross_section
        # `metrics` is a pure function of the expression -- the signal argument is
        # derived from it -- so it can be memoised. Off by default because a
        # single run scores each expression once and the cache would be dead
        # weight; on for the threshold sweep, which replays the same candidate
        # stream nine times and would otherwise recompute the horizon grid nine
        # times per candidate.
        self._metrics_cache: dict[str, CandidateMetrics] | None = {} if memoize else None
        self.panels = dict(panels) if panels is not None else build_field_panels(bars)
        self.evaluator = ExpressionEvaluator(self.panels)

        index = self.panels["close"].index
        w = config.windows
        self.fit_dates = self._slice(index, w.fit_start, w.fit_end)
        self.cheap_dates = self._slice(index, w.cheap_start, w.fit_end)
        self._validation_dates = self._slice(index, w.validation_start, w.validation_end)
        self._holdout_dates = self._slice(index, w.holdout_start, w.holdout_end)

        # forward returns, sliced per window so a scoring call never touches a
        # date outside the window it claims to be measuring
        self._fwd_fit = {
            h: forward_returns_at(self.panels, h, LAGGED).loc[self.fit_dates]
            for h in HORIZONS
        }
        self._fwd_cheap = self._fwd_fit[1].loc[self.cheap_dates]
        self._fwd_validation = forward_returns_at(self.panels, 1, LAGGED).loc[
            self._validation_dates
        ]
        self._fwd_holdout = forward_returns_at(self.panels, 1, LAGGED).loc[
            self._holdout_dates
        ]
        self._stride_dates = self.fit_dates[:: config.correlation_date_stride]
        close = self.panels["close"]
        self._close_fit = close.loc[close.index.isin(self.fit_dates)]

    @staticmethod
    def _slice(index: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        return index[mask]

    # ---- evaluation ------------------------------------------------------

    def evaluate(self, expression: str) -> pd.DataFrame:
        """The signal panel. Raises ExpressionError, which the caller logs as a gate."""
        return self.evaluator.evaluate(expression)

    def ranked_fit(self, signal: pd.DataFrame) -> np.ndarray:
        """Strided fit-window cross-sectional ranks, for the correlation gate."""
        window = signal.loc[signal.index.isin(self._stride_dates)]
        return window.rank(axis=1, pct=True).to_numpy(dtype=np.float32)

    def coverage(self, signal: pd.DataFrame) -> float:
        """Share of the investable cross-section, not of the panel.

        See `ic_panel.universe_coverage`. Measured against the panel this would
        top out near 0.42 under point-in-time membership and the 0.9 gate would
        reject everything, which is what the first calibration run showed.
        """
        window = signal.loc[signal.index.isin(self.fit_dates)]
        return universe_coverage(window, self._close_fit)

    # ---- scoring ---------------------------------------------------------

    def cheap_tstat(self, signal: pd.DataFrame) -> float:
        """|t| on the last three fit years only, used to kill hopeless candidates.

        A cheap screen has to be cheap AND has to not discard good factors. Three
        years of daily ICs is ~750 observations, so a factor with any real edge
        clears |t| = 1 there comfortably; the screen is set to remove the
        expressions with no cross-sectional information at all, which is most of
        what a random proposer produces.
        """
        ic = _daily_spearman_ic(signal, self._fwd_cheap, self.min_cross_section)
        if len(ic) < 60:
            return 0.0
        se = ic.std(ddof=1) / np.sqrt(len(ic))
        return float(abs(ic.mean() / se)) if se > 0 else 0.0

    def fit_ic(self, signal: pd.DataFrame, horizon: int = 1) -> pd.Series:
        return _daily_spearman_ic(signal, self._fwd_fit[horizon], self.min_cross_section)

    def metrics(self, expression: str, signal: pd.DataFrame) -> CandidateMetrics:
        """Every fit-window number the gates, the fitness and the feedback need."""
        if self._metrics_cache is not None and expression in self._metrics_cache:
            return self._metrics_cache[expression]
        result = self._metrics(expression, signal)
        if self._metrics_cache is not None:
            self._metrics_cache[expression] = result
        return result

    def _metrics(self, expression: str, signal: pd.DataFrame) -> CandidateMetrics:
        cfg = self.config
        ic = self.fit_ic(signal, 1)
        seed = int(candidate_id(expression)[:8], 16)
        mean_ic = float(ic.mean()) if len(ic) else float("nan")
        sign = 1 if not np.isfinite(mean_ic) or mean_ic >= 0 else -1
        oriented = ic * sign
        se = block_bootstrap_se(oriented.to_numpy(), block=cfg.bootstrap_block,
                                draws=cfg.bootstrap_draws, seed=seed)
        oriented_mean = float(oriented.mean()) if len(oriented) else float("nan")
        tstat = oriented_mean / se if se and np.isfinite(se) and se > 0 else float("nan")
        sd = float(oriented.std(ddof=1)) if len(oriented) > 1 else float("nan")
        icir = oriented_mean / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")

        by_year = {int(y): float(v) for y, v in oriented.groupby(oriented.index.year).mean().items()}
        years = range(int(cfg.windows.fit_start[:4]), int(cfg.windows.fit_end[:4]) + 1)
        positive = sum(1 for y in years if by_year.get(y, 0.0) > 0)
        stability = max(cfg.fitness.stability_floor, positive / max(len(list(years)), 1))

        turnover = daily_turnover(signal.loc[signal.index.isin(self.fit_dates)])
        curve = {h: float(self.fit_ic(signal, h).mean()) for h in HORIZONS}
        half_life = ic_half_life(curve)

        comp = at.complexity(expression).to_dict()
        weights = cfg.fitness
        complexity_penalty = weights.complexity_penalty_per_unit * (
            max(0, comp["base_fields"] - weights.free_base_fields)
            + max(0, comp["free_constants"] - weights.free_constants)
        )
        turnover_penalty = (
            weights.turnover_cost_daily * turnover / se
            if se and np.isfinite(se) and se > 0 else 0.0
        )
        fitness = (
            abs(tstat) * stability - complexity_penalty - turnover_penalty
            if np.isfinite(tstat) else float("-inf")
        )

        turnover_bucket = ("low" if turnover <= cfg.calibration.alpha101_turnover_median
                           else "high")
        return CandidateMetrics(
            expression=expression,
            mean_ic=oriented_mean,
            ic_se=float(se),
            tstat=float(tstat),
            icir=float(icir),
            coverage=self.coverage(signal),
            turnover=float(turnover),
            half_life=half_life,
            stability=float(stability),
            sign=sign,
            ic_by_year=by_year,
            complexity=comp,
            fitness=float(fitness),
            complexity_penalty=float(complexity_penalty),
            turnover_penalty=float(turnover_penalty),
            horizon_bucket=horizon_bucket(half_life),
            data_family=at.data_family(expression),
            turnover_bucket=turnover_bucket,
        )

    # ---- windows the model must never see --------------------------------
    #
    # These two are deliberately plain floats with names that say what they are,
    # returned by methods no prompt builder imports. `tests/test_evo_windows.py`
    # asserts that the feedback builder cannot reach them.

    def validation_mean_ic(self, signal: pd.DataFrame, sign: int = 1) -> float:
        ic = _daily_spearman_ic(signal, self._fwd_validation, self.min_cross_section)
        return float(ic.mean() * sign) if len(ic) else float("nan")

    def holdout_ic_series(self, signal: pd.DataFrame, sign: int = 1) -> pd.Series:
        ic = _daily_spearman_ic(signal, self._fwd_holdout, self.min_cross_section)
        return ic * sign

    def validation_ic_series(self, signal: pd.DataFrame, sign: int = 1) -> pd.Series:
        ic = _daily_spearman_ic(signal, self._fwd_validation, self.min_cross_section)
        return ic * sign


def equal_weight_rank_average(
    scorer: PanelScorer, expressions: Sequence[str], signs: Sequence[int],
) -> pd.DataFrame:
    """The combined signal of a factor set: mean of sign-corrected daily ranks.

    This is the "number that matters" in the final table -- a portfolio built
    from the whole surviving set rather than the best single expression -- so it
    lives next to the scorer both arms share.
    """
    total = None
    count = None
    for expr, sign in zip(expressions, signs):
        ranks = scorer.evaluate(expr).rank(axis=1, pct=True) * sign
        total = ranks if total is None else total.add(ranks, fill_value=0.0)
        present = ranks.notna().astype(float)
        count = present if count is None else count.add(present, fill_value=0.0)
    if total is None:
        return pd.DataFrame()
    return total / count.replace(0, np.nan)
