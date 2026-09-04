"""Every constant the evolutionary loop reads, in one place and frozen.

Two rules govern what may live here:

- Anything that could be tuned against a result is either fixed before the first
  run (windows, the round schedule, the fitness weights) or chosen by an
  explicit sweep on the VALIDATION window and then frozen (the shared-subtree
  limit and the correlation limit -- see `scripts/sp500_evo_threshold_sweep.py`).
  Nothing is tuned on the holdout, ever.
- Numbers that come from a measurement rather than a decision (the Alpha101
  turnover median that splits the archive's turnover buckets) are loaded from
  `configs/evo_calibration.json`, which a Part A script writes, so the value in
  code is a documented fallback rather than a magic number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

# quantaalpha_us/evo/ -> quantaalpha_us/ -> the repo root, where configs/ lives
REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPO_ROOT / "configs" / "evo_calibration.json"


@dataclass(frozen=True)
class Windows:
    """The three date windows, and the one the model is allowed to learn from.

    The model sees the fit window and nothing else. Validation exists for
    exactly two purposes -- the early-stop rule and the two threshold choices --
    and the holdout is read once, by one script, at the end.
    """

    fit_start: str = "2000-01-01"
    fit_end: str = "2013-12-31"
    validation_start: str = "2014-01-01"
    validation_end: str = "2017-12-31"
    holdout_start: str = "2018-01-01"
    holdout_end: str = "2025-12-31"
    # The cheap screen: a candidate is scored here first and dropped if it is
    # hopeless, before the full fit window is touched. Three years is enough to
    # kill an expression with no cross-sectional information at all, and it is
    # roughly a fifth of the cost.
    cheap_start: str = "2011-01-01"

    @property
    def fit_years(self) -> int:
        return int(self.fit_end[:4]) - int(self.fit_start[:4]) + 1

    def latest_visible_year(self) -> int:
        """The last year a prompt may mention. The audit test reads this."""
        return int(self.fit_end[:4])


THRESHOLDS_PATH = REPO_ROOT / "configs" / "evo_thresholds.json"


@dataclass(frozen=True)
class Gates:
    """Hard rejections, in the order they are checked (cheapest first)."""

    # complexity, measured by factors.ast_tools.complexity
    max_symbol_length: int = 200
    max_base_fields: int = 5
    max_free_constants: int = 4
    max_depth: int = 6
    # a candidate sharing a complete subtree of this many nodes with a published
    # alpha, or with an archive member, is a paraphrase and is rejected. Swept
    # over {4, 5, 6} on validation.
    max_shared_subtree: int = 5
    min_coverage: float = 0.9
    # swept over {0.6, 0.7, 0.8} on validation
    max_abs_corr: float = 0.7
    # the cheap-then-full screen
    cheap_min_abs_t: float = 1.0
    # a MUTATE result further than this from its parent is not a localized edit
    max_mutation_edits: int = 3
    # where the two swept values came from, recorded so a run's manifest says it
    thresholds_source: str = "defaults (configs/evo_thresholds.json absent)"
    # a CROSSOVER child must share at least this much with each parent
    min_crossover_shared: int = 3
    # SIMPLIFY must keep this share of the parent's fit-window |IC|
    simplify_min_ic_share: float = 0.8


    @classmethod
    def load(cls, path: Path | None = None) -> "Gates":
        """Defaults, overridden by the two values the validation sweep froze.

        Only `max_shared_subtree` and `max_abs_corr` may come from the file.
        Everything else here was fixed before any run, and letting a file move
        it would make the gates a tuning surface rather than a specification.
        """
        path = Path(path or THRESHOLDS_PATH)
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        chosen = payload.get("chosen", payload)
        return cls(
            max_shared_subtree=int(chosen["max_shared_subtree"]),
            max_abs_corr=float(chosen["max_abs_corr"]),
            thresholds_source=str(payload.get("source", str(path))),
        )


@dataclass(frozen=True)
class FitnessWeights:
    """The scalar. Every term is stated in t units so they can be subtracted.

    fitness = |t| * stability - complexity penalty - turnover penalty

    `t` is the mean daily rank IC over the fit window divided by a circular
    block-bootstrap standard error (block 21). `stability` is the fraction of
    fit-window years with a positive oriented IC, floored so a factor cannot be
    zeroed by one bad decade.

    The turnover term needs a stated conversion, because a cost in basis points
    and a t-statistic are not the same unit. The convention used here: charge
    `turnover_cost_bps` basis points of daily return per unit of one-way daily
    turnover, then treat that charge as if it were a reduction in the mean daily
    IC of the same magnitude, and divide by the IC's standard error to express
    it in t units. That is a convention, not a derivation -- IC points and
    return points are not interchangeable -- and it is written down here so the
    penalty can be argued with rather than reverse-engineered.
    """

    stability_floor: float = 0.5
    free_base_fields: int = 3
    free_constants: int = 1
    complexity_penalty_per_unit: float = 0.1
    turnover_cost_bps: float = 0.5

    @property
    def turnover_cost_daily(self) -> float:
        return self.turnover_cost_bps * 1e-4


@dataclass(frozen=True)
class Schedule:
    """Rounds, populations and the per-round operator budget."""

    rounds: int = 8
    population: int = 30
    elite: int = 5
    tournament: int = 3
    explore_n: int = 10
    mutate_n: int = 10
    crossover_n: int = 8
    simplify_n: int = 2
    migrate_every: int = 2
    migrate_n: int = 2
    reset_every: int = 4
    early_stop_patience: int = 2
    archive_per_niche: int = 8
    memory_max_rules: int = 12
    memory_ask_rules: int = 8
    # selection weights: a candidate in a sparsely-filled niche is worth more,
    # and one that has been sitting in the population for a while is worth less
    sparse_niche_weight: float = 2.0
    sparse_niche_threshold: int = 3
    stale_weight: float = 0.5
    stale_rounds: int = 2

    @property
    def calls_per_island_round(self) -> int:
        """Four operator calls plus one reflection call."""
        return 5

    def planned_calls(self, islands: int) -> int:
        return self.rounds * islands * self.calls_per_island_round


@dataclass(frozen=True)
class Island:
    """One sub-population and the direction its EXPLORE prompt is steered in."""

    name: str
    direction: str
    seeds: tuple[str, ...]


# Three directions, deliberately overlapping as little as the data allows. The
# seeds are ordinary, well-known expressions rather than anything mined: an
# island seeded with a good factor would make the loop look better than it is.
ISLANDS: tuple[Island, ...] = (
    Island(
        name="price_trend",
        direction=(
            "price trend and reversal: momentum over several horizons, "
            "short-horizon overreaction, the position of price inside its recent "
            "range, and how those interact with realised volatility"
        ),
        seeds=(
            "TS_DELTA($close, 21) / (TS_STD($close, 21) + 1e-8)",
            "-TS_DELTA($close, 5) / (TS_STD($close, 5) + 1e-8)",
            "($close - TS_MIN($low, 21)) / (TS_MAX($high, 21) - TS_MIN($low, 21) + 1e-8)",
        ),
    ),
    Island(
        name="liquidity_volume",
        direction=(
            "liquidity and volume: turnover and dollar volume levels, volume "
            "acceleration against its own baseline, the relationship between "
            "volume and returns, and intraday range as a liquidity proxy"
        ),
        seeds=(
            "TS_MEAN($dollar_volume, 21)",
            "-TS_CORR($return, $volume, 10)",
            "($high - $low) / ($close + 1e-8)",
        ),
    ),
    Island(
        name="fundamental_quality",
        direction=(
            "fundamentals and quality: profitability, margins, balance-sheet "
            "risk, asset turnover, valuation against price, and changes in those "
            "quantities, optionally interacted with price behaviour"
        ),
        seeds=(
            "RANK($roa)",
            "RANK($book_per_share / ($close + 1e-8))",
            "-RANK($leverage)",
        ),
    ),
)


@dataclass(frozen=True)
class Calibration:
    """Measured constants, not chosen ones.

    `alpha101_turnover_median` splits the archive's turnover axis. The fallback
    is the figure measured by `scripts/sp500_evo_calibrate.py` on the fit
    window; the file is authoritative when present.
    """

    alpha101_turnover_median: float = 0.5
    source: str = "fallback (configs/evo_calibration.json absent)"

    @classmethod
    def load(cls, path: Path | None = None) -> "Calibration":
        path = path or CALIBRATION_PATH
        if not Path(path).exists():
            return cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            alpha101_turnover_median=float(payload["alpha101_turnover_median"]),
            source=str(payload.get("source", str(path))),
        )


@dataclass(frozen=True)
class EvoConfig:
    """Everything one arm of the experiment needs to be reproducible."""

    arm: str = "mock"
    model: str = "mock"
    effort: str = "low"
    seed: int = 20260904
    # "loop" runs the full round schedule. "one_shot" is the control from B9: a
    # single EXPLORE-style call per island asking for the same TOTAL number of
    # candidates a full loop would generate, through the same gates and the same
    # archive, with no rounds, no feedback and no reflection. It isolates
    # iteration from the model.
    mode: str = "loop"
    # "full" is the B7 feedback block. "scalar" reduces it to the fitness number
    # and nothing else, which is the feedback ablation arm.
    feedback_mode: str = "full"
    windows: Windows = field(default_factory=Windows)
    gates: Gates = field(default_factory=Gates.load)
    fitness: FitnessWeights = field(default_factory=FitnessWeights)
    schedule: Schedule = field(default_factory=Schedule)
    islands: tuple[Island, ...] = ISLANDS
    calibration: Calibration = field(default_factory=Calibration.load)
    # rank-space correlation is compared on every k-th fit date. Correlation is
    # a pooled statistic over hundreds of thousands of pairs either way, and the
    # stride is what keeps a full archive's cached rank matrices inside memory.
    correlation_date_stride: int = 10
    bootstrap_draws: int = 200
    bootstrap_block: int = 21
    # The IC half-life exists to sort a factor into one of three horizon
    # buckets, and it is a ratio of mean ICs, so it does not need every date.
    # The four extra horizon ICs were otherwise the single most expensive thing
    # in scoring a candidate. Rather than a fixed stride, the curve keeps at
    # least this many dates: on the production fit window (3,521 days) that is a
    # stride of 5, and on a short window it is no stride at all. A fixed stride
    # was tried first and put a factor two buckets away from its
    # full-resolution answer on a 315-day window, because 63 observations of a
    # mean IC is noise. The one-day IC that drives fitness is always measured on
    # the full window.
    half_life_min_dates: int = 700

    @property
    def one_shot_candidates(self) -> int:
        """How many a one-shot island must ask for to match a loop island's total."""
        schedule = self.schedule
        per_round = (schedule.explore_n + schedule.mutate_n
                     + schedule.crossover_n + schedule.simplify_n)
        return schedule.rounds * per_round

    def with_thresholds(self, *, max_shared_subtree: int, max_abs_corr: float) -> "EvoConfig":
        """A copy with the two swept thresholds set. Used by the sweep only."""
        return replace(self, gates=replace(
            self.gates, max_shared_subtree=max_shared_subtree, max_abs_corr=max_abs_corr
        ))

    def to_dict(self) -> dict:
        return {
            "arm": self.arm, "model": self.model, "effort": self.effort, "seed": self.seed,
            "mode": self.mode, "feedback_mode": self.feedback_mode,
            "windows": self.windows.__dict__,
            "gates": self.gates.__dict__,
            "fitness": self.fitness.__dict__,
            "schedule": self.schedule.__dict__,
            "islands": [i.name for i in self.islands],
            "calibration": self.calibration.__dict__,
            "correlation_date_stride": self.correlation_date_stride,
            "bootstrap_draws": self.bootstrap_draws,
            "bootstrap_block": self.bootstrap_block,
        }
