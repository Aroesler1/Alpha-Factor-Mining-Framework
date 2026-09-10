"""A MAP-Elites archive over (horizon, data family, turnover).

Why an archive rather than a top-N list. A plain "keep the best 20" loop
converges on one idea restated twenty ways, because the best idea's neighbours
are the easiest improvements to find. MAP-Elites keeps the best member of each
BEHAVIOURAL niche instead, so a slow fundamental factor is not competing with a
fast price factor for the same slot and the search is pushed into the parts of
the space that are still empty.

The three axes are chosen to be things a portfolio actually cares about and that
this repo can measure:

  horizon        fast (IC half-life < 5 days), medium (5-21), slow (> 21).
                 Calibrated by scripts/sp500_ic_horizon_curve.py.
  data family    price and volume only, fundamentals only, or mixed.
  turnover       low or high, split at the median daily turnover of the 50
                 transcribed Alpha101 formulas on the fit window, so "high" means
                 "faster than half the published alphas" rather than a guess.

3 x 3 x 2 = 18 niches, 8 members each, so the archive tops out at 144 factors.

Correlation dedup is NOT done here: it is a gate, applied against the whole
archive before admission, because a candidate too close to an existing member
should be rejected and logged rather than silently dropped at insertion time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from quantaalpha_us.evo.config import EvoConfig
from quantaalpha_us.factors import ast_tools as at
from quantaalpha_us.factors.factor_research import (
    mean_daily_rank_correlation,
    mean_daily_rank_correlation_many,
)

HORIZON_BUCKETS = ("fast", "medium", "slow")
DATA_FAMILIES = ("price_volume", "fundamental", "mixed")
TURNOVER_BUCKETS = ("low", "high")
ALL_NICHES: tuple[tuple[str, str, str], ...] = tuple(
    product(HORIZON_BUCKETS, DATA_FAMILIES, TURNOVER_BUCKETS)
)


@dataclass
class ArchiveEntry:
    """One admitted factor and everything needed to audit where it came from."""

    id: str
    expression: str
    fitness: float
    niche: tuple[str, str, str]
    round: int
    island: str
    operator: str
    parent_ids: tuple[str, ...]
    rationale: str
    metrics: dict = field(default_factory=dict)
    admitted_at: int = 0

    def to_record(self) -> dict:
        record = asdict(self)
        record["niche"] = list(self.niche)
        record["parent_ids"] = list(self.parent_ids)
        return record


@dataclass
class Admission:
    admitted: bool
    reason: str
    evicted: str | None = None


class Archive:
    """Bounded per-niche storage, written to disk after every admission."""

    def __init__(self, config: EvoConfig, path: Path | None = None) -> None:
        self.config = config
        self.path = Path(path) if path is not None else None
        self.entries: dict[str, ArchiveEntry] = {}
        self.ranks: dict[str, np.ndarray] = {}
        # subtree hash -> node count, per member, computed once at admission.
        # Without it the paraphrase gate re-enumerates every archive member's
        # tree for every candidate, which is the dominant cost of the cheap
        # half of the gate stack once the archive is full.
        self._subtrees: dict[str, dict[str, int]] = {}
        self._sequence = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---- reads -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def members(self) -> list[ArchiveEntry]:
        """Admission order. Deterministic, which is what replay equality needs."""
        return sorted(self.entries.values(), key=lambda e: e.admitted_at)

    def in_niche(self, niche: tuple[str, str, str]) -> list[ArchiveEntry]:
        members = [e for e in self.entries.values() if e.niche == tuple(niche)]
        return sorted(members, key=lambda e: (-e.fitness, e.admitted_at))

    SEED_OPERATOR = "seed"

    def top(self, n: int, *, exclude_seeds: bool = False) -> list[ArchiveEntry]:
        """The n fittest members.

        `exclude_seeds` drops the hand-written island seeds. They are admitted
        so the paraphrase and correlation gates can see them -- without that a
        candidate could re-derive a seed and enter as a discovery -- but they
        were not found by anything, and every arm starts from the same three, so
        counting them in a reported top-20 would flatter all arms equally and
        blur the comparison the experiment is for.
        """
        members = self.entries.values()
        if exclude_seeds:
            members = [e for e in members if e.operator != self.SEED_OPERATOR]
        return sorted(members, key=lambda e: (-e.fitness, e.admitted_at))[:n]

    def discovered(self) -> list[ArchiveEntry]:
        """Archive members that an operator produced, seeds excluded."""
        return [e for e in self.members() if e.operator != self.SEED_OPERATOR]

    def top_from_islands(self, n: int, *, exclude_island: str) -> list[ArchiveEntry]:
        members = [e for e in self.entries.values() if e.island != exclude_island]
        return sorted(members, key=lambda e: (-e.fitness, e.admitted_at))[:n]

    def niches_filled(self) -> int:
        return len({e.niche for e in self.entries.values()})

    def niche_summary(self) -> list[dict]:
        """Counts and best fitness per niche, INCLUDING the empty ones.

        The empty ones are the point: they are what the EXPLORE prompt is asked
        to aim at, and an archive summary that only lists what already exists
        cannot steer a search toward what does not.
        """
        out = []
        for niche in ALL_NICHES:
            members = self.in_niche(niche)
            out.append({
                "niche": "/".join(niche),
                "count": len(members),
                "best_fitness": round(members[0].fitness, 3) if members else None,
                "capacity": self.config.schedule.archive_per_niche,
            })
        return out

    def sparse_niches(self, threshold: int) -> set[tuple[str, str, str]]:
        return {n for n in ALL_NICHES if len(self.in_niche(n)) < threshold}

    # ---- gate helpers ----------------------------------------------------

    def _subtree_map(self, entry: ArchiveEntry) -> dict[str, int]:
        cached = self._subtrees.get(entry.id)
        if cached is None:
            try:
                cached = at.subtree_sizes(at.parse(entry.expression))
            except at.ParseError:
                cached = {}
            self._subtrees[entry.id] = cached
        return cached

    def max_shared_subtree(self, expression: str) -> tuple[int, str]:
        """Largest complete subtree shared with any member, and which member."""
        try:
            candidate = at.subtree_sizes(at.parse(expression))
        except at.ParseError:
            return 0, ""
        best, who = 0, ""
        for entry in self.members():
            shared = at.largest_shared_from_maps(candidate, self._subtree_map(entry))
            if shared > best:
                best, who = shared, entry.expression
        return best, who

    def max_abs_correlation(self, ranked: np.ndarray,
                            *, exclude: str | None = None) -> tuple[float, str]:
        """Highest |mean daily rank correlation| against any member.

        Mean-of-daily rather than pooled, matching the measure
        `factor_research.mean_daily_rank_correlation` documents as the one that
        answers "is this the same signal on a typical day" -- which is the
        question a dedup gate is asking.
        """
        rows = [e for e in self.members()
                if e.id != exclude and self.ranks.get(e.id) is not None
                and self.ranks[e.id].shape == ranked.shape]
        if not rows:
            return 0.0, ""
        stack = np.stack([self.ranks[e.id] for e in rows])
        correlations = np.abs(mean_daily_rank_correlation_many(ranked, stack))
        if not np.isfinite(correlations).any():
            return 0.0, ""
        best_index = int(np.nanargmax(correlations))
        best = float(correlations[best_index])
        return best, rows[best_index].expression

    # ---- writes ----------------------------------------------------------

    def try_admit(self, entry: ArchiveEntry, ranked: np.ndarray | None = None) -> Admission:
        """Insert if the niche has room, or if this beats the niche's weakest."""
        if entry.id in self.entries:
            return Admission(False, "duplicate_of_archive_member")
        if not np.isfinite(entry.fitness):
            return Admission(False, "fitness_not_finite")

        members = self.in_niche(entry.niche)
        capacity = self.config.schedule.archive_per_niche
        evicted = None
        if len(members) >= capacity:
            weakest = members[-1]
            if entry.fitness <= weakest.fitness:
                return Admission(False, "niche_full_and_not_better")
            evicted = weakest.id
            self.entries.pop(weakest.id, None)
            self.ranks.pop(weakest.id, None)
            self._subtrees.pop(weakest.id, None)

        self._sequence += 1
        entry.admitted_at = self._sequence
        self.entries[entry.id] = entry
        if ranked is not None:
            self.ranks[entry.id] = ranked
        self._append(entry, evicted)
        return Admission(True, "admitted", evicted=evicted)

    def _append(self, entry: ArchiveEntry, evicted: str | None) -> None:
        if self.path is None:
            return
        record = {"event": "admit", "evicted": evicted, **entry.to_record()}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    # ---- persistence -----------------------------------------------------

    def snapshot(self) -> str:
        """The archive as a canonical string, for byte-for-byte replay checks."""
        lines = [json.dumps(e.to_record(), sort_keys=True, separators=(",", ":"))
                 for e in self.members()]
        return "\n".join(lines) + ("\n" if lines else "")

    @classmethod
    def from_records(cls, config: EvoConfig, records: Iterable[dict],
                     path: Path | None = None) -> "Archive":
        """Rebuild from a JSONL admission log, replaying evictions in order."""
        archive = cls(config, path=path)
        for record in records:
            if record.get("event") not in (None, "admit"):
                continue
            evicted = record.get("evicted")
            if evicted:
                archive.entries.pop(evicted, None)
                archive.ranks.pop(evicted, None)
                archive._subtrees.pop(evicted, None)
            entry = ArchiveEntry(
                id=record["id"], expression=record["expression"],
                fitness=float(record["fitness"]), niche=tuple(record["niche"]),
                round=int(record["round"]), island=record["island"],
                operator=record["operator"],
                parent_ids=tuple(record.get("parent_ids", [])),
                rationale=record.get("rationale", ""),
                metrics=record.get("metrics", {}),
                admitted_at=int(record.get("admitted_at", 0)),
            )
            archive.entries[entry.id] = entry
            archive._sequence = max(archive._sequence, entry.admitted_at)
        return archive


def load_archive(config: EvoConfig, path: Path) -> Archive:
    path = Path(path)
    if not path.exists():
        return Archive(config, path=path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    return Archive.from_records(config, records, path=path)


def selection_weight(entry: ArchiveEntry, *, current_round: int,
                     sparse: Sequence[tuple[str, str, str]], config: EvoConfig) -> float:
    """Tournament sampling weight: 2x in a sparse niche, 0.5x when stale.

    The two pulls are deliberately opposed. The sparse-niche bonus pushes the
    search outward into empty behaviour space; the staleness discount stops the
    same three winners being drawn as parents for eight rounds, which is how an
    island collapses onto one idea.
    """
    schedule = config.schedule
    weight = 1.0
    if tuple(entry.niche) in {tuple(n) for n in sparse}:
        weight *= schedule.sparse_niche_weight
    if current_round - entry.round > schedule.stale_rounds:
        weight *= schedule.stale_weight
    return weight
