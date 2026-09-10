"""Island populations: who gets to be a parent, and what survives a round.

Three sub-populations, each steered by a different direction prompt, each
carrying its own reflection memory. Islands are the cheapest defence against the
failure mode that kills a single-population loop: everything converges on the
first good idea, because every parent is drawn from the same pool and the pool is
already full of that idea's children.

Three mechanisms keep them apart and then deliberately mix them:

- selection is within an island (plus the shared archive), so an island's search
  is mostly its own;
- migration every `migrate_every` rounds hands each island the best archive
  members that came from the OTHER islands, so a good idea can cross;
- a reset every `reset_every` rounds reseeds whichever island has the lowest
  mean fitness, so a dead island does not spend the remaining rounds proposing
  variations on nothing.

Selection is a size-3 tournament over the island's population plus the archive,
with sampling weights from `archive.selection_weight`: double for a member in a
sparsely filled niche, half for one that has been sitting around for more than
`stale_rounds`.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Sequence

from quantaalpha_us.evo.archive import Archive, ArchiveEntry, selection_weight
from quantaalpha_us.evo.config import EvoConfig, Island


@dataclass
class Member:
    """A scored candidate held in an island's working population."""

    id: str
    expression: str
    fitness: float
    round: int
    niche: tuple[str, str, str]
    island: str
    metrics: dict = field(default_factory=dict)

    @classmethod
    def from_archive(cls, entry: ArchiveEntry) -> "Member":
        return cls(id=entry.id, expression=entry.expression, fitness=entry.fitness,
                   round=entry.round, niche=tuple(entry.niche), island=entry.island,
                   metrics=entry.metrics)


def round_rng(config: EvoConfig, round_index: int, island: str, tag: str = "") -> random.Random:
    """A seeded RNG that depends on the round and island, not on call order.

    Deriving the stream from a hash rather than from one long shared generator
    is what makes `--resume` produce the same selections as an uninterrupted
    run: resuming at round 5 cannot replay the four rounds of draws that a
    single global generator would have consumed by then.
    """
    key = f"{config.seed}|{config.arm}|{round_index}|{island}|{tag}".encode("utf-8")
    return random.Random(int(hashlib.blake2b(key, digest_size=8).hexdigest(), 16))


@dataclass
class IslandState:
    """One island's working population, direction and reseed history."""

    island: Island
    population: list[Member] = field(default_factory=list)
    reseeded_at: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        return self.island.name

    def direction_text(self) -> str:
        """The direction the EXPLORE prompt is given, plus any reseed instruction."""
        if not self.reseeded_at:
            return self.island.direction
        return (
            f"{self.island.direction}\n\n"
            "This island was reseeded because its population had the lowest mean "
            "fitness. The earlier line of attack did not work; take a different "
            "one within the same direction rather than refining what is above."
        )

    def mean_fitness(self) -> float:
        if not self.population:
            return float("-inf")
        return sum(m.fitness for m in self.population) / len(self.population)

    def elites(self, n: int) -> list[Member]:
        return sorted(self.population, key=lambda m: (-m.fitness, m.id))[:n]

    def best(self, n: int) -> list[Member]:
        return self.elites(n)


def tournament_select(
    rng: random.Random,
    state: IslandState,
    archive: Archive,
    config: EvoConfig,
    *,
    current_round: int,
    count: int,
) -> list[Member]:
    """`count` parents, each the fittest of a weighted size-3 draw.

    Parents may repeat across draws -- that is what a tournament does, and a
    strong parent being mutated twice in a round is a feature -- but a draw
    never contains the same candidate twice.
    """
    schedule = config.schedule
    sparse = archive.sparse_niches(schedule.sparse_niche_threshold)
    pool: dict[str, Member] = {}
    for member in state.population:
        pool[member.id] = member
    for entry in archive.members():
        pool.setdefault(entry.id, Member.from_archive(entry))
    if not pool:
        return []

    members = [pool[key] for key in sorted(pool)]
    weights = []
    for member in members:
        entry = archive.entries.get(member.id)
        if entry is not None:
            weights.append(selection_weight(entry, current_round=current_round,
                                            sparse=sorted(sparse), config=config))
        else:
            weight = 1.0
            if tuple(member.niche) in sparse:
                weight *= schedule.sparse_niche_weight
            if current_round - member.round > schedule.stale_rounds:
                weight *= schedule.stale_weight
            weights.append(weight)

    chosen: list[Member] = []
    for _ in range(count):
        size = min(schedule.tournament, len(members))
        contestants: list[Member] = []
        available = list(range(len(members)))
        available_weights = list(weights)
        for _ in range(size):
            total = sum(available_weights[i] for i in available)
            if total <= 0:
                pick = rng.choice(available)
            else:
                threshold = rng.random() * total
                cumulative = 0.0
                pick = available[-1]
                for i in available:
                    cumulative += available_weights[i]
                    if cumulative >= threshold:
                        pick = i
                        break
            contestants.append(members[pick])
            available.remove(pick)
        chosen.append(max(contestants, key=lambda m: (m.fitness, m.id)))
    return chosen


def next_population(
    state: IslandState,
    scored: Sequence[Member],
    archive: Archive,
    config: EvoConfig,
) -> list[Member]:
    """Top `elite` survive unconditionally; every other slot is re-drawn.

    "Re-drawn" means: filled from this round's scored candidates by fitness, and
    topped up from the archive if the round produced too few. Carrying the old
    population forward instead would make the loop pure elitism, which is how a
    population stops moving.
    """
    schedule = config.schedule
    keep = state.elites(schedule.elite)
    seen = {m.id for m in keep}
    out = list(keep)
    for member in sorted(scored, key=lambda m: (-m.fitness, m.id)):
        if len(out) >= schedule.population:
            break
        if member.id in seen:
            continue
        seen.add(member.id)
        out.append(member)
    for entry in archive.top(schedule.population):
        if len(out) >= schedule.population:
            break
        if entry.id in seen:
            continue
        seen.add(entry.id)
        out.append(Member.from_archive(entry))
    return out


def migrate(states: Sequence[IslandState], archive: Archive, config: EvoConfig) -> dict[str, int]:
    """Each island receives the top archive members that came from other islands."""
    moved: dict[str, int] = {}
    for state in states:
        incoming = archive.top_from_islands(config.schedule.migrate_n,
                                            exclude_island=state.name)
        present = {m.id for m in state.population}
        added = 0
        for entry in incoming:
            if entry.id in present:
                continue
            state.population.append(Member.from_archive(entry))
            present.add(entry.id)
            added += 1
        moved[state.name] = added
    return moved


def reset_weakest(states: Sequence[IslandState], archive: Archive, config: EvoConfig,
                  round_index: int) -> str | None:
    """Reseed the lowest-mean-fitness island from a spread of archive niches."""
    if not states:
        return None
    weakest = min(states, key=lambda s: (s.mean_fitness(), s.name))
    seeds: list[Member] = []
    seen: set[str] = set()
    # one per niche first, so a reseed does not simply reinstall the archive's
    # single best idea under a new island name
    for niche_members in (archive.in_niche(n) for n in sorted({e.niche for e in archive.members()})):
        if not niche_members:
            continue
        entry = niche_members[0]
        if entry.id in seen:
            continue
        seen.add(entry.id)
        seeds.append(Member.from_archive(entry))
    for entry in archive.top(config.schedule.population):
        if len(seeds) >= config.schedule.population:
            break
        if entry.id in seen:
            continue
        seen.add(entry.id)
        seeds.append(Member.from_archive(entry))
    weakest.population = seeds
    weakest.reseeded_at = weakest.reseeded_at + (round_index,)
    return weakest.name
