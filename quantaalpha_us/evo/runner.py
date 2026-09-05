"""The loop. One round: propose, gate, score, admit, reflect, reshuffle.

Read the round in `_run_island` and everything else is bookkeeping. The parts
worth knowing before reading the code:

Cost order. A proposal is checked as text first (sanitizer, complexity, two
paraphrase gates), then evaluated once, then gated on coverage and on
correlation against the archive, then screened on three cheap years, and only
then scored on the full fit window and the horizon grid. Most rejections happen
before a DataFrame is touched, which is what makes hundreds of candidates per
arm affordable.

What the model sees. Prompts are built from the archive, the population, the
reflection memory and fit-window metrics. The validation ICs the early-stop rule
reads live in `self._validation_ic`, are never written into an archive entry's
metrics, and are never passed to `feedback.candidate_feedback`. That is checked
by a test rather than left to discipline.

Failure handling. Every raw response is written to disk before it is parsed, so
a parse error costs nothing but the parse. A round is recorded in `rounds.jsonl`
only once all of its islands finished, so `--resume` re-does at most one round.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from quantaalpha_us.evo import operators as ops
from quantaalpha_us.evo.archive import Archive, ArchiveEntry, load_archive
from quantaalpha_us.evo.backends import Backend, BackendReply
from quantaalpha_us.evo.config import EvoConfig
from quantaalpha_us.evo.feedback import StructuralContext, candidate_feedback
from quantaalpha_us.evo.fitness import GateRunner, GateOutcome
from quantaalpha_us.evo.islands import (
    IslandState,
    Member,
    migrate,
    next_population,
    reset_weakest,
    round_rng,
    tournament_select,
)
from quantaalpha_us.evo.memory import ReflectionMemory
from quantaalpha_us.evo.persistence import RunStore
from quantaalpha_us.evo.scoring import CandidateMetrics, PanelScorer, candidate_id
from quantaalpha_us.factors import alpha101 as alpha101_module
from quantaalpha_us.factors.expression_evaluator import ExpressionError
from quantaalpha_us.llm.budget import RunBudget


@dataclass
class RoundSummary:
    round: int
    proposals: int
    scored: int
    admitted: int
    archive_size: int
    niches_filled: int
    validation_top20_ic: float
    best_fitness: float
    rejections: dict[str, int]
    migrated: dict[str, int] = field(default_factory=dict)
    reseeded: str | None = None
    calls: int = 0
    total_tokens: int = 0
    seconds: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict) -> "RoundSummary":
        """Rebuild from `rounds.jsonl`. JSON has no NaN, so nulls come back as NaN."""
        def number(key: str) -> float:
            value = payload.get(key)
            return float("nan") if value is None else float(value)

        return cls(
            round=int(payload["round"]), proposals=int(payload.get("proposals", 0)),
            scored=int(payload.get("scored", 0)), admitted=int(payload.get("admitted", 0)),
            archive_size=int(payload.get("archive_size", 0)),
            niches_filled=int(payload.get("niches_filled", 0)),
            validation_top20_ic=number("validation_top20_ic"),
            best_fitness=number("best_fitness"),
            rejections=dict(payload.get("rejections", {})),
            migrated=dict(payload.get("migrated", {})),
            reseeded=payload.get("reseeded"),
            calls=int(payload.get("calls", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            seconds=float(payload.get("seconds", 0.0)),
        )

    def to_dict(self) -> dict:
        return {
            "round": self.round, "proposals": self.proposals, "scored": self.scored,
            "admitted": self.admitted, "archive_size": self.archive_size,
            "niches_filled": self.niches_filled,
            "validation_top20_ic": (None if not np.isfinite(self.validation_top20_ic)
                                    else self.validation_top20_ic),
            "best_fitness": (None if not np.isfinite(self.best_fitness)
                             else self.best_fitness),
            "rejections": dict(sorted(self.rejections.items())),
            "migrated": dict(sorted(self.migrated.items())),
            "reseeded": self.reseeded, "calls": self.calls,
            "total_tokens": self.total_tokens, "seconds": round(self.seconds, 1),
        }


@dataclass
class RunResult:
    archive: Archive
    rounds: list[RoundSummary]
    stopped_early: bool
    stop_reason: str
    calls: int
    total_tokens: int
    seconds: float

    @property
    def validation_curve(self) -> list[float]:
        return [r.validation_top20_ic for r in self.rounds]


class EvolutionRunner:
    """Drives one arm from an empty archive to a stopping condition."""

    def __init__(
        self,
        config: EvoConfig,
        scorer: PanelScorer,
        backend: Backend,
        store: RunStore,
        *,
        alpha101: Sequence[str] | None = None,
        budget: RunBudget | None = None,
        library: ops.PromptLibrary | None = None,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.scorer = scorer
        self.backend = backend
        self.store = store
        self.verbose = verbose
        self.library = library or ops.PromptLibrary()

        alphas = list(alpha101) if alpha101 is not None else [
            a.expression for a in alpha101_module.load()
        ]
        self.alpha_names = {a.expression: a.name for a in alpha101_module.load()}
        self.gates = GateRunner(config, alphas)
        self.ban_list = ops.alpha101_ban_list(alphas)

        planned = (len(config.islands) if config.mode == "one_shot"
                   else config.schedule.planned_calls(len(config.islands)))
        self.budget = budget or RunBudget(max_requests=planned + 10,
                                          max_total_tokens=10**9)
        if self.budget.max_requests < planned:
            raise ValueError(
                f"budget allows {self.budget.max_requests} requests but a full run needs "
                f"{planned} ({config.schedule.rounds} rounds x {len(config.islands)} "
                f"islands x {config.schedule.calls_per_island_round} calls). Raise the "
                "cap or lower the schedule; starting a run that cannot finish wastes "
                "every call it does make."
            )

        self.archive = Archive(config, path=store.archive_path)
        self.states = [IslandState(island=i) for i in config.islands]
        self.memories = {i.name: ReflectionMemory(capacity=config.schedule.memory_max_rules)
                         for i in config.islands}
        self._metrics: dict[str, CandidateMetrics] = {}
        self._validation_ic: dict[str, float] = {}
        self._seen: set[str] = set()
        self._salvaged: dict[str, dict] = {}
        self.reused_calls = 0
        self._admissions = 0
        self.calls = 0
        self.total_tokens = 0

    # ---- setup -----------------------------------------------------------

    def _seed_islands(self) -> None:
        """Score each island's hand-written seeds so round 1 has parents.

        The seeds are ordinary textbook expressions, deliberately: seeding with
        anything mined would hand the loop a head start the one-shot control does
        not get, and the comparison is the whole experiment.
        """
        for state in self.states:
            members: list[Member] = []
            for expression in state.island.seeds:
                member, _ = self._score_candidate(
                    expression, operator="seed", island=state.name, round_index=0,
                    parents=(), rationale="island seed", changed_node="",
                )
                if member is not None:
                    members.append(member)
            state.population = members

    def _resume(self) -> int:
        last = self.store.last_completed_round()
        self.store.reset_to_round(last)
        self._salvaged = self.store.salvaged_responses()
        if self._salvaged and self.verbose:
            print(f"{len(self._salvaged)} reply/replies salvaged from the interrupted "
                  "round; they will be reused where the prompt is identical", flush=True)
        if last <= 0:
            if self.verbose:
                print("nothing complete to resume from; starting the run over", flush=True)
            return 0
        self.archive = load_archive(self.config, self.store.archive_path)
        self._seen |= self.store.known_candidate_ids()
        for entry in self.archive.members():
            metrics = CandidateMetrics.from_dict(entry.metrics) if entry.metrics else None
            if metrics is not None:
                self._metrics[entry.id] = metrics
            self._seen.add(entry.id)
        for state in self.states:
            self.memories[state.name] = ReflectionMemory.load(
                self.store.memory_path(state.name), self.config.schedule.memory_max_rules
            )
            state.population = [Member.from_archive(e) for e in
                                self.archive.top(self.config.schedule.population)
                                if e.island == state.name] or [
                Member.from_archive(e) for e in self.archive.top(self.config.schedule.elite)
            ]
        if not any(s.population for s in self.states):
            self._seed_islands()
        self._rebuild_archive_ranks()
        if self.verbose:
            print(f"resuming after round {last}: archive {len(self.archive)}, "
                  f"{self.archive.niches_filled()} niches filled", flush=True)
        return last

    def _rebuild_archive_ranks(self) -> None:
        """Re-evaluate archive members so the correlation gate has something to compare.

        The rank matrices are the one piece of run state that is not on disk --
        they are hundreds of megabytes and they are derived -- so a resumed run
        recomputes them. Skipping this would leave the correlation gate passing
        everything for the rest of the run, which is a silent loss of a gate
        rather than a crash.
        """
        for entry in self.archive.members():
            if entry.id in self.archive.ranks:
                continue
            try:
                signal = self.scorer.evaluate(entry.expression)
            except ExpressionError:
                continue
            self.archive.ranks[entry.id] = self.scorer.ranked_fit(signal)
            metrics = self._metrics.get(entry.id)
            sign = metrics.sign if metrics is not None else 1
            self._validation_ic[entry.id] = self.scorer.validation_mean_ic(signal, sign)

    # ---- scoring one proposal -------------------------------------------

    def _reject(self, *, expression: str, outcome: GateOutcome, operator: str,
                island: str, round_index: int, counter: dict[str, int]) -> None:
        counter[outcome.gate] = counter.get(outcome.gate, 0) + 1
        self.store.record_rejection({
            "round": round_index, "island": island, "operator": operator,
            "expression": expression, "gate": outcome.gate, "detail": outcome.detail,
        })

    def _score_candidate(
        self,
        expression: str,
        *,
        operator: str,
        island: str,
        round_index: int,
        parents: Sequence[str],
        rationale: str,
        changed_node: str,
        counter: dict[str, int] | None = None,
        parent_metrics: CandidateMetrics | None = None,
    ) -> tuple[Member | None, GateOutcome]:
        counter = counter if counter is not None else {}
        identifier = candidate_id(expression)
        if identifier in self._seen:
            outcome = GateOutcome(False, "duplicate", "already proposed in this run")
            self._reject(expression=expression, outcome=outcome, operator=operator,
                         island=island, round_index=round_index, counter=counter)
            return None, outcome
        self._seen.add(identifier)

        outcome = self.gates.text_gates(expression, self.archive)
        if not outcome.passed:
            self._reject(expression=expression, outcome=outcome, operator=operator,
                         island=island, round_index=round_index, counter=counter)
            return None, outcome

        try:
            signal = self.scorer.evaluate(expression)
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad. ExpressionError is the contract, but a hole in
            # the evaluator that raises something else must cost one candidate,
            # not a run: this exact case (a panel reaching float() inside BOUND)
            # killed a GP arm at round 5 after four hours of scoring.
            kind = "evaluation" if isinstance(exc, ExpressionError) else "evaluation_crash"
            outcome = GateOutcome(False, kind, f"{type(exc).__name__}: {exc}"[:200])
            self._reject(expression=expression, outcome=outcome, operator=operator,
                         island=island, round_index=round_index, counter=counter)
            return None, outcome

        outcome = self.gates.coverage_gate(self.scorer.coverage(signal))
        if not outcome.passed:
            self._reject(expression=expression, outcome=outcome, operator=operator,
                         island=island, round_index=round_index, counter=counter)
            return None, outcome

        ranked = self.scorer.ranked_fit(signal)
        outcome = self.gates.correlation_gate(ranked, self.archive)
        if not outcome.passed:
            self._reject(expression=expression, outcome=outcome, operator=operator,
                         island=island, round_index=round_index, counter=counter)
            return None, outcome

        outcome = self.gates.cheap_screen(self.scorer.cheap_tstat(signal))
        if not outcome.passed:
            self._reject(expression=expression, outcome=outcome, operator=operator,
                         island=island, round_index=round_index, counter=counter)
            return None, outcome

        metrics = self.scorer.metrics(expression, signal)
        if operator == ops.SIMPLIFY and parent_metrics is not None:
            outcome = self.gates.simplify_is_an_improvement(
                metrics.complexity, parent_metrics.complexity,
                abs(metrics.mean_ic), abs(parent_metrics.mean_ic),
            )
            if not outcome.passed:
                self._reject(expression=expression, outcome=outcome, operator=operator,
                             island=island, round_index=round_index, counter=counter)
                return None, outcome

        self._metrics[identifier] = metrics
        self.store.record_candidate({
            "round": round_index, "island": island, "operator": operator,
            "id": identifier, "expression": expression, "rationale": rationale,
            "changed_node": changed_node, "parents": list(parents),
            **metrics.to_dict(),
        })

        entry = ArchiveEntry(
            id=identifier, expression=expression, fitness=metrics.fitness,
            niche=metrics.niche, round=round_index, island=island, operator=operator,
            parent_ids=tuple(candidate_id(p) for p in parents), rationale=rationale,
            metrics=metrics.to_dict(),
        )
        admission = self.archive.try_admit(entry, ranked)
        if not admission.admitted:
            counter[f"archive:{admission.reason}"] = counter.get(
                f"archive:{admission.reason}", 0) + 1
        else:
            self._admissions += 1
            # validation is read here and ONLY here, for the early-stop curve.
            # It never enters an archive entry, a feedback block or a prompt.
            self._validation_ic[identifier] = self.scorer.validation_mean_ic(
                signal, metrics.sign
            )
        return Member(id=identifier, expression=expression, fitness=metrics.fitness,
                      round=round_index, niche=metrics.niche, island=island,
                      metrics=metrics.to_dict()), GateOutcome(True)

    # ---- prompts ---------------------------------------------------------

    def _ranked_for(self, member: Member) -> np.ndarray | None:
        """Fit-window ranks for a population member, reusing the archive's copy.

        Most parents are archive members, and the archive already holds their
        strided rank matrix for the correlation gate. Only the ones that are not
        pay for a re-evaluation, which keeps the per-round cost of building
        feedback at a handful of panel passes rather than one per parent.
        """
        cached = self.archive.ranks.get(member.id)
        if cached is not None:
            return cached
        try:
            return self.scorer.ranked_fit(self.scorer.evaluate(member.expression))
        except ExpressionError:
            return None

    def _feedback_for(self, member: Member) -> tuple[str, str]:
        metrics = self._metrics.get(member.id)
        if metrics is None:
            metrics = CandidateMetrics.from_dict(member.metrics)
        if self.config.feedback_mode == "scalar":
            # the ablation arm: the fitness number and nothing else. Deliberately
            # not "a shorter block" -- the question is whether the DIAGNOSTICS
            # help, so everything diagnostic has to go.
            return member.expression, (
                f"id: {member.id}\nexpression: {member.expression}\n"
                f"  fitness {member.fitness:.3f}"
            )
        ranked = self._ranked_for(member)
        if ranked is None:
            corr, who = float("nan"), ""
        else:
            corr, who = self.archive.max_abs_correlation(ranked, exclude=member.id)
        shared, alpha_expr = self.gates.max_shared_alpha101(member.expression)
        distance, nearest_expr = self.gates.nearest_alpha101_edit(member.expression)
        context = StructuralContext(
            nearest_archive_expression=who,
            nearest_archive_correlation=corr,
            nearest_alpha_name=self.alpha_names.get(alpha_expr or nearest_expr, ""),
            nearest_alpha_shared_nodes=shared,
            nearest_alpha_edit_distance=distance,
        )
        return member.expression, candidate_feedback(
            metrics, context, self.config, identifier=member.id
        )

    def _requests_for(self, state: IslandState, round_index: int) -> list[ops.OperatorRequest]:
        cfg = self.config
        rng = round_rng(cfg, round_index, state.name, "select")
        memory_block = self.memories[state.name].render()
        if cfg.mode == "one_shot":
            # one call, asking for everything a full loop would have produced on
            # this island, with no memory to draw on because there were no
            # earlier rounds to write one
            return [ops.build_explore(
                cfg, self.library, island=state.name,
                direction=state.direction_text(), round_index=round_index,
                niche_summary=self.archive.niche_summary(), ban_list=self.ban_list,
                memory_block="Working memory: not used in this run.",
                n=cfg.one_shot_candidates,
            )]
        requests = [
            ops.build_explore(
                cfg, self.library, island=state.name,
                direction=state.direction_text(), round_index=round_index,
                niche_summary=self.archive.niche_summary(), ban_list=self.ban_list,
                memory_block=memory_block,
            )
        ]

        mutate_parents = state.best(cfg.schedule.mutate_n)
        if mutate_parents:
            requests.append(ops.build_mutate(
                cfg, self.library, island=state.name, round_index=round_index,
                parent_blocks=[self._feedback_for(m) for m in mutate_parents],
                memory_block=memory_block,
            ))

        pairs = self._crossover_pairs(state, rng, round_index)
        if pairs:
            requests.append(ops.build_crossover(
                cfg, self.library, island=state.name, round_index=round_index, pairs=pairs
            ))

        complex_first = sorted(
            state.population,
            key=lambda m: (-(m.metrics.get("complexity", {}).get("depth", 0)
                             + m.metrics.get("complexity", {}).get("base_fields", 0)
                             + m.metrics.get("complexity", {}).get("free_constants", 0)),
                           m.id),
        )[:cfg.schedule.simplify_n]
        if complex_first:
            requests.append(ops.build_simplify(
                cfg, self.library, island=state.name, round_index=round_index,
                parent_blocks=[self._feedback_for(m) for m in complex_first],
            ))
        return requests

    def _crossover_pairs(self, state: IslandState, rng, round_index: int):
        """Pairs drawn from DIFFERENT niches, which is what makes a cross a cross."""
        cfg = self.config
        picked = tournament_select(rng, state, self.archive, cfg,
                                   current_round=round_index,
                                   count=cfg.schedule.crossover_n * 2)
        pairs = []
        used: set[str] = set()
        for i, left in enumerate(picked):
            if len(pairs) >= cfg.schedule.crossover_n:
                break
            partner = next(
                (m for m in picked[i + 1:]
                 if tuple(m.niche) != tuple(left.niche) and m.id != left.id
                 and (left.id, m.id) not in used),
                None,
            )
            if partner is None:
                continue
            used.add((left.id, partner.id))
            pairs.append((left.id, left.expression, partner.id, partner.expression))
        return pairs

    # ---- the round -------------------------------------------------------

    def _call(self, request: ops.OperatorRequest) -> BackendReply:
        salvaged = self._salvaged.pop(request.prompt_hash, None)
        if salvaged is not None:
            # Same prompt, already paid for. Recorded again into the live
            # response file so replay still sees one record per call, in order.
            reply = BackendReply(
                payload=salvaged.get("payload") or {},
                raw_text=salvaged.get("raw_text", ""),
                model=salvaged.get("model", ""), effort=salvaged.get("effort", ""),
                input_tokens=int(salvaged.get("input_tokens", 0) or 0),
                output_tokens=int(salvaged.get("output_tokens", 0) or 0),
                total_tokens=int(salvaged.get("total_tokens", 0) or 0),
                cost_usd=salvaged.get("cost_usd"),
                duration_ms=int(salvaged.get("duration_ms", 0) or 0),
            )
            self.store.record_response(request, reply)
            self.reused_calls += 1
            if self.verbose:
                print(f"  reused a salvaged {request.operator}/{request.island} reply",
                      flush=True)
            return reply
        if not self.budget.can_request():
            # Say WHICH limit stopped the run. "Budget exhausted" was reported
            # for a run that had used 107 of a million allowed requests: the
            # real cause was three consecutive failures, and the message sent
            # the diagnosis in the wrong direction entirely.
            budget = self.budget
            if budget.requests_used >= budget.max_requests:
                reason = (f"request cap reached ({budget.requests_used}/"
                          f"{budget.max_requests})")
            elif budget.total_tokens_used > budget.max_total_tokens:
                reason = (f"token cap reached ({budget.total_tokens_used:,}/"
                          f"{budget.max_total_tokens:,})")
            else:
                reason = (f"{budget.consecutive_failures} consecutive failed calls "
                          f"(limit {budget.max_consecutive_failures}); the last one "
                          f"was {request.operator}/{request.island} in round "
                          f"{request.round}")
            raise RuntimeError(
                f"stopping after {budget.requests_used} requests: {reason}. "
                "Re-run with --resume once the cause is cleared."
            )
        reply = self.backend.call(request)
        # written BEFORE parsing: a parse failure must not lose a paid call
        self.store.record_response(request, reply)
        self.budget.record_request(tokens_used=reply.total_tokens, success=not reply.error)
        self.calls += 1
        self.total_tokens += reply.total_tokens
        return reply

    def _run_island(self, state: IslandState, round_index: int,
                    counter: dict[str, int]) -> tuple[list[Member], int, list[str], list[str]]:
        scored: list[Member] = []
        proposals_seen = 0
        accepted_lines: list[str] = []
        rejected_lines: list[str] = []

        for request in self._requests_for(state, round_index):
            reply = self._call(request)
            if reply.error:
                rejected_lines.append(f"  {request.operator}: call failed: {reply.error[:120]}")
                continue
            proposals = ops.parse_proposals(reply.payload, request)
            proposals_seen += len(proposals)
            # Parents are addressable by the id the prompt showed, and also by
            # the expression itself, because a model that echoes the expression
            # instead of the id has still identified its parent unambiguously.
            parent_lookup = {}
            offered: list[list[str]] = []
            for expr in request.parents:
                parent_lookup[candidate_id(expr)] = expr
                parent_lookup[expr] = expr
                offered.append([expr])
            for pair in request.parent_pairs:
                for expr in pair:
                    parent_lookup[candidate_id(expr)] = expr
                    parent_lookup[expr] = expr
                offered.append(list(pair))

            for index, proposal in enumerate(proposals):
                parents = [parent_lookup[pid] for pid in proposal.parent_ids
                           if pid in parent_lookup]
                if not parents and offered:
                    # Fall back on position ONLY when the batch came back the
                    # size it was asked for. A short batch means the positions
                    # no longer line up, and pairing a child with the wrong
                    # parent would silently invert the locality and fusion
                    # gates rather than fail.
                    if len(proposals) == len(offered) and index < len(offered):
                        parents = list(offered[index])
                    else:
                        outcome = GateOutcome(
                            False, "unresolved_parent",
                            f"parent_ids {list(proposal.parent_ids)} match none of the "
                            f"{len(offered)} offered, and the batch came back "
                            f"{len(proposals)} of {len(offered)} so position cannot "
                            "be trusted",
                        )
                        self._reject(expression=proposal.expression, outcome=outcome,
                                     operator=request.operator, island=state.name,
                                     round_index=round_index, counter=counter)
                        rejected_lines.append(
                            f"  {proposal.expression[:110]}\n      {outcome.reason[:150]}")
                        continue

                outcome = self._operator_check(request.operator, proposal.expression, parents)
                if not outcome.passed:
                    self._reject(expression=proposal.expression, outcome=outcome,
                                 operator=request.operator, island=state.name,
                                 round_index=round_index, counter=counter)
                    rejected_lines.append(
                        f"  {proposal.expression[:110]}\n      {outcome.reason[:150]}")
                    continue

                parent_metrics = None
                if parents:
                    parent_metrics = self._metrics.get(candidate_id(parents[0]))
                member, outcome = self._score_candidate(
                    proposal.expression, operator=request.operator, island=state.name,
                    round_index=round_index, parents=parents,
                    rationale=proposal.rationale, changed_node=proposal.changed_node,
                    counter=counter, parent_metrics=parent_metrics,
                )
                if member is None:
                    rejected_lines.append(
                        f"  {proposal.expression[:110]}\n      {outcome.reason[:150]}")
                    continue
                scored.append(member)
                accepted_lines.append(
                    f"  fitness {member.fitness:7.3f}  niche {'/'.join(member.niche)}  "
                    f"{member.expression[:100]}")

        if self.config.mode == "one_shot":
            return scored, proposals_seen, accepted_lines, rejected_lines

        # reflection, once per island per round
        reflect_request = ops.build_reflection(
            self.config, self.library, island=state.name, round_index=round_index,
            accepted=accepted_lines[:20], rejected=rejected_lines[:20],
        )
        reply = self._call(reflect_request)
        if not reply.error:
            rules = ops.parse_rules(reply.payload, self.config.schedule.memory_ask_rules)
            self.memories[state.name].add(rules)
            self.memories[state.name].save(self.store.memory_path(state.name))
        return scored, proposals_seen, accepted_lines, rejected_lines

    def _operator_check(self, operator: str, expression: str,
                        parents: Sequence[str]) -> GateOutcome:
        if operator == ops.MUTATE and parents:
            return self.gates.mutation_is_local(expression, parents[0])
        if operator == ops.CROSSOVER and len(parents) >= 2:
            return self.gates.crossover_uses_both_parents(expression, parents[:2])
        return GateOutcome(True)

    # ---- the curve and the stop -----------------------------------------

    def _validation_top20(self, n: int = 20) -> float:
        """Equal-weighted mean of the top-n archive members' validation ICs.

        Equal-weighted mean of the individual ICs, not the IC of a combined
        portfolio: the combined version is what the final holdout table reports,
        and computing it every round would mean re-evaluating twenty panels per
        round for a number that only has to be monotone-ish.

        Note the number is only comparable across rounds once the archive holds
        at least `n` members. Below that the mean is taken over fewer, stronger
        factors, so it FALLS as the archive fills even when the search is going
        well; `run` therefore does not count a stall until the archive is full
        enough for the comparison to mean something.
        """
        top = self.archive.top(n, exclude_seeds=True)
        values = [self._validation_ic.get(e.id) for e in top]
        values = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    # ---- the run ---------------------------------------------------------

    def run(self, *, resume: bool = False) -> RunResult:
        started = time.time()
        self.store.write_manifest({
            "config": self.config.to_dict(),
            "backend": getattr(self.backend, "name", "unknown"),
            "model": getattr(self.backend, "model", self.config.model),
            "effort": getattr(self.backend, "effort", self.config.effort),
            "budget": self.budget.to_dict(),
        })
        completed = self._resume() if resume else 0
        if completed == 0:
            self._seed_islands()

        summaries = ([RoundSummary.from_dict(r) for r in self.store.rounds()]
                     if resume else [])
        best_curve = max((s.validation_top20_ic for s in summaries
                          if np.isfinite(s.validation_top20_ic)), default=float("-inf"))
        stalled = 0
        stopped_early, stop_reason = False, "reached the round limit"

        for round_index in range(completed + 1, self.config.effective_rounds + 1):
            round_started = time.time()
            counter: dict[str, int] = {}
            calls_before, tokens_before = self.calls, self.total_tokens
            admissions_before = self._admissions
            scored_all: list[Member] = []
            proposals_total = 0

            for state in self.states:
                scored, proposals, _, _ = self._run_island(state, round_index, counter)
                proposals_total += proposals
                scored_all.extend(scored)
                state.population = next_population(state, scored, self.archive, self.config)

            migrated: dict[str, int] = {}
            if round_index % self.config.schedule.migrate_every == 0:
                migrated = migrate(self.states, self.archive, self.config)
            reseeded = None
            if round_index % self.config.schedule.reset_every == 0:
                reseeded = reset_weakest(self.states, self.archive, self.config, round_index)

            curve = self._validation_top20()
            summary = RoundSummary(
                round=round_index, proposals=proposals_total, scored=len(scored_all),
                admitted=self._admissions - admissions_before,
                archive_size=len(self.archive.discovered()),
                niches_filled=self.archive.niches_filled(),
                validation_top20_ic=curve,
                best_fitness=max((m.fitness for m in self.archive.discovered()),
                                 default=float("nan")),
                rejections=counter, migrated=migrated, reseeded=reseeded,
                calls=self.calls - calls_before,
                total_tokens=self.total_tokens - tokens_before,
                seconds=time.time() - round_started,
            )
            summaries.append(summary)
            self.store.record_round(summary.to_dict())
            if self.verbose:
                print(f"round {round_index}: {proposals_total} proposals, "
                      f"{len(scored_all)} scored, archive {len(self.archive)} in "
                      f"{self.archive.niches_filled()}/18 niches, "
                      f"validation top-20 IC {curve:.5f}, "
                      f"{summary.calls} calls, {summary.seconds:.0f}s", flush=True)

            # The curve is a mean over min(20, archive size) members, so while
            # the archive is still filling it drops for a reason that has
            # nothing to do with the search stalling: each new admission is
            # weaker than the ones already there. Only start counting stalls
            # once there are twenty to average.
            archive_full_enough = len(self.archive.top(20, exclude_seeds=True)) >= 20
            if np.isfinite(curve) and curve > best_curve + 1e-12:
                best_curve, stalled = curve, 0
            elif archive_full_enough:
                stalled += 1
            if (self.config.mode != "one_shot" and archive_full_enough
                    and stalled >= self.config.schedule.early_stop_patience):
                stopped_early = True
                stop_reason = (
                    f"validation top-20 IC did not improve for "
                    f"{self.config.schedule.early_stop_patience} consecutive rounds"
                )
                break

        return RunResult(
            archive=self.archive, rounds=summaries, stopped_early=stopped_early,
            stop_reason=stop_reason, calls=self.calls, total_tokens=self.total_tokens,
            seconds=time.time() - started,
        )
