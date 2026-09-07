"""Component tests for the loop's parts, each checked where it can go wrong quietly.

The end-to-end test proves the machine runs and finds a planted signal. It does
not prove that the archive evicts the right member, that the tournament actually
weights sparse niches, that a gate fires under its own name, or that a mutation
is really localized -- all of which could be wrong while the loop still produced
a plausible-looking archive.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from quantaalpha_us.evo import gp
from quantaalpha_us.evo import operators as ops
from quantaalpha_us.evo.archive import (
    ALL_NICHES,
    Archive,
    ArchiveEntry,
    selection_weight,
)
from quantaalpha_us.evo.config import EvoConfig, Island, Schedule
from quantaalpha_us.evo.feedback import StructuralContext, candidate_feedback
from quantaalpha_us.evo.fitness import GateRunner
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
from quantaalpha_us.evo.scoring import CandidateMetrics, candidate_id
from quantaalpha_us.factors import alpha101
from quantaalpha_us.factors import ast_tools as at
from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer


PARENTS = (
    "TS_DELTA($close, 21) / (TS_STD($close, 21) + 1e-8)",
    "TS_MEAN($dollar_volume, 21)",
    "($high - $low) / ($close + 1e-8)",
    "RANK($roa)",
    "-TS_CORR($return, $volume, 10)",
)


# ---- genetic operators ---------------------------------------------------


def test_every_mutation_is_within_the_gate_and_actually_changes_something():
    rng = random.Random(0)
    seen = 0
    for parent in PARENTS:
        for _ in range(40):
            result = gp.mutate(rng, parent, max_depth=6)
            if result is None:
                continue
            child, changed = result
            seen += 1
            assert child != parent
            assert changed
            assert at.edit_distance_from_parent(child, parent) <= 3
    assert seen > 50, "the mutation operator almost never fires; the test proves nothing"


def test_mutations_keep_window_slots_as_integer_literals():
    """The trap `random_expressions` documents: an untyped edit puts a panel in
    a window slot, passes the sanitizer, and dies inside the evaluator."""
    rng = random.Random(1)
    for parent in PARENTS:
        for _ in range(30):
            result = gp.mutate(rng, parent, max_depth=6)
            if result is None:
                continue
            tree = at.parse(result[0], canonical=False)
            kinds = at.slot_kinds(tree)
            for path, kind in kinds.items():
                node = at.at_path(tree, path)
                if kind == at.WINDOW_SLOT:
                    assert node.label.startswith("#") and not node.children
                    assert float(node.label[1:]).is_integer()
                if kind == at.SCALAR_SLOT:
                    assert node.label.startswith("#")


def test_every_generated_expression_passes_the_sanitizer():
    rng = random.Random(2)
    sanitizer = ExpressionSanitizer()
    produced = []
    produced += gp.random_expressions(rng, 20, reference=list(PARENTS))
    for parent in PARENTS:
        m = gp.mutate(rng, parent, max_depth=6)
        if m:
            produced.append(m[0])
        s = gp.simplify(rng, parent)
        if s:
            produced.append(s)
    for left in PARENTS:
        for right in PARENTS:
            c = gp.crossover(rng, left, right, max_depth=6)
            if c:
                produced.append(c)
    assert len(produced) > 25
    for expr in produced:
        assert sanitizer.sanitize(expr).valid, expr


def test_crossover_keeps_a_real_piece_of_both_parents():
    rng = random.Random(3)
    made = 0
    for left in PARENTS:
        for right in PARENTS:
            if left == right:
                continue
            child = gp.crossover(rng, left, right, max_depth=6, min_shared=3)
            if child is None:
                continue
            made += 1
            assert at.largest_shared_subtree_expr(child, left) >= 3
            assert at.largest_shared_subtree_expr(child, right) >= 3
    assert made >= 5


def test_simplify_shrinks_the_tree_and_keeps_a_field():
    rng = random.Random(4)
    made = 0
    for parent in ("RANK(TS_DELTA($close, 21) / (TS_STD($close, 21) + 1e-8))",
                   "CS_RANK(TS_CORR(RANK($close), RANK($volume), 5))"):
        for _ in range(20):
            child = gp.simplify(rng, parent)
            if child is None:
                continue
            made += 1
            assert at.size(at.parse(child)) < at.size(at.parse(parent))
            assert any(n.label.startswith("$") for n in at.parse(child))
    assert made > 0


def test_the_operators_are_seeded():
    a = gp.mutate(random.Random(9), PARENTS[0], max_depth=6)
    b = gp.mutate(random.Random(9), PARENTS[0], max_depth=6)
    assert a == b


# ---- gates ---------------------------------------------------------------


@pytest.fixture(scope="module")
def gates():
    return GateRunner(EvoConfig(), [a.expression for a in alpha101.load()])


def test_each_gate_fires_under_its_own_name(gates):
    archive = Archive(EvoConfig())
    assert gates.text_gates("TS_MEAN(close, 21)", archive).gate == "sanitizer"
    deep = "RANK(RANK(RANK(RANK(RANK(RANK(RANK($close)))))))"
    assert gates.text_gates(deep, archive).gate == "complexity"
    wide = "$open + $high + $low + $close + $volume + $return"
    assert gates.text_gates(wide, archive).gate == "complexity"


def test_the_alpha101_paraphrase_gate_catches_a_transcribed_alpha(gates):
    """The strongest possible case: a published alpha, submitted verbatim."""
    archive = Archive(EvoConfig())
    published = "-1 * TS_CORR(CS_RANK($open), CS_RANK($volume), 10)"
    outcome = gates.text_gates(published, archive)
    assert not outcome.passed
    assert outcome.gate == "alpha101_subtree"


def test_the_gates_run_cheapest_first(gates):
    """A candidate that fails several gates must be reported under the first one,
    so the rejection ledger says where the search is actually going wrong."""
    archive = Archive(EvoConfig())
    # unparsable AND far too deep AND a paraphrase: the sanitizer must win
    outcome = gates.text_gates("TS_MEAN(close, 21", archive)
    assert outcome.gate == "sanitizer"


def test_the_archive_paraphrase_gate_uses_the_archive(gates):
    config = EvoConfig()
    archive = Archive(config)
    expression = "TS_DECAY_LINEAR(TS_MEAN($accrual_gap, 21), 5)"
    assert gates.text_gates(expression, archive).passed
    archive.try_admit(ArchiveEntry(
        id=candidate_id(expression), expression=expression, fitness=1.0,
        niche=("fast", "fundamental", "low"), round=1, island="i", operator="explore",
        parent_ids=(), rationale="",
    ))
    outcome = gates.text_gates(expression + " * 2", archive)
    assert not outcome.passed and outcome.gate == "archive_subtree"


def test_the_coverage_and_cheap_screens_report_their_numbers(gates):
    out = gates.coverage_gate(0.42)
    assert not out.passed and out.gate == "coverage" and "0.42" in out.detail
    out = gates.cheap_screen(0.3)
    assert not out.passed and out.gate == "cheap_screen"
    assert gates.cheap_screen(4.0).passed


def test_mutation_locality_and_crossover_fusion_gates(gates):
    parent = "TS_MEAN($close, 21)"
    assert gates.mutation_is_local("TS_MEAN($close, 63)", parent).passed
    far = "CS_RANK(TS_CORR($high, $volume, 5)) / ($low + 1e-8)"
    assert not gates.mutation_is_local(far, parent).passed

    left, right = "TS_MEAN($close, 21)", "TS_STD($volume, 63)"
    fused = "TS_MEAN($close, 21) / (TS_STD($volume, 63) + 1e-8)"
    assert gates.crossover_uses_both_parents(fused, [left, right]).passed
    assert not gates.crossover_uses_both_parents("RANK($close)", [left, right]).passed


def test_simplify_gate_needs_both_simpler_and_still_working(gates):
    parent = {"base_fields": 3, "free_constants": 2, "depth": 5}
    simpler = {"base_fields": 2, "free_constants": 1, "depth": 3}
    assert gates.simplify_is_an_improvement(simpler, parent, 0.009, 0.010).passed
    assert not gates.simplify_is_an_improvement(simpler, parent, 0.004, 0.010).passed
    assert not gates.simplify_is_an_improvement(parent, parent, 0.010, 0.010).passed


# ---- archive -------------------------------------------------------------


def _entry(name: str, fitness: float, niche=("fast", "price_volume", "low"),
           island="a", round_index=1) -> ArchiveEntry:
    return ArchiveEntry(id=name, expression=f"TS_MEAN($close, {len(name)})",
                        fitness=fitness, niche=niche, round=round_index, island=island,
                        operator="explore", parent_ids=(), rationale="")


def test_the_archive_fills_a_niche_then_evicts_only_the_weakest():
    config = EvoConfig(schedule=Schedule(archive_per_niche=2))
    archive = Archive(config)
    assert archive.try_admit(_entry("a", 1.0)).admitted
    assert archive.try_admit(_entry("b", 2.0)).admitted
    weak = archive.try_admit(_entry("c", 0.5))
    assert not weak.admitted and weak.reason == "niche_full_and_not_better"
    strong = archive.try_admit(_entry("d", 3.0))
    assert strong.admitted and strong.evicted == "a"
    assert {e.id for e in archive.in_niche(("fast", "price_volume", "low"))} == {"b", "d"}


def test_the_archive_refuses_a_duplicate_and_a_non_finite_fitness():
    archive = Archive(EvoConfig())
    archive.try_admit(_entry("a", 1.0))
    assert archive.try_admit(_entry("a", 5.0)).reason == "duplicate_of_archive_member"
    assert archive.try_admit(_entry("b", float("nan"))).reason == "fitness_not_finite"


def test_niches_are_independent():
    config = EvoConfig(schedule=Schedule(archive_per_niche=1))
    archive = Archive(config)
    archive.try_admit(_entry("a", 1.0, niche=("fast", "price_volume", "low")))
    assert archive.try_admit(_entry("b", 0.1, niche=("slow", "fundamental", "high"))).admitted
    assert len(archive) == 2 and archive.niches_filled() == 2


def test_the_niche_summary_lists_the_empty_niches_too():
    archive = Archive(EvoConfig())
    summary = archive.niche_summary()
    assert len(summary) == len(ALL_NICHES) == 18
    assert all(row["count"] == 0 and row["best_fitness"] is None for row in summary)


def test_the_archive_round_trips_through_its_jsonl(tmp_path):
    config = EvoConfig(schedule=Schedule(archive_per_niche=2))
    path = tmp_path / "archive.jsonl"
    archive = Archive(config, path=path)
    for name, fitness in (("a", 1.0), ("b", 2.0), ("d", 3.0)):
        archive.try_admit(_entry(name, fitness))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    rebuilt = Archive.from_records(config, records)
    assert rebuilt.snapshot() == archive.snapshot()


def test_the_snapshot_is_stable_under_insertion_order_of_equal_fitness():
    """Replay equality depends on this: the snapshot is admission-ordered, so two
    runs that admit the same things in the same order agree exactly."""
    archive = Archive(EvoConfig())
    for name in ("a", "b", "c"):
        archive.try_admit(_entry(name, 1.0, niche=("fast", "price_volume", "low")))
    assert [e.id for e in archive.members()] == ["a", "b", "c"]


def test_selection_weight_rewards_sparse_niches_and_punishes_staleness():
    config = EvoConfig()
    sparse = [("fast", "price_volume", "low")]
    fresh = _entry("a", 1.0, round_index=5)
    assert selection_weight(fresh, current_round=5, sparse=sparse, config=config) == 2.0
    assert selection_weight(fresh, current_round=5, sparse=[], config=config) == 1.0
    stale = _entry("b", 1.0, round_index=1)
    assert selection_weight(stale, current_round=5, sparse=[], config=config) == 0.5
    assert selection_weight(stale, current_round=5, sparse=sparse, config=config) == 1.0


# ---- islands -------------------------------------------------------------


def _member(name: str, fitness: float, niche=("fast", "price_volume", "low"),
            round_index=1) -> Member:
    return Member(id=name, expression=f"$close * {len(name)}", fitness=fitness,
                  round=round_index, niche=niche, island="a")


def test_tournament_prefers_the_fitter_of_the_drawn_contestants():
    config = EvoConfig()
    state = IslandState(island=Island("a", "d", ()),
                        population=[_member(f"m{i}", float(i)) for i in range(10)])
    rng = round_rng(config, 1, "a")
    picks = tournament_select(rng, state, Archive(config), config,
                              current_round=1, count=200)
    # the best of three draws from a uniform pool should be above the median
    assert np.mean([float(p.id[1:]) for p in picks]) > 5.5


def test_the_tournament_is_reproducible_for_a_round():
    config = EvoConfig()
    state = IslandState(island=Island("a", "d", ()),
                        population=[_member(f"m{i}", float(i)) for i in range(10)])
    first = tournament_select(round_rng(config, 3, "a"), state, Archive(config), config,
                              current_round=3, count=8)
    second = tournament_select(round_rng(config, 3, "a"), state, Archive(config), config,
                               current_round=3, count=8)
    assert [m.id for m in first] == [m.id for m in second]
    other = tournament_select(round_rng(config, 4, "a"), state, Archive(config), config,
                              current_round=4, count=8)
    assert [m.id for m in first] != [m.id for m in other]


def test_next_population_keeps_the_elites_and_redraws_the_rest():
    config = EvoConfig(schedule=Schedule(population=6, elite=2))
    old = [_member(f"o{i}", float(i)) for i in range(6)]
    state = IslandState(island=Island("a", "d", ()), population=old)
    fresh = [_member(f"n{i}", 100.0 + i) for i in range(6)]
    out = next_population(state, fresh, Archive(config), config)
    assert len(out) == 6
    kept = {m.id for m in out} & {m.id for m in old}
    assert kept == {"o5", "o4"}, "only the top 2 of the old population may survive"


def test_migration_moves_members_from_other_islands_only():
    config = EvoConfig(schedule=Schedule(migrate_n=2))
    archive = Archive(config)
    archive.try_admit(_entry("from_b", 5.0, island="b"))
    archive.try_admit(_entry("from_a", 9.0, island="a",
                             niche=("slow", "mixed", "high")))
    states = [IslandState(island=Island("a", "d", ())), IslandState(island=Island("b", "d", ()))]
    moved = migrate(states, archive, config)
    assert moved == {"a": 1, "b": 1}
    assert [m.id for m in states[0].population] == ["from_b"]
    assert [m.id for m in states[1].population] == ["from_a"]


def test_reset_picks_the_weakest_island_and_reseeds_it_across_niches():
    config = EvoConfig()
    archive = Archive(config)
    archive.try_admit(_entry("x", 9.0, niche=("fast", "price_volume", "low")))
    archive.try_admit(_entry("y", 8.0, niche=("slow", "fundamental", "high")))
    strong = IslandState(island=Island("strong", "d", ()), population=[_member("s", 10.0)])
    weak = IslandState(island=Island("weak", "d", ()), population=[_member("w", -5.0)])
    name = reset_weakest([strong, weak], archive, config, round_index=4)
    assert name == "weak"
    assert {m.id for m in weak.population} == {"x", "y"}
    assert weak.reseeded_at == (4,)
    assert "reseeded" in weak.direction_text()


def test_an_empty_island_is_the_weakest():
    config = EvoConfig()
    empty = IslandState(island=Island("empty", "d", ()))
    assert empty.mean_fitness() == float("-inf")


# ---- operators and memory ------------------------------------------------


def test_fill_refuses_to_leave_a_placeholder_behind():
    with pytest.raises(KeyError, match="MISSING"):
        ops.fill("a <<PRESENT>> b <<MISSING>>", {"PRESENT": 1})


def test_fill_survives_dollars_and_braces_in_the_substituted_text():
    out = ops.fill("<<X>>", {"X": 'TS_MEAN($close, 21) and {"a": 1}'})
    assert "$close" in out and '{"a": 1}' in out


def test_the_ban_list_reports_patterns_shared_across_alphas():
    alphas = [a.expression for a in alpha101.load()]
    ban = ops.alpha101_ban_list(alphas, min_size=3, min_alphas=3, top=10)
    assert 1 <= len(ban) <= 10
    assert all("published alphas" in row for row in ban)
    # the delayed close and the 20-day average volume are the set's workhorses
    assert any("DELAY($close, 1)" in row for row in ban)


def test_parse_proposals_drops_malformed_items_without_guessing():
    request = ops.OperatorRequest(ops.EXPLORE, "a", 1, "p", ops.CANDIDATE_SCHEMA, 3)
    payload = {"candidates": [
        {"expression": "RANK($close)", "rationale": "r", "parent_ids": ["x"],
         "changed_node": "n"},
        {"expression": "   ", "rationale": "empty"},
        {"rationale": "no expression"},
        "not an object",
        {"expression": "TS_MEAN($close, 5)", "rationale": "", "parent_ids": "nope",
         "changed_node": ""},
    ]}
    out = ops.parse_proposals(payload, request)
    assert [p.expression for p in out] == ["RANK($close)", "TS_MEAN($close, 5)"]
    assert out[0].parent_ids == ("x",)
    assert out[1].parent_ids == ()
    assert all(p.operator == ops.EXPLORE and p.round == 1 for p in out)


def test_parse_proposals_tolerates_a_reply_that_is_not_the_expected_shape():
    request = ops.OperatorRequest(ops.EXPLORE, "a", 1, "p", ops.CANDIDATE_SCHEMA, 3)
    assert ops.parse_proposals({}, request) == []
    assert ops.parse_proposals({"candidates": "no"}, request) == []
    assert ops.parse_proposals("[]", request) == []


def test_parse_rules_trims_and_caps():
    assert ops.parse_rules({"rules": ["  a  b ", "", "c"]}, 5) == ["a b", "c"]
    assert len(ops.parse_rules({"rules": [str(i) for i in range(20)]}, 3)) == 3
    assert ops.parse_rules({"nope": []}, 3) == []


def test_reflection_memory_is_fifo_and_deduped():
    memory = ReflectionMemory(capacity=3)
    memory.add(["a", "b", "a"])
    assert memory.rules == ["a", "b"]
    dropped = memory.add(["c", "d"])
    assert memory.rules == ["b", "c", "d"] and dropped == ["a"]


def test_memory_round_trips_through_disk(tmp_path):
    memory = ReflectionMemory(capacity=4, rules=["one", "two"])
    memory.save(tmp_path / "m.json")
    assert ReflectionMemory.load(tmp_path / "m.json").rules == ["one", "two"]
    assert ReflectionMemory.load(tmp_path / "absent.json").rules == []


# ---- feedback ------------------------------------------------------------


def test_a_feedback_block_carries_every_field_b7_asks_for():
    metrics = CandidateMetrics(
        expression="TS_MEAN($close, 21)", mean_ic=0.0081, ic_se=0.0011, tstat=7.4,
        icir=0.052, coverage=0.981, turnover=0.183, half_life=8.4, stability=0.79,
        sign=1, ic_by_year={2000 + i: 0.01 for i in range(14)},
        complexity={"symbol_length": 19, "base_fields": 1, "free_constants": 0, "depth": 1},
        fitness=5.6, complexity_penalty=0.0, turnover_penalty=0.08,
        horizon_bucket="medium", data_family="price_volume", turnover_bucket="low",
    )
    context = StructuralContext("RANK($close)", 0.41, "alpha006", 3, 9)
    block = candidate_feedback(metrics, context, EvoConfig(), identifier="abc")
    for needle in ("mean IC", "ICIR", "coverage", "turnover", "half-life",
                   "IC by year", "complexity", "nearest in archive",
                   "nearest published", "alpha006"):
        assert needle in block, needle
    assert block.count(":") > 14  # the fourteen yearly ICs are all there
    assert "2013" in block and "2014" not in block


def test_a_half_life_beyond_the_grid_is_stated_not_faked():
    metrics = CandidateMetrics(
        expression="$close", mean_ic=0.001, ic_se=0.001, tstat=1.0, icir=0.01,
        coverage=1.0, turnover=0.0, half_life=float("inf"), stability=0.5, sign=1,
    )
    block = candidate_feedback(metrics, StructuralContext(), EvoConfig())
    assert "does not halve" in block


def test_the_pruned_nearest_alpha_search_returns_the_exact_minimum(gates):
    """The size prune is only sound because unit-cost tree edit distance is at
    least the difference in node counts. If that ever stopped holding, the
    feedback block would quietly cite the wrong published alpha."""
    exhaustive_gates = GateRunner(EvoConfig(), [a.expression for a in alpha101.load()])
    for expression in (*PARENTS,
                       "CS_RANK(TS_CORR(RANK($close), RANK($volume), 5))",
                       "$close",
                       "IF_ELSE($close > $open, TS_MEAN($volume, 5), -TS_STD($return, 10))"):
        tree = at.parse(expression)
        brute = min(at.tree_edit_distance(tree, other)
                    for _, other in exhaustive_gates.alpha101)
        distance, name = gates.nearest_alpha101_edit(expression)
        assert distance == brute, (expression, distance, brute)
        assert name


def test_the_nearest_alpha_lookup_is_memoised(gates):
    first = gates.nearest_alpha101_edit(PARENTS[0])
    assert PARENTS[0] in gates._nearest_edit_cache
    assert gates.nearest_alpha101_edit(PARENTS[0]) == first


def test_the_strided_half_life_curve_puts_factors_in_the_same_bucket(tmp_path):
    """The half-life is measured on every fifth fit date to keep scoring
    affordable. It only has to decide fast/medium/slow, so the check is that the
    bucket agrees with the full-resolution curve, not that the number does."""
    import numpy as np
    from dataclasses import replace as dc_replace

    from quantaalpha_us.evo.scoring import PanelScorer
    from quantaalpha_us.factors.ic_panel import HORIZONS, horizon_bucket, ic_half_life
    from quantaalpha_us.factors.factor_research import _daily_spearman_ic
    from tests.synthetic_panel import PLANTED_EXPRESSION, planted_bars
    from tests.test_evo_end_to_end import make_config

    config = make_config()
    scorer = PanelScorer(planted_bars(n_days=500, n_symbols=50), config)
    for expression in (PLANTED_EXPRESSION, "TS_MEAN($volume, 21)", "RANK($high - $low)"):
        signal = scorer.evaluate(expression)
        strided = scorer.metrics(expression, signal).half_life
        full = ic_half_life({h: float(scorer.fit_ic(signal, h).mean()) for h in HORIZONS})
        assert horizon_bucket(strided) == horizon_bucket(full), (
            expression, strided, full
        )


def test_the_batched_correlation_matches_the_per_pair_reference():
    """The archive gate compares against up to 144 members at once. The batched
    form expands the per-day correlation into raw sums rather than centring, so
    it has to be pinned against the definition it replaced."""
    from quantaalpha_us.factors.factor_research import (
        mean_daily_rank_correlation,
        mean_daily_rank_correlation_many,
    )

    rng = np.random.default_rng(0)
    a = rng.random((60, 40)).astype(np.float32)
    stack = rng.random((7, 60, 40)).astype(np.float32)
    # a realistic missing pattern: each matrix defined on its own subset
    a[rng.random(a.shape) < 0.3] = np.nan
    stack[rng.random(stack.shape) < 0.3] = np.nan
    stack[0] = a  # one exact duplicate, which must come back as 1.0

    batched = mean_daily_rank_correlation_many(a, stack, min_cross_section=5)
    for i in range(stack.shape[0]):
        reference = mean_daily_rank_correlation(a, stack[i], min_cross_section=5)
        assert batched[i] == pytest.approx(reference, abs=1e-9, nan_ok=True), i
    assert batched[0] == pytest.approx(1.0)


def test_the_batched_correlation_handles_an_empty_and_a_mismatched_stack():
    from quantaalpha_us.factors.factor_research import mean_daily_rank_correlation_many

    a = np.zeros((5, 4), dtype=np.float32)
    assert mean_daily_rank_correlation_many(a, np.zeros((0, 5, 4), np.float32)).shape == (0,)
    mismatched = mean_daily_rank_correlation_many(a, np.zeros((2, 9, 9), np.float32))
    assert np.isnan(mismatched).all()


def test_a_published_alpha_is_not_compared_against_itself(gates):
    """The Alpha101 baseline reported a median shared subtree of 16 nodes and a
    median edit distance of 0, which only said that a formula matches itself."""
    published = "-1 * TS_CORR(CS_RANK($open), CS_RANK($volume), 10)"
    shared, distance = gates.nearest_alpha101_excluding(published)
    assert distance > 0, "a published alpha matched itself"
    assert shared < at.size(at.parse(published))
    # and the self-inclusive version still reports the trivial self-match
    assert gates.max_shared_alpha101(published)[0] == at.size(at.parse(published))
    assert gates.nearest_alpha101_edit(published)[0] == 0


# ---- rejection accounting ------------------------------------------------


def _store_with(tmp_path, *, candidates, gate_rejections, rounds):
    from quantaalpha_us.evo.persistence import RunStore

    store = RunStore(tmp_path)
    for record in candidates:
        store.record_candidate(record)
    for record in gate_rejections:
        store.record_rejection(record)
    for record in rounds:
        store.record_round(record)
    return store


def test_archive_refusals_are_counted_even_though_no_gate_logged_them(tmp_path):
    """A candidate that passed every gate, was scored, and was still turned away
    because its niche was full was only ever counted in memory. Reading
    rejections.jsonl alone understated the tally by 5 to 38 per arm."""
    from quantaalpha_us.evo.report import rejection_tally

    store = _store_with(
        tmp_path,
        candidates=[{"round": 1, "id": "a", "expression": "$close"},
                    {"round": 1, "id": "b", "expression": "$open"}],
        gate_rejections=[{"round": 1, "gate": "coverage", "expression": "x"},
                         {"round": 1, "gate": "coverage", "expression": "y"},
                         {"round": 1, "gate": "cheap_screen", "expression": "z"}],
        rounds=[{"round": 1, "rejections": {"coverage": 2, "cheap_screen": 1,
                                            "archive:niche_full_and_not_better": 4}}],
    )
    gates, archive = rejection_tally(store)
    assert gates == {"coverage": 2, "cheap_screen": 1}
    assert archive == {"archive:niche_full_and_not_better": 4}
    assert sum(gates.values()) + sum(archive.values()) == 7


def test_the_two_kinds_of_rejection_are_not_merged(tmp_path):
    """They answer different questions, and merging them would also imply an
    archive refusal was an extra proposal when it is a scored candidate."""
    from quantaalpha_us.evo.report import candidates_generated, rejection_tally

    store = _store_with(
        tmp_path,
        candidates=[{"round": 1, "id": str(i), "expression": "$close"} for i in range(5)],
        gate_rejections=[{"round": 1, "gate": "complexity", "expression": "x"}],
        rounds=[{"round": 1, "rejections": {"complexity": 1,
                                            "archive:niche_full_and_not_better": 3}}],
    )
    gates, archive = rejection_tally(store)
    assert sum(gates.values()) == 1
    assert sum(archive.values()) == 3
    # the three refused candidates were scored, so they are inside `generated`
    # already; counting them again would make generated 9 instead of 6
    assert candidates_generated(store) == 6


def test_rounds_without_a_rejections_block_do_not_break_the_tally(tmp_path):
    from quantaalpha_us.evo.report import rejection_tally

    store = _store_with(tmp_path, candidates=[], gate_rejections=[],
                        rounds=[{"round": 1}, {"round": 2, "rejections": None}])
    assert rejection_tally(store) == ({}, {})


def test_every_real_arm_has_archive_refusals_that_only_rounds_jsonl_knows(tmp_path):
    """Applied uniformly: every arm in this experiment has some, and none of
    them appear in rejections.jsonl."""
    from pathlib import Path

    from quantaalpha_us.evo.persistence import RunStore
    from quantaalpha_us.evo.report import rejection_tally

    runs = Path(__file__).resolve().parent.parent / "data" / "evo_runs"
    arms = [d for d in sorted(runs.glob("*"))
            if d.is_dir() and not d.name.endswith("-raw") and (d / "rounds.jsonl").exists()]
    if not arms:
        pytest.skip("no run directories present")
    total_archive = 0
    for arm in arms:
        gates, archive = rejection_tally(RunStore(arm))
        total_archive += sum(archive.values())
        # The invariant that matters, and it must hold for EVERY arm including
        # one that is mid-run: an archive refusal must never also be in
        # rejections.jsonl, or the tally counts it twice.
        assert not any(key.startswith("archive:") for key in gates), (
            f"{arm.name} has archive refusals in rejections.jsonl, so they would double count"
        )
    # Asserted over the set rather than per arm: a run still in progress has a
    # partially rebuilt rounds.jsonl, and this test should not fail for that.
    assert total_archive > 0, "no arm logged an archive refusal, so the tally proves nothing"
