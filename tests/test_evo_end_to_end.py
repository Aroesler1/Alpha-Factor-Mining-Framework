"""The loop, end to end, on a synthetic panel with a known answer.

This is the test that has to exist before any paid call: it runs two full rounds
through the real gates, the real fitness, the real archive and the real
persistence layer, using a mock backend, and asserts that the factor actually
planted in the panel comes out the other end.

A test that only asserted "the loop ran" would pass with every gate inverted.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from quantaalpha_us.evo.archive import ALL_NICHES
from quantaalpha_us.evo.backends import MockBackend
from quantaalpha_us.evo.config import EvoConfig, Island, Schedule, Windows
from quantaalpha_us.evo.persistence import RunStore, token_totals
from quantaalpha_us.evo.report import write_run_report
from quantaalpha_us.evo.runner import EvolutionRunner
from quantaalpha_us.evo.scoring import PanelScorer, candidate_id
from quantaalpha_us.factors import alpha101
from tests.synthetic_panel import PLANTED_EXPRESSION, planted_bars_with_fundamentals


# The synthetic panel is 500 business days from 2010-01-04, so the three windows
# are compressed to fit it. Everything else is the production configuration.
TEST_WINDOWS = Windows(
    fit_start="2010-01-01", fit_end="2011-03-31",
    validation_start="2011-04-01", validation_end="2011-08-31",
    holdout_start="2011-09-01", holdout_end="2011-12-31",
    cheap_start="2011-01-01",
)

TEST_ISLANDS = (
    Island(name="price_trend",
           direction="price trend and reversal",
           seeds=("TS_DELTA($close, 21) / (TS_STD($close, 21) + 1e-8)",
                  "($close - TS_MIN($low, 21)) / (TS_MAX($high, 21) - TS_MIN($low, 21) + 1e-8)")),
    Island(name="liquidity_volume",
           direction="liquidity and volume",
           seeds=("TS_MEAN($dollar_volume, 21)", "-TS_CORR($return, $volume, 10)")),
)


def make_config(**overrides) -> EvoConfig:
    schedule = Schedule(rounds=2, population=12, elite=3, explore_n=6, mutate_n=4,
                        crossover_n=3, simplify_n=2, archive_per_niche=4)
    base = dict(arm="mock", model="mock", effort="low", seed=7,
                windows=TEST_WINDOWS, schedule=schedule, islands=TEST_ISLANDS,
                correlation_date_stride=3, bootstrap_draws=40)
    base.update(overrides)
    return EvoConfig(**base)


@pytest.fixture(scope="module")
def bars():
    return planted_bars_with_fundamentals(n_days=500, n_symbols=50)


@pytest.fixture(scope="module")
def alphas():
    return [a.expression for a in alpha101.load()]


def build_runner(tmp_path, bars, alphas, *, config=None, backend=None, panels=None):
    config = config or make_config()
    scorer = PanelScorer(bars, config, panels=panels)
    backend = backend or MockBackend(seed=3, seed_expressions=(PLANTED_EXPRESSION,))
    store = RunStore(tmp_path / config.arm)
    return EvolutionRunner(config, scorer, backend, store, alpha101=alphas, verbose=False), store


def test_two_mock_rounds_find_the_planted_factor(tmp_path, bars, alphas):
    runner, store = build_runner(tmp_path, bars, alphas)
    result = runner.run()

    assert len(result.archive) > 0, "the archive is empty; nothing survived the gates"
    expressions = {e.expression for e in result.archive.members()}
    assert PLANTED_EXPRESSION in expressions, (
        "the factor the panel was built around did not reach the archive"
    )

    planted = result.archive.entries[candidate_id(PLANTED_EXPRESSION)]
    assert planted.metrics["mean_ic"] > 0.03
    # and it should be near the top: the panel has one real signal in it
    assert planted.fitness >= result.archive.top(3)[-1].fitness

    report = write_run_report(result.archive, store)
    assert (store.root / "archive_table.csv").exists()
    assert (store.root / "rounds_table.csv").exists()
    assert report["niches_possible"] == len(ALL_NICHES)
    # the reported archive counts DISCOVERIES; the island seeds are admitted so
    # the paraphrase and correlation gates can see them, but they were not found
    # by anything and every arm starts from the same ones
    assert report["archive_size"] == len(result.archive.discovered())
    assert report["archive_size_with_seeds"] == len(result.archive)
    seeds = [e for e in result.archive.members() if e.operator == "seed"]
    assert seeds, "no seed reached the archive, so the exclusion is untested"
    assert all(e.operator != "seed" for e in result.archive.top(5, exclude_seeds=True))


def test_every_raw_response_is_on_disk_with_its_prompt(tmp_path, bars, alphas):
    runner, store = build_runner(tmp_path, bars, alphas)
    runner.run()
    responses = store.all_responses()
    assert responses, "no responses were persisted"
    for (round_index, island, operator), records in responses.items():
        assert round_index >= 1 and island and operator
        for record in records:
            assert record["prompt"], "a response was saved without the prompt that caused it"
            assert record["prompt_hash"]
            assert "payload" in record
    assert token_totals(store)["calls"] == runner.calls


def test_rejections_are_logged_with_a_gate_and_a_reason(tmp_path, bars, alphas):
    runner, store = build_runner(tmp_path, bars, alphas)
    runner.run()
    rejections = store.rejections()
    assert rejections, "a random proposer produced nothing the gates rejected, which is wrong"
    gates = {r["gate"] for r in rejections}
    assert gates - {""} == gates
    for record in rejections:
        assert record["expression"]
        assert record["gate"]
    # the cheap screen and the structural gates should both be doing work
    assert len(gates) >= 2


def test_the_archive_respects_its_niche_capacity(tmp_path, bars, alphas):
    config = make_config()
    runner, _ = build_runner(tmp_path, bars, alphas, config=config)
    result = runner.run()
    for niche in ALL_NICHES:
        assert len(result.archive.in_niche(niche)) <= config.schedule.archive_per_niche


def test_two_runs_with_the_same_seed_produce_identical_archives(tmp_path, bars, alphas):
    """B10f. Determinism is what makes replay and resume meaningful; without it
    a difference between two runs proves nothing about either."""
    first, _ = build_runner(tmp_path / "a", bars, alphas,
                            backend=MockBackend(seed=3, seed_expressions=(PLANTED_EXPRESSION,)))
    second, _ = build_runner(tmp_path / "b", bars, alphas,
                             backend=MockBackend(seed=3, seed_expressions=(PLANTED_EXPRESSION,)))
    a = first.run().archive.snapshot()
    b = second.run().archive.snapshot()
    assert a == b


def test_a_different_seed_produces_a_different_archive(tmp_path, bars, alphas):
    first, _ = build_runner(tmp_path / "a", bars, alphas, backend=MockBackend(seed=3))
    second, _ = build_runner(tmp_path / "b", bars, alphas, backend=MockBackend(seed=99))
    assert first.run().archive.snapshot() != second.run().archive.snapshot()


def test_the_manifest_records_what_produced_the_run(tmp_path, bars, alphas):
    runner, store = build_runner(tmp_path, bars, alphas)
    runner.run()
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "mock"
    assert manifest["config"]["seed"] == 7
    assert manifest["config"]["windows"]["fit_end"] == "2011-03-31"
    assert manifest["config"]["gates"]["max_shared_subtree"] == 5


def test_a_budget_below_the_planned_call_count_refuses_to_start(tmp_path, bars, alphas):
    """B10d. Starting a run that cannot finish wastes every call it does make."""
    from quantaalpha_us.llm.budget import RunBudget

    config = make_config()
    scorer = PanelScorer(bars, config)
    store = RunStore(tmp_path / "short")
    with pytest.raises(ValueError, match="cannot finish|needs"):
        EvolutionRunner(config, scorer, MockBackend(seed=1), store, alpha101=alphas,
                        budget=RunBudget(max_requests=2), verbose=False)
