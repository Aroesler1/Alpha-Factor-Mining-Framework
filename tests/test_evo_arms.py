"""The control arms, and the dry run that has to precede any paid call.

B9 and B10d. The one-shot and scalar-feedback arms exist to attribute a gain:
one-shot removes iteration while keeping the model, scalar feedback removes the
diagnostics while keeping iteration. Both must run through the SAME gates,
fitness and archive as the full loop, or a difference between them measures the
scaffolding instead of the thing being ablated.
"""

from __future__ import annotations

import pytest

from quantaalpha_us.evo import operators as ops
from quantaalpha_us.evo.backends import GPBackend, MockBackend, dry_run_commands
from quantaalpha_us.evo.persistence import RunStore
from quantaalpha_us.evo.runner import EvolutionRunner
from quantaalpha_us.evo.scoring import PanelScorer
from quantaalpha_us.factors import alpha101
from tests.synthetic_panel import planted_bars_with_fundamentals
from tests.test_evo_end_to_end import make_config


@pytest.fixture(scope="module")
def bars():
    return planted_bars_with_fundamentals(n_days=500, n_symbols=50)


@pytest.fixture(scope="module")
def alphas():
    return [a.expression for a in alpha101.load()]


def _runner(tmp_path, bars, alphas, config, backend, name="arm"):
    store = RunStore(tmp_path / name)
    return EvolutionRunner(config, PanelScorer(bars, config), backend, store,
                           alpha101=alphas, verbose=False), store


# ---- one-shot ------------------------------------------------------------


def test_one_shot_asks_for_a_full_loops_worth_of_candidates_in_one_call(tmp_path, bars, alphas):
    config = make_config(mode="one_shot")
    runner, store = _runner(tmp_path, bars, alphas, config, MockBackend(seed=3))
    runner._seed_islands()
    requests = runner._requests_for(runner.states[0], 1)
    assert len(requests) == 1 and requests[0].operator == ops.EXPLORE
    schedule = config.schedule
    expected = schedule.rounds * (schedule.explore_n + schedule.mutate_n
                                  + schedule.crossover_n + schedule.simplify_n)
    assert requests[0].n == expected == config.one_shot_candidates
    assert f"Propose {expected} NEW factor" in requests[0].prompt


def test_one_shot_makes_exactly_one_call_per_island_and_no_reflection(tmp_path, bars, alphas):
    config = make_config(mode="one_shot")
    backend = MockBackend(seed=3)
    runner, store = _runner(tmp_path, bars, alphas, config, backend)
    runner.run()
    keys = set(store.all_responses())
    assert {op for _, _, op in keys} == {ops.EXPLORE}
    assert len({island for _, island, _ in keys}) == len(config.islands)


def test_one_shot_runs_the_same_gates_as_the_loop(tmp_path, bars, alphas):
    """If the arms did not share gates, a difference between them would measure
    the gates rather than the proposer."""
    config = make_config(mode="one_shot")
    runner, store = _runner(tmp_path, bars, alphas, config, MockBackend(seed=3))
    runner.run()
    assert store.rejections(), "the one-shot arm rejected nothing, so it is not gated"
    gates = {r["gate"] for r in store.rejections()}
    assert gates & {"cheap_screen", "coverage", "complexity", "sanitizer",
                    "correlation", "alpha101_subtree", "archive_subtree"}


def test_one_shot_budget_is_one_call_per_island(tmp_path, bars, alphas):
    from quantaalpha_us.llm.budget import RunBudget

    config = make_config(mode="one_shot")
    store = RunStore(tmp_path / "tight")
    # exactly enough is fine; one fewer is not
    EvolutionRunner(config, PanelScorer(bars, config), MockBackend(seed=1), store,
                    alpha101=alphas, budget=RunBudget(max_requests=len(config.islands)),
                    verbose=False)
    with pytest.raises(ValueError):
        EvolutionRunner(config, PanelScorer(bars, config), MockBackend(seed=1), store,
                        alpha101=alphas,
                        budget=RunBudget(max_requests=len(config.islands) - 1),
                        verbose=False)


# ---- feedback ablation ---------------------------------------------------


def test_scalar_feedback_shows_the_number_and_nothing_else(tmp_path, bars, alphas):
    config = make_config(feedback_mode="scalar")
    runner, store = _runner(tmp_path, bars, alphas, config, MockBackend(seed=3))
    runner.run()
    mutate_prompts = [r["prompt"] for key, records in store.all_responses().items()
                      for r in records if key[2] == ops.MUTATE]
    assert mutate_prompts, "no MUTATE calls happened, so the ablation is untested"
    for prompt in mutate_prompts:
        assert "fitness" in prompt
        # the labels the full feedback block prints its numbers under; none of
        # them may survive into the ablation arm
        for absent in ("fit-window mean IC", "IC by year", "IC half-life",
                       "nearest in archive", "nearest published", "sign as fitted",
                       "daily turnover"):
            assert absent not in prompt, absent
        assert "not given per-year" in prompt, (
            "the ablation prompt still tells the model to read diagnostics it "
            "was not given"
        )


def test_full_feedback_shows_the_diagnostics(tmp_path, bars, alphas):
    config = make_config(feedback_mode="full")
    runner, store = _runner(tmp_path, bars, alphas, config, MockBackend(seed=3))
    runner.run()
    mutate_prompts = [r["prompt"] for key, records in store.all_responses().items()
                      for r in records if key[2] == ops.MUTATE]
    assert mutate_prompts
    joined = "\n".join(mutate_prompts)
    for needle in ("IC by year", "half-life", "coverage", "turnover"):
        assert needle in joined, needle


# ---- the GP control ------------------------------------------------------


def test_the_gp_arm_is_labelled_distinctly_in_the_manifest(tmp_path, bars, alphas):
    """A control reported as a model run would invalidate the comparison, so the
    two backends are separate classes and the manifest records which ran."""
    import json

    config = make_config(arm="gp")
    runner, store = _runner(tmp_path, bars, alphas, config, GPBackend(seed=3))
    runner.run()
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "gp"
    assert manifest["model"] == "gp-random-grammar"


def test_the_gp_arm_fills_an_archive_without_any_model(tmp_path, bars, alphas):
    config = make_config(arm="gp")
    runner, _ = _runner(tmp_path, bars, alphas, config, GPBackend(seed=3))
    result = runner.run()
    assert len(result.archive) > 0
    assert all(e.operator in ("seed", ops.EXPLORE, ops.MUTATE, ops.CROSSOVER, ops.SIMPLIFY)
               for e in result.archive.members())


# ---- dry run -------------------------------------------------------------


def test_dry_run_renders_a_command_line_per_request():
    """The command lines are what a reviewer checks before any tokens are spent:
    the model, the effort, the schema and the prompt, exactly as sent."""
    class FakeCLI:
        name = "claude-code"
        model = "claude-sonnet-5"
        effort = "max"

        def command(self, request):
            return ["claude", "-p", request.prompt, "--model", self.model,
                    "--effort", self.effort, "--output-format", "json",
                    "--json-schema", "{}"]

        def call(self, request):  # pragma: no cover - never called in a dry run
            raise AssertionError("a dry run must not call the model")

    request = ops.OperatorRequest(ops.EXPLORE, "a", 1, "propose factors",
                                  ops.CANDIDATE_SCHEMA, 10)
    lines = dry_run_commands(FakeCLI(), [request])
    assert len(lines) == 1
    assert "--effort max" in lines[0]
    assert "--model claude-sonnet-5" in lines[0]
    assert "'propose factors'" in lines[0]


def test_dry_run_on_an_offline_backend_says_so_instead_of_inventing_a_command():
    request = ops.OperatorRequest(ops.EXPLORE, "a", 1, "p", ops.CANDIDATE_SCHEMA, 3)
    lines = dry_run_commands(MockBackend(seed=0), [request])
    assert "offline backend" in lines[0]


def test_the_stop_message_names_the_limit_that_was_hit(tmp_path, bars, alphas):
    """"Budget exhausted" was reported for a run that had used 107 of a million
    allowed requests; the real cause was three consecutive failures, and the
    message pointed the diagnosis the wrong way."""
    from quantaalpha_us.evo.backends import BackendReply, MockBackend
    from quantaalpha_us.llm.budget import RunBudget

    class AlwaysFails(MockBackend):
        def call(self, request):
            self.calls += 1
            return BackendReply(payload={}, error="simulated upstream failure")

    config = make_config()
    store = RunStore(tmp_path / "failing")
    runner = EvolutionRunner(
        config, PanelScorer(bars, config), AlwaysFails(seed=1), store,
        alpha101=alphas,
        budget=RunBudget(max_requests=10**6, max_total_tokens=10**9,
                         max_consecutive_failures=3),
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="consecutive failed calls"):
        runner.run()

    # The constructor refuses a cap below the planned count, so the request-cap
    # message is reached by lowering the cap after the run has started -- which
    # is what a subscription cap looks like from inside a run.
    tight = RunStore(tmp_path / "tight")
    runner = EvolutionRunner(
        config, PanelScorer(bars, config), MockBackend(seed=1), tight,
        alpha101=alphas, verbose=False,
    )
    runner.budget.max_requests = 3
    with pytest.raises(RuntimeError, match="request cap reached"):
        runner.run()


def test_one_shot_runs_exactly_one_round(tmp_path, bars, alphas):
    """Regression, and an expensive one. One-shot skipped the reflection call
    but still iterated all 8 rounds, so it made 3 calls a round until the budget
    stopped it -- at roughly 20 minutes and several dollars a call."""
    config = make_config(mode="one_shot")
    assert config.effective_rounds == 1
    assert config.schedule.rounds > 1, "the schedule still sizes the one-shot batch"

    backend = MockBackend(seed=3)
    runner, store = _runner(tmp_path, bars, alphas, config, backend, name="oneshot")
    result = runner.run()
    assert len(result.rounds) == 1
    assert backend.calls == len(config.islands)
    assert {key[0] for key in store.all_responses()} == {1}


def test_the_loop_mode_still_iterates_its_full_schedule(tmp_path, bars, alphas):
    config = make_config()
    assert config.effective_rounds == config.schedule.rounds == 2
