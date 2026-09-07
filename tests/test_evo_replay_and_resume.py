"""Replay and resume: the two properties that make a paid run recoverable.

B10b and B10c. Both matter for the same reason: an eight-round run at max effort
is roughly 120 calls, and losing it to a parse error, a subscription cap or a
laptop lid is not an acceptable failure mode. Replay proves the saved responses
are sufficient to rebuild the whole result with zero model calls; resume proves
a run killed part-way through a round comes back to a clean state rather than a
corrupted one.
"""

from __future__ import annotations

import json

import pytest

from quantaalpha_us.evo.backends import MockBackend, ReplayBackend
from quantaalpha_us.evo.persistence import RunStore
from quantaalpha_us.evo.runner import EvolutionRunner
from quantaalpha_us.evo.scoring import PanelScorer
from quantaalpha_us.factors import alpha101
from tests.synthetic_panel import PLANTED_EXPRESSION, planted_bars_with_fundamentals
from tests.test_evo_end_to_end import make_config


@pytest.fixture(scope="module")
def bars():
    return planted_bars_with_fundamentals(n_days=500, n_symbols=50)


@pytest.fixture(scope="module")
def alphas():
    return [a.expression for a in alpha101.load()]


def _runner(tmp_path, bars, alphas, *, backend, name="run", config=None):
    config = config or make_config()
    scorer = PanelScorer(bars, config)
    store = RunStore(tmp_path / name)
    return EvolutionRunner(config, scorer, backend, store, alpha101=alphas,
                           verbose=False), store


# ---- replay --------------------------------------------------------------


def test_replay_reproduces_the_archive_byte_for_byte(tmp_path, bars, alphas):
    live, store = _runner(tmp_path, bars, alphas,
                          backend=MockBackend(seed=3, seed_expressions=(PLANTED_EXPRESSION,)),
                          name="live")
    original = live.run().archive.snapshot()

    replay_backend = ReplayBackend.from_store(store)
    replayed, _ = _runner(tmp_path, bars, alphas, backend=replay_backend, name="replay")
    result = replayed.run()

    assert not replay_backend.missing, (
        f"replay asked for responses that were never saved: {replay_backend.missing}"
    )
    assert result.archive.snapshot() == original


def test_replay_makes_no_model_calls_beyond_the_saved_ones(tmp_path, bars, alphas):
    live, store = _runner(tmp_path, bars, alphas, backend=MockBackend(seed=5), name="live")
    live.run()
    saved = sum(len(v) for v in store.all_responses().values())

    replay_backend = ReplayBackend.from_store(store)
    replayed, _ = _runner(tmp_path, bars, alphas, backend=replay_backend, name="replay")
    replayed.run()
    assert replay_backend.calls == saved


def test_replay_reproduces_the_rejection_ledger(tmp_path, bars, alphas):
    live, store = _runner(tmp_path, bars, alphas, backend=MockBackend(seed=11), name="live")
    live.run()
    replayed, replay_store = _runner(tmp_path, bars, alphas,
                                     backend=ReplayBackend.from_store(store), name="replay")
    replayed.run()

    def key(records):
        return sorted((r["round"], r["island"], r["gate"], r["expression"])
                      for r in records)

    assert key(replay_store.rejections()) == key(store.rejections())


# ---- resume --------------------------------------------------------------


class ExplodingBackend(MockBackend):
    """A mock that dies part-way through a round, the way a real run dies."""

    fail_after: int = 10**9

    def call(self, request):
        if self.calls >= self.fail_after:
            raise KeyboardInterrupt("simulated kill mid-round")
        return super().call(request)


def test_a_run_killed_mid_round_resumes_from_a_clean_state(tmp_path, bars, alphas):
    config = make_config()
    scorer = PanelScorer(bars, config)
    store = RunStore(tmp_path / "killed")
    killer = ExplodingBackend(seed=3, seed_expressions=(PLANTED_EXPRESSION,))
    killer.fail_after = 4  # part-way through round 1, after some islands answered
    runner = EvolutionRunner(config, scorer, killer, store, alpha101=alphas, verbose=False)
    with pytest.raises(KeyboardInterrupt):
        runner.run()

    assert store.last_completed_round() == 0
    assert store.all_responses(), "the killed run saved nothing, so there is nothing to resume"

    resumed = EvolutionRunner(config, PanelScorer(bars, config),
                              MockBackend(seed=3, seed_expressions=(PLANTED_EXPRESSION,)),
                              store, alpha101=alphas, verbose=False)
    result = resumed.run(resume=True)
    assert len(result.rounds) == config.schedule.rounds
    assert len(result.archive) > 0
    # the partial round's responses were discarded rather than left to be read
    # twice: every (round, island, operator) key holds exactly one record
    for records in store.all_responses().values():
        assert len(records) == 1


def test_resuming_after_a_completed_round_keeps_that_round_and_continues(tmp_path, bars, alphas):
    one_round = make_config(schedule=make_config().schedule.__class__(
        rounds=1, population=12, elite=3, explore_n=6, mutate_n=4, crossover_n=3,
        simplify_n=2, archive_per_niche=4))
    scorer = PanelScorer(bars, one_round)
    store = RunStore(tmp_path / "partial")
    first = EvolutionRunner(one_round, scorer, MockBackend(seed=3), store,
                            alpha101=alphas, verbose=False)
    first.run()
    assert store.last_completed_round() == 1
    archive_after_one = {e.expression for e in first.archive.members()}

    two_rounds = make_config()
    resumed = EvolutionRunner(two_rounds, PanelScorer(bars, two_rounds),
                              MockBackend(seed=3), store, alpha101=alphas, verbose=False)
    result = resumed.run(resume=True)

    assert store.last_completed_round() == 2
    assert archive_after_one <= {e.expression for e in result.archive.members()}, (
        "resuming lost factors that round 1 had already admitted"
    )


def test_resume_restores_the_correlation_gate(tmp_path, bars, alphas):
    """The rank matrices are the one piece of state not on disk. A resume that
    forgot to rebuild them would leave the correlation gate passing everything
    for the rest of the run -- a silently missing gate, not a crash."""
    one_round = make_config(schedule=make_config().schedule.__class__(
        rounds=1, population=12, elite=3, explore_n=6, mutate_n=4, crossover_n=3,
        simplify_n=2, archive_per_niche=4))
    store = RunStore(tmp_path / "corr")
    first = EvolutionRunner(one_round, PanelScorer(bars, one_round), MockBackend(seed=3),
                            store, alpha101=alphas, verbose=False)
    first.run()

    two_rounds = make_config()
    resumed = EvolutionRunner(two_rounds, PanelScorer(bars, two_rounds),
                              MockBackend(seed=3), store, alpha101=alphas, verbose=False)
    resumed.run(resume=True)
    assert resumed.archive.ranks, "no rank matrices were rebuilt on resume"
    assert set(resumed.archive.ranks) >= {e.id for e in first.archive.members()
                                          if e.id in resumed.archive.entries}


def test_a_partly_finished_round_reuses_the_calls_it_already_paid_for(tmp_path, bars, alphas):
    """The expensive case. A round that dies half way through has already paid
    for the calls that succeeded; deleting them and asking again is the most
    costly possible response at 20 to 30 minutes and several dollars a call."""
    config = make_config()
    scorer = PanelScorer(bars, config)
    store = RunStore(tmp_path / "salvage")
    killer = ExplodingBackend(seed=3, seed_expressions=(PLANTED_EXPRESSION,))
    killer.fail_after = 4
    runner = EvolutionRunner(config, scorer, killer, store, alpha101=alphas, verbose=False)
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    paid_for = sum(len(v) for v in store.all_responses().values())
    assert paid_for >= 3, "the interrupted run saved too little to test salvage"

    resumed = EvolutionRunner(config, PanelScorer(bars, config),
                              MockBackend(seed=3, seed_expressions=(PLANTED_EXPRESSION,)),
                              store, alpha101=alphas, verbose=False)
    result = resumed.run(resume=True)
    assert resumed.reused_calls > 0, "no salvaged reply was reused"
    assert len(result.rounds) == config.schedule.rounds
    # replay integrity survives: still exactly one record per call
    for records in store.all_responses().values():
        assert len(records) == 1


def test_a_salvaged_reply_is_only_reused_for_an_identical_prompt(tmp_path, bars, alphas):
    """Reuse is keyed on the prompt hash, so a resumed run that asks a different
    question gets a real answer rather than a stale one."""
    config = make_config()
    store = RunStore(tmp_path / "hash")
    killer = ExplodingBackend(seed=3)
    killer.fail_after = 4
    runner = EvolutionRunner(config, PanelScorer(bars, config), killer, store,
                             alpha101=alphas, verbose=False)
    with pytest.raises(KeyboardInterrupt):
        runner.run()

    store.salvage_partial_rounds(store.last_completed_round())
    salvaged = store.salvaged_responses()
    assert salvaged, "nothing was salvaged"
    assert all(len(digest) == 16 for digest in salvaged), "salvage is keyed by prompt hash"

    # Rewrite every salvaged reply under a hash that answers no question the
    # resumed run will ask. Reuse must then be zero, and the run must complete
    # by calling the model instead of silently pairing a stale answer with a
    # prompt it never saw.
    for path in sorted(store.salvage_root.rglob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for record in records:
            record["prompt_hash"] = "deadbeefdeadbeef"
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))

    resumed = EvolutionRunner(config, PanelScorer(bars, config), MockBackend(seed=3),
                              store, alpha101=alphas, verbose=False)
    result = resumed.run(resume=True)
    assert resumed.reused_calls == 0, "a reply was reused for a prompt it did not answer"
    assert len(result.rounds) == config.schedule.rounds


def test_resuming_a_fully_saved_run_makes_no_model_calls_at_all(tmp_path, bars, alphas):
    """The property the whole salvage design rests on. If every reply is on
    disk, resuming must cost CPU and nothing else -- a single re-bought call
    here means the resumed run built a different prompt, which means its state
    diverged from the run it claims to be continuing."""
    config = make_config()
    store = RunStore(tmp_path / "full")
    first = EvolutionRunner(config, PanelScorer(bars, config),
                            MockBackend(seed=3, seed_expressions=(PLANTED_EXPRESSION,)),
                            store, alpha101=alphas, verbose=False)
    original = first.run().archive.snapshot()

    class NeverCall(MockBackend):
        def call(self, request):
            raise AssertionError(
                f"resume re-bought {request.operator}/{request.island} in round "
                f"{request.round}; its prompt did not match the saved one"
            )

    resumed = EvolutionRunner(config, PanelScorer(bars, config), NeverCall(seed=3),
                              store, alpha101=alphas, verbose=False)
    result = resumed.run(resume=True)
    assert resumed.reused_calls > 0
    assert result.archive.snapshot() == original, "resume did not reproduce the archive"
