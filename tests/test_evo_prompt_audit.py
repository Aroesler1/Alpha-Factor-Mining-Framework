"""The wall between the model and everything after 2013.

B1 and B10e. The experiment's whole claim is that the loop's factors were
selected without seeing the validation or holdout windows, so this is the test
that claim rests on. It scans every prompt the loop actually wrote to disk, not
just the templates, because a template can be clean and a filled prompt can
still carry a year through a feedback block or an archive summary.

Three things are checked:

1. no prompt mentions a year later than the configured fit end;
2. no prompt contains the string "2018" -- the first holdout year, called out
   separately because it is the one number a reader will look for;
3. no prompt contains any number the runner computed on the validation window,
   at any of the precisions those numbers are ever printed with.

Plus a structural check: the module that builds feedback blocks does not import
the scorer, so there is no code path from a prompt to a validation number even
if someone adds a field later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from quantaalpha_us.evo import feedback as feedback_module
from quantaalpha_us.evo.backends import MockBackend
from quantaalpha_us.evo.operators import PROMPT_DIR
from quantaalpha_us.evo.persistence import RunStore
from quantaalpha_us.evo.runner import EvolutionRunner
from quantaalpha_us.evo.scoring import PanelScorer
from quantaalpha_us.factors import alpha101
from tests.synthetic_panel import PLANTED_EXPRESSION, planted_bars_with_fundamentals
from tests.test_evo_end_to_end import make_config

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    bars = planted_bars_with_fundamentals(n_days=500, n_symbols=50)
    alphas = [a.expression for a in alpha101.load()]
    config = make_config()
    store = RunStore(tmp_path_factory.mktemp("audit") / "mock")
    runner = EvolutionRunner(config, PanelScorer(bars, config),
                             MockBackend(seed=3, seed_expressions=(PLANTED_EXPRESSION,)),
                             store, alpha101=alphas, verbose=False)
    runner.run()
    return runner, store, config


def all_prompts(store: RunStore) -> list[tuple[tuple, str]]:
    out = []
    for key, records in store.all_responses().items():
        for record in records:
            out.append((key, record["prompt"]))
    return out


def test_the_run_actually_wrote_prompts(finished_run):
    _, store, _ = finished_run
    prompts = all_prompts(store)
    assert len(prompts) >= 8, "too few prompts to be auditing anything"


def test_no_prompt_mentions_a_year_after_the_fit_window(finished_run):
    _, store, config = finished_run
    limit = config.windows.latest_visible_year()
    for key, prompt in all_prompts(store):
        years = {int(y) for y in YEAR.findall(prompt)}
        late = {y for y in years if y > limit}
        assert not late, f"{key} mentions {sorted(late)}, past the fit end {limit}"


def test_no_prompt_contains_the_first_holdout_year(finished_run):
    _, store, _ = finished_run
    for key, prompt in all_prompts(store):
        assert "2018" not in prompt, f"{key} contains the string 2018"


def test_no_prompt_contains_a_validation_number(finished_run):
    """The early-stop rule reads validation ICs. If one of them ever reached a
    prompt, the loop would be selecting on validation and the holdout would be
    the only clean window left."""
    runner, store, _ = finished_run
    values = [v for v in runner._validation_ic.values() if v == v]
    assert values, "the run computed no validation numbers, so this proves nothing"
    prompts = [p for _, p in all_prompts(store)]
    # Five decimals and up. Below that the check stops being a fingerprint and
    # starts being a coincidence: prompts legitimately print fit-window IC,
    # fitness and penalty figures, and "0.003" collides with a turnover charge
    # rather than proving a leak. Feedback prints ICs at five decimals, so a
    # real leak of an IC-shaped number is caught at exactly this precision.
    for value in values:
        for digits in (5, 6, 8):
            rendered = f"{value:.{digits}f}"
            if float(rendered) == 0.0:
                continue
            for prompt in prompts:
                assert rendered not in prompt, (
                    f"a validation IC ({rendered}) appears in a prompt"
                )


def test_the_prompt_templates_on_disk_are_clean():
    for path in sorted(Path(PROMPT_DIR).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        years = {int(y) for y in YEAR.findall(text)}
        assert not years, (
            f"{path.name} hard-codes the year(s) {sorted(years)}; the fit window "
            "must come from the config so a short run cannot advertise a long one"
        )
        assert "2018" not in text


def test_the_feedback_builder_imports_nothing_that_can_see_validation():
    """Structural, not behavioural: if `feedback` imported the scorer, a future
    field could pull a validation number into a prompt and every test above
    would still pass until someone did. Checked on the import statements rather
    than on the file text, so the module can talk about the rule in its prose.
    """
    import ast

    source = Path(feedback_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    forbidden = {"quantaalpha_us.evo.scoring.PanelScorer",
                 "quantaalpha_us.evo.runner",
                 "quantaalpha_us.factors.factor_research"}
    assert not (imported & forbidden), sorted(imported & forbidden)
    # CandidateMetrics is allowed -- it is the fit-window-only carrier -- and it
    # must not have grown a field from a later window
    from quantaalpha_us.evo.scoring import CandidateMetrics

    fields = set(CandidateMetrics.__dataclass_fields__)
    assert not {f for f in fields if "valid" in f or "holdout" in f}


def test_the_archive_records_carry_no_validation_number(finished_run):
    runner, store, _ = finished_run
    for record in store.archive_records():
        keys = set(record.get("metrics", {}))
        assert not {k for k in keys if "valid" in k or "holdout" in k}
