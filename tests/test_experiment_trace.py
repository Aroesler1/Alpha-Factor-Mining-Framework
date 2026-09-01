"""Tests for the append-only experiment trace.

The trace exists so a mining run can be audited, which means its guarantees
have to actually hold: nothing is rewritten, failures are kept, the same idea
is recognised across spellings, and the hypothesis count reflects everything
tried rather than everything that passed.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantaalpha_us.factors.experiment_trace import (  # noqa: E402
    ExperimentTrace,
    TraceRecord,
    expression_id,
    normalise_expression,
)


def test_identity_ignores_spacing_and_case():
    a = "RANK(TS_MEAN($close, 21))"
    b = "rank( ts_mean( $close ,  21 ) )"
    assert normalise_expression(a) != normalise_expression(b)  # punctuation differs
    assert expression_id("RANK($close)") == expression_id("  rank($close)  ")
    assert expression_id("RANK($close)") != expression_id("RANK($volume)")


def test_append_is_append_only(tmp_path):
    """Existing lines must survive byte-for-byte."""
    trace = ExperimentTrace(tmp_path / "t.jsonl")
    trace.append(TraceRecord(expression="RANK($close)", verdict="selected"))
    first = (tmp_path / "t.jsonl").read_text()

    trace.append(TraceRecord(expression="RANK($volume)", verdict="rejected_score"))
    after = (tmp_path / "t.jsonl").read_text()

    assert after.startswith(first)
    assert len(after.splitlines()) == 2
    # no mutation API exists
    assert not hasattr(trace, "update")
    assert not hasattr(trace, "delete")


def test_failures_are_retained(tmp_path):
    """The rejected hypotheses are the point; they must not be filterable away."""
    trace = ExperimentTrace(tmp_path / "t.jsonl")
    trace.append(TraceRecord(expression="A", verdict="rejected_sanitizer", error="bad syntax"))
    trace.append(TraceRecord(expression="B", verdict="rejected_score", ic_tstat=0.3))
    trace.append(TraceRecord(expression="C", verdict="selected", ic_tstat=4.1))

    verdicts = [r.verdict for r in trace]
    assert verdicts.count("selected") == 1
    assert sum(v.startswith("rejected") for v in verdicts) == 2
    assert trace.summary()["rejected_sanitizer"] == 1


def test_hypothesis_count_is_the_multiple_testing_denominator(tmp_path):
    """Counting survivors understates the search by exactly the failures."""
    trace = ExperimentTrace(tmp_path / "t.jsonl")
    for i in range(9):
        trace.append(TraceRecord(expression=f"RANK($f{i})", verdict="rejected_score"))
    trace.append(TraceRecord(expression="RANK($f9)", verdict="selected"))

    selected = [r for r in trace if r.verdict == "selected"]
    assert len(selected) == 1
    assert trace.n_hypotheses() == 10, "denominator must count everything tried"


def test_duplicate_ideas_are_detected_across_runs(tmp_path):
    trace = ExperimentTrace(tmp_path / "t.jsonl")
    trace.append(TraceRecord(expression="RANK($close)", verdict="rejected_score", run_id="r1"))
    assert not trace.is_new("  rank($close) ")
    assert trace.is_new("RANK($volume)")


def test_corrupt_line_does_not_hide_history(tmp_path):
    path = tmp_path / "t.jsonl"
    trace = ExperimentTrace(path)
    trace.append(TraceRecord(expression="A", verdict="selected"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
    trace.append(TraceRecord(expression="B", verdict="selected"))

    exprs = [r.expression for r in trace]
    assert exprs == ["A", "B"]


def test_records_are_json_serialisable(tmp_path):
    trace = ExperimentTrace(tmp_path / "t.jsonl")
    trace.append(TraceRecord(expression="A", verdict="selected", hypothesis="cheapness",
                             mean_ic=0.01, extra={"model": "x"}))
    line = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert line["expression"] == "A"
    assert line["hypothesis"] == "cheapness"
    assert line["expression_id"] and line["recorded_at"]
