"""Append-only experiment trace for factor mining.

A mining loop that keeps only its winners cannot be audited. The rejected
hypotheses are the part that shows whether a search was disciplined or whether
it simply ran until something passed, and they are exactly what gets discarded
when results are written as a leaderboard.

This implements the trace the constrained-agent literature argues for
(arXiv 2604.26747): every proposed hypothesis is recorded with the reasoning
that produced it, the expression it became, the scores it earned, and the
verdict -- including, and especially, the failures. Two properties make it
useful rather than decorative:

- **Append-only.** Records are written as JSON Lines and never rewritten. A
  trace that can be edited after the fact provides no evidence about what was
  tried, so `append` refuses to modify existing lines and there is no update or
  delete operation.
- **Content-addressed.** Each record carries a hash of the normalised
  expression, so the same idea proposed twice is detectable even when it is
  spelled differently across runs. That is what makes the trace usable as
  search memory rather than just a log.

The trace is also the multiple-testing denominator. A DSR computed against the
factors that survived understates the search; computed against every hypothesis
the trace has ever recorded, it reflects what was actually tried.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

Verdict = Literal["proposed", "rejected_sanitizer", "rejected_evaluator",
                  "rejected_score", "rejected_correlated", "selected"]

_WHITESPACE = re.compile(r"\s+")


def normalise_expression(expression: str) -> str:
    """Whitespace- and case-normalised form used for identity.

    Deliberately conservative: it collapses spacing and lowercases, but does not
    attempt algebraic equivalence. Two expressions that differ only in spacing
    are the same idea; two that differ in structure are treated as distinct even
    if they might reduce to the same signal, because claiming otherwise would
    require a solver this does not have.
    """
    return _WHITESPACE.sub(" ", str(expression).strip()).lower()


def expression_id(expression: str) -> str:
    return hashlib.sha256(normalise_expression(expression).encode("utf-8")).hexdigest()[:16]


@dataclass
class TraceRecord:
    expression: str
    verdict: Verdict
    hypothesis: str = ""
    run_id: str = ""
    model: str = ""
    mean_ic: float | None = None
    ic_tstat: float | None = None
    signal_autocorr: float | None = None
    coverage: float | None = None
    error: str | None = None
    expression_id: str = ""
    recorded_at: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.expression_id:
            self.expression_id = expression_id(self.expression)
        if not self.recorded_at:
            self.recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExperimentTrace:
    """Append-only JSON Lines trace of every hypothesis considered."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TraceRecord) -> TraceRecord:
        """Write one record. Existing lines are never touched."""
        # opened in append mode with an explicit flush so a crashed run still
        # leaves behind everything it had already decided
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def append_many(self, records: list[TraceRecord]) -> int:
        for record in records:
            self.append(record)
        return len(records)

    def __iter__(self) -> Iterator[TraceRecord]:
        if not self.path.exists():
            return iter(())
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(TraceRecord(**json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    # a corrupt line must not hide the rest of the history
                    continue
        return iter(records)

    def records(self) -> list[TraceRecord]:
        return list(self)

    def seen(self) -> set[str]:
        """Expression ids already tried, in any run and with any verdict."""
        return {r.expression_id for r in self}

    def is_new(self, expression: str) -> bool:
        return expression_id(expression) not in self.seen()

    def n_hypotheses(self) -> int:
        """Distinct expressions ever proposed.

        This is the multiple-testing denominator. Counting only the survivors
        understates the search by exactly the number of failures, which is the
        error the trace exists to prevent.
        """
        return len(self.seen())

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self:
            counts[record.verdict] = counts.get(record.verdict, 0) + 1
        counts["distinct_expressions"] = self.n_hypotheses()
        return counts
