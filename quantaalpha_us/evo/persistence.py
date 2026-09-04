"""Run directories: every raw response on disk before it is parsed, and resume.

The rule this module exists to enforce: a model call that has been paid for is
written to disk BEFORE anything tries to understand it. A parser that throws
after a 30k-token call at max effort has otherwise burned the call, and on a
subscription that is not a retryable failure.

Layout:

    runs/<arm>/manifest.json                     config, seed, model, effort
    runs/<arm>/<round>/<island>/<operator>.jsonl raw responses, one per call
    runs/<arm>/archive.jsonl                     every admission, in order
    runs/<arm>/rounds.jsonl                      per-round summary + the curve
    runs/<arm>/memory/<island>.json              the reflection rules
    runs/<arm>/rejections.jsonl                  every gate rejection with reason

`rounds.jsonl` is what --resume reads: a round appears there only after all of
its islands finished, so resuming re-does at most one round.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from quantaalpha_us.evo.backends import BackendReply
from quantaalpha_us.evo.operators import OperatorRequest


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                            default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


@dataclass
class RunStore:
    """One arm's directory."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # ---- layout ----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def archive_path(self) -> Path:
        return self.root / "archive.jsonl"

    @property
    def rounds_path(self) -> Path:
        return self.root / "rounds.jsonl"

    @property
    def rejections_path(self) -> Path:
        return self.root / "rejections.jsonl"

    @property
    def candidates_path(self) -> Path:
        return self.root / "candidates.jsonl"

    def memory_path(self, island: str) -> Path:
        return self.root / "memory" / f"{island}.json"

    def response_path(self, round_index: int, island: str, operator: str) -> Path:
        return self.root / str(round_index) / island / f"{operator}.jsonl"

    # ---- writes ----------------------------------------------------------

    def write_manifest(self, payload: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

    def record_response(self, request: OperatorRequest, reply: BackendReply) -> None:
        """Called immediately after the backend returns, before any parsing."""
        _append_jsonl(self.response_path(request.round, request.island, request.operator),
                      reply.to_record(request))

    def record_rejection(self, record: dict) -> None:
        _append_jsonl(self.rejections_path, record)

    def record_candidate(self, record: dict) -> None:
        _append_jsonl(self.candidates_path, record)

    def record_round(self, record: dict) -> None:
        _append_jsonl(self.rounds_path, record)

    # ---- reads -----------------------------------------------------------

    def manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def rounds(self) -> list[dict]:
        return _read_jsonl(self.rounds_path)

    def last_completed_round(self) -> int:
        rounds = self.rounds()
        return max((int(r.get("round", 0)) for r in rounds), default=0)

    def rejections(self) -> list[dict]:
        return _read_jsonl(self.rejections_path)

    def candidates(self) -> list[dict]:
        return _read_jsonl(self.candidates_path)

    def archive_records(self) -> list[dict]:
        return _read_jsonl(self.archive_path)

    def all_responses(self) -> dict[tuple[int, str, str], list[dict]]:
        """Every saved response, keyed by (round, island, operator), in file order."""
        out: dict[tuple[int, str, str], list[dict]] = {}
        for path in sorted(self.root.rglob("*.jsonl")):
            if path.parent == self.root:
                continue  # archive/rounds/rejections live at the top level
            try:
                round_index = int(path.parent.parent.name)
            except ValueError:
                continue
            key = (round_index, path.parent.name, path.stem)
            out.setdefault(key, []).extend(_read_jsonl(path))
        return out

    # ---- resume ----------------------------------------------------------

    def _remove_round_dir(self, path: Path) -> None:
        for child in sorted(path.rglob("*")):
            if child.is_file():
                child.unlink()
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        path.rmdir()

    def _filter_by_round(self, path: Path, through_round: int) -> None:
        if not path.exists():
            return
        kept = [r for r in _read_jsonl(path) if int(r.get("round", 0)) <= through_round]
        with path.open("w", encoding="utf-8") as fh:
            for record in kept:
                fh.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    default=str) + "\n")

    def reset_to_round(self, round_index: int) -> None:
        """Make the directory look exactly as it did at the end of `round_index`.

        A killed run leaves a partial round: some islands answered, others did
        not, and the archive already carries admissions from the ones that did.
        Resuming on top of that would produce an archive no complete run could
        produce, and would leave two response records under one (round, island,
        operator) key -- which then breaks replay, silently, much later.

        `round_index <= 0` is a full reset: the seed population is re-scored on
        every start, so keeping its records would make the re-scored seeds look
        like duplicates and leave every island empty.
        """
        for path in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not path.is_dir():
                continue
            try:
                index = int(path.name)
            except ValueError:
                continue
            if index > round_index:
                self._remove_round_dir(path)
        for path in (self.archive_path, self.candidates_path,
                     self.rejections_path, self.rounds_path):
            if round_index <= 0:
                if path.exists():
                    path.unlink()
            else:
                self._filter_by_round(path, round_index)

    def known_candidate_ids(self) -> set[str]:
        """Ids already proposed, so a resumed run does not re-propose them."""
        out = {str(r["id"]) for r in self.candidates() if r.get("id")}
        out |= {str(r["id"]) for r in self.archive_records() if r.get("id")}
        return out


def token_totals(store: RunStore) -> dict[str, Any]:
    """Token and wall-clock totals for an arm, read back off the saved envelopes."""
    calls = 0
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    cost = 0.0
    duration_ms = 0
    have_cost = False
    for records in store.all_responses().values():
        for record in records:
            calls += 1
            for key in totals:
                totals[key] += int(record.get(key, 0) or 0)
            duration_ms += int(record.get("duration_ms", 0) or 0)
            if isinstance(record.get("cost_usd"), (int, float)):
                cost += float(record["cost_usd"])
                have_cost = True
    return {
        "calls": calls, **totals,
        "cost_usd": round(cost, 4) if have_cost else None,
        "model_wall_clock_minutes": round(duration_ms / 60000.0, 2),
    }
