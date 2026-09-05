"""Where proposals come from: the CLI, a deterministic mock, or a saved run.

Three backends behind one interface, and the interface is the point. The runner
cannot tell them apart, so the mock exercises the real gate/score/archive path
in CI, the replay backend re-derives an archive from disk with zero model calls,
and the live backend is the only one that costs anything.

`ClaudeCodeBackend` here is a sibling of the one in `factors.llm_client`, not a
replacement: that one predates the loop, returns `{"factors": [...]}`, and has no
notion of effort, per-call token accounting, or an operator-specific schema. All
three are things this experiment has to report, so rather than widen a contract
the existing miner depends on, the loop gets its own thin wrapper and the two
stay independently testable.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from quantaalpha_us.evo import gp
from quantaalpha_us.evo.operators import (
    CROSSOVER,
    EXPLORE,
    MUTATE,
    REFLECT,
    SIMPLIFY,
    OperatorRequest,
)
from quantaalpha_us.evo.scoring import candidate_id

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass
class BackendReply:
    """One model call's result, including everything needed to report its cost."""

    payload: dict
    raw_text: str = ""
    envelope: dict = field(default_factory=dict)
    model: str = ""
    effort: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    duration_ms: int = 0
    error: str = ""

    def to_record(self, request: OperatorRequest) -> dict:
        return {
            "timestamp": time.time(),
            "operator": request.operator,
            "island": request.island,
            "round": request.round,
            "prompt_hash": request.prompt_hash,
            "prompt": request.prompt,
            "model": self.model,
            "effort": self.effort,
            "raw_text": self.raw_text,
            "payload": self.payload,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class Backend(Protocol):
    name: str

    def call(self, request: OperatorRequest) -> BackendReply: ...


# --------------------------------------------------------------------------
# live: the Claude Code CLI
# --------------------------------------------------------------------------


@dataclass
class ClaudeCodeEvoBackend:
    """`claude -p --model M --effort E --output-format json --json-schema S`.

    The envelope is parsed BEFORE the exit code is inspected, for the reason
    `factors.llm_client` documents at length: the CLI reports an expired OAuth
    session as exit 1 with an empty stderr and the actual reason inside stdout's
    JSON, so branching on the return code first produces an error message with
    nothing in it.
    """

    model: str = "claude-sonnet-5"
    effort: str = "max"
    binary: str = "claude"
    # Measured, not guessed: a loop call at max effort runs a few minutes, but a
    # one-shot call asking for 240 expressions ran 18 to 30 minutes and produced
    # 100k to 190k output tokens. At 1800 s one of those was killed by the
    # timeout AFTER the model had done the work and been billed for it, which is
    # the worst possible way to lose a call.
    timeout_seconds: int = 5400
    name: str = "claude-code"

    def __post_init__(self) -> None:
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}, got {self.effort!r}")
        resolved = shutil.which(self.binary)
        if resolved is None:
            raise RuntimeError(f"{self.binary!r} is not on PATH")
        self.binary = resolved

    def command(self, request: OperatorRequest) -> list[str]:
        return [
            self.binary, "-p", request.prompt,
            "--model", self.model,
            "--effort", self.effort,
            "--output-format", "json",
            "--json-schema", json.dumps(request.schema),
        ]

    def call(self, request: OperatorRequest) -> BackendReply:
        started = time.time()
        proc = subprocess.run(self.command(request), capture_output=True, text=True,
                              timeout=self.timeout_seconds)
        elapsed = int((time.time() - started) * 1000)

        envelope: Any = None
        if proc.stdout.strip():
            try:
                envelope = json.loads(proc.stdout)
            except json.JSONDecodeError:
                envelope = None
        reply = BackendReply(payload={}, model=self.model, effort=self.effort,
                             duration_ms=elapsed)
        if isinstance(envelope, dict):
            reply.envelope = envelope
            usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
            reply.input_tokens = int(usage.get("input_tokens", 0) or 0)
            reply.output_tokens = int(usage.get("output_tokens", 0) or 0)
            # cache reads and writes are billed input; counting them keeps the
            # reported total comparable to what the subscription actually spends
            reply.total_tokens = (
                reply.input_tokens + reply.output_tokens
                + int(usage.get("cache_creation_input_tokens", 0) or 0)
                + int(usage.get("cache_read_input_tokens", 0) or 0)
            )
            cost = envelope.get("total_cost_usd")
            reply.cost_usd = float(cost) if isinstance(cost, (int, float)) else None
            reply.raw_text = str(envelope.get("result", ""))
            if envelope.get("is_error"):
                reply.error = str(envelope.get("result")
                                  or envelope.get("api_error_status")
                                  or envelope.get("subtype") or "unknown")[:400]
                return reply
            structured = envelope.get("structured_output")
            if isinstance(structured, dict):
                reply.payload = structured
            else:
                reply.payload = _loose_json(reply.raw_text)
            return reply

        reply.raw_text = proc.stdout
        reply.error = (proc.stderr.strip() or proc.stdout.strip()
                       or "no output on either stream")[:400]
        if proc.returncode == 0:
            reply.error = f"claude CLI returned non-JSON output: {reply.error}"
        else:
            reply.error = f"claude CLI failed (rc={proc.returncode}): {reply.error}"
        return reply


def _loose_json(text: str) -> dict:
    """Last-resort parse for a CLI build that predates `structured_output`."""
    import re

    text = (text or "").strip()
    for candidate in (text,):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {}


# --------------------------------------------------------------------------
# offline: the deterministic mock
# --------------------------------------------------------------------------


@dataclass
class MockBackend:
    """Deterministic proposals, so CI exercises the real loop with no tokens.

    Two sources, in order. `scripted` supplies canned payloads keyed by operator
    and is popped from first, which is what a test uses to plant a specific
    expression or a specific malformed reply. When it runs out, proposals come
    from the same typed GP operators the control arm uses -- so the mock exhibits
    the real distribution of accepted and rejected candidates rather than a
    stream of one hand-written string that would never exercise a gate.
    """

    seed: int = 0
    scripted: dict[str, list[dict]] = field(default_factory=dict)
    seed_expressions: tuple[str, ...] = ()
    reference: tuple[str, ...] = ("TS_MEAN($close, 21) / (TS_STD($close, 21) + 1e-8)",)
    model: str = "mock"
    effort: str = "low"
    name: str = "mock"
    calls: int = 0
    _rng: random.Random = field(init=False)
    _seeded_used: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def call(self, request: OperatorRequest) -> BackendReply:
        self.calls += 1
        queue = self.scripted.get(request.operator)
        if queue:
            payload = queue.pop(0)
            return BackendReply(payload=payload, raw_text=json.dumps(payload),
                                model=self.model, effort=self.effort,
                                input_tokens=len(request.prompt) // 4,
                                output_tokens=64,
                                total_tokens=len(request.prompt) // 4 + 64)
        payload = self._generate(request)
        return BackendReply(payload=payload, raw_text=json.dumps(payload),
                            model=self.model, effort=self.effort,
                            input_tokens=len(request.prompt) // 4, output_tokens=64,
                            total_tokens=len(request.prompt) // 4 + 64)

    def _generate(self, request: OperatorRequest) -> dict:
        if request.operator == REFLECT:
            return {"rules": [
                f"round {request.round}: keep windows on the ladder the archive already uses",
            ]}
        candidates: list[dict] = []
        if request.operator == EXPLORE:
            planted = [e for e in self.seed_expressions if not self._seeded_used]
            self._seeded_used = bool(planted) or self._seeded_used
            for expr in planted:
                candidates.append({"expression": expr,
                                   "rationale": "planted by the mock backend",
                                   "parent_ids": [], "changed_node": ""})
            drawn = gp.random_expressions(self._rng, max(request.n - len(candidates), 0),
                                          reference=list(self.reference))
            candidates.extend({"expression": e, "rationale": "random grammar draw",
                               "parent_ids": [], "changed_node": ""} for e in drawn)
        elif request.operator == MUTATE:
            for parent in request.parents:
                result = gp.mutate(self._rng, parent, max_depth=6)
                if result is None:
                    continue
                expression, changed = result
                candidates.append({"expression": expression, "rationale": "single-node edit",
                                   "parent_ids": [candidate_id(parent)],
                                   "changed_node": changed})
        elif request.operator == CROSSOVER:
            for left, right in request.parent_pairs:
                child = gp.crossover(self._rng, left, right, max_depth=6)
                if child is None:
                    continue
                candidates.append({"expression": child, "rationale": "subtree fusion",
                                   "parent_ids": [candidate_id(left), candidate_id(right)],
                                   "changed_node": ""})
        elif request.operator == SIMPLIFY:
            for parent in request.parents:
                child = gp.simplify(self._rng, parent)
                if child is None:
                    continue
                candidates.append({"expression": child, "rationale": "pruned a subtree",
                                   "parent_ids": [candidate_id(parent)],
                                   "changed_node": "pruned"})
        return {"candidates": candidates}


# --------------------------------------------------------------------------
# offline: the GP control arm
# --------------------------------------------------------------------------


@dataclass
class GPBackend(MockBackend):
    """The control. Identical to the mock minus any planted expression.

    Kept as a distinct class rather than a flag so a run's manifest says plainly
    which one produced it, and so nobody can accidentally report a mock run as
    the GP control.
    """

    name: str = "gp"
    model: str = "gp-random-grammar"


# --------------------------------------------------------------------------
# offline: replay
# --------------------------------------------------------------------------


@dataclass
class ReplayBackend:
    """Serves the responses a previous run saved, in the order it saved them."""

    responses: dict[tuple[int, str, str], list[dict]]
    name: str = "replay"
    calls: int = 0
    missing: list[tuple[int, str, str]] = field(default_factory=list)

    @classmethod
    def from_store(cls, store) -> "ReplayBackend":
        return cls(responses=store.all_responses())

    def call(self, request: OperatorRequest) -> BackendReply:
        self.calls += 1
        key = (request.round, request.island, request.operator)
        queue = self.responses.get(key)
        if not queue:
            self.missing.append(key)
            return BackendReply(payload={}, error=f"no saved response for {key}")
        record = queue.pop(0)
        return BackendReply(
            payload=record.get("payload") or {},
            raw_text=record.get("raw_text", ""),
            model=record.get("model", ""), effort=record.get("effort", ""),
            input_tokens=int(record.get("input_tokens", 0) or 0),
            output_tokens=int(record.get("output_tokens", 0) or 0),
            total_tokens=int(record.get("total_tokens", 0) or 0),
            cost_usd=record.get("cost_usd"),
            duration_ms=int(record.get("duration_ms", 0) or 0),
            error=record.get("error", ""),
        )


def dry_run_commands(backend: Backend, requests: Sequence[OperatorRequest]) -> list[str]:
    """The exact command lines a live run would issue, for --dry-run."""
    if not hasattr(backend, "command"):
        return [f"[{backend.name}] {r.operator}/{r.island} round {r.round} "
                f"(offline backend, no command line)" for r in requests]
    out = []
    for request in requests:
        parts = backend.command(request)  # type: ignore[attr-defined]
        rendered = " ".join(
            f"'{p}'" if (" " in p or "\n" in p or '"' in p) else p for p in parts
        )
        out.append(rendered)
    return out
