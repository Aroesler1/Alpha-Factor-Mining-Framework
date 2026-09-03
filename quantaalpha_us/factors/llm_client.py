"""Pluggable LLM backends for factor mining.

The mining loop needs one thing from a model: a list of candidate factor
expressions in this repo's DSL. Two backends provide it, because the two ways of
reaching Claude have different credentials and different audiences.

`ClaudeCodeBackend` shells out to the Claude Code CLI (`claude -p`), which
authenticates with an interactive Claude subscription rather than an API key.
This is the path for running the miner on your own machine at no marginal cost.
It is deliberately NOT the default: Anthropic does not permit third-party
products to offer claude.ai login, so a public repo should not require a
reviewer to have one.

`AnthropicAPIBackend` uses the official `anthropic` SDK with `ANTHROPIC_API_KEY`.
This is the reproducible path: anyone who clones the repo and holds an API key
can rerun the mining step and get comparable output.

Both return the same shape, so `sp500_run_factor_mining.py` does not care which
is in use, and swapping backends cannot change the downstream contract.

Selection order:
    1. explicit `backend=` argument
    2. LLM_BACKEND environment variable ("claude_code" | "anthropic")
    3. ANTHROPIC_API_KEY present -> anthropic
    4. `claude` on PATH -> claude_code
    5. raise, rather than silently degrading to a stub
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Protocol

# Structured output contract. Both backends are asked for exactly this shape so
# the caller never has to parse prose, and a malformed reply fails loudly.
FACTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "factors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "hypothesis": {"type": "string"},
                },
                "required": ["expression", "hypothesis"],
            },
        }
    },
    "required": ["factors"],
}

DEFAULT_MODEL = "claude-opus-5"


class LLMBackend(Protocol):
    def generate(self, prompt: str, model: str) -> dict[str, Any]:
        """Return {"factors": [{"expression": ..., "hypothesis": ...}, ...]}."""


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a reply that may carry prose around it."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"no JSON object in model reply: {text[:200]!r}")


@dataclass
class ClaudeCodeBackend:
    """Claude Code CLI in non-interactive mode; uses the subscription login.

    `--bare` is deliberately omitted: bare mode does not read OAuth credentials
    and would require an API key, which defeats the purpose of this backend.
    """

    binary: str = "claude"
    timeout_seconds: int = 300

    def generate(self, prompt: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
        cmd = [
            self.binary, "-p", prompt,
            "--model", model,
            "--output-format", "json",
            "--json-schema", json.dumps(FACTOR_SCHEMA),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout_seconds
        )

        # Parse the envelope BEFORE branching on the exit code. The CLI puts its
        # diagnostic in stdout's JSON, not stderr, and it does so on failing exit
        # codes too: an expired OAuth session exits 1 with an EMPTY stderr and
        # {"is_error": true, "result": "Failed to authenticate: OAuth session
        # expired..."} on stdout. Reporting stderr on a nonzero exit therefore
        # produced "claude CLI failed (rc=1): " with no reason in it at all.
        # Measured against CLI 2.1.239.
        envelope: Any = None
        if proc.stdout.strip():
            try:
                envelope = json.loads(proc.stdout)
            except json.JSONDecodeError:
                envelope = None

        if isinstance(envelope, dict) and envelope.get("is_error"):
            # `subtype` is a poor fallback -- it reads "success" even on a failed
            # authentication -- so prefer `result`, then the API error status.
            detail = str(
                envelope.get("result")
                or envelope.get("api_error_status")
                or envelope.get("subtype")
                or "unknown"
            )
            raise RuntimeError(f"claude CLI reported an error: {detail[:300]}")

        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or "no output on either stream"
            raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {detail[:300]}")

        if envelope is None:
            raise RuntimeError(
                f"claude CLI returned non-JSON output: {proc.stdout[:200]!r}"
            )

        # --json-schema puts the typed value in structured_output; fall back to
        # parsing the text result for CLI versions that predate that field.
        if isinstance(envelope.get("structured_output"), dict):
            return envelope["structured_output"]
        return _extract_json(envelope.get("result", ""))


@dataclass
class AnthropicAPIBackend:
    """Official Anthropic SDK; requires ANTHROPIC_API_KEY."""

    timeout_seconds: int = 300
    max_tokens: int = 8000

    def generate(self, prompt: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
        import anthropic

        client = anthropic.Anthropic(timeout=float(self.timeout_seconds))
        response = client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": FACTOR_SCHEMA,
                }
            },
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return _extract_json(text)


def get_backend(backend: str | None = None) -> LLMBackend:
    choice = (backend or os.getenv("LLM_BACKEND") or "").strip().lower()

    if not choice:
        if os.getenv("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        elif shutil.which("claude"):
            choice = "claude_code"
        else:
            raise RuntimeError(
                "No LLM backend available. Set ANTHROPIC_API_KEY for the API "
                "backend, or install the Claude Code CLI for the subscription "
                "backend, or pass backend= explicitly."
            )

    if choice == "claude_code":
        binary = shutil.which("claude")
        if binary is None:
            raise RuntimeError("LLM_BACKEND=claude_code but `claude` is not on PATH")
        return ClaudeCodeBackend(binary=binary)
    if choice == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is unset")
        return AnthropicAPIBackend()
    raise ValueError(f"unknown LLM backend: {choice!r}")


def make_call_model(backend: str | None = None) -> Callable[[str, str], dict[str, Any]]:
    """Adapter matching the mining runtime's expected call_model(model, prompt).

    The runtime consumes an OpenAI-style envelope, so the structured reply is
    re-wrapped rather than changing the runtime's contract. Expressions are
    flattened to one per line, which is what the sanitizer expects; the
    hypotheses are preserved alongside so a rejected factor can still be audited
    against the reasoning that produced it.
    """
    impl = get_backend(backend)

    def call_model(model: str, prompt: str) -> dict[str, Any]:
        payload = impl.generate(prompt, model=model or DEFAULT_MODEL)
        factors = payload.get("factors", [])
        expressions = [f["expression"] for f in factors if f.get("expression")]
        return {
            "choices": [
                {"message": {"content": json.dumps({"factors": expressions})}}
            ],
            "hypotheses": {f["expression"]: f.get("hypothesis", "") for f in factors},
            "model": model,
        }

    return call_model
