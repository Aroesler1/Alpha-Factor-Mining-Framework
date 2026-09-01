"""Tests for the pluggable LLM backends.

None of these call a model. They pin the parts that fail silently in
production: envelope error handling, reply parsing, and backend selection.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantaalpha_us.factors.llm_client import (  # noqa: E402
    AnthropicAPIBackend,
    ClaudeCodeBackend,
    _extract_json,
    get_backend,
    make_call_model,
)


class _FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_extract_json_handles_bare_fenced_and_prose():
    assert _extract_json('{"factors": []}') == {"factors": []}
    assert _extract_json('sure:\n```json\n{"factors": [1]}\n```') == {"factors": [1]}
    assert _extract_json('here you go {"factors": [2]} hope that helps') == {"factors": [2]}
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


def test_cli_error_envelope_raises_even_on_exit_code_zero(monkeypatch):
    """The CLI reports expired auth and rate limits with is_error and rc=0.

    Checking only returncode would turn those into an empty factor list and
    quietly poison a mining run with silence instead of an error.
    """
    envelope = {
        "is_error": True,
        "result": "Failed to authenticate: OAuth session expired",
        "subtype": "success",
    }
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _FakeProc(stdout=json.dumps(envelope))
    )
    with pytest.raises(RuntimeError, match="OAuth session expired"):
        ClaudeCodeBackend().generate("prompt")


def test_cli_prefers_structured_output(monkeypatch):
    envelope = {
        "is_error": False,
        "structured_output": {"factors": [{"expression": "RANK($close)", "hypothesis": "h"}]},
        "result": "ignored when structured_output is present",
    }
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _FakeProc(stdout=json.dumps(envelope))
    )
    out = ClaudeCodeBackend().generate("prompt")
    assert out["factors"][0]["expression"] == "RANK($close)"


def test_cli_falls_back_to_parsing_result_text(monkeypatch):
    envelope = {"is_error": False, "result": '{"factors": [{"expression": "X", "hypothesis": "y"}]}'}
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _FakeProc(stdout=json.dumps(envelope))
    )
    assert ClaudeCodeBackend().generate("prompt")["factors"][0]["expression"] == "X"


def test_call_model_adapter_shape_and_hypotheses(monkeypatch):
    """The adapter must preserve the runtime's envelope AND keep hypotheses."""
    envelope = {
        "is_error": False,
        "structured_output": {
            "factors": [
                {"expression": "RANK($close)", "hypothesis": "cheapness"},
                {"expression": "TS_STD($return, 21)", "hypothesis": "low vol"},
            ]
        },
    }
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _FakeProc(stdout=json.dumps(envelope))
    )
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    call = make_call_model("claude_code")
    out = call("claude-opus-5", "prompt")

    payload = json.loads(out["choices"][0]["message"]["content"])
    assert payload["factors"] == ["RANK($close)", "TS_STD($return, 21)"]
    assert out["hypotheses"]["RANK($close)"] == "cheapness"


def test_backend_selection_never_silently_stubs(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="No LLM backend available"):
        get_backend()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(get_backend(), AnthropicAPIBackend)

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    assert isinstance(get_backend(), ClaudeCodeBackend)


def test_explicit_backend_requires_its_credential(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is unset"):
        get_backend("anthropic")
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="not on PATH"):
        get_backend("claude_code")
