from quantaalpha_us.llm.budget import RunBudget
from quantaalpha_us.llm.mining import FactorMiningRuntime


def test_factor_mining_runtime_applies_sanitizer_and_budget() -> None:
    budget = RunBudget(max_requests=5, max_total_tokens=5000, max_consecutive_failures=3)
    runtime = FactorMiningRuntime(budget=budget, max_batch_failure_rate=0.8)

    def fake_call(model: str, prompt: str):  # noqa: ANN001
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"factors":["TS_MEAN($close,20)","import os"]}',
                    }
                }
            ],
            "usage": {"total_tokens": 50},
        }

    factors, stats = runtime.run(
        prompts=["p1", "p2"],
        models=["m1"],
        call_model=fake_call,
        estimated_tokens_per_request=32,
    )
    assert len(factors) >= 1
    assert stats.valid_factors >= 1
    assert stats.invalid_factors >= 1


def test_factor_mining_runtime_extracts_json_from_fenced_content_and_dedupes() -> None:
    budget = RunBudget(max_requests=5, max_total_tokens=5000, max_consecutive_failures=3)
    runtime = FactorMiningRuntime(budget=budget, max_batch_failure_rate=0.8)

    def fake_call(model: str, prompt: str):  # noqa: ANN001
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": '```json\n{"factors":[{"expression":"TS_MEAN($close,20)"},{"expression":"TS_MEAN($close,20)"}]}\n```',
                            }
                        ],
                    }
                }
            ],
            "usage": {"total_tokens": 50},
        }

    factors, stats = runtime.run(
        prompts=["p1", "p2"],
        models=["m1"],
        call_model=fake_call,
        estimated_tokens_per_request=32,
        target_valid_factors=1,
    )
    assert factors == ["TS_MEAN($close,20)"]
    assert stats.valid_factors == 1


def test_default_mining_path_does_not_send_endpoint_models_to_claude() -> None:
    from scripts.sp500_run_factor_mining import select_models

    cfg = {"primary_model": "gpt-5.4", "fallback_models": ["gemini-3.1-pro"]}
    assert select_models(cfg, live_call=True) == ["gpt-5.4", "gemini-3.1-pro"]
    default = select_models(cfg, live_call=False)
    assert default and all(m.startswith("claude-") for m in default)
    assert select_models({**cfg, "claude_models": ["claude-sonnet-5"]}, live_call=False) == [
        "claude-sonnet-5"]
