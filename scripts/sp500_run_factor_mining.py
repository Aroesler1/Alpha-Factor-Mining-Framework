#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.paths import resolve_from_us_root

from quantaalpha_us.llm.budget import RunBudget  # noqa: E402
from quantaalpha_us.llm.mining import FactorMiningRuntime  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _probe_available_models(api_key: str, base_url: str) -> set[str] | None:
    endpoint = base_url.rstrip("/") + "/models"
    req = Request(
        endpoint,
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
        data = payload.get("data")
        if not isinstance(data, list):
            return None
        models = {
            str(item.get("id")).strip()
            for item in data
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
        return models or None
    except URLError as exc:
        host = (urlparse(base_url).hostname or "").lower()
        if host in {"127.0.0.1", "localhost"}:
            raise RuntimeError(
                f"OpenAI-compatible proxy is not reachable at {endpoint}: {exc.reason}"
            ) from exc
        return None
    except HTTPError as exc:
        if exc.code == 404:
            return None
        return None
    except Exception:
        return None


def _build_live_call_model(api_key: str, base_url: str, timeout_seconds: int):
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def _call(model: str, prompt: str) -> Any:
        req_payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON only with exactly one top-level key named "
                        "\"factors\" whose value is an array of expression strings. "
                        "Do not return markdown, explanations, rationale, or extra keys."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        def _send(payload: dict[str, Any]) -> Any:
            body = json.dumps(payload).encode("utf-8")
            req = Request(endpoint, data=body, method="POST", headers=headers)
            with urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)

        try:
            return _send(req_payload)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            incompatible = "response_format" in detail or "json_object" in detail or exc.code == 422
            if incompatible:
                fallback_payload = dict(req_payload)
                fallback_payload.pop("response_format", None)
                return _send(fallback_payload)
            raise
        except URLError:
            raise

    return _call


def _clean_prompt_text(text: str) -> str:
    return re.sub(r"\s+\n", "\n", str(text).strip())


def _build_prompt_batches(
    base_prompts: Iterable[str],
    *,
    target_factors: int,
    factors_per_request: int,
) -> list[str]:
    prompt_seeds = [_clean_prompt_text(p) for p in base_prompts if str(p).strip()]
    if not prompt_seeds:
        prompt_seeds = ["Generate robust US large-cap daily OHLCV factors."]

    themes = [
        "momentum and trend persistence",
        "short-horizon reversal and mean reversion",
        "volatility compression and expansion",
        "liquidity and dollar-volume dynamics",
        "range, gap, and intraday location effects",
        "interaction effects between price and volume",
        "robust cross-sector behavior with simple logic",
        "regime-agnostic factors avoiding unstable tails",
    ]
    request_count = max(1, (max(int(target_factors), 1) + max(int(factors_per_request), 1) - 1) // max(int(factors_per_request), 1))

    prompts: list[str] = []
    for idx in range(request_count):
        seed = prompt_seeds[idx % len(prompt_seeds)]
        theme = themes[idx % len(themes)]
        prompts.append(
            (
                f"{seed}\n\n"
                f"Batch {idx + 1}/{request_count}.\n"
                f"Return exactly {int(factors_per_request)} distinct factor expressions.\n"
                f"Focus this batch on: {theme}.\n"
                "Keep formulas diverse and avoid near-duplicates of earlier common ideas.\n"
                'Return only JSON in the form {"factors":["expr1","expr2",...]}.\n'
                "Expressions only."
            )
        )
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run US factor mining runtime with budget/sanitizer guards.")
    parser.add_argument("--config", default="configs/llm_sp500.yaml")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--prompt", action="append", default=[], help="Optional mining prompt(s)")
    parser.add_argument("--live-call", action="store_true", help="Use OpenAI-compatible chat endpoint instead of dummy model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_dotenv(resolve_from_us_root(".env", US_ROOT))
    cfg_path = resolve_from_us_root(args.config, US_ROOT)
    cfg = _load_yaml(cfg_path)
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg.get("llm"), dict) else {}
    budget_cfg = llm_cfg.get("budget", {}) if isinstance(llm_cfg.get("budget"), dict) else {}
    factor_cfg = llm_cfg.get("factor_generation", {}) if isinstance(llm_cfg.get("factor_generation"), dict) else {}

    prompt_file = llm_cfg.get("prompt_file")
    prompts = [p for p in args.prompt if str(p).strip()]
    if prompt_file:
        prompt_path = resolve_from_us_root(str(prompt_file), US_ROOT)
        if prompt_path.exists():
            prompts.append(prompt_path.read_text(encoding="utf-8"))
    if not prompts:
        prompts = ["Generate 5 robust US OHLCV factors for S&P 500."]

    target_factors = int(factor_cfg.get("target_factors_per_run", 200))
    factors_per_request = int(factor_cfg.get("factors_per_request", 12))
    prompts = _build_prompt_batches(
        prompts,
        target_factors=target_factors,
        factors_per_request=factors_per_request,
    )

    models = [str(llm_cfg.get("primary_model", "kimi-k2.5"))]
    fallback = llm_cfg.get("fallback_models")
    if isinstance(fallback, list):
        models.extend(str(m) for m in fallback)
    models = [m for m in models if m]

    budget = RunBudget(
        max_requests=int(budget_cfg.get("max_requests_per_run", 250)),
        max_total_tokens=int(budget_cfg.get("max_total_tokens_per_run", 250000)),
        max_consecutive_failures=int(budget_cfg.get("max_consecutive_failures", 3)),
    )
    runtime = FactorMiningRuntime(
        budget=budget,
        max_batch_failure_rate=float(factor_cfg.get("max_batch_failure_rate", 0.50)),
    )

    if args.live_call:
        api_key = os.environ.get(args.api_key_env)
        base_url = os.environ.get(args.base_url_env, "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError(f"Missing API key in env var: {args.api_key_env}")
        available = _probe_available_models(api_key=api_key, base_url=base_url)
        if available is not None:
            require_exact = bool(llm_cfg.get("require_exact_model_catalog", True))
            missing = [model for model in models if model not in available]
            if require_exact and missing:
                raise RuntimeError(
                    "Configured factor-mining models are not all advertised by the OpenAI-compatible endpoint. "
                    f"missing={missing} configured={models} available_sample={sorted(list(available))[:20]}"
                )
            filtered = [model for model in models if model in available]
            if not filtered:
                raise RuntimeError(
                    "Configured models are not advertised by the OpenAI-compatible endpoint. "
                    f"Configured={models} available_sample={sorted(list(available))[:20]}"
                )
            models = filtered
        call_model = _build_live_call_model(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=int(llm_cfg.get("request_timeout_seconds", 120)),
        )
    else:
        # Real backend: Claude Code CLI on a subscription, or the Anthropic SDK
        # with an API key. Selection and its failure modes live in llm_client;
        # there is deliberately no stub fallback, because a miner that silently
        # returns canned factors is worse than one that refuses to run.
        from quantaalpha_us.factors.llm_client import make_call_model

        call_model = make_call_model(llm_cfg.get("backend"))

    valid, stats = runtime.run(
        prompts=prompts,
        models=models,
        call_model=call_model,
        estimated_tokens_per_request=256,
        target_valid_factors=target_factors,
    )

    payload = {
        "config": str(cfg_path),
        "models": models,
        "prompt_batches": len(prompts),
        "target_factors": target_factors,
        "valid_factors": valid,
        "stats": stats.to_dict(),
    }

    if args.output_file:
        out_path = resolve_from_us_root(args.output_file, US_ROOT)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = resolve_from_us_root(f"data/results/research/factor_mining_{ts}.json", US_ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "valid_factors": len(valid), "stats": stats.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
