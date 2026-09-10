#!/usr/bin/env python3
"""One real model call, before any expensive one: does the plumbing hold?

B10g. Everything else in the loop is exercised offline, which proves the logic
and proves nothing about the CLI contract. This makes exactly one call, with the
cheapest model and the lowest effort, and checks the four things that would
otherwise be discovered 30 minutes into a max-effort run:

  1. `--effort` is accepted by this CLI build
  2. the JSON envelope parses and carries usage
  3. `--json-schema` populates `structured_output` rather than only prose
  4. the sanitizer accepts at least one of the returned expressions

No panel, no data, no WRDS entitlement needed.

    python scripts/sp500_evo_smoke.py --model claude-haiku-4-5-20251001 --effort low
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.evo import operators as ops  # noqa: E402
from quantaalpha_us.evo.archive import Archive  # noqa: E402
from quantaalpha_us.evo.backends import ClaudeCodeEvoBackend  # noqa: E402
from quantaalpha_us.evo.config import ISLANDS, EvoConfig  # noqa: E402
from quantaalpha_us.factors import alpha101  # noqa: E402
from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer  # noqa: E402

# Fields an envelope may carry that should never be pasted into a report.
# Deliberately NOT a bare "token": the usage block is full of `*_tokens`
# counters, and redacting those would remove the only numbers this smoke test
# exists to show.
REDACT = ("api_key", "apikey", "authorization", "auth_token", "access_token",
          "refresh_token", "session_id", "uuid", "user_id", "account_uuid",
          "email", "cwd", "credential", "secret")


def redact(value, key: str = ""):
    if isinstance(value, dict):
        return {k: ("<redacted>" if any(r.lower() in k.lower() for r in REDACT)
                    else redact(v, k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--effort", default="low",
                        choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", default=str(US_ROOT / "data" / "evo_runs" / "smoke.json"))
    args = parser.parse_args()

    config = EvoConfig(arm="smoke", model=args.model, effort=args.effort)
    library = ops.PromptLibrary()
    alphas = [a.expression for a in alpha101.load()]
    request = ops.build_explore(
        config, library, island=ISLANDS[0].name, direction=ISLANDS[0].direction,
        round_index=1, niche_summary=Archive(config).niche_summary(),
        ban_list=ops.alpha101_ban_list(alphas), memory_block="Working memory: empty.",
        n=args.n,
    )
    backend = ClaudeCodeEvoBackend(model=args.model, effort=args.effort)
    print(f"prompt {len(request.prompt):,} chars, hash {request.prompt_hash}")
    print(f"calling: {' '.join(backend.command(request)[:8])} ... "
          f"--json-schema <{len(json.dumps(request.schema))} chars>", flush=True)

    reply = backend.call(request)
    envelope = redact(reply.envelope)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"prompt_hash": request.prompt_hash, "model": args.model, "effort": args.effort,
         "envelope": envelope}, indent=2, sort_keys=True), encoding="utf-8")

    print("\n--- envelope (credential-shaped fields redacted) ---")
    trimmed = dict(envelope)
    if isinstance(trimmed.get("result"), str) and len(trimmed["result"]) > 600:
        trimmed["result"] = trimmed["result"][:600] + " ...<truncated>"
    print(json.dumps(trimmed, indent=2, sort_keys=True)[:4000])

    checks = []
    checks.append(("--effort accepted", not reply.error or "effort" not in reply.error))
    checks.append(("envelope parsed", bool(reply.envelope)))
    checks.append(("no error reported", not reply.error))
    checks.append(("usage present", reply.total_tokens > 0))
    structured = isinstance(reply.envelope.get("structured_output"), dict)
    checks.append(("structured_output present", structured))

    proposals = ops.parse_proposals(reply.payload, request)
    checks.append(("candidates parsed", len(proposals) > 0))
    sanitizer = ExpressionSanitizer()
    accepted = [p for p in proposals if sanitizer.sanitize(p.expression).valid]
    checks.append(("sanitizer accepts at least one", len(accepted) > 0))

    print("\n--- expressions returned ---")
    for proposal in proposals:
        result = sanitizer.sanitize(proposal.expression)
        mark = "ok  " if result.valid else "FAIL"
        print(f"  [{mark}] {proposal.expression}")
        if not result.valid:
            print(f"         {'; '.join(result.errors)[:160]}")
        if proposal.rationale:
            print(f"         rationale: {proposal.rationale[:130]}")

    print("\n--- checks ---")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\ntokens: in {reply.input_tokens:,}, out {reply.output_tokens:,}, "
          f"total {reply.total_tokens:,}"
          + (f", cost ${reply.cost_usd:.4f}" if reply.cost_usd is not None else ""))
    print(f"duration: {reply.duration_ms / 1000:.1f}s")
    print(f"-> {args.out}")
    if reply.error:
        print(f"\nERROR: {reply.error}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
