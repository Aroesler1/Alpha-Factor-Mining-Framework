#!/usr/bin/env python3
"""Run one arm of the evolutionary factor-mining experiment.

    # no tokens: the genetic-programming control
    python scripts/sp500_run_evolution.py --arm gp-loop --backend gp

    # print every command line a real run would issue, and the token estimate
    python scripts/sp500_run_evolution.py --arm sonnet5-loop --backend claude \
        --model claude-sonnet-5 --effort max --dry-run

    # the real thing
    python scripts/sp500_run_evolution.py --arm sonnet5-loop --backend claude \
        --model claude-sonnet-5 --effort max --budget 200

    # continue after a subscription cap, or re-derive a run from its saved replies
    python scripts/sp500_run_evolution.py --arm sonnet5-loop --backend claude --resume
    python scripts/sp500_run_evolution.py --arm sonnet5-loop --replay data/evo_runs/sonnet5-loop

Arms differ only in the backend and two flags. Gates, fitness, archive, islands,
windows and the round schedule are shared, which is the whole point: a
difference between two arms has to be the proposer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.evo.backends import (  # noqa: E402
    ClaudeCodeEvoBackend,
    GPBackend,
    MockBackend,
    ReplayBackend,
    dry_run_commands,
)
from quantaalpha_us.evo.config import EvoConfig, Schedule  # noqa: E402
from quantaalpha_us.evo.persistence import RunStore, token_totals  # noqa: E402
from quantaalpha_us.evo.report import write_run_report  # noqa: E402
from quantaalpha_us.evo.runner import EvolutionRunner  # noqa: E402
from quantaalpha_us.evo.scoring import PanelScorer  # noqa: E402
from quantaalpha_us.factors import alpha101  # noqa: E402
from quantaalpha_us.factors.ic_panel import apply_membership_filter  # noqa: E402
from quantaalpha_us.llm.budget import RunBudget  # noqa: E402


def build_backend(args, config: EvoConfig, store: RunStore):
    if args.replay:
        return ReplayBackend.from_store(RunStore(Path(args.replay)))
    if args.backend == "gp":
        return GPBackend(seed=config.seed)
    if args.backend == "mock":
        return MockBackend(seed=config.seed)
    return ClaudeCodeEvoBackend(model=args.model, effort=args.effort)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, help="name of this run; also its directory")
    parser.add_argument("--backend", default="gp", choices=("gp", "mock", "claude"))
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--effort", default="max",
                        choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--mode", default="loop", choices=("loop", "one_shot"))
    parser.add_argument("--feedback", default="full", choices=("full", "scalar"),
                        help="'scalar' is the feedback-ablation arm: parents are shown "
                             "their fitness number and nothing else")
    parser.add_argument("--rounds", type=int, default=Schedule.rounds)
    parser.add_argument("--seed", type=int, default=EvoConfig.seed)
    parser.add_argument("--budget", type=int, default=0,
                        help="hard cap on model requests for this arm; 0 means "
                             "exactly the planned count plus a small margin")
    parser.add_argument("--token-estimate", type=int, default=30000,
                        help="tokens assumed per call by --dry-run. At max effort a "
                             "call on these prompts runs 20k-40k including thinking.")
    parser.add_argument("--bars", default=str(US_ROOT / "data" / "us_equities" / "processed"
                                              / "daily_bars_fundamentals.parquet"))
    parser.add_argument("--membership",
                        default=str(US_ROOT / "data" / "us_equities" / "reference"
                                    / "sp500_membership_daily.parquet"))
    parser.add_argument("--runs-dir", default=str(US_ROOT / "data" / "evo_runs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay", default=None,
                        help="re-derive an arm from a saved run directory, zero model calls")
    args = parser.parse_args()

    schedule = Schedule(rounds=args.rounds)
    config = EvoConfig(
        arm=args.arm,
        model=(args.model if args.backend == "claude" else args.backend),
        effort=(args.effort if args.backend == "claude" else "n/a"),
        seed=args.seed, schedule=schedule, mode=args.mode, feedback_mode=args.feedback,
    )
    if args.mode == "one_shot":
        print(f"one-shot arm: {len(config.islands)} calls asking for "
              f"{config.one_shot_candidates} candidates each, matching the "
              f"{config.one_shot_candidates * len(config.islands)} a full loop generates")

    store = RunStore(Path(args.runs_dir) / args.arm)
    if not args.resume and store.archive_path.exists():
        # Starting fresh on top of an existing run appends to its archive and
        # leaves two response records under one (round, island, operator) key,
        # which corrupts replay silently and much later.
        raise SystemExit(
            f"{store.root} already holds a run ({len(store.archive_records())} archive "
            f"records, {store.last_completed_round()} completed rounds).\n"
            "Pass --resume to continue it, use a different --arm, or delete the "
            "directory to start over."
        )
    backend = build_backend(args, config, store)

    started = time.time()
    print(f"loading {args.bars} ...", flush=True)
    bars = pd.read_parquet(args.bars)
    before = len(bars)
    bars = apply_membership_filter(bars, args.membership)
    print(f"membership filter: {before:,} -> {len(bars):,} rows", flush=True)
    scorer = PanelScorer(bars, config)
    print(f"windows: fit {scorer.fit_dates.min().date()} -> {scorer.fit_dates.max().date()} "
          f"({len(scorer.fit_dates):,} days), cheap screen from "
          f"{scorer.cheap_dates.min().date()} ({len(scorer.cheap_dates):,} days). "
          f"Validation and holdout are loaded but never returned to a prompt.",
          flush=True)
    print(f"calibration: turnover split at "
          f"{config.calibration.alpha101_turnover_median:.4f} "
          f"[{config.calibration.source}]", flush=True)

    planned = (len(config.islands) if config.mode == "one_shot"
               else schedule.planned_calls(len(config.islands)))
    budget = RunBudget(max_requests=args.budget or planned + 10, max_total_tokens=10**9)
    runner = EvolutionRunner(config, scorer, backend, store,
                             alpha101=[a.expression for a in alpha101.load()],
                             budget=budget)

    if args.dry_run:
        runner._seed_islands()
        requests = []
        for state in runner.states:
            requests.extend(runner._requests_for(state, 1))
        lines = dry_run_commands(backend, requests)
        print("\n" + "=" * 96)
        print(f"DRY RUN: round 1 would issue {len(requests)} operator calls "
              f"({len(config.islands)} islands)")
        print("=" * 96)
        for request, line in zip(requests, lines):
            print(f"\n--- {request.operator} / {request.island} "
                  f"(prompt {len(request.prompt):,} chars, hash {request.prompt_hash}) ---")
            print(line[:4000] + (" ..." if len(line) > 4000 else ""))
        per_round = len(requests) + (0 if config.mode == "one_shot" else len(config.islands))
        total = planned
        print("\n" + "-" * 96)
        print(f"planned calls for the whole arm: {total}")
        print(f"  round 1 measured: {len(requests)} operator calls + "
              f"{0 if config.mode == 'one_shot' else len(config.islands)} reflection "
              f"= {per_round}")
        print(f"token estimate at {args.token_estimate:,}/call: "
              f"{total * args.token_estimate:,} tokens for the arm")
        print(f"budget cap: {budget.max_requests} requests")
        print("\nLater rounds are not printed: their prompts depend on what earlier "
              "rounds find,\nso they do not exist yet. The call COUNT above is exact.")
        return 0

    result = runner.run(resume=args.resume)
    report = write_run_report(result.archive, store)
    tokens = token_totals(store)

    print("\n" + "=" * 96)
    print(f"ARM {args.arm}  ({config.model}, effort {config.effort}, mode {config.mode}, "
          f"feedback {config.feedback_mode})".center(96))
    print("=" * 96)
    print(f"\nrounds run          {len(result.rounds)}/{schedule.rounds}"
          f"{'  (stopped early)' if result.stopped_early else ''}")
    print(f"stop reason         {result.stop_reason}")
    print(f"archive             {report['archive_size']} factors in "
          f"{report['niches_filled']}/{report['niches_possible']} niches")
    print(f"model calls         {tokens['calls']}")
    print(f"tokens              {tokens['total_tokens']:,} "
          f"(in {tokens['input_tokens']:,}, out {tokens['output_tokens']:,})")
    if tokens["cost_usd"] is not None:
        print(f"reported cost       ${tokens['cost_usd']:.2f}")
    print(f"wall clock          {(time.time() - started) / 60:.1f} min total, "
          f"{tokens['model_wall_clock_minutes']:.1f} min inside model calls")

    print("\nvalidation curve (archive top-20 equal-weight mean IC), the early-stop signal:")
    for summary in result.rounds:
        print(f"  round {summary.round}: {summary.validation_top20_ic:.5f}   "
              f"archive {summary.archive_size:3d}  niches {summary.niches_filled:2d}  "
              f"proposals {summary.proposals:3d}  admitted {summary.admitted:3d}")

    rejections = pd.read_csv(report["rejections"])
    if not rejections.empty:
        print("\nrejections by gate:")
        for _, row in rejections.iterrows():
            print(f"  {row['gate']:<28}{int(row['count']):>6}")

    archive_table = pd.read_csv(report["archive"])
    if not archive_table.empty:
        # the seeds are in the CSV (marked by operator) so the run is auditable,
        # but they are not discoveries and every arm starts from the same ones
        seeds = int((archive_table["operator"] == "seed").sum())
        archive_table = archive_table[archive_table["operator"] != "seed"]
        print(f"\ntop 10 discovered by fitness ({seeds} island seed(s) excluded; "
              "they are in the CSV):")
        with pd.option_context("display.width", 200, "display.max_colwidth", 62):
            print(archive_table.nlargest(10, "fitness")[
                ["fitness", "mean_ic", "tstat", "turnover", "niche", "operator",
                 "expression"]].to_string(index=False))
    for key in ("archive", "rounds", "rejections", "niches"):
        print(f"-> {report[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
