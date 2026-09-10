#!/usr/bin/env python3
"""Re-derive every arm's archive from its saved replies and check it byte for byte.

This is the claim the whole run record rests on: that `data/evo_runs/` contains
enough to reconstruct each result without calling a model again. An archive that
cannot be reproduced from its own responses is not evidence, it is an artefact
of one process that no longer exists.

Every arm is replayed in ONE process, sharing a single loaded panel and
evaluator. Six separate processes would each pay ~30 s and ~2.5 GB to rebuild
the same panel, and on a machine that has been swapping that is the difference
between a check that finishes and one that thrashes.

    python scripts/sp500_evo_verify_replay.py

Zero model calls, by construction: the backend is ReplayBackend, which has no
way to reach one.
"""
from __future__ import annotations

import argparse
import difflib
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.evo.archive import load_archive  # noqa: E402
from quantaalpha_us.evo.backends import ReplayBackend  # noqa: E402
from quantaalpha_us.evo.config import EvoConfig, Schedule  # noqa: E402
from quantaalpha_us.evo.persistence import RunStore  # noqa: E402
from quantaalpha_us.evo.runner import EvolutionRunner  # noqa: E402
from quantaalpha_us.evo.scoring import PanelScorer  # noqa: E402
from quantaalpha_us.factors import alpha101  # noqa: E402
from quantaalpha_us.factors.ic_panel import apply_membership_filter  # noqa: E402
from quantaalpha_us.llm.budget import RunBudget  # noqa: E402


def config_for(arm: str, store: RunStore) -> EvoConfig:
    """Rebuild the arm's config from its own manifest, not from today's defaults.

    An arm run before a threshold was frozen must be replayed under the
    thresholds it actually used, or the replay is a different experiment that
    happens to read the same replies.
    """
    manifest = store.manifest()
    saved = manifest.get("config", {})
    schedule_fields = set(Schedule.__dataclass_fields__)
    schedule = Schedule(**{k: v for k, v in (saved.get("schedule") or {}).items()
                           if k in schedule_fields})
    from dataclasses import replace

    from quantaalpha_us.evo.config import Gates

    gate_fields = set(Gates.__dataclass_fields__)
    gates = Gates(**{k: v for k, v in (saved.get("gates") or {}).items() if k in gate_fields})
    base = EvoConfig(
        arm=arm,
        model=saved.get("model", "replay"),
        effort=saved.get("effort", "n/a"),
        seed=int(saved.get("seed", EvoConfig.seed)),
        mode=saved.get("mode", "loop"),
        feedback_mode=saved.get("feedback_mode", "full"),
        schedule=schedule,
        gates=gates,
    )
    return replace(base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(US_ROOT / "data" / "evo_runs"))
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--bars", default=str(US_ROOT / "data" / "us_equities" / "processed"
                                              / "daily_bars_fundamentals.parquet"))
    parser.add_argument("--membership",
                        default=str(US_ROOT / "data" / "us_equities" / "reference"
                                    / "sp500_membership_daily.parquet"))
    parser.add_argument("--work-dir", default=None,
                        help="where the replayed copies are written; defaults to a "
                             "_replay_check directory beside the runs, removed at the end")
    args = parser.parse_args()

    runs = Path(args.runs_dir)
    arms = args.arms or [d.name for d in sorted(runs.iterdir())
                         if d.is_dir() and not d.name.endswith("-raw")
                         and not d.name.startswith("_")
                         and (d / "archive.jsonl").exists()]
    work = Path(args.work_dir) if args.work_dir else runs.parent / "_replay_check"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    print(f"verifying {len(arms)} arm(s): {', '.join(arms)}", flush=True)
    started = time.time()
    bars = apply_membership_filter(pd.read_parquet(args.bars), args.membership)
    alphas = [a.expression for a in alpha101.load()]
    print(f"panel loaded in {time.time() - started:.0f}s; it is shared by every replay",
          flush=True)

    rows = []
    for arm in arms:
        arm_started = time.time()
        source = RunStore(runs / arm)
        config = config_for(arm, source)
        scorer = PanelScorer(bars, config)
        backend = ReplayBackend.from_store(source)
        target = RunStore(work / arm)
        runner = EvolutionRunner(
            config, scorer, backend, target, alpha101=alphas,
            budget=RunBudget(max_requests=10**6, max_total_tokens=10**9,
                             max_consecutive_failures=10**6),
            verbose=False,
        )
        result = runner.run()
        original = load_archive(config, source.archive_path).snapshot()
        replayed = result.archive.snapshot()
        identical = original == replayed
        rows.append({
            "arm": arm,
            "identical": identical,
            "archive_members": len(result.archive),
            "original_members": original.count("\n"),
            "replayed_bytes": len(replayed),
            "saved_replies_used": backend.calls,
            "missing_replies": len(backend.missing),
            "seconds": round(time.time() - arm_started, 1),
        })
        verdict = "IDENTICAL" if identical else "DIFFERS"
        print(f"  {arm:<18}{verdict:<10}{len(result.archive):>4} members  "
              f"{backend.calls:>4} saved replies used  "
              f"{len(backend.missing):>3} missing  "
              f"{time.time() - arm_started:>6.0f}s", flush=True)
        if not identical:
            diff = list(difflib.unified_diff(
                original.splitlines(), replayed.splitlines(),
                fromfile=f"{arm}/original", tofile=f"{arm}/replayed", lineterm="", n=1))
            print("    first divergence:")
            for line in diff[:12]:
                print(f"      {line[:160]}")

    table = pd.DataFrame(rows)
    out = US_ROOT / "data" / "factor_zoo" / "evo_replay_verification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print("\n" + "=" * 96)
    print(table.to_string(index=False))
    print("=" * 96)
    passed = int(table["identical"].sum())
    print(f"\n{passed}/{len(table)} archives reproduced byte for byte from saved replies, "
          f"with zero model calls")
    print(f"-> {out}")
    print(f"total wall clock: {(time.time() - started) / 60:.1f} min")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if passed == len(table) else 1


if __name__ == "__main__":
    raise SystemExit(main())
