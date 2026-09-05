#!/usr/bin/env python3
"""Choose the two swept gate thresholds on validation, then freeze them.

Two numbers in the gate stack are judgement calls rather than specifications:
how large a shared sub-expression counts as a paraphrase (4, 5 or 6 nodes) and
how correlated is too correlated (0.6, 0.7 or 0.8). Picking them by eye on the
final result would be exactly the overfitting this repo spends its time
measuring, so they are chosen once, on the VALIDATION window, and then frozen.

The sweep costs no model calls. It replays a completed run's saved responses
through nine threshold combinations, rebuilds an archive under each, and scores
each archive's top 20 on validation. Replaying the GP arm rather than a model
arm is deliberate: the thresholds should not be tuned on the thing being
measured.

    python scripts/sp500_run_evolution.py --arm gp-loop --backend gp   # first
    python scripts/sp500_evo_threshold_sweep.py --run data/evo_runs/gp-loop
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.evo.backends import ReplayBackend  # noqa: E402
from quantaalpha_us.evo.config import THRESHOLDS_PATH, EvoConfig, Schedule  # noqa: E402
from quantaalpha_us.evo.persistence import RunStore  # noqa: E402
from quantaalpha_us.evo.runner import EvolutionRunner  # noqa: E402
from quantaalpha_us.evo.scoring import PanelScorer  # noqa: E402
from quantaalpha_us.factors import alpha101  # noqa: E402
from quantaalpha_us.factors.ic_panel import apply_membership_filter  # noqa: E402
from quantaalpha_us.llm.budget import RunBudget  # noqa: E402

SUBTREE_GRID = (4, 5, 6)
CORRELATION_GRID = (0.6, 0.7, 0.8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(US_ROOT / "data" / "evo_runs" / "gp-loop"),
                        help="a completed run directory to replay")
    parser.add_argument("--bars", default=str(US_ROOT / "data" / "us_equities" / "processed"
                                              / "daily_bars_fundamentals.parquet"))
    parser.add_argument("--membership",
                        default=str(US_ROOT / "data" / "us_equities" / "reference"
                                    / "sp500_membership_daily.parquet"))
    parser.add_argument("--out", default=str(THRESHOLDS_PATH))
    parser.add_argument("--report-dir", default=str(US_ROOT / "data" / "factor_zoo"))
    args = parser.parse_args()

    source = RunStore(Path(args.run))
    manifest = source.manifest()
    if not manifest:
        raise SystemExit(f"{args.run} has no manifest; run that arm first")
    rounds = int(manifest.get("config", {}).get("schedule", {}).get("rounds", Schedule.rounds))
    seed = int(manifest.get("config", {}).get("seed", EvoConfig.seed))
    print(f"replaying {args.run} (arm {manifest.get('config', {}).get('arm')}, "
          f"{rounds} rounds, backend {manifest.get('backend')})")

    bars = apply_membership_filter(pd.read_parquet(args.bars), args.membership)
    base = EvoConfig(arm="sweep", seed=seed, schedule=Schedule(rounds=rounds))
    # one scorer, memoised: the same candidate stream is replayed nine times and
    # its fit-window metrics do not depend on the thresholds
    scorer = PanelScorer(bars, base, memoize=True)
    alphas = [a.expression for a in alpha101.load()]
    started = time.time()

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for subtree in SUBTREE_GRID:
            for correlation in CORRELATION_GRID:
                config = base.with_thresholds(max_shared_subtree=subtree,
                                              max_abs_corr=correlation)
                store = RunStore(Path(tmp) / f"s{subtree}_c{int(correlation * 100)}")
                backend = ReplayBackend.from_store(source)
                runner = EvolutionRunner(
                    config, scorer, backend, store, alpha101=alphas,
                    # A missing saved response is EXPECTED here and is not a
                    # failure to stop on: under a different threshold an island
                    # can ask for an operator the original run never called (no
                    # cross-niche pair existed for CROSSOVER, say). Three of
                    # those in a row tripped the consecutive-failure limit and
                    # killed the sweep at its second combination. The count is
                    # reported per combination instead.
                    budget=RunBudget(max_requests=10**6, max_total_tokens=10**9,
                                     max_consecutive_failures=10**6),
                    verbose=False,
                )
                result = runner.run()
                top = result.archive.top(20)
                values = [runner._validation_ic.get(e.id) for e in top]
                values = [v for v in values if v is not None and np.isfinite(v)]
                rows.append({
                    "max_shared_subtree": subtree,
                    "max_abs_corr": correlation,
                    "archive_size": len(result.archive),
                    "niches_filled": result.archive.niches_filled(),
                    "top20_validation_ic": float(np.mean(values)) if values else np.nan,
                    "rounds_run": len(result.rounds),
                    "missing_responses": len(backend.missing),
                })
                print(f"  subtree {subtree}, corr {correlation:.1f}: "
                      f"archive {rows[-1]['archive_size']:3d}, "
                      f"niches {rows[-1]['niches_filled']:2d}, "
                      f"validation top-20 IC {rows[-1]['top20_validation_ic']:.5f}"
                      + (f"  [{rows[-1]['missing_responses']} missing replies]"
                         if rows[-1]["missing_responses"] else ""), flush=True)

    grid = pd.DataFrame(rows)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(report_dir / "evo_threshold_sweep.csv", index=False)

    # Ties are real here, not hypothetical: a shared-subtree limit of 4 and one
    # of 5 produced byte-identical archives on this candidate stream, because no
    # candidate ever shared a subtree of exactly 4 nodes. `idxmax` would break
    # that by row order, which is not a reason. Break it toward the STRICTER
    # gate instead: a tie means the looser threshold bought nothing, and the
    # stricter one makes the weaker claim about novelty.
    tolerance = 1e-6
    top = grid["top20_validation_ic"].max()
    tied = grid[grid["top20_validation_ic"] >= top - tolerance]
    best = tied.sort_values(["max_shared_subtree", "max_abs_corr"]).iloc[0]
    if len(tied) > 1:
        print(f"\n{len(tied)} cells within {tolerance} of the best validation IC; "
              "broke the tie toward the stricter gate")
    payload = {
        "chosen": {
            "max_shared_subtree": int(best["max_shared_subtree"]),
            "max_abs_corr": float(best["max_abs_corr"]),
        },
        "criterion": "archive top-20 equal-weight mean IC on the validation window, "
                     "ties broken toward the stricter gate",
        "replayed_run": str(args.run),
        "grid": rows,
        "source": "scripts/sp500_evo_threshold_sweep.py, frozen after this sweep",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")

    print("\n" + "=" * 78)
    print("B4  THRESHOLD SWEEP ON VALIDATION".center(78))
    print("=" * 78)
    pivot = grid.pivot(index="max_shared_subtree", columns="max_abs_corr",
                       values="top20_validation_ic")
    print("\narchive top-20 validation IC\n")
    print(pivot.to_string(float_format=lambda v: f"{v:.5f}"))
    print("\narchive size\n")
    print(grid.pivot(index="max_shared_subtree", columns="max_abs_corr",
                     values="archive_size").to_string())
    print(f"\nchosen: shared-subtree limit {int(best['max_shared_subtree'])}, "
          f"correlation limit {best['max_abs_corr']:.1f} "
          f"(validation IC {best['top20_validation_ic']:.5f}); frozen in {args.out}")
    print(f"sweep wall clock: {(time.time() - started) / 60:.1f} min, zero model calls")
    print(f"-> {report_dir / 'evo_threshold_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
