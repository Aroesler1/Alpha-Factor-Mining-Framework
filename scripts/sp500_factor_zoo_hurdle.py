#!/usr/bin/env python3
"""How many factors survive a correction that counts the whole search?

The repo's baseline table already shows that "best |t| = 11.6 out of 54" is not
evidence, because random expressions from the same grammar reach the same
number. This script asks the next question one factor at a time: pool every
scored candidate -- the two LLM sets, the fundamental set, five random-grammar
draws and Alpha101 -- and ask which individual factors clear a hurdle that
knows how many were tried.

Four hurdles, run on the same pool and the same bootstrap:

  random-max   |t| above the 95th percentile of the bootstrapped MAXIMUM |t|
               across the random-grammar pool. The empirical bar: a factor
               clears it only by beating what a same-sized search over the same
               grammar produces from nothing.
  HLZ          |t| > 3.0 (Harvey, Liu and Zhu 2016).
  BH           Benjamini-Hochberg at 5% across the pool.
  Romano-Wolf  bootstrapped stepdown, family-wise error at 5%.

Everything is measured with a Newey-West t (Bartlett, 21 lags) on the daily
rank-IC series, NOT the iid t the scoring CLI prints. That single change is
most of the story, and the script reports both so the gap is visible.

    python scripts/sp500_build_factor_pool.py      # once, ~1h
    python scripts/sp500_factor_zoo_hurdle.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.factors.ic_panel import LAGGED, load_ic_table  # noqa: E402
from quantaalpha_us.factors.multiple_testing import (  # noqa: E402
    DEFAULT_BLOCK,
    DEFAULT_NW_LAG,
    HLZ_TSTAT_HURDLE,
    evaluate_pool,
    newey_west_tstats,
)

HURDLES = ("random_max", "hlz", "bh", "romano_wolf")


def _iid_tstats(frame: pd.DataFrame) -> pd.Series:
    """The t-stat `score_expressions` reports: mean / (sd / sqrt(n)), iid."""
    n = frame.notna().sum()
    mean = frame.mean()
    sd = frame.std(ddof=1)
    return mean / (sd / np.sqrt(n))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo-dir", default=str(US_ROOT / "data" / "factor_zoo"))
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--lag", type=int, default=DEFAULT_NW_LAG)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--min-days-share", type=float, default=0.9,
                        help="a factor whose training IC covers less than this share of the "
                             "pool's best day count is reported but kept out of the "
                             "complete-case matrix the bootstrap runs on")
    args = parser.parse_args()

    zoo = Path(args.zoo_dir)
    table = load_ic_table(zoo / "cache")
    pool = pd.read_csv(zoo / "pool.csv")
    ic_all = table.ic[(LAGGED, 1)]

    train = ic_all.loc[ic_all.index <= pd.Timestamp(args.train_end)]
    hold = ic_all.loc[ic_all.index > pd.Timestamp(args.train_end)]
    print(f"training window {train.index.min().date()} -> {train.index.max().date()} "
          f"({len(train):,} days); holdout {hold.index.min().date()} -> "
          f"{hold.index.max().date()} ({len(hold):,} days)")

    scorable = pool[pool["error"].isna()].copy()
    days = train[scorable["expression"]].notna().sum()
    keep = days >= args.min_days_share * days.max()
    dropped = scorable[~scorable["expression"].map(keep).astype(bool)]
    if len(dropped):
        print(f"\n{len(dropped)} factor(s) held out of the complete-case matrix for a short "
              f"training sample (< {args.min_days_share:.0%} of {int(days.max()):,} days):")
        for _, row in dropped.iterrows():
            print(f"  [{row['set']}] {int(days[row['expression']]):5,} days  "
                  f"{row['expression'][:74]}")
    members = scorable[scorable["expression"].map(keep).astype(bool)].reset_index(drop=True)

    matrix = train[members["expression"].tolist()].dropna(axis=0, how="any")
    print(f"\npool: {len(members)} factors, complete-case on {len(matrix):,} of "
          f"{len(train):,} training days "
          f"({matrix.index.min().date()} -> {matrix.index.max().date()}). "
          f"The gap is rolling-window warm-up, and it is shared by every factor "
          f"so the maximum below is a maximum over one sample.")

    is_random = (members["group"] == "random").to_numpy()
    print(f"empirical null drawn from {int(is_random.sum())} random-grammar factors; "
          f"{args.draws:,} circular block draws, block {args.block}, NW lag {args.lag}",
          flush=True)
    verdict = evaluate_pool(
        matrix.to_numpy(), random_columns=is_random, block=args.block,
        draws=args.draws, lag=args.lag, alpha=args.alpha, seed=args.seed,
    )

    out = members.copy()
    out["train_days"] = len(matrix)
    out["mean_ic_train"] = matrix.mean().to_numpy()
    out["iid_t_train"] = _iid_tstats(matrix).to_numpy()
    out["nw_t_train"] = verdict.nw_tstat
    out["p_value"] = verdict.p_value
    out["passes_random_max"] = verdict.passes_random_max
    out["passes_hlz"] = verdict.passes_hlz
    out["passes_bh"] = verdict.passes_bh
    out["passes_romano_wolf"] = verdict.passes_romano_wolf
    out["rw_critical"] = verdict.romano_wolf_critical

    # Holdout, with the training sign frozen. Orienting by the training sign is
    # what makes "kept its sign" mean something: refitting the sign out of
    # sample is the same overfitting one layer down.
    sign = np.sign(out["mean_ic_train"]).replace(0, 1).to_numpy()
    hold_matrix = hold[members["expression"].tolist()]
    oriented_hold = hold_matrix * sign
    out["mean_ic_holdout"] = oriented_hold.mean().to_numpy()
    out["nw_t_holdout"] = newey_west_tstats(
        oriented_hold.fillna(0.0).to_numpy(), lag=args.lag
    )
    out["holdout_days"] = hold_matrix.notna().sum().to_numpy()
    out["sign_held"] = out["mean_ic_holdout"] > 0
    out["retention"] = out["mean_ic_holdout"] / (out["mean_ic_train"] * sign)

    zoo.mkdir(parents=True, exist_ok=True)
    out.to_csv(zoo / "zoo_hurdle.csv", index=False)

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 100)
    print("A1  FACTOR-ZOO HURDLE".center(100))
    print("=" * 100)
    print(f"\ncritical values at alpha = {args.alpha:.2f}")
    print(f"  random-grammar max |t| (95th pct of the bootstrap)  {verdict.random_max_critical:6.2f}")
    print(f"  Harvey-Liu-Zhu                                      {HLZ_TSTAT_HURDLE:6.2f}")
    print(f"  Benjamini-Hochberg p threshold                      "
          f"{verdict.bh_threshold:.3e}" if verdict.bh_threshold else
          "  Benjamini-Hochberg p threshold                      nothing rejected")
    finite_rw = out["rw_critical"][np.isfinite(out["rw_critical"])]
    if len(finite_rw):
        print(f"  Romano-Wolf, first step / last step used            "
              f"{finite_rw.max():6.2f} / {finite_rw.min():6.2f}")

    print("\nthe HAC correction on its own:")
    print(f"  max |iid t| over the pool   {out['iid_t_train'].abs().max():6.2f}")
    print(f"  max |NW  t| over the pool   {out['nw_t_train'].abs().max():6.2f}")
    print(f"  median |iid t| / |NW t|     "
          f"{(out['iid_t_train'].abs() / out['nw_t_train'].abs()).median():6.2f}")

    print("\nsurvivors by group\n")
    header = f"{'group':<10}{'set':<24}{'n':>5}" + "".join(f"{h:>14}" for h in HURDLES)
    print(header)
    print("-" * len(header))
    for group in ("model", "random", "alpha101"):
        sub_g = out[out["group"] == group]
        if sub_g.empty:
            continue
        for set_name, sub in sub_g.groupby("set"):
            line = f"{group:<10}{set_name:<24}{len(sub):>5}"
            for h in HURDLES:
                line += f"{int(sub['passes_' + h].sum()):>14}"
            print(line)
        line = f"{group:<10}{'ALL ' + group:<24}{len(sub_g):>5}"
        for h in HURDLES:
            line += f"{int(sub_g['passes_' + h].sum()):>14}"
        print(line)
        print("-" * len(header))

    print("\nsurvivors on the holdout (training sign frozen)\n")
    print(f"{'hurdle':<14}{'group':<10}{'survivors':>10}{'sign held':>12}"
          f"{'median retention':>19}{'median NW t OOS':>18}")
    print("-" * 83)
    for h in HURDLES:
        for group in ("model", "random", "alpha101"):
            sub = out[(out["group"] == group) & out["passes_" + h]]
            if sub.empty:
                print(f"{h:<14}{group:<10}{0:>10}{'-':>12}{'-':>19}{'-':>18}")
                continue
            held = f"{int(sub['sign_held'].sum())}/{len(sub)}"
            print(f"{h:<14}{group:<10}{len(sub):>10}{held:>12}"
                  f"{sub['retention'].median():>19.1%}"
                  f"{sub['nw_t_holdout'].median():>18.2f}")
        print("-" * 83)

    strict = out[out["passes_romano_wolf"] & out["passes_random_max"]]
    print(f"\nfactors clearing BOTH the empirical random-grammar bar and Romano-Wolf: "
          f"{len(strict)} of {len(out)}")
    if len(strict):
        by_group = strict.groupby("group").size()
        print("  by group: " + ", ".join(f"{k} {v}" for k, v in by_group.items()))
        # The survivor list is dominated by restatements of one idea, so a count
        # of survivors overstates how many distinct effects cleared the bar.
        # $dollar_volume is the giveaway: it is size, spelled many ways.
        touches_dv = strict["expression"].str.contains(r"\$dollar_volume", regex=True)
        print(f"  of which mention $dollar_volume: {int(touches_dv.sum())} "
              f"({touches_dv.mean():.0%}); their median holdout retention is "
              f"{strict.loc[touches_dv, 'retention'].median():.0%} against "
              f"{strict.loc[~touches_dv, 'retention'].median():.0%} for the rest")
        print("\n  strongest 15 by |NW t| on the training window:\n")
        with pd.option_context("display.width", 200, "display.max_colwidth", 56):
            print(strict.reindex(
                strict["nw_t_train"].abs().sort_values(ascending=False).index).head(15)[
                ["set", "expression", "nw_t_train", "mean_ic_train", "mean_ic_holdout",
                 "retention", "sign_held"]].to_string(index=False))
    print(f"\n-> {zoo / 'zoo_hurdle.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
