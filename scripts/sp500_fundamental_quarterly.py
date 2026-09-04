#!/usr/bin/env python3
"""Re-score the fundamental factors on non-overlapping quarters.

The reason, stated up front because it is the whole point: Compustat
fundamentals change on the report date and then sit still. A factor built on
`$roa` takes ONE new value per firm per quarter, but the daily IC series scores
it on every trading day, so the same ~63 observations of the same number are
counted 63 times. The mean IC is unaffected -- it is an average of per-day
cross-sections and each of those is a legitimate cross-section -- but the
t-statistic is not: it divides by sqrt(n) with an n that counts repeats.

So this script scores the same expressions on one IC date per quarter, which is
the coarsest grid on which consecutive observations carry genuinely new
fundamental information, and reports the two t-statistics side by side. If the
daily figure is honest, the quarterly one should fall by roughly the square
root of the number of days per quarter, and no further.

    python scripts/sp500_fundamental_quarterly.py
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
    DEFAULT_NW_LAG,
    newey_west_tstats,
)


# One IC date per quarter, at three different positions inside it. Quarter-end
# is not a neutral day -- window dressing, index rebalances and the earnings
# calendar all cluster there -- so a result that only holds on the last day of
# the quarter is a calendar effect, not a statement about fundamentals. Running
# all three is what separates the two.
QUARTER_POSITIONS = ("first", "mid", "last")


def quarterly_dates(index: pd.DatetimeIndex, position: str = "last") -> pd.DatetimeIndex:
    """One IC date per calendar quarter, at `position` within it."""
    frame = pd.DataFrame({"date": pd.DatetimeIndex(index)})
    frame["q"] = frame["date"].dt.to_period("Q")
    if position == "last":
        picked = frame.groupby("q")["date"].max()
    elif position == "first":
        picked = frame.groupby("q")["date"].min()
    elif position == "mid":
        picked = frame.groupby("q")["date"].apply(
            lambda s: s.sort_values().iloc[len(s) // 2]
        )
    else:
        raise ValueError(f"unknown quarter position: {position!r}")
    return pd.DatetimeIndex(picked.sort_values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo-dir", default=str(US_ROOT / "data" / "factor_zoo"))
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--lag", type=int, default=DEFAULT_NW_LAG)
    parser.add_argument("--sets", nargs="*", default=["fundamental-2026-09"])
    args = parser.parse_args()

    zoo = Path(args.zoo_dir)
    table = load_ic_table(zoo / "cache")
    pool = pd.read_csv(zoo / "pool.csv")
    ic = table.ic[(LAGGED, 1)]
    cut = pd.Timestamp(args.train_end)

    members = pool[pool["set"].isin(args.sets) & pool["error"].isna()]
    if members.empty:
        raise SystemExit(f"no scorable factors in sets {args.sets}")

    rows = []
    for _, meta in members.iterrows():
        expr = meta["expression"]
        daily = ic[expr]
        daily = daily.loc[daily.index <= cut].dropna()
        # one observation per quarter is already non-overlapping under the
        # one-day forward label, so the HAC correction has nothing left to do;
        # lag 0 is the iid t and is reported as such
        record = {
            "expression": expr,
            "set": meta["set"],
            "daily_days": len(daily),
            "daily_mean_ic": float(daily.mean()),
            "daily_iid_t": float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))),
            "daily_nw_t": float(newey_west_tstats(daily.to_numpy()[:, None], lag=args.lag)[0]),
        }
        for position in QUARTER_POSITIONS:
            picked = daily.loc[daily.index.isin(quarterly_dates(daily.index, position))]
            record[f"q_{position}_obs"] = len(picked)
            record[f"q_{position}_mean_ic"] = float(picked.mean())
            record[f"q_{position}_t"] = float(
                picked.mean() / (picked.std(ddof=1) / np.sqrt(len(picked)))
            ) if len(picked) > 2 else float("nan")
        record["quarterly_obs"] = record["q_last_obs"]
        record["quarterly_mean_ic"] = record["q_last_mean_ic"]
        record["quarterly_t"] = record["q_last_t"]
        # the honest headline is the WEAKEST of the three positions, so a result
        # cannot be carried by whichever day of the quarter happened to work
        record["quarterly_t_min_abs"] = min(
            abs(record[f"q_{p}_t"]) for p in QUARTER_POSITIONS
        )
        rows.append(record)

    out = pd.DataFrame(rows)
    out["ic_retention"] = out["quarterly_mean_ic"] / out["daily_mean_ic"]
    out["t_retention_vs_iid"] = out["quarterly_t"].abs() / out["daily_iid_t"].abs()
    out["t_retention_vs_nw"] = out["quarterly_t"].abs() / out["daily_nw_t"].abs()
    zoo.mkdir(parents=True, exist_ok=True)
    out.to_csv(zoo / "fundamental_quarterly.csv", index=False)

    days_per_obs = out["daily_days"].iloc[0] / out["quarterly_obs"].iloc[0]
    print("\n" + "=" * 112)
    print("A5  FUNDAMENTAL FACTORS ON NON-OVERLAPPING QUARTERS".center(112))
    print("=" * 112)
    print(f"\ntraining window through {cut.date()}: {int(out['daily_days'].max()):,} daily "
          f"IC dates against {int(out['quarterly_obs'].max())} quarter-end dates "
          f"({days_per_obs:.0f} days per quarter).")
    print(f"If the daily t were honest, the quarterly t would be about "
          f"1/sqrt({days_per_obs:.0f}) = {1 / np.sqrt(days_per_obs):.2f} of it, "
          f"and no smaller.\n")
    show = out.copy()
    show["expression"] = show["expression"].str.slice(0, 44)
    for col in ("daily_mean_ic", "quarterly_mean_ic"):
        show[col] = show[col].map(lambda v: f"{v:+.5f}")
    for col in ("ic_retention", "t_retention_vs_iid", "t_retention_vs_nw"):
        show[col] = show[col].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "-")
    with pd.option_context("display.width", 220):
        print(show[["expression", "daily_mean_ic", "daily_iid_t", "daily_nw_t",
                    "quarterly_mean_ic", "quarterly_t", "ic_retention",
                    "t_retention_vs_iid", "t_retention_vs_nw"]].to_string(index=False))

    print("\nthe same test at three positions inside the quarter, because "
          "quarter-end is not a neutral day\n")
    print(f"{'position':<12}{'median |t|':>12}{'|t| > 2':>10}{'median mean IC':>18}")
    print("-" * 52)
    for position in QUARTER_POSITIONS:
        column = out[f"q_{position}_t"]
        significant = f"{int((column.abs() > 2).sum())}/{len(out)}"
        print(f"{position:<12}{column.abs().median():>12.2f}{significant:>10}"
              f"{out[f'q_{position}_mean_ic'].median():>+18.5f}")
    significant = f"{int((out['quarterly_t_min_abs'] > 2).sum())}/{len(out)}"
    print(f"{'weakest':<12}{out['quarterly_t_min_abs'].median():>12.2f}{significant:>10}")

    print(f"\nmedian IC retention (quarterly / daily): {out['ic_retention'].median():.2f}")
    print(f"median t retention against the iid daily t: "
          f"{out['t_retention_vs_iid'].median():.2f}  "
          f"(sqrt-rule reference {1 / np.sqrt(days_per_obs):.2f})")
    print(f"median t retention against the NW daily t:  "
          f"{out['t_retention_vs_nw'].median():.2f}")
    print(f"factors with |quarterly t| > 2: "
          f"{int((out['quarterly_t'].abs() > 2).sum())}/{len(out)}   "
          f"|daily NW t| > 2: {int((out['daily_nw_t'].abs() > 2).sum())}/{len(out)}   "
          f"|daily iid t| > 2: {int((out['daily_iid_t'].abs() > 2).sum())}/{len(out)}")
    print(f"\n-> {zoo / 'fundamental_quarterly.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
