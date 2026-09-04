#!/usr/bin/env python3
"""How far ahead does each factor see, and where does its IC halve?

Two curves, because "score it at horizon h" has two readings and they answer
different questions (see `quantaalpha_us.factors.ic_panel`):

  lagged      IC against the single day from open T+h to open T+h+1. Decays.
              The half-life is read off this one.
  cumulative  IC against open T+1 to open T+1+h. Usually rises with h, because
              the return accumulates faster than the signal decays. This is the
              curve that tells you how long to hold.

The half-life becomes the horizon axis of the evolutionary loop's archive
(fast < 5 days, medium 5-21, slow > 21), so this script is where that axis gets
its calibration rather than a guess.

    python scripts/sp500_ic_horizon_curve.py
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

from quantaalpha_us.factors.ic_panel import (  # noqa: E402
    CUMULATIVE,
    FAST_MAX_HALF_LIFE,
    HORIZONS,
    LAGGED,
    MEDIUM_MAX_HALF_LIFE,
    horizon_bucket,
    ic_half_life,
    load_ic_table,
)
from quantaalpha_us.factors.multiple_testing import DEFAULT_NW_LAG, newey_west_tstats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo-dir", default=str(US_ROOT / "data" / "factor_zoo"))
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--lag", type=int, default=DEFAULT_NW_LAG)
    args = parser.parse_args()

    zoo = Path(args.zoo_dir)
    table = load_ic_table(zoo / "cache")
    pool = pd.read_csv(zoo / "pool.csv")
    hurdle_path = zoo / "zoo_hurdle.csv"
    hurdle = pd.read_csv(hurdle_path) if hurdle_path.exists() else None
    if hurdle is None:
        print(f"note: {hurdle_path.name} not found, so nothing is marked as a survivor. "
              "Run sp500_factor_zoo_hurdle.py first for the survivor tables.")

    cut = pd.Timestamp(args.train_end)
    rows = []
    for _, meta in pool[pool["error"].isna()].iterrows():
        expr = meta["expression"]
        record = {"expression": expr, "set": meta["set"], "group": meta["group"]}
        curves: dict[str, dict[int, float]] = {LAGGED: {}, CUMULATIVE: {}}
        for kind in (LAGGED, CUMULATIVE):
            for h in HORIZONS:
                key = (LAGGED, h) if (kind == CUMULATIVE and h == min(HORIZONS)) else (kind, h)
                series = table.ic[key][expr]
                series = series.loc[series.index <= cut].dropna()
                if series.empty:
                    continue
                curves[kind][h] = float(series.mean())
                record[f"ic_{kind}_{h}"] = curves[kind][h]
                record[f"nw_t_{kind}_{h}"] = float(
                    newey_west_tstats(series.to_numpy()[:, None], lag=args.lag)[0]
                )
        # sign is frozen on the one-day horizon, so a curve that flips sign
        # further out shows up as a negative IC rather than as a fresh fit
        sign = np.sign(curves[LAGGED].get(min(HORIZONS), 0.0)) or 1.0
        for kind in (LAGGED, CUMULATIVE):
            for h, v in curves[kind].items():
                record[f"oriented_ic_{kind}_{h}"] = v * sign
        record["half_life"] = ic_half_life(curves[LAGGED])
        record["horizon_bucket"] = horizon_bucket(record["half_life"])
        rows.append(record)

    flag_names = ["passes_random_max", "passes_hlz", "passes_bh", "passes_romano_wolf"]
    if hurdle is not None:
        flags = hurdle.set_index("expression")[flag_names].astype(bool)
        for name in flag_names:
            lookup = flags[name].to_dict()
            for record in rows:
                record[name] = bool(lookup.get(record["expression"], False))
    for record in rows:
        record["survivor"] = any(record.get(name, False) for name in flag_names)
    out = pd.DataFrame(rows)
    out.to_csv(zoo / "ic_horizon_curve.csv", index=False)

    print("\n" + "=" * 96)
    print("A3  IC HORIZON CURVE".center(96))
    print("=" * 96)
    print(f"\nmeasured on the training window only ({out.shape[0]} scorable factors, "
          f"IC dates through {cut.date()})")

    def curve_table(sub: pd.DataFrame, label: str) -> None:
        if sub.empty:
            print(f"\n{label}: none")
            return
        print(f"\n{label}  (n = {len(sub)}, median |mean IC| at each horizon)\n")
        header = f"{'convention':<14}" + "".join(f"{f'h={h}':>11}" for h in HORIZONS)
        print(header)
        print("-" * len(header))
        for kind in (LAGGED, CUMULATIVE):
            line = f"{kind:<14}"
            for h in HORIZONS:
                col = f"oriented_ic_{kind}_{h}"
                line += f"{sub[col].abs().median():>11.5f}" if col in sub else f"{'-':>11}"
            print(line)
        line = f"{'median NW t':<14}"
        for h in HORIZONS:
            col = f"nw_t_{LAGGED}_{h}"
            line += f"{sub[col].abs().median():>11.2f}" if col in sub else f"{'-':>11}"
        print(line)
        finite = sub["half_life"].replace([np.inf], np.nan).dropna()
        never = int(np.isinf(sub["half_life"]).sum())
        print(f"\n  half-life (lagged convention): median "
              f"{finite.median():.1f} days over the {len(finite)} that halve inside the "
              f"63-day grid; {never} of {len(sub)} never halve")
        counts = sub["horizon_bucket"].value_counts()
        print("  buckets: " + ", ".join(
            f"{b} {int(counts.get(b, 0))}" for b in ("fast", "medium", "slow")))

    survivors = out[out["survivor"]]
    curve_table(survivors, "SURVIVORS of any A1 hurdle")
    for group in ("model", "random", "alpha101"):
        curve_table(out[out["group"] == group], f"all {group} factors")

    print("\nnote on reading these two curves together: a factor whose lagged IC halves\n"
          "quickly but whose cumulative IC keeps rising is a fast signal worth holding\n"
          "anyway, because the accumulating return outruns the decaying edge. A factor\n"
          "with a flat lagged curve and a flat cumulative curve is not a horizon story\n"
          "at all -- it is a slow-moving characteristic.")
    print(f"\n-> {zoo / 'ic_horizon_curve.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
