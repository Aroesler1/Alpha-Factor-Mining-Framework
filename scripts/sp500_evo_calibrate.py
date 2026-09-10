#!/usr/bin/env python3
"""Measure the one constant the archive's turnover axis needs.

The archive splits factors into low- and high-turnover niches. Any split point
is arbitrary unless it is anchored to something, so it is anchored to the median
daily turnover of the 50 transcribed Alpha101 formulas over the FIT window --
which makes "high turnover" mean "faster than half of the published alphas"
rather than "faster than a number someone picked".

Measured on the fit window only, so the calibration itself cannot smuggle in
information from validation or holdout.

    python scripts/sp500_evo_calibrate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.evo.config import CALIBRATION_PATH, EvoConfig  # noqa: E402
from quantaalpha_us.evo.scoring import PanelScorer  # noqa: E402
from quantaalpha_us.factors import alpha101  # noqa: E402
from quantaalpha_us.factors.expression_evaluator import ExpressionError  # noqa: E402
from quantaalpha_us.factors.ic_panel import (  # noqa: E402
    apply_membership_filter,
    daily_turnover,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", default=str(US_ROOT / "data" / "us_equities" / "processed"
                                              / "daily_bars_fundamentals.parquet"))
    parser.add_argument("--membership",
                        default=str(US_ROOT / "data" / "us_equities" / "reference"
                                    / "sp500_membership_daily.parquet"))
    parser.add_argument("--out", default=str(CALIBRATION_PATH))
    args = parser.parse_args()

    config = EvoConfig()
    bars = apply_membership_filter(pd.read_parquet(args.bars), args.membership)
    scorer = PanelScorer(bars, config)
    print(f"fit window {scorer.fit_dates.min().date()} -> {scorer.fit_dates.max().date()} "
          f"({len(scorer.fit_dates):,} days)", flush=True)

    rows = []
    for alpha in alpha101.load():
        try:
            signal = scorer.evaluate(alpha.expression)
        except ExpressionError as exc:
            print(f"  {alpha.name}: {exc}")
            continue
        window = signal.loc[signal.index.isin(scorer.fit_dates)]
        rows.append({"name": alpha.name, "turnover": daily_turnover(window),
                     "coverage": float(window.notna().mean().mean())})
    frame = pd.DataFrame(rows).sort_values("turnover")
    median = float(frame["turnover"].median())

    out_dir = US_ROOT / "data" / "factor_zoo"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "alpha101_turnover.csv", index=False)
    Path(args.out).write_text(json.dumps({
        "alpha101_turnover_median": median,
        "n_alphas": len(frame),
        "fit_start": config.windows.fit_start,
        "fit_end": config.windows.fit_end,
        "source": "scripts/sp500_evo_calibrate.py on the fit window",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nAlpha101 daily one-way turnover on the fit window, {len(frame)} formulas")
    print(f"  min {frame['turnover'].min():.4f}   median {median:.4f}   "
          f"max {frame['turnover'].max():.4f}")
    print(f"  quartiles {np.percentile(frame['turnover'], [25, 75]).round(4)}")
    print(f"\nslowest 5:\n{frame.head(5).to_string(index=False)}")
    print(f"\nfastest 5:\n{frame.tail(5).to_string(index=False)}")
    print(f"\n-> {args.out}")
    print(f"-> {out_dir / 'alpha101_turnover.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
