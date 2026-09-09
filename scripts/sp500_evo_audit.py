#!/usr/bin/env python3
"""Verify completed-run aggregates and adjust five published comparisons offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantaalpha_us.evo.comparison_audit import adjust_comparisons


def build(root=ROOT):
    source = root / "data/factor_zoo"
    paired = pd.read_csv(source / "evo_paired_bootstrap.csv")
    combo = pd.read_csv(source / "evo_combined_table.csv").set_index("arm")
    expected = {(f"{m}-loop", other) for m in ("sonnet5", "opus5")
                for other in (f"{m}-oneshot", "gp-loop")}
    expected.add(("sonnet5-loop", "sonnet5-scalar"))
    if len(paired) != 5 or set(zip(paired.a, paired.b)) != expected:
        raise ValueError("The published five-comparison family is incomplete or changed")
    for row in paired.itertuples():
        gap = combo.loc[row.a, "equal_weight_holdout_ic"] - combo.loc[row.b, "equal_weight_holdout_ic"]
        if not np.isclose(gap, row.observed, atol=1e-12, rtol=0):
            raise ValueError(f"Combined IC difference does not reproduce: {row.comparison}")
    return adjust_comparisons(paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    table = build()
    path = ROOT / "reports/evo_comparison_audit.csv"
    if args.check:
        pd.testing.assert_frame_equal(table, pd.read_csv(path), check_exact=False,
                                      rtol=1e-12, atol=1e-12)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    print(f"Verified {len(table)} published comparisons; "
          f"{int(table.reject_holm_5pct.sum())} pass Holm at 5%. No holdout was rescored.")


if __name__ == "__main__":
    main()
