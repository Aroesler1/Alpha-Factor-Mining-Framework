#!/usr/bin/env python3
"""Rebuild the summary tables in reports/ from the committed result tables.

Offline: reads only CSVs under data/ (no price panel, no vendor data), so the
summaries quoted in the README can be checked against their sources.

    python scripts/sp500_build_reports.py           # rewrite reports/*.csv
    python scripts/sp500_build_reports.py --check   # verify, write nothing

Covered: baseline_comparison, publication_decay, rw_survivors,
quarterly_sampling, inference_diagnostics. evo_comparison_audit.csv is built by
sp500_evo_audit.py; the fundamentals provenance files by
sp500_fundamentals_provenance.py; literature_versions.csv is a hand-entered
record of published numbers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    zoo = root / "data" / "factor_zoo"
    baseline = pd.read_csv(root / "data" / "baseline_comparison" / "comparison_table.csv")
    decay = pd.read_csv(zoo / "alpha101_decay_summary.csv")
    hurdle = pd.read_csv(zoo / "zoo_hurdle.csv")
    quarterly = pd.read_csv(zoo / "fundamental_quarterly.csv")

    rows = []
    for group, frame in hurdle.groupby("group", sort=True):
        survivors = frame[frame["passes_romano_wolf"].astype(bool)]
        rows.append({
            "group": group,
            "factors": len(frame),
            "rw_survivors": len(survivors),
            "survival_rate": len(survivors) / len(frame),
            "sign_held": int(survivors["sign_held"].astype(bool).sum()),
            "median_retention": float(survivors["retention"].median()),
        })
    rw = pd.DataFrame(rows)

    def sampling_row(name: str, column: str) -> dict:
        t = quarterly[column].abs()
        return {"sampling": name, "factors": int(t.notna().sum()),
                "above_abs_t_2": int((t > 2).sum()), "median_abs_t": float(t.median())}

    sampling = pd.DataFrame([
        sampling_row("daily_iid", "daily_iid_t"),
        sampling_row("daily_hac", "daily_nw_t"),
        sampling_row("quarter_first", "q_first_t"),
        sampling_row("quarter_mid", "q_mid_t"),
        sampling_row("quarter_last", "q_last_t"),
    ])

    daily_days = float(quarterly["daily_days"].median())
    quarter_obs = float(quarterly["quarterly_obs"].median())
    diagnostics = pd.DataFrame({
        "metric": ["median_abs_iid_over_hac", "median_quarter_t_over_daily_iid_t",
                   "sqrt_sample_size_reference"],
        "value": [
            float((hurdle["iid_t_train"].abs() / hurdle["nw_t_train"].abs()).median()),
            float((quarterly["quarterly_t"].abs() / quarterly["daily_iid_t"].abs()).median()),
            float(np.sqrt(quarter_obs / daily_days)),
        ],
    })
    return {
        "baseline_comparison.csv": baseline,
        "publication_decay.csv": decay,
        "rw_survivors.csv": rw,
        "quarterly_sampling.csv": sampling,
        "inference_diagnostics.csv": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify reports/ without writing")
    args = parser.parse_args()
    reports = ROOT / "reports"
    tables = build()
    for name, table in tables.items():
        path = reports / name
        if args.check:
            pd.testing.assert_frame_equal(table, pd.read_csv(path), check_exact=False,
                                          rtol=1e-9, atol=1e-12, check_dtype=False)
        else:
            reports.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)
    print(f"{'Verified' if args.check else 'Wrote'} {len(tables)} report tables from data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
