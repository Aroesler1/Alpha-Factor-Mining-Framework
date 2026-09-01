#!/usr/bin/env python3
"""Score mined factor expressions against forward returns and select a set.

Pipeline stage between mining and backtesting:

    candidates (txt, one expression per line)
        -> sanitize -> evaluate -> daily rank IC vs forward open-to-open
        -> greedy selection by |IC t-stat| with correlation dedup
        -> report CSV + YAML snippet for configs (signals.mined_expressions)

Usage:
    python scripts/sp500_score_mined_factors.py \
        --bars data/us_equities/processed_bars.parquet \
        --candidates configs/mined_factors_claude_2026-08.txt \
        --out-dir data/factor_research

The bars file needs long-format daily rows: date, symbol, open, high, low,
close, adj_close, volume (csv or parquet).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.factors.factor_research import score_expressions, select_uncorrelated  # noqa: E402


def load_candidates(path: Path) -> list[str]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", required=True, help="daily bars file (csv or parquet)")
    parser.add_argument("--candidates", default=str(US_ROOT / "configs" / "mined_factors_claude_2026-08.txt"))
    parser.add_argument("--out-dir", default=str(US_ROOT / "data" / "factor_research"))
    parser.add_argument("--min-abs-tstat", type=float, default=2.0)
    parser.add_argument("--max-abs-corr", type=float, default=0.7)
    parser.add_argument("--max-factors", type=int, default=10)
    parser.add_argument("--min-cross-section", type=int, default=30)
    args = parser.parse_args()

    bars_path = Path(args.bars)
    if bars_path.suffix == ".parquet":
        bars = pd.read_parquet(bars_path)
    else:
        bars = pd.read_csv(bars_path)

    candidates = load_candidates(Path(args.candidates))
    print(f"Scoring {len(candidates)} candidate expressions on {bars_path} ...")

    report, signals = score_expressions(bars, candidates, min_cross_section=args.min_cross_section)
    selected = select_uncorrelated(
        report,
        signals,
        min_abs_tstat=args.min_abs_tstat,
        max_abs_corr=args.max_abs_corr,
        max_factors=args.max_factors,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = report.frame().sort_values("ic_tstat", key=lambda s: s.abs(), ascending=False)
    report_path = out_dir / "factor_scores.csv"
    frame.to_csv(report_path, index=False)

    print(f"\nReport -> {report_path}")
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(frame.to_string(index=False))

    print(f"\nSelected {len(selected)} factor(s) "
          f"(|t| >= {args.min_abs_tstat}, |corr| <= {args.max_abs_corr}):")
    snippet_path = out_dir / "selected_factors.yaml"
    lines = ["signals:", "  mined_expressions:"]
    for expr in selected:
        print(f"  {expr}")
        lines.append(f'    - "{expr}"')
    snippet_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nConfig snippet -> {snippet_path}")
    print("Paste the snippet into your research config; the walk-forward and "
          "promotion gates remain the final arbiter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
