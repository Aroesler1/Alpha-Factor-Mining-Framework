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

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.factors.experiment_trace import (  # noqa: E402
    ExperimentTrace,
    TraceRecord,
)
from quantaalpha_us.factors.factor_research import (  # noqa: E402
    holdout_frame,
    score_expressions,
    select_uncorrelated,
)


def load_candidates(path: Path) -> list[str]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def apply_membership_filter(bars: pd.DataFrame, membership: str) -> pd.DataFrame:
    """Restrict bars to point-in-time index membership, joined on (date, permno).

    This must run before anything is computed. The bars panel carries every name
    that was ever a constituent over the sample, which is ~745 per date against
    the index's ~502: on 2005-06-15, 315 of 814 names in the panel (39%) were
    not in the S&P 500 that day, and the surplus are future members as often as
    former ones. Ranking a factor cross-sectionally over that panel scores it on
    a universe no strategy could have held, and it inflates significance -- the
    best candidate measured |t| = 8.21 unfiltered against 7.56 filtered.

    Joined on permno rather than symbol so ticker reuse cannot reintroduce the
    identity ambiguity the PERMNO keying exists to remove.
    """
    if str(membership).lower() == "none":
        print("WARNING: scoring the RAW panel; it contains names that were not "
              "index members on the dates they are ranked against.")
        return bars

    path = Path(membership)
    if not path.exists():
        raise SystemExit(
            f"Membership file not found: {path}. Build it with "
            "scripts/sp500_build_membership.py, or pass --membership none to "
            "score the raw panel deliberately."
        )
    members = pd.read_parquet(path)
    if "active" in members.columns:
        members = members[members["active"]]
    members = members[["date", "permno"]].dropna().drop_duplicates()
    members["date"] = pd.to_datetime(members["date"])
    members["permno"] = members["permno"].astype("int64")

    before_rows, before_names = len(bars), bars["permno"].nunique()
    out = bars.assign(
        date=pd.to_datetime(bars["date"]),
        permno=bars["permno"].astype("int64"),
    ).merge(members, on=["date", "permno"], how="inner")
    print(f"Point-in-time membership filter: {before_rows:,} -> {len(out):,} rows, "
          f"{before_names} -> {out['permno'].nunique()} distinct names "
          f"({out.groupby('date')['permno'].nunique().mean():.0f} per date)")
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
    parser.add_argument("--train-end", default="2017-12-31",
                        help="last date whose IC may influence selection. Everything after "
                             "is a held-out window scored only once, with the in-sample "
                             "choices -- expression AND sign -- frozen. Pass 'none' to "
                             "select on the full sample, which is what the factor-zoo "
                             "literature is about.")
    parser.add_argument("--membership",
                        default=str(US_ROOT / "data" / "us_equities" / "reference"
                                    / "sp500_membership_daily.parquet"),
                        help="point-in-time index membership, joined on (date, permno). "
                             "Pass 'none' to score the raw panel, which is almost never "
                             "what you want -- see the note in main().")
    parser.add_argument("--model", default="unrecorded",
                        help="model that generated the candidates. Written to the report, "
                             "the YAML snippet and the trace, so two mining runs can be "
                             "compared after the fact instead of being anonymous.")
    parser.add_argument("--trace", type=Path, default=None,
                        help="append-only JSONL trace of every hypothesis considered")
    args = parser.parse_args()

    bars_path = Path(args.bars)
    if bars_path.suffix == ".parquet":
        bars = pd.read_parquet(bars_path)
    else:
        bars = pd.read_csv(bars_path)

    bars = apply_membership_filter(bars, args.membership)

    candidates = load_candidates(Path(args.candidates))
    print(f"Scoring {len(candidates)} candidate expressions on {bars_path} ...")

    all_dates = pd.to_datetime(bars["date"]).drop_duplicates().sort_values()
    if str(args.train_end).lower() == "none":
        train_dates, test_dates = None, None
        print("WARNING: selecting on the full sample; reported ICs are in-sample.")
    else:
        cut = pd.Timestamp(args.train_end)
        train_dates = pd.DatetimeIndex(all_dates[all_dates <= cut])
        test_dates = pd.DatetimeIndex(all_dates[all_dates > cut])
        if len(test_dates) < 250:
            raise SystemExit(
                f"--train-end {args.train_end} leaves only {len(test_dates)} holdout "
                "days; that is too few to say anything about decay."
            )
        print(f"Selection window: {train_dates.min().date()} -> {train_dates.max().date()} "
              f"({len(train_dates):,} days)   holdout: {test_dates.min().date()} -> "
              f"{test_dates.max().date()} ({len(test_dates):,} days)")

    report, signals = score_expressions(bars, candidates,
                                        min_cross_section=args.min_cross_section,
                                        ic_dates=train_dates)
    selected = select_uncorrelated(
        report,
        signals,
        min_abs_tstat=args.min_abs_tstat,
        max_abs_corr=args.max_abs_corr,
        max_factors=args.max_factors,
    )

    # Record every hypothesis and its verdict, failures included. The trace is
    # what makes a mining run auditable, and its distinct-expression count is
    # the honest multiple-testing denominator -- counting only survivors
    # understates the search by exactly the number of rejections.
    if args.trace is not None:
        trace = ExperimentTrace(args.trace)
        selected_set = set(selected)
        for score in report.scores:
            if score.error is not None:
                verdict = "rejected_sanitizer"
            elif score.expression in selected_set or f"-({score.expression})" in selected_set:
                verdict = "selected"
            elif not np.isfinite(score.ic_tstat) or abs(score.ic_tstat) < args.min_abs_tstat:
                verdict = "rejected_score"
            else:
                verdict = "rejected_correlated"
            trace.append(TraceRecord(
                expression=score.expression,
                verdict=verdict,
                model=args.model,
                mean_ic=None if not np.isfinite(score.mean_ic) else float(score.mean_ic),
                ic_tstat=None if not np.isfinite(score.ic_tstat) else float(score.ic_tstat),
                signal_autocorr=(None if not np.isfinite(score.signal_autocorr)
                                 else float(score.signal_autocorr)),
                coverage=float(score.coverage),
                error=score.error,
            ))
        print(f"\ntrace -> {args.trace}")
        print(f"  verdicts: {trace.summary()}")
        print(f"  distinct hypotheses ever tried: {trace.n_hypotheses()} "
              f"(use this as the DSR trial count, not {len(selected)})")

    # Score the SELECTED factors on the holdout, with the in-sample decisions --
    # which expressions, and which sign each was oriented to -- frozen. Refitting
    # the sign out of sample would be the same overfitting one layer down.
    oos = None
    if test_dates is not None and selected:
        oos = holdout_frame(bars, selected, report, test_dates=test_dates,
                            min_cross_section=args.min_cross_section, model=args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = report.frame().sort_values("ic_tstat", key=lambda s: s.abs(), ascending=False)
    # Provenance: without it two runs' scores are indistinguishable after the
    # fact, and "did the new model beat the old one" is unanswerable.
    frame.insert(1, "model", args.model)
    report_path = out_dir / "factor_scores.csv"
    frame.to_csv(report_path, index=False)

    print(f"\nReport -> {report_path}")
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(frame.to_string(index=False))

    print(f"\nSelected {len(selected)} factor(s) "
          f"(|t| >= {args.min_abs_tstat}, |corr| <= {args.max_abs_corr}):")
    snippet_path = out_dir / "selected_factors.yaml"
    lines = [
        f"# Generated by: {args.model}",
        f"# Scored on: {bars_path.name} ({len(bars):,} rows after membership filter)",
        f"# Membership: {args.membership}",
        f"# Gates: |t| >= {args.min_abs_tstat}, |corr| <= {args.max_abs_corr}, "
        f"max {args.max_factors} factors",
        f"# Candidates considered: {len(candidates)}",
        "signals:",
        "  mined_expressions:",
    ]
    for expr in selected:
        print(f"  {expr}")
        lines.append(f'    - "{expr}"')
    snippet_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if oos is not None and not oos.empty:
        oos_path = out_dir / "holdout_scores.csv"
        oos.to_csv(oos_path, index=False)
        held = int(oos["sign_held"].sum())
        median_retention = float(oos["retention"].median())
        print(f"\nHOLDOUT ({test_dates.min().date()} -> {test_dates.max().date()}, "
              f"{len(test_dates):,} days), in-sample choices frozen:")
        with pd.option_context("display.width", 170, "display.max_colwidth", 52):
            print(oos[["expression", "is_mean_ic", "oos_mean_ic", "oos_ic_tstat",
                       "sign_held"]].to_string(index=False))
        print(f"\n  factors keeping their sign out of sample: {held}/{len(oos)}")
        print(f"  median IC retention: {median_retention:.1%}")
        print(f"  -> {oos_path}")

    print(f"\nConfig snippet -> {snippet_path}")
    print("Paste the snippet into your research config; the walk-forward and "
          "promotion gates remain the final arbiter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
