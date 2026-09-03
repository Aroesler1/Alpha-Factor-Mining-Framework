#!/usr/bin/env python3
"""Does the LLM add anything? Score every candidate set against two baselines.

Runs the LLM sets, a seeded random-grammar null, and an Alpha101 baseline
through one scoring pipeline and prints a single comparison table, plus the
memorization test (how close each LLM candidate is to a published alpha).

    python scripts/sp500_run_baseline_comparison.py \
        --bars data/us_equities/processed/daily_bars.parquet

Everything is loaded and pivoted once; the per-set work is the same
`score_expressions` / `select_uncorrelated` / `holdout_frame` the scoring CLI
calls, so the numbers are comparable to a `sp500_score_mined_factors.py` run by
construction rather than by intention.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.factors.expression_evaluator import (  # noqa: E402
    ExpressionEvaluator,
    build_field_panels,
)
from quantaalpha_us.factors.factor_research import (  # noqa: E402
    _corr_flat,
    holdout_frame,
    mean_daily_rank_correlation,
    ranked_flat,
    ranked_matrix,
    score_expressions,
    select_uncorrelated,
)
from quantaalpha_us.factors.random_expressions import (  # noqa: E402
    PRICE_FIELDS,
    GrammarSampler,
    structure_profile,
)

MEMORIZATION_THRESHOLD = 0.9
# Every k-th date for the memorization correlations. Correlation is pooled over
# millions of paired observations either way -- at stride 5 each pair still
# rests on ~1.5M points, a standard error near 0.0008 against a 0.9 threshold --
# and the stride is what keeps the cached rank vectors inside memory.
MEMORIZATION_DATE_STRIDE = 5


def load_candidates(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def apply_membership_filter(bars: pd.DataFrame, membership: Path) -> pd.DataFrame:
    """Restrict bars to point-in-time index membership, joined on (date, permno)."""
    members = pd.read_parquet(membership)
    if "active" in members.columns:
        members = members[members["active"]]
    members = members[["date", "permno"]].dropna().drop_duplicates()
    members["date"] = pd.to_datetime(members["date"])
    members["permno"] = members["permno"].astype("int64")
    out = bars.assign(
        date=pd.to_datetime(bars["date"]),
        permno=bars["permno"].astype("int64"),
    ).merge(members, on=["date", "permno"], how="inner")
    print(f"Point-in-time membership filter: {len(bars):,} -> {len(out):,} rows, "
          f"{out['permno'].nunique()} distinct names")
    return out


def expected_max_abs_t(n: int) -> float:
    """E[max |t|] over n independent draws, if every candidate were pure noise.

    E[M] = integral of (1 - F(x)^n) dx with F the CDF of |N(0,1)|. This is the
    bar a search of size n has to clear before "the best factor has |t| = 4"
    means anything at all.

    It is an IDEALISATION in two ways that pull in opposite directions, and
    neither is small: candidates within a set are heavily correlated, which
    lowers the true expected maximum below this figure, while daily ICs are
    autocorrelated, which inflates the realised t-stat above a clean N(0,1).
    The random-grammar row measures the combination empirically and is the
    honest benchmark; this column is the textbook reference point.
    """
    if n < 1:
        return float("nan")
    xs = np.linspace(0.0, 12.0, 24001)
    cdf = np.array([math.erf(x / math.sqrt(2.0)) for x in xs])  # P(|Z| <= x)
    return float(np.trapezoid(1.0 - cdf ** n, xs))


def draw_random_set(seed: int, size: int, reference: list[str],
                    evaluator: ExpressionEvaluator) -> list[str]:
    """`size` random expressions that actually compute a usable cross-section.

    About a quarter of raw draws from the grammar evaluate to a constant per
    date -- BOUND clipping a count, say -- and a constant cross-section has no
    ordering, so its IC is undefined and it silently drops out of the report.
    Left alone that would hand the null a smaller effective search than the LLM
    set it is being compared against, which biases the comparison toward the
    LLM. Topping up keeps the two searches the same size. The filter looks only
    at the feature panel, never at forward returns, so it leaks nothing.
    """
    sampler = GrammarSampler(seed=seed, profile=structure_profile(reference),
                             fields=PRICE_FIELDS)
    kept: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(kept) < size and attempts < size * 40:
        attempts += 1
        expr = sampler.sample()
        if expr in seen:
            continue
        seen.add(expr)
        try:
            signal = evaluator.evaluate(expr)
        except Exception:
            continue
        if signal.nunique(axis=1).mean() < 2:
            continue
        kept.append(expr)
    if len(kept) < size:
        raise SystemExit(f"seed {seed}: only drew {len(kept)} usable of {size}")
    return kept


def summarise(name: str, candidates: list[str], bars: pd.DataFrame,
              train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
              args: argparse.Namespace) -> tuple[dict, dict[str, pd.DataFrame]]:
    report, signals = score_expressions(bars, candidates,
                                        min_cross_section=args.min_cross_section,
                                        ic_dates=train_dates)
    selected = select_uncorrelated(report, signals,
                                   min_abs_tstat=args.min_abs_tstat,
                                   max_abs_corr=args.max_abs_corr,
                                   max_factors=args.max_factors)
    scored = [s for s in report.scores if s.error is None and np.isfinite(s.ic_tstat)]
    best_t = max((abs(s.ic_tstat) for s in scored), default=float("nan"))
    oos = holdout_frame(bars, selected, report, test_dates=test_dates,
                        min_cross_section=args.min_cross_section, model=name)
    row = {
        "set": name,
        "candidates": len(candidates),
        "scorable": len(scored),
        "best_abs_t_is": best_t,
        "null_max_abs_t": expected_max_abs_t(len(scored)),
        "selected": len(selected),
        "oos_median_retention": (float(oos["retention"].median())
                                 if not oos.empty else float("nan")),
        "sign_held": (f"{int(oos['sign_held'].sum())}/{len(oos)}"
                      if not oos.empty else "0/0"),
        "sign_held_frac": (float(oos["sign_held"].mean())
                           if not oos.empty else float("nan")),
    }
    return row, signals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", default=str(US_ROOT / "data" / "us_equities"
                                              / "processed" / "daily_bars.parquet"))
    parser.add_argument("--membership",
                        default=str(US_ROOT / "data" / "us_equities" / "reference"
                                    / "sp500_membership_daily.parquet"))
    parser.add_argument("--out-dir", default=str(US_ROOT / "data" / "baseline_comparison"))
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--seeds", type=int, default=5, help="random-grammar seeds")
    parser.add_argument("--random-size", type=int, default=54,
                        help="candidates per random set; defaults to the Sonnet set size")
    parser.add_argument("--min-abs-tstat", type=float, default=2.0)
    parser.add_argument("--max-abs-corr", type=float, default=0.7)
    parser.add_argument("--max-factors", type=int, default=10)
    parser.add_argument("--min-cross-section", type=int, default=30)
    args = parser.parse_args()

    bars = pd.read_parquet(args.bars)
    bars = apply_membership_filter(bars, Path(args.membership))

    all_dates = pd.to_datetime(bars["date"]).drop_duplicates().sort_values()
    cut = pd.Timestamp(args.train_end)
    train_dates = pd.DatetimeIndex(all_dates[all_dates <= cut])
    test_dates = pd.DatetimeIndex(all_dates[all_dates > cut])
    print(f"Selection: {train_dates.min().date()} -> {train_dates.max().date()} "
          f"({len(train_dates):,} days)   holdout: {test_dates.min().date()} -> "
          f"{test_dates.max().date()} ({len(test_dates):,} days)")
    print(f"Panel ends {all_dates.max().date()}. No post-training-cutoff window exists.")

    configs = US_ROOT / "configs"
    fable = load_candidates(configs / "mined_factors_claude_2026-08.txt")
    sonnet = load_candidates(configs / "mined_factors_sonnet5_2026-09.txt")
    alpha101 = load_candidates(configs / "alpha101_us.txt")

    evaluator = ExpressionEvaluator(build_field_panels(bars))

    sets: list[tuple[str, list[str]]] = [
        ("claude-fable-5", fable),
        ("claude-sonnet-5", sonnet),
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        print(f"drawing random-grammar-seed{seed} ...", flush=True)
        drawn = draw_random_set(seed, args.random_size, sonnet, evaluator)
        # Written out so the null is inspectable, and so each set can be re-scored
        # through sp500_score_mined_factors.py --model random-grammar-seed{N}
        # exactly like a real candidate file. Deterministic given the seed.
        path = out_dir / f"random_grammar_seed{seed}.txt"
        path.write_text(f"# random-grammar-seed{seed}: {len(drawn)} expressions sampled from\n"
                        f"# ExpressionSanitizer's grammar, matched to the Sonnet set's structure.\n"
                        f"# Regenerate with scripts/sp500_run_baseline_comparison.py --seeds {args.seeds}\n"
                        + "\n".join(drawn) + "\n", encoding="utf-8")
        sets.append((f"random-grammar-seed{seed}", drawn))
    sets.append(("alpha101", alpha101))

    rows, all_signals = [], {}
    for name, candidates in sets:
        print(f"scoring {name} ({len(candidates)} candidates) ...", flush=True)
        row, signals = summarise(name, candidates, bars, train_dates, test_dates, args)
        rows.append(row)
        all_signals[name] = signals

    table = pd.DataFrame(rows)

    # --- memorization test ------------------------------------------------
    # For every LLM candidate, its closest published alpha in rank space. A
    # frontier model asked for factors on a market it has read the literature
    # about can restate a known alpha and present it as a discovery; this is
    # the check for that, and it is only meaningful because Alpha101 is scored
    # here on the same panel through the same evaluator.
    print("memorization test: ranking signals once ...", flush=True)
    alpha_rank = {expr: ranked_matrix(sig, date_stride=MEMORIZATION_DATE_STRIDE)
                  for expr, sig in all_signals["alpha101"].items()}
    mem_rows = []
    for name in ("claude-fable-5", "claude-sonnet-5"):
        for expr, signal in all_signals[name].items():
            mat = ranked_matrix(signal, date_stride=MEMORIZATION_DATE_STRIDE)
            flat = mat.ravel()
            best_daily, best_match, best_pooled = 0.0, None, float("nan")
            for a_expr, a_mat in alpha_rank.items():
                daily = mean_daily_rank_correlation(mat, a_mat)
                if np.isfinite(daily) and abs(daily) > abs(best_daily):
                    best_daily = daily
                    best_match = a_expr
                    best_pooled = _corr_flat(flat, a_mat.ravel())
            mem_rows.append({"set": name, "expression": expr,
                             "max_abs_corr_to_alpha101": abs(best_daily),
                             "pooled_corr_at_that_match": abs(best_pooled),
                             "closest_alpha101": best_match})
    mem = pd.DataFrame(mem_rows)
    mem_share = (mem.groupby("set")["max_abs_corr_to_alpha101"]
                    .apply(lambda s: float((s > MEMORIZATION_THRESHOLD).mean())))
    table["share_corr_gt_0.9_to_alpha101"] = table["set"].map(mem_share)

    table.to_csv(out_dir / "comparison_table.csv", index=False)
    mem.sort_values("max_abs_corr_to_alpha101", ascending=False).to_csv(
        out_dir / "memorization_test.csv", index=False)

    show = table.copy()
    for col in ("best_abs_t_is", "null_max_abs_t"):
        show[col] = show[col].map(lambda v: f"{v:.2f}")
    show["oos_median_retention"] = show["oos_median_retention"].map(
        lambda v: "n/a" if not np.isfinite(v) else f"{v:.1%}")
    show["share_corr_gt_0.9_to_alpha101"] = show["share_corr_gt_0.9_to_alpha101"].map(
        lambda v: "" if not np.isfinite(v) else f"{v:.0%}")
    print("\n" + "=" * 118)
    print("BASELINE COMPARISON  (selection 2000-2017, holdout 2018-2025, in-sample choices frozen)")
    print("=" * 118)
    with pd.option_context("display.width", 200):
        print(show[["set", "candidates", "scorable", "best_abs_t_is", "null_max_abs_t",
                    "selected", "oos_median_retention", "sign_held",
                    "share_corr_gt_0.9_to_alpha101"]].to_string(index=False))

    rand = table[table["set"].str.startswith("random-grammar")]
    print(f"\nrandom-grammar best |t| across {len(rand)} seeds: "
          f"min {rand['best_abs_t_is'].min():.2f}, median "
          f"{rand['best_abs_t_is'].median():.2f}, max {rand['best_abs_t_is'].max():.2f}")
    print(f"random-grammar median OOS retention across seeds: "
          f"{rand['oos_median_retention'].median():.1%}")
    print("\nMost alpha101-like LLM candidates:")
    with pd.option_context("display.width", 200, "display.max_colwidth", 58):
        print(mem.nlargest(5, "max_abs_corr_to_alpha101")[
            ["set", "expression", "max_abs_corr_to_alpha101"]].to_string(index=False))
    print(f"\n-> {out_dir / 'comparison_table.csv'}")
    print(f"-> {out_dir / 'memorization_test.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
