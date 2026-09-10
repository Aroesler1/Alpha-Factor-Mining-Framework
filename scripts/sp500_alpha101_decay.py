#!/usr/bin/env python3
"""Did Alpha101 decay after it was published?

Kakushadze posted "101 Formulaic Alphas" (arXiv:1601.00991) on 2015-12-31. That
gives a clean pre/post split that needs no judgement call: score the 50
transcribed formulas on 2000-01 to 2015-12, freeze each one's sign there, and
score the same expressions on 2016-01 to 2025-12.

McLean and Pontiff (2016, Journal of Finance 71(1), 5-32) put the benchmark at
a 58% fall in predictor returns after publication, of which they attribute
roughly 26 points to in-sample overfitting that any out-of-sample window would
expose and the rest to publication-informed trading. This is the same
experiment on one published set, on one universe, with one execution
convention.

The model's own factors run through the identical split as a control. They have
no publication date, so their pre/post ratio measures ordinary out-of-sample
decay with nothing publication-specific in it -- which is the point of having
them here.

    python scripts/sp500_alpha101_decay.py
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
    circular_block_indices,
    newey_west_tstats,
)

PUBLICATION_DATE = "2015-12-31"
MCLEAN_PONTIFF_DECAY = 0.58


def _wilcoxon_signed_rank_p(d: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test, normal approximation with tie correction.

    Written out rather than imported: the repo has no scipy dependency and this
    is the only place a rank test is needed. The normal approximation is fine at
    n = 50; the exact distribution is not worth a table here.
    """
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n < 6:
        return float("nan"), float("nan")
    order = np.argsort(np.abs(d))
    absd = np.abs(d)[order]
    # average ranks within tie groups
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    signs = np.sign(d)[order]
    w = float(ranks[signs > 0].sum())
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(absd, return_counts=True)
    tie_term = float(((counts ** 3 - counts).sum())) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    if var <= 0:
        return w, float("nan")
    z = (w - mean) / np.sqrt(var)
    from math import erfc, sqrt
    return w, float(erfc(abs(z) / sqrt(2.0)))


def _paired_t_p(d: np.ndarray) -> tuple[float, float]:
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    from math import erfc, sqrt
    return float(t), float(erfc(abs(t) / sqrt(2.0)))


def _bootstrap_mean_shift(pre: pd.DataFrame, post: pd.DataFrame, *,
                          draws: int, block: int, seed: int) -> tuple[float, float, float, float]:
    """Block-bootstrap CI and p for the change in the pool's average mean IC.

    Both windows are resampled in blocks of dates, independently of each other,
    and the same block indices apply to every factor within a window. That is
    the null that matters here: the 50 alphas are heavily correlated with one
    another, so treating them as 50 independent observations -- which the
    paired-across-alphas tests below do -- understates the standard error. This
    one does not, at the cost of saying nothing about individual alphas.
    """
    rng = np.random.default_rng(seed)
    a, b = pre.to_numpy(), post.to_numpy()
    observed = b.mean(axis=0).mean() - a.mean(axis=0).mean()
    diffs = np.empty(draws)
    for i in range(draws):
        ia = circular_block_indices(a.shape[0], block, rng)
        ib = circular_block_indices(b.shape[0], block, rng)
        diffs[i] = b[ib].mean(axis=0).mean() - a[ia].mean(axis=0).mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    centred = diffs - diffs.mean()
    p = float((np.abs(centred) >= abs(observed)).mean())
    return float(observed), float(lo), float(hi), p


def summarise(name: str, ic: pd.DataFrame, cut: pd.Timestamp, *,
              lag: int, draws: int, block: int, seed: int,
              min_days_share: float = 0.9) -> tuple[pd.DataFrame, dict]:
    pre_all = ic.loc[ic.index <= cut]
    post_all = ic.loc[ic.index > cut]

    # Complete-case across every factor in the set, so a mean and its t come
    # from one sample -- but only AFTER dropping the factors whose own sample is
    # far shorter than the rest. Without that filter one degenerate expression
    # collapses the intersection for everybody: the 270-factor random-grammar
    # set came back with 16 usable post-publication days, which is not a
    # measurement of anything.
    dropped = []
    for window in (pre_all, post_all):
        days = window.notna().sum()
        if days.max() > 0:
            dropped += [c for c in window.columns
                        if days[c] < min_days_share * days.max()]
    keep = [c for c in ic.columns if c not in set(dropped)]
    if len(keep) < len(ic.columns):
        print(f"  {name}: dropped {len(ic.columns) - len(keep)} of {len(ic.columns)} "
              f"factors with a short sample in one of the two windows")
    pre = pre_all[keep].dropna(axis=0, how="any")
    post = post_all[keep].dropna(axis=0, how="any")
    ic = ic[keep]

    sign = np.sign(pre.mean()).replace(0, 1)
    pre_o, post_o = pre * sign, post * sign

    rows = pd.DataFrame({
        "expression": ic.columns,
        "mean_ic_pre": pre_o.mean().to_numpy(),
        "nw_t_pre": newey_west_tstats(pre_o.to_numpy(), lag=lag),
        "mean_ic_post": post_o.mean().to_numpy(),
        "nw_t_post": newey_west_tstats(post_o.to_numpy(), lag=lag),
    })
    rows["ratio"] = rows["mean_ic_post"] / rows["mean_ic_pre"]
    rows["decay"] = 1.0 - rows["ratio"]
    rows["set"] = name

    d = (rows["mean_ic_post"] - rows["mean_ic_pre"]).to_numpy()
    t_stat, t_p = _paired_t_p(d)
    w_stat, w_p = _wilcoxon_signed_rank_p(d)
    obs, lo, hi, boot_p = _bootstrap_mean_shift(
        pre_o, post_o, draws=draws, block=block, seed=seed
    )
    summary = {
        "set": name,
        "n_factors": len(rows),
        "pre_days": len(pre),
        "post_days": len(post),
        "mean_ic_pre": float(rows["mean_ic_pre"].mean()),
        "mean_ic_post": float(rows["mean_ic_post"].mean()),
        "median_ratio": float(rows["ratio"].median()),
        "mean_ratio": float(rows["mean_ic_post"].mean() / rows["mean_ic_pre"].mean()),
        "share_sign_held": float((rows["mean_ic_post"] > 0).mean()),
        "paired_t": t_stat, "paired_t_p": t_p,
        "wilcoxon_W": w_stat, "wilcoxon_p": w_p,
        "boot_diff": obs, "boot_lo": lo, "boot_hi": hi, "boot_p": boot_p,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo-dir", default=str(US_ROOT / "data" / "factor_zoo"))
    parser.add_argument("--publication-date", default=PUBLICATION_DATE)
    parser.add_argument("--lag", type=int, default=DEFAULT_NW_LAG)
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    zoo = Path(args.zoo_dir)
    table = load_ic_table(zoo / "cache")
    pool = pd.read_csv(zoo / "pool.csv")
    ic = table.ic[(LAGGED, 1)]
    cut = pd.Timestamp(args.publication_date)

    groups = {
        "alpha101": pool.loc[pool["set"] == "alpha101", "expression"],
        "model (control)": pool.loc[pool["group"] == "model", "expression"],
        "random-grammar (control)": pool.loc[pool["group"] == "random", "expression"],
    }

    all_rows, summaries = [], []
    for name, exprs in groups.items():
        cols = [e for e in exprs if e in ic.columns]
        rows, summary = summarise(name, ic[cols], cut, lag=args.lag,
                                  draws=args.draws, block=args.block, seed=args.seed)
        all_rows.append(rows)
        summaries.append(summary)

    detail = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    zoo.mkdir(parents=True, exist_ok=True)
    detail.to_csv(zoo / "alpha101_decay_factors.csv", index=False)
    summary.to_csv(zoo / "alpha101_decay_summary.csv", index=False)

    print("\n" + "=" * 104)
    print("A2  ALPHA101 DECAY AROUND PUBLICATION".center(104))
    print("=" * 104)
    print(f"\nsplit at {cut.date()} (arXiv:1601.00991 posted 2015-12-31); "
          f"sign frozen on the pre window; NW lag {args.lag}\n")
    show = summary.copy()
    for col in ("mean_ic_pre", "mean_ic_post"):
        show[col] = show[col].map(lambda v: f"{v:+.5f}")
    for col in ("median_ratio", "mean_ratio", "share_sign_held"):
        show[col] = show[col].map(lambda v: f"{v:.1%}")
    print(show[["set", "n_factors", "pre_days", "post_days", "mean_ic_pre",
                "mean_ic_post", "mean_ratio", "median_ratio",
                "share_sign_held"]].to_string(index=False))

    print("\ntests of the pre -> post change in mean IC\n")
    print(f"{'set':<26}{'paired t':>10}{'p':>9}{'Wilcoxon p':>12}"
          f"{'block-bootstrap diff [95% CI]':>36}{'p':>8}")
    print("-" * 101)
    for row in summaries:
        ci = f"{row['boot_diff']:+.5f} [{row['boot_lo']:+.5f}, {row['boot_hi']:+.5f}]"
        print(f"{row['set']:<26}{row['paired_t']:>10.2f}{row['paired_t_p']:>9.3f}"
              f"{row['wilcoxon_p']:>12.3f}{ci:>36}{row['boot_p']:>8.3f}")
    print("\n  paired t and Wilcoxon treat the factors within a set as independent "
          "observations.\n  They are not -- the sets are full of restatements of the same "
          "idea -- so those two p-values\n  are optimistic. The block bootstrap resamples "
          "DATES and is the one to read.")

    a101 = summary[summary["set"] == "alpha101"].iloc[0]
    print(f"\nAlpha101 post/pre ratio of the average mean IC: {a101['mean_ratio']:.1%} "
          f"(decay {1 - a101['mean_ratio']:.1%})")
    print(f"McLean and Pontiff (2016) post-publication benchmark: "
          f"{MCLEAN_PONTIFF_DECAY:.0%} decay in predictor returns")

    print("\nper-alpha detail, largest and smallest decay\n")
    a = detail[detail["set"] == "alpha101"].sort_values("ratio")
    with pd.option_context("display.width", 200, "display.max_colwidth", 52):
        print(a.head(5)[["expression", "mean_ic_pre", "nw_t_pre", "mean_ic_post",
                         "nw_t_post", "ratio"]].to_string(index=False))
        print("  ...")
        print(a.tail(5)[["expression", "mean_ic_pre", "nw_t_pre", "mean_ic_post",
                         "nw_t_post", "ratio"]].to_string(index=False))
    print(f"\n-> {zoo / 'alpha101_decay_summary.csv'}")
    print(f"-> {zoo / 'alpha101_decay_factors.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
