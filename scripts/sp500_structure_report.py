#!/usr/bin/env python3
"""Is the model paraphrasing Alpha101, structurally rather than statistically?

The repo's existing memorization test asks the question in signal space: how
correlated is each mined factor with its closest published alpha? That misses a
restatement whose numbers happen to differ (a window changed from 10 to 12) and
flags coincidences (two unrelated ideas that co-move because everything in
equities co-moves).

This asks it in structure space, which is independent of the data entirely:

  largest shared subtree   the biggest complete sub-expression a candidate has
                           in common with ANY Alpha101 formula, after
                           canonicalisation. Reusing `TS_DELTA($volume, 1)`
                           whole is what a paraphrase looks like.
  edit distance            Zhang-Shasha, to the nearest Alpha101 formula. Small
                           means "one edit away from a published alpha".

Neither is meaningful alone -- a longer expression shares more by accident --
so both are reported against random-grammar draws of MATCHED TREE SIZE. The
random draws never saw the literature, so whatever they score is the level of
structural overlap that costs nothing to explain.

No data. No panel. Runs in a minute on a laptop with no WRDS entitlement.

    python scripts/sp500_structure_report.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.factors import ast_tools as at  # noqa: E402
from quantaalpha_us.factors.ic_panel import load_expression_file  # noqa: E402

SIZE_TOLERANCE = 2  # nodes; the width of the "matched size" band


def _permutation_p(a: np.ndarray, b: np.ndarray, *, draws: int = 20000,
                   seed: int = 20260904) -> float:
    """Two-sided permutation p for a difference in medians. No scipy needed."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    observed = abs(np.median(a) - np.median(b))
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    n = len(a)
    hits = 0
    for _ in range(draws):
        rng.shuffle(pool)
        if abs(np.median(pool[:n]) - np.median(pool[n:])) >= observed:
            hits += 1
    return (hits + 1) / (draws + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(US_ROOT / "data" / "factor_zoo"))
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()

    configs = US_ROOT / "configs"
    baseline = US_ROOT / "data" / "baseline_comparison"
    sources = [
        ("claude-fable-5", "model", configs / "mined_factors_claude_2026-08.txt"),
        ("claude-sonnet-5", "model", configs / "mined_factors_sonnet5_2026-09.txt"),
        ("fundamental-2026-09", "model", configs / "mined_factors_fundamental_2026-09.txt"),
    ] + [
        (f"random-grammar-seed{s}", "random", baseline / f"random_grammar_seed{s}.txt")
        for s in range(5)
    ]
    alpha_exprs = load_expression_file(configs / "alpha101_us.txt")
    alpha_trees = [at.parse(e) for e in alpha_exprs]
    print(f"Alpha101 reference: {len(alpha_trees)} formulas, "
          f"sizes {min(at.size(t) for t in alpha_trees)}-"
          f"{max(at.size(t) for t in alpha_trees)} nodes")

    started = time.time()
    rows = []
    for set_name, group, path in sources:
        exprs = load_expression_file(path)
        for expr in exprs:
            try:
                tree = at.parse(expr)
            except at.ParseError as exc:
                rows.append({"set": set_name, "group": group, "expression": expr,
                             "error": str(exc)})
                continue
            shared = [at.largest_shared_subtree(tree, a) for a in alpha_trees]
            dists = [at.tree_edit_distance(tree, a) for a in alpha_trees]
            best_shared = int(np.argmax(shared))
            best_dist = int(np.argmin(dists))
            comp = at.complexity(expr)
            rows.append({
                "set": set_name, "group": group, "expression": expr, "error": None,
                "tree_size": at.size(tree),
                "depth": comp.depth,
                "symbol_length": comp.symbol_length,
                "base_fields": comp.base_fields,
                "free_constants": comp.free_constants,
                "max_shared_subtree": int(max(shared)),
                "shared_subtree_share": max(shared) / at.size(tree),
                "nearest_shared_alpha": alpha_exprs[best_shared],
                "min_edit_distance": int(min(dists)),
                "min_normalized_edit_distance": float(
                    min(dists) / (at.size(tree) + at.size(alpha_trees[best_dist]))
                ),
                "nearest_edit_alpha": alpha_exprs[best_dist],
            })
        print(f"  {set_name:24s} {len(exprs):4d} done  "
              f"({time.time() - started:.0f}s elapsed)", flush=True)

    out = pd.DataFrame(rows)
    ok = out[out["error"].isna()].copy()

    # ---- size-matched pairing -------------------------------------------
    # A larger tree shares more with anything, so the model/random comparison is
    # only fair inside a size band. Each model factor is paired with the random
    # draws whose node count is within SIZE_TOLERANCE of its own.
    randoms = ok[ok["group"] == "random"].copy()
    matched = []
    for _, row in ok[ok["group"] == "model"].copy().iterrows():
        band = randoms[(randoms["tree_size"] - row["tree_size"]).abs() <= SIZE_TOLERANCE]
        shared_median = band["max_shared_subtree"].median() if len(band) else np.nan
        edit_median = band["min_edit_distance"].median() if len(band) else np.nan
        matched.append({
            "expression": row["expression"],
            "set": row["set"],
            "tree_size": row["tree_size"],
            "model_shared": row["max_shared_subtree"],
            "model_edit": row["min_edit_distance"],
            "n_matched_random": len(band),
            "random_shared_median": shared_median,
            "random_edit_median": edit_median,
            "shared_excess": row["max_shared_subtree"] - shared_median,
            "edit_excess": row["min_edit_distance"] - edit_median,
        })
    pairs = pd.DataFrame(matched)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "structure_report.csv", index=False)
    pairs.to_csv(out_dir / "structure_size_matched.csv", index=False)

    print("\n" + "=" * 104)
    print("A4  STRUCTURAL DISTANCE TO ALPHA101".center(104))
    print("=" * 104)
    print("\nby set (medians)\n")
    agg = ok.groupby(["group", "set"]).agg(
        n=("expression", "size"),
        tree_size=("tree_size", "median"),
        depth=("depth", "median"),
        base_fields=("base_fields", "median"),
        free_constants=("free_constants", "median"),
        shared=("max_shared_subtree", "median"),
        shared_share=("shared_subtree_share", "median"),
        edit=("min_edit_distance", "median"),
        norm_edit=("min_normalized_edit_distance", "median"),
    ).reset_index()
    with pd.option_context("display.width", 200):
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    model = ok[ok["group"] == "model"]
    rand = ok[ok["group"] == "random"]
    print("\nmodel against random, pooled (no size matching)\n")
    print(f"{'statistic':<34}{'model':>10}{'random':>10}{'perm p':>10}")
    print("-" * 64)
    for label, col in (("largest shared subtree (nodes)", "max_shared_subtree"),
                       ("shared subtree / own tree size", "shared_subtree_share"),
                       ("edit distance to nearest alpha", "min_edit_distance"),
                       ("normalized edit distance", "min_normalized_edit_distance"),
                       ("tree size", "tree_size")):
        p = _permutation_p(model[col].to_numpy(), rand[col].to_numpy(), draws=args.draws)
        print(f"{label:<34}{model[col].median():>10.2f}{rand[col].median():>10.2f}{p:>10.4f}")

    print(f"\nsize-matched (each model factor against random draws within "
          f"{SIZE_TOLERANCE} nodes)\n")
    usable = pairs[pairs["n_matched_random"] > 0]
    print(f"  {len(usable)}/{len(pairs)} model factors have a size-matched random band")
    print(f"  median excess shared subtree vs matched random: "
          f"{usable['shared_excess'].median():+.2f} nodes")
    print(f"  median excess edit distance  vs matched random: "
          f"{usable['edit_excess'].median():+.2f} edits")
    print(f"  model factors sharing a STRICTLY larger subtree than their matched "
          f"random median: {int((usable['shared_excess'] > 0).sum())}/{len(usable)}")

    # The evolutionary loop's gate rejects a candidate sharing a subtree of 5 or
    # more nodes with any Alpha101 formula. Counting how many EXISTING factors
    # that would reject is the only way to know whether the threshold is a real
    # constraint or decoration -- and it is the number Part B's threshold sweep
    # starts from.
    print("\nwhat the Part B shared-subtree gate would reject, by threshold\n")
    print(f"{'gate: shared subtree >=':<26}" + "".join(f"{k:>10}" for k in (3, 4, 5, 6, 7)))
    print("-" * 76)
    for label, sub in (("model", model), ("random", rand), ("alpha101 (self-excluded)", None)):
        if sub is None:
            continue
        line = f"{label:<26}"
        for k in (3, 4, 5, 6, 7):
            line += f"{(sub['max_shared_subtree'] >= k).mean():>9.0%} "
        print(line)

    print("\nmost Alpha101-like candidates by structure\n")
    with pd.option_context("display.width", 220, "display.max_colwidth", 46):
        print(model.nlargest(8, "max_shared_subtree")[
            ["set", "expression", "tree_size", "max_shared_subtree",
             "min_edit_distance", "nearest_shared_alpha"]].to_string(index=False))
    print(f"\n-> {out_dir / 'structure_report.csv'}")
    print(f"-> {out_dir / 'structure_size_matched.csv'}")
    print(f"wall clock: {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
