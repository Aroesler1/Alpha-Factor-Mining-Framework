#!/usr/bin/env python3
"""Score the whole factor zoo once and cache the daily IC series.

Every question in the factor-zoo work -- the multiple-testing hurdle, the
Alpha101 decay split, the IC horizon curve, the structural report -- is a
statistic over the same object: one daily rank-IC series per factor per forward
horizon. Evaluating the panel is the expensive step and it does not depend on
which question is being asked, so it happens here, once, and the analysis
scripts read the cache.

    python scripts/sp500_build_factor_pool.py \
        --bars data/us_equities/processed/daily_bars_fundamentals.parquet

Output: data/factor_zoo/cache/ (gitignored -- roughly 200 MB of IC matrices)
plus data/factor_zoo/pool.csv, the small committed index of what is in it.

The random-grammar sets are read from data/baseline_comparison/, not redrawn.
They are seeded and already on disk; redrawing them here would silently give
the zoo hurdle a different null than the README's baseline table.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.factors.expression_evaluator import build_field_panels  # noqa: E402
from quantaalpha_us.factors.ic_panel import (  # noqa: E402
    CUMULATIVE,
    HORIZONS,
    LAGGED,
    apply_membership_filter,
    build_ic_table,
    load_expression_file,
    save_ic_table,
)

# (set name, coarse group, path). The coarse group is what the zoo report
# aggregates by; the set name is what identifies a row.
def pool_sources(us_root: Path) -> list[tuple[str, str, Path]]:
    configs = us_root / "configs"
    baseline = us_root / "data" / "baseline_comparison"
    sources = [
        ("claude-fable-5", "model", configs / "mined_factors_claude_2026-08.txt"),
        ("claude-sonnet-5", "model", configs / "mined_factors_sonnet5_2026-09.txt"),
        ("fundamental-2026-09", "model", configs / "mined_factors_fundamental_2026-09.txt"),
    ]
    for seed in range(5):
        sources.append((f"random-grammar-seed{seed}", "random",
                        baseline / f"random_grammar_seed{seed}.txt"))
    sources.append(("alpha101", "alpha101", configs / "alpha101_us.txt"))
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", default=str(US_ROOT / "data" / "us_equities" / "processed"
                                              / "daily_bars_fundamentals.parquet"))
    parser.add_argument("--membership",
                        default=str(US_ROOT / "data" / "us_equities" / "reference"
                                    / "sp500_membership_daily.parquet"))
    parser.add_argument("--out-dir", default=str(US_ROOT / "data" / "factor_zoo"))
    parser.add_argument("--min-cross-section", type=int, default=30)
    args = parser.parse_args()

    started = time.time()
    bars = pd.read_parquet(args.bars)
    before = len(bars)
    bars = apply_membership_filter(bars, args.membership)
    print(f"membership filter: {before:,} -> {len(bars):,} rows, "
          f"{bars['permno'].nunique()} names", flush=True)

    expressions: list[str] = []
    sets: dict[str, str] = {}
    groups: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for set_name, group, path in pool_sources(US_ROOT):
        loaded = load_expression_file(path)
        print(f"  {set_name:24s} {len(loaded):4d} from {path.name}")
        for expr in loaded:
            if expr in sets:
                # An expression in two sets would collide on the IC table's
                # column key and give the pool a hidden duplicate, which is
                # exactly the thing a factor-zoo count must not have.
                duplicates.append((expr, sets[expr], set_name))
                continue
            sets[expr] = set_name
            groups[expr] = group
            expressions.append(expr)
    if duplicates:
        print(f"\n{len(duplicates)} expression(s) appear in more than one set; kept once:")
        for expr, first, second in duplicates:
            print(f"  {first} / {second}: {expr[:80]}")

    print(f"\npool: {len(expressions)} distinct expressions", flush=True)
    panels = build_field_panels(bars)
    print(f"panel: {panels['close'].shape[0]:,} dates x {panels['close'].shape[1]:,} symbols "
          f"({panels['close'].index.min().date()} -> {panels['close'].index.max().date()})",
          flush=True)

    table = build_ic_table(
        bars, expressions, groups=sets, horizons=HORIZONS,
        kinds=(LAGGED, CUMULATIVE), min_cross_section=args.min_cross_section,
        panels=panels, verbose=True,
    )

    out_dir = Path(args.out_dir)
    cache = out_dir / "cache"
    save_ic_table(table, cache)

    index = table.meta.rename(columns={"group": "set"}).copy()
    index["group"] = index["set"].map(
        {name: group for name, group, _ in pool_sources(US_ROOT)}
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(out_dir / "pool.csv", index=False)

    failed = index[index["error"].notna()]
    print(f"\nscored {len(index) - len(failed)}/{len(index)}; {len(failed)} failed")
    for _, row in failed.iterrows():
        print(f"  [{row['set']}] {row['expression'][:70]} -> {str(row['error'])[:90]}")
    print(f"\ncache -> {cache}")
    print(f"index -> {out_dir / 'pool.csv'}")
    print(f"wall clock: {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
