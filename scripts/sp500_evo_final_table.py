#!/usr/bin/env python3
"""Read the holdout once, for every arm, and print the table the experiment is for.

This is the ONLY script that touches 2018-2025. Everything upstream -- the loop,
the gates, the fitness, the early-stop rule, the two swept thresholds -- is
fitted on 2000-2013 and tuned on 2014-2017. Running this twice with different
choices in between would quietly turn the holdout into a validation set, so it
is deliberately one command with no tuning knobs.

Per arm it reports what was generated, what the gates threw away, what survived,
how far the survivors are from the published literature structurally, and how
they do out of sample -- individually and, more importantly, combined.

    python scripts/sp500_evo_final_table.py --runs-dir data/evo_runs
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

from quantaalpha_us.evo.archive import Archive, ArchiveEntry, load_archive  # noqa: E402
from quantaalpha_us.evo.config import EvoConfig  # noqa: E402
from quantaalpha_us.evo.evaluation_protocol import (claim_evaluation, input_snapshot,
                                                   write_exclusive)
from quantaalpha_us.evo.comparison_audit import adjust_comparisons
from quantaalpha_us.evo.fitness import GateRunner  # noqa: E402
from quantaalpha_us.evo.persistence import RunStore, token_totals  # noqa: E402
from quantaalpha_us.evo.report import (  # noqa: E402
    candidates_generated,
    rejection_tally,
)
from quantaalpha_us.evo.scoring import PanelScorer, candidate_id  # noqa: E402
from quantaalpha_us.factors import alpha101  # noqa: E402
from quantaalpha_us.factors.factor_research import _daily_spearman_ic  # noqa: E402
from quantaalpha_us.factors.ic_panel import LAGGED, apply_membership_filter, load_ic_table  # noqa: E402
from quantaalpha_us.factors.multiple_testing import (  # noqa: E402
    DEFAULT_BLOCK,
    DEFAULT_NW_LAG,
    bootstrap_null_tstats,
    circular_block_indices,
    newey_west_tstats,
)

TOP_N = 20
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


# --------------------------------------------------------------------------
# combining a factor set
# --------------------------------------------------------------------------


def fit_ridge(scorer: PanelScorer, expressions, signs, *, alphas=RIDGE_GRID) -> np.ndarray:
    """Ridge weights over the factors' cross-sectionally demeaned ranks.

    Closed form, in numpy: w = (X'X + lambda I)^-1 X'y. The penalty is chosen on
    the VALIDATION window, which is what validation is for; choosing it on the
    holdout would make the holdout number a fitted number.
    """
    panels = []
    for expression, sign in zip(expressions, signs):
        panel = scorer.evaluate(expression)
        ranks = panel.rank(axis=1, pct=True) * sign
        panels.append(ranks.sub(ranks.mean(axis=1), axis=0))

    def design(dates):
        columns = [p.loc[p.index.isin(dates)].to_numpy(dtype=np.float32).ravel()
                   for p in panels]
        x = np.column_stack(columns)
        fwd = scorer._fwd_fit[1] if dates is scorer.fit_dates else None
        if fwd is None:
            base = scorer._fwd_validation
            base = base.loc[base.index.isin(dates)]
        else:
            base = fwd
        y = base.sub(base.mean(axis=1), axis=0).to_numpy(dtype=np.float32).ravel()
        keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
        return x[keep].astype(np.float64), y[keep].astype(np.float64)

    x_fit, y_fit = design(scorer.fit_dates)
    x_val, y_val = design(scorer._validation_dates)
    xtx = x_fit.T @ x_fit
    xty = x_fit.T @ y_fit
    scale = float(np.trace(xtx)) / max(x_fit.shape[1], 1)

    best_weights, best_score = None, -np.inf
    for penalty in alphas:
        weights = np.linalg.solve(xtx + penalty * scale * np.eye(xtx.shape[0]), xty)
        prediction = x_val @ weights
        denom = float(np.std(prediction) * np.std(y_val))
        score = float(np.mean((prediction - prediction.mean()) *
                              (y_val - y_val.mean())) / denom) if denom > 0 else -np.inf
        if score > best_score:
            best_weights, best_score = weights, score
    return best_weights if best_weights is not None else np.zeros(len(expressions))


def combined_holdout_ic(scorer: PanelScorer, expressions, signs,
                        weights=None) -> pd.Series:
    """Daily holdout IC of one combined signal."""
    total = None
    count = None
    for i, (expression, sign) in enumerate(zip(expressions, signs)):
        weight = 1.0 if weights is None else float(weights[i])
        ranks = scorer.evaluate(expression).rank(axis=1, pct=True) * sign * weight
        total = ranks if total is None else total.add(ranks, fill_value=0.0)
        present = ranks.notna().astype(float)
        count = present if count is None else count.add(present, fill_value=0.0)
    combined = total / count.replace(0, np.nan)
    return _daily_spearman_ic(combined, scorer._fwd_holdout, scorer.min_cross_section)


# --------------------------------------------------------------------------
# per-arm summary
# --------------------------------------------------------------------------


def summarise_arm(name: str, entries, scorer: PanelScorer, gates: GateRunner,
                  store: RunStore | None, random_crit: float,
                  exclude_self: bool = False) -> dict:
    """One arm's row, its per-factor detail, and its combined IC series.

    `exclude_self` matters for exactly one arm: the Alpha101 baseline, whose
    members ARE published alphas. Comparing each of them against the published
    set including itself reported a median shared subtree of 16 nodes and a
    median edit distance of 0, which says nothing except that a formula matches
    itself.
    """
    top = sorted(entries, key=lambda e: (-e.fitness, e.admitted_at))[:TOP_N]
    expressions = [e.expression for e in top]
    signs = [int((e.metrics or {}).get("sign", 1)) for e in top]

    per_factor = []
    ic_frames = []
    for entry, sign in zip(top, signs):
        signal = scorer.evaluate(entry.expression)
        holdout = scorer.holdout_ic_series(signal, sign)
        fit_ic = float((entry.metrics or {}).get("mean_ic", np.nan))
        if exclude_self:
            shared, distance = gates.nearest_alpha101_excluding(entry.expression)
        else:
            shared, _ = gates.max_shared_alpha101(entry.expression)
            distance, _ = gates.nearest_alpha101_edit(entry.expression)
        half_life = (entry.metrics or {}).get("half_life")
        if half_life is None:
            half_life = np.inf if (entry.metrics or {}).get(
                "half_life_is_beyond_grid") else np.nan
        per_factor.append({
            "arm": name, "expression": entry.expression, "island": entry.island,
            "operator": entry.operator, "round": entry.round, "fitness": entry.fitness,
            "fit_mean_ic": fit_ic,
            "fit_tstat": float((entry.metrics or {}).get("tstat", np.nan)),
            "holdout_mean_ic": float(holdout.mean()) if len(holdout) else np.nan,
            "retention": (float(holdout.mean()) / fit_ic
                          if fit_ic and np.isfinite(fit_ic) and fit_ic != 0 else np.nan),
            "sign_held": bool(len(holdout) and holdout.mean() > 0),
            "shared_subtree_alpha101": shared,
            "edit_distance_alpha101": distance,
            "half_life": half_life,
            "turnover": float((entry.metrics or {}).get("turnover", np.nan)),
            "passes_a1_hurdle": bool(abs(float((entry.metrics or {}).get("tstat", 0)))
                                     > random_crit),
        })
        ic_frames.append(holdout.rename(entry.expression))

    equal_weight = pd.concat(ic_frames, axis=1).mean(axis=1) if ic_frames else pd.Series(dtype=float)
    detail = pd.DataFrame(per_factor)

    # One definition of "generated" for every arm: everything that was scored,
    # plus everything that was rejected. The two files partition the proposals
    # exactly once each, so nothing is double counted and nothing is dropped.
    # It runs nine higher than the round counters' proposal total for every arm,
    # and that nine is the island seeds, which are scored at round 0 and are not
    # proposals.
    generated = candidates_generated(store) if store else len(entries)

    # Two kinds of rejection, kept apart because they mean different things.
    #
    # A GATE rejection is a candidate that never earned a score: it failed the
    # sanitizer, the complexity limits, a paraphrase check, coverage,
    # correlation or the cheap screen. Those are in rejections.jsonl.
    #
    # An ARCHIVE refusal is a candidate that passed every gate, was scored, and
    # was still turned away because its niche was full and it did not beat the
    # weakest member. Those were only ever counted in memory and written into
    # each round's summary, so reading rejections.jsonl alone understated the
    # tally by 5 to 38 depending on the arm.
    #
    # An archive-refused candidate is already inside `generated` via
    # candidates.jsonl -- it WAS scored -- so it is a subset of the scored
    # candidates, not an extra proposal. Adding it to `generated` as well would
    # count it twice.
    rejections, archive_refusals = rejection_tally(store) if store else ({}, {})
    tally = {**rejections, **archive_refusals}

    niches = len({tuple(e.niche) for e in entries})
    totals = token_totals(store) if store else {}
    row = {
        "arm": name,
        "candidates_generated": generated,
        "gate_rejections": sum(rejections.values()),
        "archive_refusals": sum(archive_refusals.values()),
        "rejections_total": sum(tally.values()),
        "archive_size": len(entries),
        "niches_filled": niches,
        "top20_median_retention": float(detail["retention"].median()) if len(detail) else np.nan,
        "top20_sign_held": f"{int(detail['sign_held'].sum())}/{len(detail)}" if len(detail) else "0/0",
        "mean_holdout_ic": float(equal_weight.mean()) if len(equal_weight) else np.nan,
        "holdout_nw_t": (float(newey_west_tstats(equal_weight.to_numpy()[:, None],
                                                 lag=DEFAULT_NW_LAG)[0])
                         if len(equal_weight) > DEFAULT_NW_LAG + 1 else np.nan),
        "a1_hurdle_survivors": int(detail["passes_a1_hurdle"].sum()) if len(detail) else 0,
        "median_shared_subtree": float(detail["shared_subtree_alpha101"].median()) if len(detail) else np.nan,
        "median_edit_distance": float(detail["edit_distance_alpha101"].median()) if len(detail) else np.nan,
        "median_half_life": float(np.nanmedian(
            np.where(np.isinf(detail["half_life"]), np.nan, detail["half_life"]))) if len(detail) else np.nan,
        "share_half_life_beyond_grid": (float(np.isinf(detail["half_life"]).mean())
                                        if len(detail) else np.nan),
        "mean_turnover": float(detail["turnover"].mean()) if len(detail) else np.nan,
        # Sourced from the live run directory's saved envelopes, never from
        # manifest.json's budget block and never from summing rounds.jsonl.
        # Both of those undercount an arm that was resumed: the budget block
        # records what one process was allowed rather than what was spent, and a
        # round row only knows the calls that process made. The two salvaged
        # one-shot arms are exactly that case.
        "model_calls": totals.get("calls", 0),
        "total_tokens": totals.get("total_tokens", 0),
        "output_tokens": totals.get("output_tokens", 0),
        "cost_usd": totals.get("cost_usd"),
        "model_minutes": totals.get("model_wall_clock_minutes", 0.0),
        "rejections_by_gate": "; ".join(f"{k}={v}" for k, v in sorted(tally.items())),
    }
    return {"row": row, "detail": detail, "top": top, "expressions": expressions,
            "signs": signs, "equal_weight_ic": equal_weight}


def paired_bootstrap(a: pd.Series, b: pd.Series, *, block: int = DEFAULT_BLOCK,
                     draws: int = 2000, seed: int = 20260904) -> dict:
    """Difference in mean holdout IC, resampling the SAME date blocks for both.

    Paired on dates because the two arms trade the same universe on the same
    days: an unpaired test would attribute their common market exposure to
    sampling noise and be far too conservative.
    """
    common = a.index.intersection(b.index)
    x, y = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    diff = x - y
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    draws_out = np.empty(draws)
    for i in range(draws):
        idx = circular_block_indices(len(diff), block, rng)
        draws_out[i] = diff[idx].mean()
    lo, hi = np.quantile(draws_out, [0.025, 0.975])
    centred = draws_out - draws_out.mean()
    return {"observed": observed, "lo": float(lo), "hi": float(hi),
            "p": float((np.abs(centred) >= abs(observed)).mean()),
            "days": int(len(common))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(US_ROOT / "data" / "evo_runs"))
    parser.add_argument("--bars", default=str(US_ROOT / "data" / "us_equities" / "processed"
                                              / "daily_bars_fundamentals.parquet"))
    parser.add_argument("--membership",
                        default=str(US_ROOT / "data" / "us_equities" / "reference"
                                    / "sp500_membership_daily.parquet"))
    parser.add_argument("--zoo-dir", default=str(US_ROOT / "data" / "factor_zoo"))
    parser.add_argument("--out-dir", default=str(US_ROOT / "data" / "factor_zoo"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--freeze-evaluation", action="store_true",
                        help="freeze input hashes/settings without scoring; review this manifest before evaluation")
    args = parser.parse_args()

    started = time.time()
    config = EvoConfig(arm="final")
    runs_dir = Path(args.runs_dir)
    consumed = runs_dir / "_evaluation_consumed.json"
    if consumed.exists():
        raise SystemExit("This holdout is already consumed. Run scripts/sp500_evo_audit.py --check instead.")
    arm_paths = [p for p in sorted(runs_dir.iterdir()) if p.is_dir()
                 and not p.name.endswith("-raw") and not p.name.startswith("_")
                 and (p / "archive.jsonl").exists()]
    if not arm_paths:
        raise SystemExit("No completed arms to freeze")
    for path in arm_paths:
        saved = RunStore(path).manifest().get("config", {})
        if saved.get("label_policy") != config.label_policy:
            raise SystemExit("Historical archives used unpurged labels. Preserve their published tables; do not relabel a rescore as clean.")
    files = {"bars": Path(args.bars), "membership": Path(args.membership)}
    for path in arm_paths:
        for item in path.rglob("*"):
            if item.is_file():
                files["runs/" + str(item.relative_to(runs_dir))] = item
    for folder in (US_ROOT / "quantaalpha_us", US_ROOT / "scripts", US_ROOT / "configs"):
        for item in folder.rglob("*"):
            if item.is_file() and item.suffix in {".py", ".json", ".txt"}:
                files[str(item.relative_to(US_ROOT))] = item
    for item in Path(args.zoo_dir).rglob("*"):
        if item.is_file() and ("cache" in item.parts or item.name == "pool.csv"):
            files["zoo/" + str(item.relative_to(args.zoo_dir))] = item
    snapshot = input_snapshot(files, {"config": config.to_dict(), "draws": args.draws,
                                     "arms": args.arms, "top_n": TOP_N,
                                     "ridge_grid": list(RIDGE_GRID)})
    manifest = runs_dir / "_evaluation_manifest.json"
    if args.freeze_evaluation:
        write_exclusive(manifest, snapshot)
        print("Evaluation manifest frozen. No outcome was scored. Review it before proceeding.")
        return 0
    if Path(args.out_dir).resolve() == Path(args.zoo_dir).resolve():
        raise SystemExit("Use a separate new output directory; historical evidence must not be overwritten.")
    if Path(args.out_dir).exists():
        raise SystemExit("Evaluation output directory must not already exist")
    claim_evaluation(manifest, consumed, snapshot)
    bars = apply_membership_filter(pd.read_parquet(args.bars), args.membership)
    scorer = PanelScorer(bars, config, memoize=True)
    alphas = alpha101.load()
    gates = GateRunner(config, [a.expression for a in alphas])
    print(f"holdout {scorer._holdout_dates.min().date()} -> "
          f"{scorer._holdout_dates.max().date()} ({len(scorer._holdout_dates):,} days). "
          "This is the only script that reads it.", flush=True)

    # The A1 hurdle, re-derived on the loop's FIT window so it applies to factors
    # selected there: the 95th percentile of the bootstrapped maximum |t| across
    # the random-grammar pool, from the cached IC panel.
    table = load_ic_table(Path(args.zoo_dir) / "cache")
    pool = pd.read_csv(Path(args.zoo_dir) / "pool.csv")
    random_exprs = pool.loc[(pool["group"] == "random") & pool["error"].isna(),
                            "expression"].tolist()
    fit_window = table.window(config.windows.fit_start, config.windows.fit_end,
                              horizon=1, kind=LAGGED)[random_exprs]
    # Same guard the zoo hurdle uses: a handful of degenerate random draws have
    # a far shorter sample than the rest, and a complete-case intersection over
    # all of them would collapse the null to a few hundred days.
    days = fit_window.notna().sum()
    usable = [c for c in fit_window.columns if days[c] >= 0.9 * days.max()]
    if len(usable) < len(fit_window.columns):
        print(f"  held {len(fit_window.columns) - len(usable)} short-sample random "
              f"factors out of the null")
    fit_ic = fit_window[usable].dropna(axis=0, how="any")
    boot = bootstrap_null_tstats(fit_ic.to_numpy(), draws=args.draws, seed=1)
    random_crit = float(np.quantile(boot.max_abs_t(), 0.95))
    print(f"A1 hurdle on the fit window: |t| > {random_crit:.2f} "
          f"(95th pct of the max over {len(random_exprs)} random-grammar factors, "
          f"{len(fit_ic):,} complete-case days)", flush=True)

    runs_dir = Path(args.runs_dir)
    # A "-raw" directory is a crashed pre-fix attempt kept for the record: real
    # paid work, but not an arm. Including one would report the same arm twice
    # under two names and double its apparent token cost. Underscore-prefixed
    # directories are scratch areas (the salvage holding pen, the replay check).
    arm_dirs = sorted(p for p in runs_dir.iterdir()
                      if p.is_dir() and (p / "archive.jsonl").exists()
                      and not p.name.endswith("-raw")
                      and not p.name.startswith("_")) if runs_dir.exists() else []
    if args.arms:
        arm_dirs = [p for p in arm_dirs if p.name in args.arms]
    if not arm_dirs:
        raise SystemExit(f"no completed arms under {runs_dir}")

    summaries = {}
    for path in arm_dirs:
        print(f"scoring arm {path.name} ...", flush=True)
        store = RunStore(path)
        archive = load_archive(config, store.archive_path)
        # discovered() drops the island seeds: every arm starts from the same
        # three hand-written expressions, so counting them would flatter all
        # arms equally and blur exactly the comparison this table is for
        summaries[path.name] = summarise_arm(path.name, archive.discovered(), scorer,
                                             gates, store, random_crit)

    # The two reference sets, through the same archive rules the arms face.
    #
    # The paraphrase gate is lifted for both, for different reasons. Alpha101
    # would otherwise reject every published alpha for being itself. The random
    # pool has no reason to be judged against a literature it never read: it is
    # the null, and the null's job is to show what this grammar produces from
    # nothing, not to be filtered by novelty.
    #
    # Everything else -- coverage, correlation dedup, niche capacity, the same
    # fitness -- applies identically, which is what makes the rows comparable.
    def score_baseline(label: str, items, *, exclude_self: bool = False):
        print(f"scoring the {label} baseline through the same archive rules "
              f"({len(items)} expressions) ...", flush=True)
        baseline_config = config.with_thresholds(
            max_shared_subtree=10**6, max_abs_corr=config.gates.max_abs_corr)
        baseline_archive = Archive(baseline_config)
        baseline_gates = GateRunner(baseline_config, [])
        for expression, rationale in items:
            try:
                signal = scorer.evaluate(expression)
            except Exception:  # noqa: BLE001
                continue
            if not baseline_gates.coverage_gate(scorer.coverage(signal)).passed:
                continue
            ranked = scorer.ranked_fit(signal)
            if not baseline_gates.correlation_gate(ranked, baseline_archive).passed:
                continue
            metrics = scorer.metrics(expression, signal)
            baseline_archive.try_admit(ArchiveEntry(
                id=candidate_id(expression), expression=expression,
                fitness=metrics.fitness, niche=metrics.niche, round=0,
                island="reference", operator=label, parent_ids=(), rationale=rationale,
                metrics=metrics.to_dict(),
            ), ranked)
        return summarise_arm(label, baseline_archive.members(), scorer, gates, None,
                             random_crit, exclude_self=exclude_self)

    summaries["alpha101"] = score_baseline(
        "alpha101", [(a.expression, a.name) for a in alphas], exclude_self=True)
    summaries["random-grammar"] = score_baseline(
        "random-grammar", [(e, "random grammar draw") for e in random_exprs])

    # ---- the combinations ------------------------------------------------
    combos = []
    for name, summary in summaries.items():
        if not summary["expressions"]:
            continue
        print(f"combining {name} ...", flush=True)
        equal = combined_holdout_ic(scorer, summary["expressions"], summary["signs"])
        weights = fit_ridge(scorer, summary["expressions"], summary["signs"])
        ridge = combined_holdout_ic(scorer, summary["expressions"], summary["signs"],
                                    weights=weights)
        summary["equal_weight_combined"] = equal
        summary["ridge_combined"] = ridge
        combos.append({
            "arm": name,
            "equal_weight_holdout_ic": float(equal.mean()),
            "equal_weight_nw_t": float(newey_west_tstats(equal.to_numpy()[:, None],
                                                         lag=DEFAULT_NW_LAG)[0]),
            "ridge_holdout_ic": float(ridge.mean()),
            "ridge_nw_t": float(newey_west_tstats(ridge.to_numpy()[:, None],
                                                  lag=DEFAULT_NW_LAG)[0]),
            "ridge_weight_concentration": float(np.max(np.abs(weights))
                                                / (np.sum(np.abs(weights)) or np.nan)),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_table = pd.DataFrame([s["row"] for s in summaries.values()])
    combo_table = pd.DataFrame(combos)
    detail = pd.concat([s["detail"] for s in summaries.values()], ignore_index=True)
    main_table.to_csv(out_dir / "evo_final_table.csv", index=False)
    combo_table.to_csv(out_dir / "evo_combined_table.csv", index=False)
    detail.to_csv(out_dir / "evo_top20_detail.csv", index=False)

    # ---- paired bootstraps ----------------------------------------------
    comparisons = []
    def compare(label, a, b):
        if a in summaries and b in summaries and \
                "equal_weight_combined" in summaries[a] and \
                "equal_weight_combined" in summaries[b]:
            result = paired_bootstrap(summaries[a]["equal_weight_combined"],
                                      summaries[b]["equal_weight_combined"],
                                      draws=args.draws)
            comparisons.append({"comparison": label, "a": a, "b": b, **result})

    for model in ("sonnet5", "opus5"):
        compare(f"{model}: loop vs one-shot", f"{model}-loop", f"{model}-oneshot")
        compare(f"{model}: loop vs GP loop", f"{model}-loop", "gp-loop")
    compare("sonnet5: full vs scalar feedback", "sonnet5-loop", "sonnet5-scalar")
    comparison_table = pd.DataFrame(comparisons)
    if not comparison_table.empty:
        comparison_table.to_csv(out_dir / "evo_paired_bootstrap.csv", index=False)
        adjusted = adjust_comparisons(comparison_table)
        adjusted["evidence_status"] = "purged_labels_conditional_on_frozen_search"
        adjusted.to_csv(out_dir / "evo_comparison_audit.csv", index=False)

    # ---- lineage ---------------------------------------------------------
    lineage = ["# Lineage of every top-20 factor\n",
               "Generated by `scripts/sp500_evo_final_table.py`. For each arm, the top 20 "
               "archive members by fit-window fitness, with the rationale the proposer gave "
               "and the chain of parents it came from. Rationales are the model's own words "
               "and are not evidence of anything; they are here so a factor can be read "
               "back to the reasoning that produced it.\n"]
    for name, summary in summaries.items():
        store = RunStore(runs_dir / name)
        archive = load_archive(config, store.archive_path) if (
            runs_dir / name / "archive.jsonl").exists() else None
        by_id = {e.id: e for e in (archive.members() if archive else [])}
        lineage.append(f"\n## {name}\n")
        for rank, entry in enumerate(summary["top"], start=1):
            lineage.append(f"\n### {rank}. `{entry.expression}`\n")
            lineage.append(f"- fitness {entry.fitness:.3f}, niche "
                           f"{'/'.join(entry.niche)}, island {entry.island}, "
                           f"round {entry.round}, operator {entry.operator}")
            lineage.append(f"- rationale: {entry.rationale or '(none recorded)'}")
            chain, current, seen = [], entry, set()
            while current is not None and current.parent_ids:
                parent = by_id.get(current.parent_ids[0])
                if parent is None or parent.id in seen:
                    break
                seen.add(parent.id)
                chain.append(f"`{parent.expression}` ({parent.operator}, round {parent.round})")
                current = parent
            lineage.append("- parents: " + (" <- ".join(chain) if chain
                                            else "none (proposed from scratch)"))
    docs = US_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "lineage.md").write_text("\n".join(lineage) + "\n", encoding="utf-8")

    # ---- print -----------------------------------------------------------
    print("\n" + "=" * 150)
    print("PART C  FINAL TABLE ON THE HOLDOUT (2018-2025)".center(150))
    print("=" * 150 + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(main_table.drop(columns=["rejections_by_gate"]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
    print("\ngate rejections by reason\n")
    for _, row in main_table.iterrows():
        print(f"  {row['arm']:<18}{row['rejections_by_gate']}")

    print("\n" + "-" * 150)
    print("the number that matters: the top-20 as one signal, scored on the holdout\n")
    with pd.option_context("display.width", 200):
        print(combo_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    if not comparison_table.empty:
        print("\npaired block bootstrap of the combined holdout IC "
              f"({args.draws:,} draws, block {DEFAULT_BLOCK})\n")
        for _, row in comparison_table.iterrows():
            print(f"  {row['comparison']:<40}{row['observed']:+.5f}  "
                  f"[{row['lo']:+.5f}, {row['hi']:+.5f}]  p = {row['p']:.3f}  "
                  f"({int(row['days'])} days)")

    print("\nhow each arm stopped, and its per-round validation curve\n")
    for path in arm_dirs:
        store = RunStore(path)
        rounds = store.rounds()
        if not rounds:
            print(f"  {path.name:<18}no completed rounds")
            continue
        planned = int((store.manifest().get("config", {}).get("schedule", {})
                       or {}).get("rounds", 0))
        last = max(int(r["round"]) for r in rounds)
        mode = (store.manifest().get("config", {}) or {}).get("mode", "loop")
        # Derived, because the reason lives in the RunResult rather than on disk.
        # An arm that has response files for a round it never recorded was
        # interrupted part way through that round; one that stopped short with
        # no such files hit the coded early stop; one that reached its schedule
        # simply ran out of rounds.
        interrupted = (path / str(last + 1)).exists()
        if mode == "one_shot":
            reason = "one-shot: a single call per island, no rounds"
        elif interrupted:
            reason = (f"INTERRUPTED during round {last + 1}; its completed calls are "
                      "on disk and its admissions are in the archive")
        elif planned and last < planned:
            reason = ("early stop: archive top-20 validation IC did not improve for "
                      "2 consecutive rounds")
        else:
            reason = "reached the round limit"
        curve = " ".join(f"{float(r['validation_top20_ic']):.5f}"
                         if r.get("validation_top20_ic") is not None else "n/a"
                         for r in sorted(rounds, key=lambda r: r["round"]))
        print(f"  {path.name:<18}{last}/{planned or last} rounds  {reason}")
        print(f"  {'':<18}curve: {curve}")

    print("\ntokens and wall clock per arm, from the saved envelopes in each live "
          "run directory\n")
    for path in arm_dirs:
        totals = token_totals(RunStore(path))
        print(f"  {path.name:<18}{totals['calls']:>5} calls  "
              f"{totals['total_tokens']:>12,} tokens  "
              f"{totals['output_tokens']:>9,} out  "
              f"{totals['model_wall_clock_minutes']:>8.1f} min in model calls"
              + (f"  ${totals['cost_usd']:.2f}" if totals["cost_usd"] is not None else ""))
    print("\n  cost_usd is the CLI envelope's list-price equivalent -- what these "
          "tokens would\n  have cost through the API. These runs went through an "
          "interactive subscription,\n  so nothing here was billed; the real costs are "
          "model wall clock and usage limits.")
    print("\n  gp-loop's token and duration figures are synthetic bookkeeping. Its "
          "backend\n  draws from the grammar in process and never calls a model, so "
          "the numbers are\n  what the mock envelope recorded, not work anyone paid "
          "for. Its cost is zero.")
    print("  The two one-shot arms were rebuilt from replies bought by an earlier "
          "crashed\n  attempt. Those attempts are preserved in the -raw directories and "
          "are real paid\n  work, correctly excluded here: this table reports the arm "
          "that exists, not the\n  arithmetic of getting to it.")

    print(f"\n-> {out_dir / 'evo_final_table.csv'}")
    print(f"-> {out_dir / 'evo_combined_table.csv'}")
    print(f"-> {out_dir / 'evo_top20_detail.csv'}")
    print(f"-> {docs / 'lineage.md'}")
    print(f"scoring wall clock: {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
