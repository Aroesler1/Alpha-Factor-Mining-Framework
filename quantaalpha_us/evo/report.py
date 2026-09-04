"""Turning a finished run into tables: the archive, the curve, the rejections.

Kept apart from the runner so a run that has already happened can be re-reported
without re-running it, and so the mock end-to-end test and the real Part C
script produce the same columns rather than two similar-looking tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantaalpha_us.evo.archive import ALL_NICHES, Archive
from quantaalpha_us.evo.persistence import RunStore, token_totals


def archive_frame(archive: Archive, *, include_seeds: bool = True) -> pd.DataFrame:
    """One row per archive member, flattened for a CSV."""
    rows = []
    for entry in (archive.members() if include_seeds else archive.discovered()):
        metrics = entry.metrics or {}
        complexity = metrics.get("complexity", {}) or {}
        rows.append({
            "id": entry.id,
            "expression": entry.expression,
            "fitness": entry.fitness,
            "niche": "/".join(entry.niche),
            "horizon_bucket": entry.niche[0],
            "data_family": entry.niche[1],
            "turnover_bucket": entry.niche[2],
            "round": entry.round,
            "island": entry.island,
            "operator": entry.operator,
            "parent_ids": ";".join(entry.parent_ids),
            "rationale": entry.rationale,
            "mean_ic": metrics.get("mean_ic"),
            "tstat": metrics.get("tstat"),
            "icir": metrics.get("icir"),
            "coverage": metrics.get("coverage"),
            "turnover": metrics.get("turnover"),
            "half_life": (np.inf if metrics.get("half_life_is_beyond_grid")
                          else metrics.get("half_life")),
            "stability": metrics.get("stability"),
            "sign": metrics.get("sign"),
            "base_fields": complexity.get("base_fields"),
            "free_constants": complexity.get("free_constants"),
            "depth": complexity.get("depth"),
            "symbol_length": complexity.get("symbol_length"),
        })
    return pd.DataFrame(rows)


def rounds_frame(store: RunStore) -> pd.DataFrame:
    rounds = store.rounds()
    if not rounds:
        return pd.DataFrame()
    def flatten(value) -> str:
        return "; ".join(f"{k}={v}" for k, v in sorted((value or {}).items()))

    for record in rounds:
        for key in ("rejections", "migrated"):
            if key in record:
                record[key] = flatten(record[key])
    return pd.DataFrame(rounds)


def rejection_counts(store: RunStore) -> pd.DataFrame:
    records = store.rejections()
    if not records:
        return pd.DataFrame(columns=["gate", "count"])
    frame = pd.DataFrame(records)
    counts = frame.groupby("gate").size().reset_index(name="count")
    return counts.sort_values("count", ascending=False).reset_index(drop=True)


def niche_frame(archive: Archive) -> pd.DataFrame:
    return pd.DataFrame(archive.niche_summary())


def write_run_report(archive: Archive, store: RunStore, out_dir: Path | None = None) -> dict:
    """Write the four tables an arm is reported through. Returns the paths."""
    out = Path(out_dir) if out_dir is not None else store.root
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "archive": out / "archive_table.csv",
        "rounds": out / "rounds_table.csv",
        "rejections": out / "rejection_counts.csv",
        "niches": out / "niche_table.csv",
    }
    archive_frame(archive).to_csv(paths["archive"], index=False)
    rounds_frame(store).to_csv(paths["rounds"], index=False)
    rejection_counts(store).to_csv(paths["rejections"], index=False)
    niche_frame(archive).to_csv(paths["niches"], index=False)
    return {
        **{k: str(v) for k, v in paths.items()},
        "tokens": token_totals(store),
        "archive_size": len(archive.discovered()),
        "archive_size_with_seeds": len(archive),
        "niches_filled": archive.niches_filled(),
        "niches_possible": len(ALL_NICHES),
    }
