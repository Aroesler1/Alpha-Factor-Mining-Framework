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


def rejection_tally(store: RunStore) -> tuple[dict[str, int], dict[str, int]]:
    """(gate rejections, archive refusals) for one arm, from the two places they live.

    A GATE rejection never earned a score: sanitizer, complexity, a paraphrase
    check, coverage, correlation or the cheap screen. Those are appended to
    rejections.jsonl as they happen.

    An ARCHIVE refusal passed every gate and was scored, and was still turned
    away because its niche was full and it did not beat the weakest member.
    Those were only ever incremented in memory and folded into the round
    summary, so a tally read from rejections.jsonl alone understates the total
    by 5 to 38 depending on the arm.

    They are returned separately rather than merged because they answer
    different questions -- how much of the search was unusable, against how much
    was usable but crowded out -- and because an archive refusal is already
    counted among the SCORED candidates, so adding it to a proposal count would
    count it twice.
    """
    gates: dict[str, int] = {}
    for record in store.rejections():
        gate = record.get("gate", "")
        gates[gate] = gates.get(gate, 0) + 1
    archive: dict[str, int] = {}
    for round_record in store.rounds():
        for gate, count in (round_record.get("rejections") or {}).items():
            if gate.startswith("archive:"):
                archive[gate] = archive.get(gate, 0) + int(count)
    return gates, archive


def candidates_generated(store: RunStore) -> int:
    """Every proposal an arm produced, counted exactly once.

    The two files partition the proposals: a candidate is either scored and
    written to candidates.jsonl, or rejected at a gate and written to
    rejections.jsonl. Never both. The figure runs nine above the round
    counters' proposal total for every arm in this experiment, and that nine is
    the island seeds, which are scored at round 0 and are not proposals.
    """
    return len(store.candidates()) + len(store.rejections())


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
