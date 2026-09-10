"""What a candidate is told about itself, and the wall that keeps it fit-window only.

The MUTATE and SIMPLIFY prompts are the only place the loop hands a model
numbers, and the whole experiment depends on those numbers coming from the fit
window and nowhere else. So the feedback block is built from a
`CandidateMetrics` -- an object with no validation or holdout field to leak --
plus structural facts that are computed from text. Nothing here takes a
`PanelScorer`, so there is no path from a prompt to `validation_mean_ic`.

The fields are the ones B7 specifies: mean IC, t, ICIR, the fourteen yearly ICs,
coverage, turnover, IC half-life, the nearest archive member and how correlated
it is, the nearest published alpha by name and shared-subtree size, the
complexity numbers against their limits, and -- for a rejected candidate -- the
gate that stopped it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantaalpha_us.evo.config import EvoConfig
from quantaalpha_us.evo.scoring import CandidateMetrics


@dataclass
class StructuralContext:
    """The text-derived half of a feedback block."""

    nearest_archive_expression: str = ""
    nearest_archive_correlation: float = float("nan")
    nearest_alpha_name: str = ""
    nearest_alpha_shared_nodes: int = 0
    nearest_alpha_edit_distance: int = -1


def _fmt(value: float, digits: int = 5) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _half_life(value: float) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    if np.isinf(value):
        return "> 63 days (does not halve inside the measured grid)"
    return f"{value:.1f} days"


def candidate_feedback(
    metrics: CandidateMetrics,
    context: StructuralContext,
    config: EvoConfig,
    *,
    identifier: str = "",
    rejected_by: str = "",
) -> str:
    """One parent's diagnostics, as the block that goes into a prompt."""
    gates = config.gates
    comp = metrics.complexity
    years = " ".join(
        f"{year}:{value:+.4f}" for year, value in sorted(metrics.ic_by_year.items())
    )
    lines = [
        f"id: {identifier or 'unknown'}",
        f"expression: {metrics.expression}",
        f"  fit-window mean IC  {_fmt(metrics.mean_ic)}   "
        f"t {_fmt(metrics.tstat, 2)}   ICIR {_fmt(metrics.icir, 3)}   "
        f"fitness {_fmt(metrics.fitness, 3)}",
        f"  sign as fitted      {'+1' if metrics.sign >= 0 else '-1'}   "
        f"stability {_fmt(metrics.stability, 2)} of fit-window years with positive IC",
        f"  coverage            {_fmt(metrics.coverage, 3)} "
        f"(limit >= {gates.min_coverage:.2f})",
        f"  daily turnover      {_fmt(metrics.turnover, 3)} one-way, "
        f"charged {_fmt(metrics.turnover_penalty, 3)} of fitness",
        f"  IC half-life        {_half_life(metrics.half_life)}",
        f"  IC by year          {years}",
        f"  complexity          symbol length {comp.get('symbol_length', 0)}"
        f"/{gates.max_symbol_length}, fields {comp.get('base_fields', 0)}"
        f"/{gates.max_base_fields}, free constants {comp.get('free_constants', 0)}"
        f"/{gates.max_free_constants}, depth {comp.get('depth', 0)}/{gates.max_depth}"
        f", charged {_fmt(metrics.complexity_penalty, 3)} of fitness",
    ]
    if context.nearest_archive_expression:
        lines.append(
            f"  nearest in archive  |rho| {_fmt(context.nearest_archive_correlation, 3)} "
            f"against {context.nearest_archive_expression}"
        )
    else:
        lines.append("  nearest in archive  nothing correlated above the noise")
    if context.nearest_alpha_name:
        lines.append(
            f"  nearest published   {context.nearest_alpha_name}: shares "
            f"{context.nearest_alpha_shared_nodes} nodes "
            f"(limit < {gates.max_shared_subtree}), "
            f"{context.nearest_alpha_edit_distance} tree edits away"
        )
    if rejected_by:
        lines.append(f"  REJECTED BY         {rejected_by}")
    return "\n".join(lines)


def rejection_note(expression: str, reason: str) -> str:
    """A one-line record for a candidate that never got as far as being scored."""
    return f"  rejected: {expression[:120]}\n      reason: {reason}"
