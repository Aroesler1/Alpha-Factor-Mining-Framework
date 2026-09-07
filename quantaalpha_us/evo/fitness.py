"""The hard gates, in cost order, each rejection logged with a reason.

A candidate has to clear six gates before it is scored at all, and the order is
the point: the first four read the expression as text and cost microseconds, the
last two need the panel. Roughly nine in ten rejections happen before anything
touches a DataFrame.

  1 sanitizer          the repo's existing gate: syntax, identifiers, arity
  2 complexity         symbol length, base fields, free constants, depth
  3 alpha101 subtree   a complete subtree of >= N nodes shared with a published
                       alpha is a paraphrase, not a discovery
  4 archive subtree    the same test against what the loop has already found,
                       so an island cannot re-derive its own best factor
  5 coverage           at least 90% of the fit panel, or the IC is measured on a
                       corner of the universe
  6 correlation        |rank correlation| against every archive member

Then the cheap-then-full screen: three years first, and the full fit window only
for what survives.

The two thresholds in gates 3 and 6 are the ones chosen on validation; every
other number here is fixed before any run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from quantaalpha_us.evo.archive import Archive
from quantaalpha_us.evo.config import EvoConfig
from quantaalpha_us.factors import ast_tools as at
from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer

GATE_NAMES = (
    "sanitizer", "complexity", "alpha101_subtree", "archive_subtree",
    "evaluation", "coverage", "correlation", "cheap_screen",
)


@dataclass
class GateOutcome:
    passed: bool
    gate: str = ""
    detail: str = ""

    @property
    def reason(self) -> str:
        return "" if self.passed else f"{self.gate}: {self.detail}"


PASSED = GateOutcome(True)


class GateRunner:
    """All six gates plus the cheap screen, sharing one parsed Alpha101 set."""

    def __init__(self, config: EvoConfig, alpha101: Sequence[str],
                 sanitizer: ExpressionSanitizer | None = None) -> None:
        self.config = config
        self.sanitizer = sanitizer or ExpressionSanitizer()
        self.alpha101: list[tuple[str, at.Node]] = []
        self._alpha_subtrees: list[tuple[str, dict[str, int]]] = []
        self._alpha_sizes: list[tuple[str, at.Node, int]] = []
        # memo for the edit-distance lookup, which is the single most expensive
        # thing the feedback block asks for and is asked for the same parents
        # round after round (elites survive)
        self._nearest_edit_cache: dict[str, tuple[int, str]] = {}
        for expr in alpha101:
            try:
                tree = at.parse(expr)
            except at.ParseError:
                continue
            self.alpha101.append((expr, tree))
            self._alpha_subtrees.append((expr, at.subtree_sizes(tree)))
            self._alpha_sizes.append((expr, tree, at.size(tree)))

    # ---- text-only -------------------------------------------------------

    def text_gates(self, expression: str, archive: Archive) -> GateOutcome:
        result = self.sanitizer.sanitize(expression)
        if not result.valid:
            return GateOutcome(False, "sanitizer", "; ".join(result.errors)[:200])

        gates = self.config.gates
        try:
            comp = at.complexity(expression)
        except at.ParseError as exc:
            return GateOutcome(False, "sanitizer", str(exc)[:200])
        for label, value, limit in (
            ("symbol_length", comp.symbol_length, gates.max_symbol_length),
            ("base_fields", comp.base_fields, gates.max_base_fields),
            ("free_constants", comp.free_constants, gates.max_free_constants),
            ("depth", comp.depth, gates.max_depth),
        ):
            if value > limit:
                return GateOutcome(False, "complexity", f"{label} {value} > {limit}")

        shared, who = self.max_shared_alpha101(expression)
        if shared >= gates.max_shared_subtree:
            return GateOutcome(False, "alpha101_subtree",
                               f"shares {shared} nodes with `{who[:70]}`")

        shared, who = archive.max_shared_subtree(expression)
        if shared >= gates.max_shared_subtree:
            return GateOutcome(False, "archive_subtree",
                               f"shares {shared} nodes with `{who[:70]}`")
        return PASSED

    def max_shared_alpha101(self, expression: str) -> tuple[int, str]:
        try:
            candidate = at.subtree_sizes(at.parse(expression))
        except at.ParseError:
            return 0, ""
        best, who = 0, ""
        for expr, other in self._alpha_subtrees:
            shared = at.largest_shared_from_maps(candidate, other)
            if shared > best:
                best, who = shared, expr
        return best, who

    def nearest_alpha101_edit(self, expression: str) -> tuple[int, str]:
        """Edit distance to the closest published alpha, for the feedback block.

        Exact, but it does not compute all 50 distances. Two things make that
        affordable, and neither changes the answer:

        - a memo, because the feedback block is built for the same elite parents
          round after round;
        - a size prune. Zhang-Shasha with unit costs cannot return less than the
          difference in node counts, so once a distance of d has been found,
          every formula whose size differs by d or more can be skipped. Visiting
          the alphas in order of size difference makes that prune bite
          immediately: on this Alpha101 set it removes about four in five of the
          comparisons.

        Without it a single feedback block cost 50 full tree-edit computations
        against formulas of up to 51 nodes, which was most of a round's wall
        clock once the islands had real populations.
        """
        cached = self._nearest_edit_cache.get(expression)
        if cached is not None:
            return cached
        try:
            tree = at.parse(expression)
        except at.ParseError:
            return -1, ""
        own = at.size(tree)
        candidates = sorted(self._alpha_sizes, key=lambda row: abs(row[2] - own))
        best, who = None, ""
        for expr, other, size in candidates:
            lower_bound = abs(size - own)
            if best is not None and lower_bound >= best:
                break  # sorted by the bound, so nothing later can beat `best`
            distance = at.tree_edit_distance(tree, other)
            if best is None or distance < best:
                best, who = distance, expr
        result = ((best if best is not None else -1), who)
        self._nearest_edit_cache[expression] = result
        return result

    def nearest_alpha101_excluding(self, expression: str) -> tuple[int, int]:
        """(largest shared subtree, edit distance) against every alpha BUT itself.

        Only the Alpha101 baseline needs this, and it needs it badly: comparing
        a published alpha against a set that contains it reports that it matches
        itself, which is not a statistic.
        """
        try:
            tree = at.parse(expression)
            candidate = at.subtree_sizes(tree)
        except at.ParseError:
            return 0, -1
        own = at.canonical_text(tree)
        best_shared, best_distance = 0, None
        for expr, other in self.alpha101:
            try:
                if at.canonical_text(other) == own:
                    continue
            except at.ParseError:
                continue
            shared = at.largest_shared_from_maps(candidate, at.subtree_sizes(other))
            best_shared = max(best_shared, shared)
            distance = at.tree_edit_distance(tree, other)
            if best_distance is None or distance < best_distance:
                best_distance = distance
        return best_shared, (best_distance if best_distance is not None else -1)

    # ---- panel-backed ----------------------------------------------------

    def coverage_gate(self, coverage: float) -> GateOutcome:
        limit = self.config.gates.min_coverage
        if not np.isfinite(coverage) or coverage < limit:
            return GateOutcome(False, "coverage", f"{coverage:.3f} < {limit:.2f}")
        return PASSED

    def correlation_gate(self, ranked: np.ndarray, archive: Archive) -> GateOutcome:
        limit = self.config.gates.max_abs_corr
        corr, who = archive.max_abs_correlation(ranked)
        if corr >= limit:
            return GateOutcome(False, "correlation",
                               f"|rho| {corr:.3f} >= {limit:.2f} against `{who[:70]}`")
        return PASSED

    def cheap_screen(self, tstat: float) -> GateOutcome:
        limit = self.config.gates.cheap_min_abs_t
        if not np.isfinite(tstat) or abs(tstat) < limit:
            return GateOutcome(False, "cheap_screen",
                               f"|t| {tstat:.2f} < {limit:.2f} on the cheap window")
        return PASSED

    # ---- operator-specific structural checks -----------------------------

    def mutation_is_local(self, child: str, parent: str) -> GateOutcome:
        limit = self.config.gates.max_mutation_edits
        try:
            distance = at.edit_distance_from_parent(child, parent)
        except at.ParseError as exc:
            return GateOutcome(False, "sanitizer", str(exc)[:200])
        if distance > limit:
            return GateOutcome(False, "mutation_not_local",
                               f"{distance} edits from the parent > {limit}")
        return PASSED

    def crossover_uses_both_parents(self, child: str, parents: Sequence[str]) -> GateOutcome:
        need = self.config.gates.min_crossover_shared
        try:
            tree = at.parse(child)
        except at.ParseError as exc:
            return GateOutcome(False, "sanitizer", str(exc)[:200])
        for parent in parents:
            try:
                shared = at.largest_shared_subtree(tree, at.parse(parent))
            except at.ParseError:
                return GateOutcome(False, "crossover_parent_unparsable", parent[:70])
            if shared < need:
                return GateOutcome(False, "crossover_not_a_fusion",
                                   f"shares only {shared} nodes with `{parent[:60]}`")
        return PASSED

    def simplify_is_an_improvement(self, child_complexity: dict, parent_complexity: dict,
                                   child_abs_ic: float, parent_abs_ic: float) -> GateOutcome:
        """Simpler AND still working. Either alone is not a simplification."""
        share = self.config.gates.simplify_min_ic_share
        child_cost = child_complexity["base_fields"] + child_complexity["free_constants"] \
            + child_complexity["depth"]
        parent_cost = parent_complexity["base_fields"] + parent_complexity["free_constants"] \
            + parent_complexity["depth"]
        if child_cost >= parent_cost:
            return GateOutcome(False, "simplify_not_simpler",
                               f"complexity {child_cost} >= parent {parent_cost}")
        if not np.isfinite(parent_abs_ic) or parent_abs_ic <= 0:
            return PASSED
        if child_abs_ic < share * parent_abs_ic:
            return GateOutcome(
                False, "simplify_lost_too_much_ic",
                f"|IC| {child_abs_ic:.5f} < {share:.0%} of parent {parent_abs_ic:.5f}",
            )
        return PASSED
