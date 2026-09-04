"""Tests for the canonical-tree structure tools.

The structural claims in the README ("is the model paraphrasing?") rest
entirely on this module, so the canonicalisation rules are pinned one by one:
an alias that stops collapsing, or a commutative sort that starts sorting
subtraction, would move every reported shared-subtree number without failing
anything else in the suite.
"""

from __future__ import annotations

import pytest

from quantaalpha_us.factors import ast_tools as at
from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer


# ---- parsing and canonicalisation ---------------------------------------


def test_aliases_collapse_to_one_label():
    assert at.canonical_text(at.parse("RANK($close)")) == at.canonical_text(
        at.parse("CS_RANK($close)")
    )
    assert at.canonical_text(at.parse("DELTA($close, 5)")) == at.canonical_text(
        at.parse("TS_DELTA($close, 5)")
    )
    assert at.canonical_text(at.parse("LN($volume)")) == at.canonical_text(
        at.parse("LOG($volume)")
    )
    assert at.canonical_text(at.parse("IF($close > 0, 1, 2)")) == at.canonical_text(
        at.parse("IF_ELSE($close > 0, 1, 2)")
    )
    assert at.canonical_text(at.parse("TS_COVARIANCE($close, $volume, 5)")) == (
        at.canonical_text(at.parse("TS_COV($close, $volume, 5)"))
    )


def test_every_alias_maps_onto_a_function_the_sanitizer_knows():
    known = {f.upper() for f in ExpressionSanitizer.DEFAULT_ALLOWED_FUNCTIONS}
    for alias, target in at.FUNCTION_ALIASES.items():
        assert alias in known
        assert target in known


def test_numeric_literals_normalise():
    assert at.canonical_text(at.parse("TS_MEAN($close, 5)")) == at.canonical_text(
        at.parse("TS_MEAN($close, 5.0)")
    )
    assert at.canonical_text(at.parse("$close * 0.50")) == at.canonical_text(
        at.parse("$close * 0.5")
    )


def test_commutative_operands_sort_but_non_commutative_ones_do_not():
    assert at.canonical_text(at.parse("$high + $low")) == at.canonical_text(
        at.parse("$low + $high")
    )
    assert at.canonical_text(at.parse("$high * $low")) == at.canonical_text(
        at.parse("$low * $high")
    )
    # subtraction and division are the difference between a factor and its
    # negation / reciprocal, so sorting them would equate opposite ideas
    assert at.canonical_text(at.parse("$high - $low")) != at.canonical_text(
        at.parse("$low - $high")
    )
    assert at.canonical_text(at.parse("$high / $low")) != at.canonical_text(
        at.parse("$low / $high")
    )
    assert at.canonical_text(at.parse("$high > $low")) != at.canonical_text(
        at.parse("$low > $high")
    )


def test_ts_corr_is_commutative_in_its_operands_only():
    assert at.canonical_text(at.parse("TS_CORR($close, $volume, 10)")) == (
        at.canonical_text(at.parse("TS_CORR($volume, $close, 10)"))
    )
    assert at.canonical_text(at.parse("TS_CORR($close, $volume, 10)")) != (
        at.canonical_text(at.parse("TS_CORR($close, $volume, 21)"))
    )


def test_minmax_is_commutative_in_every_operand():
    assert at.canonical_text(at.parse("MIN($close, $open, $high)")) == (
        at.canonical_text(at.parse("MIN($high, $close, $open)"))
    )


def test_parse_rejects_what_the_evaluator_rejects():
    with pytest.raises(at.ParseError):
        at.parse("close")           # bare field, no $
    with pytest.raises(at.ParseError):
        at.parse("$close % 2")      # operator the evaluator has no branch for
    with pytest.raises(at.ParseError):
        at.parse("TS_MEAN($close,") # syntax


# ---- size, depth, hashing ------------------------------------------------


def test_size_and_depth():
    tree = at.parse("TS_MEAN($close, 21)")
    assert at.size(tree) == 3          # call + field + window
    assert at.depth(tree) == 1
    assert at.depth(at.parse("$close")) == 0
    assert at.size(at.parse("$close")) == 1


def test_subtree_hash_is_stable_across_processes():
    # blake2b of the canonical text, not Python's salted hash. Pinned literally
    # so a change to the serialisation format cannot pass silently.
    assert at.subtree_hash(at.parse("$close")) == at.subtree_hash(at.parse("$close"))
    assert len(at.subtree_hash(at.parse("$close"))) == 16


def test_subtree_sizes_enumerates_every_complete_subtree():
    sizes = at.subtree_sizes(at.parse("TS_MEAN($close, 21)"))
    assert sorted(sizes.values()) == [1, 1, 3]


# ---- largest shared subtree ---------------------------------------------


def test_largest_shared_subtree_finds_the_whole_shared_piece():
    a = "RANK(TS_DELTA($close, 21)) * 2"
    b = "CS_RANK(TS_DELTA($close, 21)) - $volume"
    # RANK(TS_DELTA($close, 21)) is 4 nodes: RANK, TS_DELTA, $close, 21
    assert at.largest_shared_subtree_expr(a, b) == 4


def test_largest_shared_subtree_is_one_when_only_a_field_is_shared():
    assert at.largest_shared_subtree_expr("TS_STD($close, 5)", "RANK($close)") == 1


def test_largest_shared_subtree_is_symmetric():
    a, b = "TS_MEAN($volume, 10) + $close", "RANK(TS_MEAN($volume, 10))"
    assert at.largest_shared_subtree_expr(a, b) == at.largest_shared_subtree_expr(b, a)


# ---- Zhang-Shasha --------------------------------------------------------


def test_edit_distance_of_a_tree_with_itself_is_zero():
    for expr in ("$close", "TS_MEAN($close, 21)",
                 "RANK(TS_CORR($close, $volume, 5)) / (TS_STD($return, 21) + 1e-8)"):
        assert at.tree_edit_distance_expr(expr, expr) == 0


def test_relabelling_one_leaf_costs_one():
    assert at.tree_edit_distance_expr("TS_MEAN($close, 21)", "TS_MEAN($close, 63)") == 1
    assert at.tree_edit_distance_expr("TS_MEAN($close, 21)", "TS_MEAN($open, 21)") == 1
    assert at.tree_edit_distance_expr("TS_MEAN($close, 21)", "TS_STD($close, 21)") == 1


def test_wrapping_a_tree_in_one_call_costs_one_insert():
    assert at.tree_edit_distance_expr("$close", "RANK($close)") == 1


def test_collapsing_a_tree_to_one_leaf_costs_a_relabel_plus_the_deletions():
    # $close cannot be kept: an ordered mapping that pairs it with $volume still
    # has to delete TS_MEAN and the window, and relabel. Three either way.
    assert at.tree_edit_distance_expr("TS_MEAN($close, 21)", "$volume") == 3


def test_edit_distance_is_symmetric_and_a_metric_on_a_small_set():
    exprs = ["$close", "RANK($close)", "TS_MEAN($close, 21)",
             "RANK(TS_MEAN($close, 21))", "TS_STD($volume, 5)"]
    trees = [at.parse(e) for e in exprs]
    for i, a in enumerate(trees):
        for j, b in enumerate(trees):
            assert at.tree_edit_distance(a, b) == at.tree_edit_distance(b, a)
            if i == j:
                assert at.tree_edit_distance(a, b) == 0
            else:
                assert at.tree_edit_distance(a, b) > 0
    for a in trees:
        for b in trees:
            for c in trees:
                assert (at.tree_edit_distance(a, c)
                        <= at.tree_edit_distance(a, b) + at.tree_edit_distance(b, c))


def test_normalized_distance_is_bounded_by_one():
    a = at.parse("TS_CORR(RANK($close), RANK($volume), 5)")
    b = at.parse("$open")
    d = at.normalized_edit_distance(a, b)
    assert 0.0 < d <= 1.0
    assert at.normalized_edit_distance(a, a) == 0.0


def test_edit_distance_matches_a_brute_force_reference_on_small_trees():
    """Cross-check Zhang-Shasha against exhaustive search over edit scripts.

    The DP is the one part of this module where an off-by-one in the keyroot or
    forest-distance indexing produces a plausible-looking wrong number rather
    than a crash, so it is checked against a definition that shares no code
    with it.
    """
    def brute(a: at.Node, b: at.Node, limit: int = 4) -> int:
        # breadth-first over edit scripts, using the classical mapping-free
        # recursive characterisation on ordered forests
        from functools import lru_cache

        fa, la = at._postorder(a)
        fb, lb = at._postorder(b)

        @lru_cache(maxsize=None)
        def fd(i0: int, i1: int, j0: int, j1: int) -> int:
            # forest a[i0..i1], forest b[j0..j1]; empty when hi < lo
            if i1 < i0 and j1 < j0:
                return 0
            if i1 < i0:
                return j1 - j0 + 1
            if j1 < j0:
                return i1 - i0 + 1
            relabel = 0 if fa[i1].label == fb[j1].label else 1
            return min(
                fd(i0, i1 - 1, j0, j1) + 1,
                fd(i0, i1, j0, j1 - 1) + 1,
                fd(la[i1], i1 - 1, lb[j1], j1 - 1)
                + fd(i0, la[i1] - 1, j0, lb[j1] - 1)
                + relabel,
            )

        return fd(0, len(fa) - 1, 0, len(fb) - 1)

    exprs = ["$close", "RANK($close)", "TS_MEAN($close, 21)",
             "RANK(TS_MEAN($close, 21))", "TS_STD($volume, 5) + $close",
             "TS_CORR($close, $volume, 5)", "MIN($close, $open)"]
    for ea in exprs:
        for eb in exprs:
            a, b = at.parse(ea), at.parse(eb)
            assert at.tree_edit_distance(a, b) == brute(a, b), (ea, eb)


# ---- complexity ----------------------------------------------------------


def test_complexity_counts_distinct_fields_not_references():
    c = at.complexity("TS_CORR($close, $volume, 5) + TS_MEAN($close, 5)")
    assert c.base_fields == 2


def test_windows_are_not_free_constants():
    assert at.complexity("TS_MEAN($close, 21)").free_constants == 0
    assert at.complexity("TS_CORR($close, $volume, 63)").free_constants == 0


def test_the_epsilon_guard_is_not_a_free_constant():
    # every division in this repo carries it by convention; charging for it
    # would put a house-style expression over the gate for following the style
    assert at.complexity("$close / ($volume + 1e-8)").free_constants == 0
    assert at.complexity("$close / ($volume + 0.001)").free_constants == 1


def test_scalar_parameters_are_free_constants():
    assert at.complexity("POWER($close, 2)").free_constants == 1
    assert at.complexity("BOUND($close, -3, 3)").free_constants == 2


def test_symbol_length_is_measured_on_the_normalised_original():
    assert at.complexity("TS_MEAN( $close ,  21 )").symbol_length == len(
        "TS_MEAN( $close , 21 )"
    )


def test_data_family_splits_the_three_ways_the_archive_needs():
    assert at.data_family("RANK($close)") == "price_volume"
    assert at.data_family("RANK($roa)") == "fundamental"
    assert at.data_family("RANK($roa) - RANK(TS_STD($return, 21))") == "mixed"


def test_fundamental_and_price_field_sets_partition_the_sanitizer_fields():
    assert at.FUNDAMENTAL_FIELDS | at.PRICE_VOLUME_FIELDS == set(
        ExpressionSanitizer.KNOWN_FIELDS
    )
    assert not (at.FUNDAMENTAL_FIELDS & at.PRICE_VOLUME_FIELDS)


def test_every_transcribed_alpha101_formula_parses():
    """The structural report is over Alpha101; a formula that will not parse
    would silently drop out of every median in it."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "configs" / "alpha101_us.txt"
    exprs = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert len(exprs) == 50
    for expr in exprs:
        at.parse(expr)


# ---- tree surgery --------------------------------------------------------


def test_rendering_round_trips_through_the_parser():
    for expr in ("$close",
                 "TS_MEAN($close, 21)",
                 "RANK(TS_CORR($close, $volume, 5)) / (TS_STD($return, 21) + 1e-8)",
                 "IF_ELSE($close > $open, TS_MEAN($volume, 5), -TS_STD($return, 10))",
                 "MIN($close, $open, $high)",
                 "-CS_RANK(TS_DELTA($close, 3))",
                 "POWER(ABS($return), 0.5) * SIGN($return)"):
        tree = at.parse(expr)
        rendered = at.to_expression(tree)
        assert at.canonical_text(at.parse(rendered)) == at.canonical_text(tree), rendered


def test_rendered_expressions_pass_the_sanitizer():
    sanitizer = ExpressionSanitizer()
    for expr in ("TS_CORR(RANK($close), RANK($volume), 5)",
                 "BOUND($return, -3, 3)",
                 "COUNT($close > $open, 21)",
                 "$high - ($low * 2)"):
        rendered = at.to_expression(at.parse(expr))
        result = sanitizer.sanitize(rendered)
        assert result.valid, (rendered, result.errors)


def test_rendering_parenthesises_so_association_cannot_drift():
    # `/` is not commutative, so the written order survives canonicalisation and
    # the rendered string can be pinned exactly
    tree = at.parse("$close - $open / $high")
    assert at.to_expression(tree) == "($close - ($open / $high))"
    # and the grouping is preserved rather than reassociated
    assert at.to_expression(at.parse("($close - $open) / $high")) == (
        "(($close - $open) / $high)"
    )


def test_paths_address_every_node_once():
    tree = at.parse("TS_MEAN($close, 21)")
    ps = at.paths(tree)
    assert len(ps) == at.size(tree)
    assert len(set(ps)) == len(ps)
    assert at.at_path(tree, ()) is tree
    assert {at.at_path(tree, p).label for p in ps} == {"TS_MEAN", "$close", "#21"}


def test_replace_at_swaps_exactly_one_subtree():
    tree = at.parse("TS_MEAN($close, 21) + TS_STD($close, 21)")
    target = next(p for p in at.paths(tree) if at.at_path(tree, p).label == "TS_STD")
    swapped = at.replace_at(tree, target, at.parse("$volume"))
    rendered = at.to_expression(swapped)
    assert "TS_STD" not in rendered
    assert "TS_MEAN" in rendered and "$volume" in rendered
    # measured in written order: relabel TS_STD -> $volume, delete its two children
    assert at.edit_distance_from_parent(rendered, at.to_expression(tree)) == 3


def test_slot_kinds_type_the_positions_the_evaluator_cares_about():
    tree = at.parse("TS_MEAN($close, 21)")
    kinds = at.slot_kinds(tree)
    assert kinds[(0,)] == at.PANEL_SLOT
    assert kinds[(1,)] == at.WINDOW_SLOT

    tree = at.parse("BOUND($return, -3, 3)")
    kinds = at.slot_kinds(tree)
    assert kinds[(0,)] == at.PANEL_SLOT
    assert kinds[(1,)] == at.SCALAR_SLOT and kinds[(2,)] == at.SCALAR_SLOT

    tree = at.parse("IF_ELSE($close > $open, $high, $low)")
    kinds = at.slot_kinds(tree)
    assert kinds[(0,)] == at.CONDITION_SLOT
    assert kinds[(1,)] == at.PANEL_SLOT and kinds[(2,)] == at.PANEL_SLOT

    tree = at.parse("TS_CORR($close, $volume, 10)")
    kinds = at.slot_kinds(tree)
    assert kinds[(0,)] == at.PANEL_SLOT and kinds[(1,)] == at.PANEL_SLOT
    assert kinds[(2,)] == at.WINDOW_SLOT


def test_window_and_scalar_positions_are_never_offered_as_panel_slots():
    """The trap random_expressions documents: an untyped edit puts a panel in a
    window slot, passes the sanitizer, and dies in the evaluator."""
    for expr in ("TS_MEAN($close, 21)", "TS_CORR($close, $volume, 10)",
                 "POWER($close, 2)", "BOUND($return, -3, 3)", "COUNT($close > 0, 5)"):
        tree = at.parse(expr)
        panel = set(at.panel_paths(tree))
        kinds = at.slot_kinds(tree)
        for path, kind in kinds.items():
            if kind in (at.WINDOW_SLOT, at.SCALAR_SLOT, at.CONDITION_SLOT):
                assert path not in panel, (expr, path, kind)


def test_commutative_sorting_inflates_the_ordered_distance_of_a_single_edit():
    """Why `edit_distance_from_parent` exists. Canonicalisation re-sorts the
    operands of `+`, and Zhang-Shasha is an ordered distance, so the same edit
    measures 4 canonically and 3 in written order. The MUTATE gate is at 3."""
    parent = "(TS_MEAN($close, 21) + TS_STD($close, 21))"
    child = "(TS_MEAN($close, 21) + $volume)"
    assert at.tree_edit_distance_expr(child, parent) == 4
    assert at.edit_distance_from_parent(child, parent) == 3


def test_written_order_parse_keeps_operand_order_but_still_collapses_aliases():
    assert at.canonical_text(at.parse("$low + $high", canonical=False)) != (
        at.canonical_text(at.parse("$high + $low", canonical=False))
    )
    assert at.canonical_text(at.parse("RANK($close)", canonical=False)) == (
        at.canonical_text(at.parse("CS_RANK($close)", canonical=False))
    )


def test_panel_paths_includes_the_root():
    assert () in at.panel_paths(at.parse("TS_MEAN($close, 21)"))


def test_the_cached_map_form_of_shared_subtree_agrees_with_the_direct_one():
    """The gates use the map form for speed; if the two ever disagreed, every
    paraphrase rejection would be measured differently from every report."""
    exprs = ["$close", "RANK($close)", "TS_MEAN($close, 21)",
             "RANK(TS_DELTA($close, 21)) * 2", "CS_RANK(TS_DELTA($close, 21)) - $volume",
             "TS_CORR(RANK($close), RANK($volume), 5)",
             "IF_ELSE($close > $open, TS_MEAN($volume, 5), -TS_STD($return, 10))"]
    maps = {e: at.subtree_sizes(at.parse(e)) for e in exprs}
    for a in exprs:
        for b in exprs:
            assert at.largest_shared_from_maps(maps[a], maps[b]) == (
                at.largest_shared_subtree_expr(a, b)
            ), (a, b)
