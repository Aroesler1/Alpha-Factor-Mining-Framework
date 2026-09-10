"""Tree-level genetic operators: the GP control arm, and the mock backend's mouth.

Two callers, one implementation on purpose.

The GP control (Part B9) is the experiment's most important comparison. It runs
the SAME loop -- same gates, same fitness, same archive, same islands, same
round schedule -- and swaps only the proposal step for random draws from the
sanitizer's grammar. If the model arm beats it, the gain is the model. If it
does not, the gain was the scaffolding, and saying so is the point of building
the control at all.

The mock backend used by every offline test also proposes through these, so the
test suite exercises the real operator plumbing rather than a stub that always
returns the same string.

Every operator is typed. A window slot keeps an integer literal, a scalar slot
keeps a scalar, a condition slot keeps a comparison, and only panel-valued
positions are ever cut or grafted -- the same discipline `random_expressions`
documents for the null sampler, and for the same reason: an untyped edit passes
the sanitizer and then dies inside the evaluator, which silently biases the
control toward whatever survives by accident.
"""

from __future__ import annotations

import random
from typing import Sequence

from quantaalpha_us.factors import ast_tools as at
from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer
from quantaalpha_us.factors.random_expressions import (
    DEFAULT_WINDOWS,
    PRICE_FIELDS,
    SIGNATURES,
    GrammarSampler,
    structure_profile,
)

# Fundamentals are available to the loop, unlike the price-only null in
# `random_expressions`. The fundamental-quality island has to be able to reach
# them, and the archive's data-family niche is empty without them.
ALL_FIELDS: tuple[str, ...] = PRICE_FIELDS + tuple(sorted(at.FUNDAMENTAL_FIELDS))

# Unary panel -> panel functions, canonical names only. Used to wrap and unwrap.
UNARY_PANEL = ("ABS", "CS_DEMEAN", "LOG", "RANK", "SIGN", "SQRT", "ZSCORE")

SCALARS = ("0.25", "0.5", "1", "1.5", "2", "3")

MUTATION_KINDS = ("window", "field", "relabel", "wrap", "unwrap", "scalar")


def _canonical_by_signature() -> dict[str, list[str]]:
    """Canonical function names grouped by argument shape, for relabelling."""
    out: dict[str, list[str]] = {}
    for name, sig in SIGNATURES.items():
        canonical = at.FUNCTION_ALIASES.get(name, name)
        out.setdefault(sig, [])
        if canonical not in out[sig]:
            out[sig].append(canonical)
    for names in out.values():
        names.sort()
    return out


BY_SIGNATURE = _canonical_by_signature()


def _signature_of(label: str) -> str | None:
    for name in (label, *(k for k, v in at.FUNCTION_ALIASES.items() if v == label)):
        sig = SIGNATURES.get(name)
        if sig is not None:
            return sig
    return None


def _accept(tree: at.Node, sanitizer: ExpressionSanitizer,
            *, forbid: str | None = None) -> str | None:
    """Render, sanitize, and re-parse as a check. Returns the expression or None.

    The rendering deliberately keeps the tree's WRITTEN operand order rather
    than re-emitting the canonical sort. Two reasons:

    - the mutation-locality gate measures tree edit distance from the parent,
      and re-sorting a commutative node turns a one-node edit into three or
      four (see `ast_tools.parse`'s note);
    - nothing downstream needs the string to be canonical, because identity is
      carried by `canonical_text`'s hash, which already ignores operand order.

    `forbid` rejects a result that is the same factor as the parent, compared
    through that order-insensitive canonical form so a reordering does not read
    as a new candidate.
    """
    try:
        rendered = at.to_expression(tree)
    except Exception:  # noqa: BLE001 - a malformed graft should not kill a round
        return None
    if not sanitizer.sanitize(rendered).valid:
        return None
    try:
        reparsed = at.parse(rendered)
    except at.ParseError:
        return None
    if forbid is not None:
        try:
            if at.canonical_text(reparsed) == at.canonical_text(at.parse(forbid)):
                return None
        except at.ParseError:
            return None
    return rendered


# --------------------------------------------------------------------------
# EXPLORE
# --------------------------------------------------------------------------


def random_expressions(
    rng: random.Random,
    n: int,
    *,
    reference: Sequence[str],
    fields: Sequence[str] = ALL_FIELDS,
    sanitizer: ExpressionSanitizer | None = None,
) -> list[str]:
    """`n` fresh draws from the grammar, shaped like `reference`.

    Matched to the reference set's call-count distribution for the same reason
    the baseline null is: "best of N" depends on how expressive each of the N
    is, and a control made of two-node expressions is not a control.
    """
    sanitizer = sanitizer or ExpressionSanitizer()
    sampler = GrammarSampler(
        seed=rng.randrange(2**31), profile=structure_profile(reference),
        fields=tuple(fields), windows=DEFAULT_WINDOWS, sanitizer=sanitizer,
    )
    out: list[str] = []
    seen: set[str] = set()
    for _ in range(n * 40):
        if len(out) >= n:
            break
        try:
            drawn = sampler.sample()
        except RuntimeError:
            break
        accepted = _accept(at.parse(drawn), sanitizer)
        if accepted is None or accepted in seen:
            continue
        seen.add(accepted)
        out.append(accepted)
    return out


# --------------------------------------------------------------------------
# MUTATE
# --------------------------------------------------------------------------


def mutate(
    rng: random.Random,
    parent: str,
    *,
    fields: Sequence[str] = ALL_FIELDS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    max_depth: int | None = None,
    sanitizer: ExpressionSanitizer | None = None,
    attempts: int = 40,
) -> tuple[str, str] | None:
    """One typed single-node edit. Returns (expression, what changed), or None.

    Six kinds, all of which move the tree by at most a couple of edits: retune a
    lookback window, swap a base field, relabel a function to another of the
    same shape, wrap a sub-expression in a unary transform, unwrap one, or
    retune a scalar parameter.
    """
    sanitizer = sanitizer or ExpressionSanitizer()
    try:
        tree = at.parse(parent, canonical=False)
    except at.ParseError:
        return None
    kinds = at.slot_kinds(tree)
    all_paths = at.paths(tree)

    for _ in range(attempts):
        kind = rng.choice(MUTATION_KINDS)
        candidates: list[tuple[int, ...]] = []
        if kind == "window":
            candidates = [p for p in all_paths if kinds.get(p) == at.WINDOW_SLOT]
        elif kind == "scalar":
            candidates = [p for p in all_paths if kinds.get(p) == at.SCALAR_SLOT]
        elif kind == "field":
            candidates = [p for p in all_paths if at.at_path(tree, p).label.startswith("$")]
        elif kind == "relabel":
            candidates = [p for p in all_paths
                          if _signature_of(at.at_path(tree, p).label) is not None]
        elif kind == "wrap":
            candidates = at.panel_paths(tree)
        elif kind == "unwrap":
            candidates = [p for p in all_paths
                          if _signature_of(at.at_path(tree, p).label) == "P"]
        if not candidates:
            continue

        path = rng.choice(candidates)
        node = at.at_path(tree, path)

        if kind == "window":
            options = [w for w in windows if f"#{w}" != node.label]
            if not options:
                continue
            new = at.Node(f"#{rng.choice(options)}")
        elif kind == "scalar":
            options = [s for s in SCALARS if f"#{float(s):g}" != node.label]
            new = at.Node(f"#{rng.choice(options)}")
        elif kind == "field":
            options = [f for f in fields if f"${f}" != node.label]
            if not options:
                continue
            new = at.Node(f"${rng.choice(options)}")
        elif kind == "relabel":
            sig = _signature_of(node.label)
            # only same-shape functions, and only when the node's actual arity
            # matches the signature -- MIN/MAX are variadic and a three-operand
            # MIN cannot become a two-argument TS_CORR
            options = ([n for n in BY_SIGNATURE.get(sig, []) if n != node.label]
                       if len(node.children) == len(sig) else [])
            if not options:
                continue
            new = at.Node(rng.choice(options), node.children)
        elif kind == "wrap":
            new = at.Node(rng.choice(UNARY_PANEL), (node,))
        else:  # unwrap
            new = node.children[0]

        mutated = at.replace_at(tree, path, new)
        if max_depth is not None and at.depth(mutated) > max_depth:
            continue
        accepted = _accept(mutated, sanitizer, forbid=parent)
        if accepted is not None:
            return accepted, f"{kind}@{'.'.join(map(str, path)) or 'root'}"
    return None


# --------------------------------------------------------------------------
# CROSSOVER
# --------------------------------------------------------------------------


def crossover(
    rng: random.Random,
    left: str,
    right: str,
    *,
    max_depth: int | None = None,
    min_shared: int = 3,
    sanitizer: ExpressionSanitizer | None = None,
    attempts: int = 60,
) -> str | None:
    """Graft a panel-valued subtree of `right` into a panel slot of `left`.

    The root of `left` is excluded as a destination: grafting at the root just
    returns a subtree of `right`, which is not a combination of two parents and
    would let the operator quietly degenerate into copying.

    The child must share a complete subtree of at least `min_shared` nodes with
    EACH parent -- the same structural check the loop applies to the model's
    CROSSOVER replies. Applying it here too is what keeps the GP arm a fair
    control: a control that is allowed to submit non-crossovers while the model
    is not would be measuring the gate, not the proposer. Donors are drawn
    largest-first among those big enough to satisfy it, so the requirement
    shapes the search instead of only filtering it.
    """
    sanitizer = sanitizer or ExpressionSanitizer()
    try:
        a = at.parse(left, canonical=False)
        b = at.parse(right, canonical=False)
    except at.ParseError:
        return None
    destinations = [p for p in at.panel_paths(a) if p != ()]
    donor_paths = at.panel_paths(b)
    if not destinations or not donor_paths:
        return None
    big_donors = [p for p in donor_paths if at.size(at.at_path(b, p)) >= min_shared]
    donor_pool = big_donors or donor_paths

    for _ in range(attempts):
        dest = rng.choice(destinations)
        donor = at.at_path(b, rng.choice(donor_pool))
        child = at.replace_at(a, dest, donor)
        if max_depth is not None and at.depth(child) > max_depth:
            continue
        accepted = _accept(child, sanitizer, forbid=left)
        if accepted is None:
            continue
        reparsed = at.parse(accepted)
        if at.canonical_text(reparsed) == at.canonical_text(at.parse(right)):
            continue
        if at.largest_shared_subtree(reparsed, at.parse(left)) < min_shared:
            continue
        if at.largest_shared_subtree(reparsed, at.parse(right)) < min_shared:
            continue
        return accepted
    return None


# --------------------------------------------------------------------------
# SIMPLIFY
# --------------------------------------------------------------------------


def simplify(
    rng: random.Random,
    expression: str,
    *,
    sanitizer: ExpressionSanitizer | None = None,
    attempts: int = 40,
) -> str | None:
    """Prune: replace an internal node with one of its panel-valued children."""
    sanitizer = sanitizer or ExpressionSanitizer()
    try:
        tree = at.parse(expression, canonical=False)
    except at.ParseError:
        return None
    kinds = at.slot_kinds(tree)
    internal = [p for p in at.paths(tree) if at.at_path(tree, p).children]
    if not internal:
        return None

    for _ in range(attempts):
        path = rng.choice(internal)
        node = at.at_path(tree, path)
        child_slots = [i for i in range(len(node.children))
                       if kinds.get(path + (i,)) == at.PANEL_SLOT]
        if not child_slots:
            continue
        pruned = at.replace_at(tree, path, node.children[rng.choice(child_slots)])
        # pruning to a subtree with no field left is not a simpler factor, it is
        # a constant, and a constant cross-section has no ordering to score
        if not any(n.label.startswith("$") for n in pruned):
            continue
        accepted = _accept(pruned, sanitizer, forbid=expression)
        if accepted is not None and at.size(at.parse(accepted)) < at.size(at.parse(expression)):
            return accepted
    return None
