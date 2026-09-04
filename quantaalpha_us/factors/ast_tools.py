"""Structural tools for factor expressions: canonical trees, shared subtrees, edit distance.

Three questions this repo could not previously ask:

- Is a mined factor a paraphrase of a published one? The memorization test in
  `sp500_run_baseline_comparison.py` answers that in *signal* space (rank
  correlation). Signal correlation misses a restatement whose numbers happen to
  differ, and flags two unrelated ideas that happen to co-move. Structure is the
  independent axis: a factor sharing an eight-node subtree with alpha012 is a
  paraphrase whatever its correlation says.
- How complex is a candidate, in terms an evolutionary loop can gate on?
- How far did a mutation move? The MUTATE operator in `quantaalpha_us.evo` is
  only "localized" if something measures the edit.

Everything here is text-only: no pandas, no panel, no data. That keeps it
testable without a WRDS entitlement and cheap enough to call inside a gate.

Canonicalisation
----------------
Two expressions that mean the same thing must produce the same tree, or every
structural statistic below is noise. So, before anything is compared:

- alias functions collapse to one name (`RANK`/`CS_RANK`, `DELTA`/`TS_DELTA`,
  `LOG`/`LN`, `IF`/`IF_ELSE`, `TS_COV`/`TS_COVARIANCE`, and the lowercase
  spellings the sanitizer also accepts);
- numeric literals normalise, so `5`, `5.0` and `5.00` are one label;
- commutative operands sort by their own canonical form, so `$high - $low`
  keeps its order but `$high + $low` and `$low + $high` do not differ. `TS_CORR`
  and `TS_COV` are commutative in their first two arguments only, and `MIN`/
  `MAX` are commutative in all of theirs.

Subtraction, division and comparison are NOT sorted; they are not commutative
and sorting them would equate a factor with its own negation.

Tree edit distance is Zhang-Shasha (1989) with unit insert/delete/relabel costs,
implemented here rather than taken from a dependency.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer
from quantaalpha_us.factors.random_expressions import SIGNATURES

# Functions the evaluator treats as the same operation under two names. Mapping
# them to one label is what lets `RANK($close)` and `CS_RANK($close)` register
# as the same structure, which they are: `_call` dispatches both to the same
# branch of ExpressionEvaluator.
FUNCTION_ALIASES: dict[str, str] = {
    "CS_RANK": "RANK",
    "CS_ZSCORE": "ZSCORE",
    "DELTA": "TS_DELTA",
    "TS_COVARIANCE": "TS_COV",
    "IF": "IF_ELSE",
    "LN": "LOG",
}

# Commutative in every argument.
FULLY_COMMUTATIVE = {"MIN", "MAX"}
# Commutative in the first two arguments only; the trailing window is positional.
PAIRWISE_COMMUTATIVE = {"TS_CORR", "TS_COV"}

# Numeric literals that are boilerplate rather than a tuned parameter. Every
# division in this repo carries the same epsilon guard by convention (see the
# prompt and the Alpha101 transcription header), so counting it as a free
# constant would penalise an expression for following the house style and would
# put every guarded factor over a "free constants <= 4" gate for no reason.
BOILERPLATE_CONSTANTS = {1e-8}


@dataclass(frozen=True)
class Node:
    """One canonical tree node. `label` is already alias- and order-normalised."""

    label: str
    children: tuple["Node", ...] = ()

    def __iter__(self) -> Iterator["Node"]:
        """Pre-order walk over this node and its descendants."""
        yield self
        for child in self.children:
            yield from child


class ParseError(ValueError):
    """Raised when an expression cannot be turned into a canonical tree."""


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _normalize_number(value: float | int) -> str:
    f = float(value)
    if f.is_integer() and abs(f) < 1e15:
        return f"#{int(f)}"
    return f"#{f!r}"


def _canonical_function(name: str) -> str:
    upper = name.upper()
    return FUNCTION_ALIASES.get(upper, upper)


def _sort_key(node: Node) -> str:
    return canonical_text(node)


def _from_ast(node: ast.AST, sort_commutative: bool = True) -> Node:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ParseError(f"only numeric literals allowed, got {node.value!r}")
        return Node(_normalize_number(node.value))

    if isinstance(node, ast.Name):
        if not node.id.startswith("field_"):
            raise ParseError(f"unknown identifier: {node.id}")
        return Node(f"${node.id[len('field_'):].lower()}")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return Node("neg", (_from_ast(node.operand, sort_commutative),))

    if isinstance(node, ast.BinOp):
        ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
        label = ops.get(type(node.op))
        if label is None:
            raise ParseError(f"operator {type(node.op).__name__} not allowed")
        children = (_from_ast(node.left, sort_commutative),
                    _from_ast(node.right, sort_commutative))
        if sort_commutative and label in ("+", "*"):
            children = tuple(sorted(children, key=_sort_key))
        return Node(label, children)

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ParseError("chained comparisons not allowed")
        ops = {ast.Gt: ">", ast.Lt: "<", ast.GtE: ">=", ast.LtE: "<="}
        label = ops.get(type(node.ops[0]))
        if label is None:
            raise ParseError(f"comparison {type(node.ops[0]).__name__} not allowed")
        return Node(label, (_from_ast(node.left, sort_commutative),
                            _from_ast(node.comparators[0], sort_commutative)))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ParseError("only plain function calls allowed")
        name = _canonical_function(node.func.id)
        children = tuple(_from_ast(a, sort_commutative) for a in node.args)
        if sort_commutative and name in FULLY_COMMUTATIVE:
            children = tuple(sorted(children, key=_sort_key))
        elif sort_commutative and name in PAIRWISE_COMMUTATIVE and len(children) == 3:
            head = tuple(sorted(children[:2], key=_sort_key))
            children = head + children[2:]
        return Node(name, children)

    raise ParseError(f"syntax not allowed: {type(node).__name__}")


def parse(expression: str, *, canonical: bool = True) -> Node:
    """Tree for one expression string.

    Parsed through the same `$x -> field_x` rewrite and `ast.parse` the
    evaluator and the sanitizer's arity check use, so all three agree about
    what an expression is before any of them disagree about whether it is good.

    `canonical=False` skips ONLY the commutative-operand sort, keeping the
    written order. That matters for one measurement: how far a mutation moved.
    Zhang-Shasha is an ORDERED tree distance, so a canonical sort can make a
    genuine single-node edit look like several -- replacing `TS_STD(...)` with
    `$volume` under a `+` re-sorts the operands and measures 4 edits, not 3.
    Paraphrase detection wants the sort (`$high + $low` and `$low + $high` are
    one idea); the MUTATE locality check does not.
    """
    text = re.sub(r"\s+", " ", str(expression).strip())
    transformed = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"field_\1", text)
    try:
        tree = ast.parse(transformed, mode="eval")
    except SyntaxError as exc:
        raise ParseError(f"cannot parse expression: {exc.msg}") from exc
    return _from_ast(tree.body, canonical)


# --------------------------------------------------------------------------
# shape statistics
# --------------------------------------------------------------------------


def canonical_text(node: Node) -> str:
    """Deterministic serialisation. Equal text <=> equal canonical tree."""
    if not node.children:
        return node.label
    inner = ",".join(canonical_text(c) for c in node.children)
    return f"{node.label}({inner})"


def subtree_hash(node: Node) -> str:
    """Stable content hash of a subtree.

    blake2b rather than `hash()`: Python salts string hashing per process, and a
    structure statistic that changes between runs is not a statistic. 16 hex
    characters is 64 bits, which is far past collision risk for the few thousand
    subtrees any run of this repo enumerates.
    """
    return hashlib.blake2b(canonical_text(node).encode("utf-8"), digest_size=8).hexdigest()


def size(node: Node) -> int:
    """Number of nodes."""
    return 1 + sum(size(c) for c in node.children)


def depth(node: Node) -> int:
    """Edges on the longest root-to-leaf path; a bare field has depth 0."""
    return 0 if not node.children else 1 + max(depth(c) for c in node.children)


def subtree_sizes(node: Node) -> dict[str, int]:
    """{subtree hash: node count} over every complete subtree, root included."""
    out: dict[str, int] = {}
    for sub in node:
        out[subtree_hash(sub)] = size(sub)
    return out


def largest_shared_subtree(a: Node, b: Node) -> int:
    """Node count of the largest complete subtree the two trees have in common.

    "Complete" means a node together with all of its descendants, which is the
    unit a paraphrase actually reuses: an LLM restating alpha012 keeps
    `TS_DELTA($volume, 1)` whole, it does not keep a scattered handful of
    unrelated nodes. Returns 0 when nothing is shared, which cannot happen in
    practice because both trees draw leaves from the same eight fields -- so a
    result of 1 means "shares a field name and nothing else".
    """
    left = subtree_sizes(a)
    right = subtree_sizes(b)
    common = set(left) & set(right)
    return max((left[h] for h in common), default=0)


def largest_shared_subtree_expr(a: str, b: str) -> int:
    return largest_shared_subtree(parse(a), parse(b))


def largest_shared_from_maps(left: dict[str, int], right: dict[str, int]) -> int:
    """`largest_shared_subtree` when both subtree maps are already computed.

    The gates compare each candidate against 50 published formulas and up to 144
    archive members, so enumerating and hashing the same reference trees on
    every comparison is the dominant cost of the cheap half of the gate stack.
    Callers cache the reference maps once and pass them here; a test asserts
    this returns exactly what the two-tree version does.
    """
    if len(left) > len(right):
        left, right = right, left
    best = 0
    for key, size in left.items():
        if size > best and key in right:
            best = size
    return best


# --------------------------------------------------------------------------
# Zhang-Shasha tree edit distance
# --------------------------------------------------------------------------


@lru_cache(maxsize=8192)
def _postorder(root: Node) -> tuple[list[Node], list[int]]:
    """Post-order node list plus, per node, the post-order index of its leftmost leaf.

    Memoised on the node itself. The structural report compares every candidate
    against all 50 Alpha101 formulas, so each Alpha101 tree would otherwise be
    decomposed 400 times. `Node` is a frozen dataclass over a tuple of `Node`,
    so it hashes by content and the cache keys on structure, not identity.
    """
    nodes: list[Node] = []
    leftmost: list[int] = []

    def walk(node: Node) -> tuple[int, int]:
        # returns (own post-order index, post-order index of own leftmost leaf).
        # Returning the child's own index instead of the child's leftmost leaf
        # is the classic way to get this wrong: it makes every internal node
        # look like its own keyroot, which silently understates the distance
        # (`$close` against `RANK(TS_MEAN($close, 21))` came back 2, not 3).
        first_child_leftmost = None
        for child in node.children:
            _, child_leftmost = walk(child)
            if first_child_leftmost is None:
                first_child_leftmost = child_leftmost
        nodes.append(node)
        here = len(nodes) - 1
        own_leftmost = here if first_child_leftmost is None else first_child_leftmost
        leftmost.append(own_leftmost)
        return here, own_leftmost

    walk(root)
    return nodes, leftmost


def _keyroots(leftmost: list[int]) -> list[int]:
    """The highest node for each distinct leftmost-leaf, in increasing order.

    Zhang-Shasha only needs a full forest-distance table anchored at these,
    which is what turns the naive O(n^4)-ish recursion into the published bound.
    """
    seen: dict[int, int] = {}
    for i, l in enumerate(leftmost):
        seen[l] = i  # later index wins: the highest node with this leftmost leaf
    return sorted(seen.values())


def tree_edit_distance(a: Node, b: Node) -> int:
    """Zhang-Shasha (1989) edit distance with unit insert, delete and relabel costs.

    Implemented here rather than pulled from a package: the algorithm is 40
    lines, and this repo's dependency list is short on purpose.
    """
    nodes_a, l_a = _postorder(a)
    nodes_b, l_b = _postorder(b)
    n, m = len(nodes_a), len(nodes_b)
    tree_dist = [[0] * m for _ in range(n)]

    for i in _keyroots(l_a):
        for j in _keyroots(l_b):
            # forest distance over the sub-forests [l_a[i]..i] and [l_b[j]..j],
            # offset by one so index 0 means "empty forest"
            oi, oj = l_a[i], l_b[j]
            rows, cols = i - oi + 2, j - oj + 2
            fd = [[0] * cols for _ in range(rows)]
            for x in range(1, rows):
                fd[x][0] = fd[x - 1][0] + 1          # delete
            for y in range(1, cols):
                fd[0][y] = fd[0][y - 1] + 1          # insert
            for x in range(1, rows):
                for y in range(1, cols):
                    ni, nj = oi + x - 1, oj + y - 1
                    if l_a[ni] == oi and l_b[nj] == oj:
                        relabel = 0 if nodes_a[ni].label == nodes_b[nj].label else 1
                        fd[x][y] = min(
                            fd[x - 1][y] + 1,
                            fd[x][y - 1] + 1,
                            fd[x - 1][y - 1] + relabel,
                        )
                        tree_dist[ni][nj] = fd[x][y]
                    else:
                        px, py = l_a[ni] - oi, l_b[nj] - oj
                        fd[x][y] = min(
                            fd[x - 1][y] + 1,
                            fd[x][y - 1] + 1,
                            fd[px][py] + tree_dist[ni][nj],
                        )
    return tree_dist[n - 1][m - 1]


def normalized_edit_distance(a: Node, b: Node) -> float:
    """Edit distance scaled by the two trees' combined size, in [0, 1].

    Dividing by |a| + |b| rather than by max(|a|, |b|) keeps the measure bounded
    by 1: deleting one tree entirely and inserting the other costs exactly
    |a| + |b|, which is the worst any edit script can do.
    """
    total = size(a) + size(b)
    return 0.0 if total == 0 else tree_edit_distance(a, b) / total


def tree_edit_distance_expr(a: str, b: str, *, canonical: bool = True) -> int:
    return tree_edit_distance(parse(a, canonical=canonical), parse(b, canonical=canonical))


def edit_distance_from_parent(child: str, parent: str) -> int:
    """How far a mutation moved, in edits, ignoring commutative reordering.

    This is the number the MUTATE operator is gated on, so it uses the
    written-order parse: a localized edit must not be rejected for the
    bookkeeping reason that its parent happened to sit under a `+`.
    """
    return tree_edit_distance_expr(child, parent, canonical=False)


# --------------------------------------------------------------------------
# complexity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Complexity:
    """The four numbers the evolutionary loop's complexity gate reads."""

    symbol_length: int
    base_fields: int
    free_constants: int
    depth: int

    def to_dict(self) -> dict[str, int]:
        return {
            "symbol_length": self.symbol_length,
            "base_fields": self.base_fields,
            "free_constants": self.free_constants,
            "depth": self.depth,
        }


def _window_slot_positions(label: str) -> set[int]:
    """Argument positions of `label` that hold a lookback window, not a parameter.

    Read off `random_expressions.SIGNATURES`, which the sampler already keeps in
    lockstep with the sanitizer's arity tables, so a function added there
    without a signature fails loudly in one place instead of silently
    miscounting constants here.
    """
    for name in (label, *(k for k, v in FUNCTION_ALIASES.items() if v == label)):
        sig = SIGNATURES.get(name)
        if sig is not None:
            return {i for i, kind in enumerate(sig) if kind == "W"}
    return set()


def _count_free_constants(node: Node) -> int:
    """Numeric literals that are a tuned choice rather than structure.

    Excluded: lookback windows (they are the operator's argument, and an
    expression is not more complex for saying 21 instead of 20), and the
    epsilon guard every division in this repo carries.
    """
    total = 0
    window_positions = _window_slot_positions(node.label)
    for i, child in enumerate(node.children):
        if child.label.startswith("#") and not child.children:
            if i in window_positions:
                continue
            value = float(child.label[1:])
            if value in BOILERPLATE_CONSTANTS:
                continue
            total += 1
        else:
            total += _count_free_constants(child)
    if not node.children and node.label.startswith("#"):
        # a bare literal at the root; cannot be a window because it has no parent
        value = float(node.label[1:])
        if value not in BOILERPLATE_CONSTANTS:
            total += 1
    return total


def complexity(expression: str) -> Complexity:
    """Symbol length, distinct base fields, free constants and depth.

    `symbol_length` is measured on the whitespace-normalised ORIGINAL text, the
    same normalisation `ExpressionSanitizer` applies, so the number the gate
    checks is the number a reader counts in the candidate file. Everything else
    is measured on the canonical tree.
    """
    node = parse(expression)
    fields = {n.label for n in node if n.label.startswith("$")}
    return Complexity(
        symbol_length=len(re.sub(r"\s+", " ", str(expression).strip())),
        base_fields=len(fields),
        free_constants=_count_free_constants(node),
        depth=depth(node),
    )


def base_fields(expression: str) -> set[str]:
    """The distinct `$field` names an expression reads, without the `$`."""
    return {n.label[1:] for n in parse(expression) if n.label.startswith("$")}


# The fundamental fields, as `build_field_panels` attaches them. Used to sort a
# factor into the archive's data-family niche.
FUNDAMENTAL_FIELDS = frozenset({
    "roa", "roe", "operating_margin", "leverage", "asset_turnover",
    "book_per_share", "earnings_per_share", "accrual_gap",
})
PRICE_VOLUME_FIELDS = frozenset(ExpressionSanitizer.KNOWN_FIELDS) - FUNDAMENTAL_FIELDS


def data_family(expression: str) -> str:
    """"price_volume", "fundamental" or "mixed" -- the archive's second niche axis."""
    fields = base_fields(expression)
    has_fund = bool(fields & FUNDAMENTAL_FIELDS)
    has_px = bool(fields & PRICE_VOLUME_FIELDS)
    if has_fund and has_px:
        return "mixed"
    if has_fund:
        return "fundamental"
    return "price_volume"


@lru_cache(maxsize=4096)
def parse_cached(expression: str) -> Node:
    """`parse` with memoisation, for the gate loops that re-parse the archive."""
    return parse(expression)


# --------------------------------------------------------------------------
# tree surgery: rendering, addressing and typed slots
#
# The evolutionary loop's MUTATE, CROSSOVER and SIMPLIFY operators all do the
# same three things -- find a position, put something else there, and render the
# result back to an expression string the sanitizer accepts. Doing that on the
# canonical tree rather than on text is what keeps a "single-node edit" actually
# single-node, and what lets the loop MEASURE the edit it just made.
# --------------------------------------------------------------------------

# Binary labels rendered infix. Everything else renders as a call.
_INFIX = {"+", "-", "*", "/", "**", ">", "<", ">=", "<="}


def to_expression(node: Node) -> str:
    """Render a canonical tree back to an expression string.

    Every infix operation is fully parenthesised. That is uglier than
    precedence-aware output and it is the right trade: a rendering bug that
    silently reassociates `a - b * c` changes the factor, and nothing
    downstream would notice.
    """
    if not node.children:
        if node.label.startswith("#"):
            return node.label[1:]
        return node.label  # a $field
    if node.label == "neg":
        return f"-({to_expression(node.children[0])})"
    if node.label in _INFIX:
        left, right = (to_expression(c) for c in node.children)
        return f"({left} {node.label} {right})"
    inner = ", ".join(to_expression(c) for c in node.children)
    return f"{node.label}({inner})"


def paths(node: Node, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    """Every node address, as a tuple of child indices from the root."""
    out = [prefix]
    for i, child in enumerate(node.children):
        out.extend(paths(child, prefix + (i,)))
    return out


def at_path(node: Node, path: tuple[int, ...]) -> Node:
    for i in path:
        node = node.children[i]
    return node


def replace_at(node: Node, path: tuple[int, ...], replacement: Node) -> Node:
    """A copy of `node` with the subtree at `path` swapped for `replacement`.

    Note this returns the raw substituted tree, NOT a re-canonicalised one: a
    commutative parent whose children were sorted before the swap may now be out
    of order. Callers that care re-parse `to_expression(...)`, which is what the
    operators do -- round-tripping through the parser is the only way to be sure
    the result is both canonical and something the sanitizer accepts.
    """
    if not path:
        return replacement
    i, rest = path[0], path[1:]
    children = list(node.children)
    children[i] = replace_at(children[i], rest, replacement)
    return Node(node.label, tuple(children))


# Slot kinds, from `random_expressions.SIGNATURES`:
#   P panel, C condition, W integer window literal, S numeric scalar
PANEL_SLOT = "P"
CONDITION_SLOT = "C"
WINDOW_SLOT = "W"
SCALAR_SLOT = "S"


def slot_kinds(node: Node) -> dict[tuple[int, ...], str]:
    """The type each position in the tree must keep, addressed by path.

    A mutation that puts a panel where a window literal belongs produces an
    expression that passes the sanitizer -- which does not know that `TS_MEAN`
    needs an integer literal -- and then dies inside the evaluator. Same trap
    `random_expressions` documents for the null sampler. Typing the slots up
    front is how the operators avoid generating candidates that can only fail.
    """
    out: dict[tuple[int, ...], str] = {(): PANEL_SLOT}

    def walk(current: Node, prefix: tuple[int, ...]) -> None:
        if current.label in _INFIX or current.label == "neg":
            # arithmetic and comparison operands: a panel is always legal there,
            # and so is a scalar, so the permissive label is the correct one
            kinds = [PANEL_SLOT] * len(current.children)
        else:
            sig = None
            for name in (current.label,
                         *(k for k, v in FUNCTION_ALIASES.items() if v == current.label)):
                sig = SIGNATURES.get(name)
                if sig is not None:
                    break
            if sig is None:
                kinds = [PANEL_SLOT] * len(current.children)
            elif current.label in FULLY_COMMUTATIVE:
                kinds = [sig[0]] * len(current.children)  # variadic, all one kind
            else:
                kinds = [sig[i] if i < len(sig) else PANEL_SLOT
                         for i in range(len(current.children))]
        for i, child in enumerate(current.children):
            path = prefix + (i,)
            out[path] = kinds[i]
            walk(child, path)

    walk(node, ())
    return out


def panel_paths(node: Node) -> list[tuple[int, ...]]:
    """Positions holding a panel-valued sub-expression, root included.

    These are the only positions a subtree may be lifted from or dropped into
    without changing what the expression means type-wise, so they are the
    address space CROSSOVER and SIMPLIFY work in.
    """
    kinds = slot_kinds(node)
    return [p for p, kind in kinds.items() if kind == PANEL_SLOT]
