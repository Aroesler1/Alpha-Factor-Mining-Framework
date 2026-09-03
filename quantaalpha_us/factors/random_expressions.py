"""Random-grammar baseline: sample valid expressions from the sanitizer's own grammar.

Why this exists
---------------
Every candidate set in this repo is LLM-authored, and the models involved were
trained on data covering the holdout window. A set of factors that scores well
therefore proves nothing on its own: the question is not "do these factors
work?" but "do they work better than expressions drawn at random from the same
grammar, given the same search budget?". Without that comparison a mining run
cannot distinguish a real signal from the best of N draws.

This module is the null. It samples expressions from exactly the grammar
`ExpressionSanitizer` accepts -- the function set, arities and field names are
read out of `FUNCTION_ARITY`, `VARIADIC_MIN_ARITY` and `KNOWN_FIELDS` rather
than restated here, so the null cannot silently drift away from the thing it is
a null for. Samples are matched to a reference candidate set's structural
distribution (call count and nesting depth), so the comparison is against a
search of the same size and shape, not a strawman of trivially short
expressions.

Sampling is typed. The sanitizer checks syntax, identifiers and arity, but it
does not know that `TS_MEAN` needs an integer literal window or that `TS_CORR`
needs two panel operands -- those are evaluator-level constraints. An untyped
sampler passes `sanitize()` and then dies in `evaluate()`, which would quietly
bias the null: the malformed draws vanish from the report as errors and the
survivors are no longer a random sample. So the generator tracks whether each
position wants a panel, a scalar, a positive integer window, or a condition,
and only ever emits something evaluable there.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer

# Fields the price-only candidate sets use. Kept as a subset of the sanitizer's
# KNOWN_FIELDS (asserted below) so the null draws from the same alphabet the
# LLM was given, minus the fundamentals that only the fundamental set may use.
PRICE_FIELDS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "dollar_volume", "return",
)

# Windows the LLM sets actually use. Drawing from the same ladder keeps the
# comparison about expression structure rather than about one side getting
# luckier lookback lengths.
DEFAULT_WINDOWS: tuple[int, ...] = (3, 5, 10, 21, 63, 252)

# How each function consumes its arguments. Signatures are evaluator-level
# facts the arity table cannot express; `_check_signature_coverage` asserts this
# map stays exactly in step with the sanitizer's tables, so adding a function
# there without describing it here fails loudly instead of silently narrowing
# the null.
#   P = panel, W = positive integer window literal, S = numeric scalar, C = condition
SIGNATURES: dict[str, str] = {
    "ABS": "P", "CS_DEMEAN": "P", "CS_RANK": "P", "CS_ZSCORE": "P",
    "LN": "P", "LOG": "P", "RANK": "P", "SIGN": "P", "SQRT": "P", "ZSCORE": "P",
    "DELAY": "PW", "DELTA": "PW", "EMA": "PW", "TS_ARGMAX": "PW",
    "TS_ARGMIN": "PW", "TS_DECAY_LINEAR": "PW", "TS_DELTA": "PW",
    "TS_MAX": "PW", "TS_MEAN": "PW", "TS_MIN": "PW", "TS_PRODUCT": "PW",
    "TS_RANK": "PW", "TS_STD": "PW", "TS_SUM": "PW",
    "TS_CORR": "PPW", "TS_COV": "PPW", "TS_COVARIANCE": "PPW",
    "POWER": "PS",
    "BOUND": "PSS",
    "COUNT": "CW",
    "IF": "CPP", "IF_ELSE": "CPP",
    # variadic; sampled at their minimum arity
    "MIN": "PP", "MAX": "PP",
}

_EPS = "1e-8"


def _check_signature_coverage() -> None:
    """Fail loudly if SIGNATURES and the sanitizer's tables disagree."""
    table = set(ExpressionSanitizer.FUNCTION_ARITY) | set(ExpressionSanitizer.VARIADIC_MIN_ARITY)
    described = set(SIGNATURES)
    missing = table - described
    extra = described - table
    if missing or extra:
        raise RuntimeError(
            "random_expressions.SIGNATURES is out of step with ExpressionSanitizer: "
            f"missing {sorted(missing)}, unknown {sorted(extra)}"
        )
    for name, sig in SIGNATURES.items():
        if name in ExpressionSanitizer.VARIADIC_MIN_ARITY:
            expected = ExpressionSanitizer.VARIADIC_MIN_ARITY[name]
            if len(sig) < expected:
                raise RuntimeError(f"{name} signature {sig!r} is below its minimum arity {expected}")
            continue
        expected = ExpressionSanitizer.FUNCTION_ARITY[name]
        if len(sig) != expected:
            raise RuntimeError(f"{name} signature {sig!r} does not match arity {expected}")


_check_signature_coverage()


@dataclass(frozen=True)
class StructureProfile:
    """Structural fingerprint of a candidate set, used as the sampling target."""

    call_counts: tuple[int, ...]
    max_depth: int

    @property
    def mean_calls(self) -> float:
        return sum(self.call_counts) / len(self.call_counts)


def _ast_of(expression: str) -> ast.Expression:
    return ast.parse(re.sub(r"\$([A-Za-z_]\w*)", r"field_\1", expression), mode="eval")


def _depth(node: ast.AST, d: int = 0) -> int:
    children = list(ast.iter_child_nodes(node))
    return d if not children else max(_depth(c, d + 1) for c in children)


def expression_structure(expression: str) -> tuple[int, int]:
    """(number of function calls, AST depth) for one expression."""
    tree = _ast_of(expression)
    calls = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call))
    return calls, _depth(tree.body)


def structure_profile(expressions: Iterable[str]) -> StructureProfile:
    """Fingerprint a reference set so the null can be drawn to match it.

    Matching matters because "best |t| over N candidates" is a function of both
    N and how expressive each candidate is. A null of bare `$close` terms would
    lose to any LLM set for reasons that have nothing to do with the LLM.
    """
    counts, depths = [], []
    for expr in expressions:
        calls, depth = expression_structure(expr)
        counts.append(calls)
        depths.append(depth)
    if not counts:
        raise ValueError("Cannot profile an empty candidate set")
    return StructureProfile(call_counts=tuple(counts), max_depth=max(depths))


class GrammarSampler:
    """Draw syntactically valid, evaluable expressions from the sanitizer's grammar."""

    def __init__(
        self,
        *,
        seed: int,
        profile: StructureProfile,
        fields: Sequence[str] = PRICE_FIELDS,
        windows: Sequence[int] = DEFAULT_WINDOWS,
        sanitizer: ExpressionSanitizer | None = None,
        binop_prob: float = 0.35,
    ) -> None:
        unknown = set(fields) - ExpressionSanitizer.KNOWN_FIELDS
        if unknown:
            raise ValueError(f"Fields not in the sanitizer's KNOWN_FIELDS: {sorted(unknown)}")
        self.rng = random.Random(seed)
        self.profile = profile
        self.fields = tuple(fields)
        self.windows = tuple(windows)
        self.sanitizer = sanitizer or ExpressionSanitizer()
        self.binop_prob = float(binop_prob)
        # Leave headroom under the sanitizer's own nesting cap: the emitted
        # string carries parentheses the AST does not, so an AST-depth budget
        # equal to the cap can still overflow it.
        self.max_depth = min(profile.max_depth, self.sanitizer.max_nesting_depth - 2)
        self._by_shape = self._index_signatures()

    def _index_signatures(self) -> dict[str, list[str]]:
        """Group function names by the argument shape they need."""
        out: dict[str, list[str]] = {}
        for name, sig in SIGNATURES.items():
            out.setdefault(sig, []).append(name)
        for names in out.values():
            names.sort()
        return out

    # ---- terminals -------------------------------------------------------

    def _field(self) -> str:
        return f"${self.rng.choice(self.fields)}"

    def _window(self) -> str:
        return str(self.rng.choice(self.windows))

    def _scalar(self) -> str:
        return self.rng.choice(("0.5", "1", "2", "1.5", "0.25", "3"))

    def _atom(self) -> str:
        """A zero-call panel term: a field, or a cheap arithmetic pair of fields."""
        if self.rng.random() < 0.45:
            a, b = self._field(), self._field()
            op = self.rng.choice(("-", "+", "*"))
            return f"({a} {op} {b})"
        return self._field()

    # ---- recursive generation -------------------------------------------

    def _panel(self, budget: int, depth: int) -> str:
        """Emit a panel-valued expression placing exactly `budget` function calls."""
        if budget <= 0 or depth >= self.max_depth:
            return self._atom()

        # spend the budget across an arithmetic combination rather than nesting
        if budget >= 2 and self.rng.random() < self.binop_prob:
            left_budget = self.rng.randint(1, budget - 1)
            left = self._panel(left_budget, depth + 1)
            right = self._panel(budget - left_budget, depth + 1)
            op = self.rng.choice(("-", "+", "*", "/"))
            if op == "/":
                # every division in the LLM sets is epsilon-guarded; without it a
                # near-zero denominator produces inf and the signal's rank is
                # decided by a rounding artefact
                return f"({left} / ({right} + {_EPS}))"
            return f"({left} {op} {right})"

        shapes = [s for s in self._by_shape if self._shape_fits(s, budget)]
        shape = self.rng.choice(shapes)
        name = self.rng.choice(self._by_shape[shape])
        return self._call(name, shape, budget - 1, depth + 1)

    def _shape_fits(self, shape: str, budget: int) -> bool:
        """A shape is usable when its panel/condition slots can absorb the budget."""
        slots = sum(1 for c in shape if c in "PC")
        return slots >= 1 and budget - 1 <= slots * self._max_calls_per_slot()

    def _max_calls_per_slot(self) -> int:
        return max(self.profile.call_counts) if self.profile.call_counts else 1

    def _call(self, name: str, shape: str, budget: int, depth: int) -> str:
        """Render one call, distributing `budget` remaining calls over its slots."""
        slot_positions = [i for i, c in enumerate(shape) if c in "PC"]
        allocation = self._split(budget, len(slot_positions))
        args: list[str] = []
        alloc_iter = iter(allocation)
        for kind in shape:
            if kind == "P":
                args.append(self._panel(next(alloc_iter), depth))
            elif kind == "C":
                args.append(self._condition(next(alloc_iter), depth))
            elif kind == "W":
                args.append(self._window())
            elif kind == "S":
                args.append(self._scalar())
            else:  # pragma: no cover - guarded by _check_signature_coverage
                raise RuntimeError(f"Unknown signature slot {kind!r} in {name}")
        if name == "BOUND":
            # BOUND(x, lo, hi) needs lo < hi; two independent scalar draws do not
            args[1], args[2] = "-3", "3"
        return f"{name}({', '.join(args)})"

    def _condition(self, budget: int, depth: int) -> str:
        """A comparison, which is what IF/IF_ELSE/COUNT need in their first slot."""
        left = self._panel(budget, depth)
        op = self.rng.choice((">", "<", ">=", "<="))
        if self.rng.random() < 0.5:
            return f"({left} {op} 0)"
        return f"({left} {op} {self._atom()})"

    def _split(self, budget: int, slots: int) -> list[int]:
        """Randomly partition `budget` calls across `slots` positions."""
        if slots <= 0:
            return []
        out = [0] * slots
        for _ in range(max(0, budget)):
            out[self.rng.randrange(slots)] += 1
        return out

    # ---- public API ------------------------------------------------------

    def sample(self) -> str:
        """One expression that passes the sanitizer, matched to the profile."""
        for _ in range(200):
            target = self.rng.choice(self.profile.call_counts)
            expr = self._panel(target, 0)
            if self.sanitizer.sanitize(expr).valid:
                return expr
        raise RuntimeError("Could not draw a sanitizing expression in 200 attempts")

    def sample_many(self, n: int) -> list[str]:
        """`n` distinct expressions, each passing the sanitizer.

        Distinctness is on the expression string. Two structurally different
        draws can still compute the same signal; that is the null's problem to
        have, and it is the same problem the LLM sets have.
        """
        out: list[str] = []
        seen: set[str] = set()
        attempts = 0
        while len(out) < n:
            attempts += 1
            if attempts > 400 * n:
                raise RuntimeError(f"Only drew {len(out)} distinct expressions of {n} requested")
            expr = self.sample()
            if expr in seen:
                continue
            seen.add(expr)
            out.append(expr)
        return out


def sample_random_set(
    *,
    seed: int,
    n: int,
    reference: Sequence[str],
    fields: Sequence[str] = PRICE_FIELDS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> list[str]:
    """Draw `n` expressions matched to `reference`'s structure. Seeded and pure."""
    sampler = GrammarSampler(
        seed=seed, profile=structure_profile(reference), fields=fields, windows=windows
    )
    return sampler.sample_many(n)
