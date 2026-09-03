from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class SanitizeResult:
    valid: bool
    cleaned: str
    errors: list[str] = field(default_factory=list)


class ExpressionSanitizer:
    """Pre-validates LLM-generated expressions before parser evaluation."""

    DEFAULT_ALLOWED_FUNCTIONS = {
        # Time-series
        "TS_MEAN",
        "TS_STD",
        "TS_MAX",
        "TS_MIN",
        "TS_RANK",
        "TS_DELTA",
        "TS_CORR",
        "TS_COV",
        "TS_COVARIANCE",
        "TS_SUM",
        "TS_PRODUCT",
        "TS_ARGMAX",
        "TS_ARGMIN",
        "TS_DECAY_LINEAR",
        "EMA",
        "DELAY",
        "DELTA",
        # Cross-sectional
        "RANK",
        "ZSCORE",
        "CS_RANK",
        "CS_ZSCORE",
        "CS_DEMEAN",
        # Logic/math
        "IF",
        "IF_ELSE",
        "MIN",
        "MAX",
        "ABS",
        "SIGN",
        "LOG",
        "LN",
        "SQRT",
        "POWER",
        "BOUND",
        "COUNT",
        # lowercase aliases
        "rank",
        "zscore",
        "ts_mean",
        "ts_std",
        "ts_max",
        "ts_min",
        "ts_rank",
        "ts_delta",
        "ts_corr",
        "ts_cov",
        "ts_sum",
        "ts_product",
        "ts_argmax",
        "ts_argmin",
        "ts_decay_linear",
        "if_else",
        "log",
        "abs",
        "sign",
        "power",
        "sqrt",
        "min",
        "max",
        "count",
        "bound",
    }

    # The evaluator resolves fields only through a leading "$" (see
    # expression_evaluator's module docstring). A bare `close` is not a field to
    # it, it is an unknown identifier, so an expression using bare names parses
    # here and then dies at evaluation time. These names exist to make the
    # resulting error say which field was meant.
    KNOWN_FIELDS = {
        "open", "high", "low", "close", "adj_close", "volume",
        "dollar_volume", "return",
        # fundamentals, per build_field_panels
        "roa", "roe", "operating_margin", "leverage", "asset_turnover",
        "book_per_share", "earnings_per_share", "accrual_gap",
    }

    # Argument counts the evaluator enforces. Kept here rather than imported
    # from expression_evaluator so this gate stays dependency-free (that module
    # pulls in pandas/numpy); test_arity_table_matches_the_evaluator asserts the
    # two never drift apart.
    # MIN/MAX are variadic (elementwise across N operands), so they carry a
    # MINIMUM arity rather than a fixed one. Without this they fell outside the
    # arity table entirely: MIN() passed the gate and then raised a bare
    # ValueError("min() iterable argument is empty") from inside the evaluator.
    VARIADIC_MIN_ARITY = {"MIN": 2, "MAX": 2}

    FUNCTION_ARITY = {
        "ABS": 1, "BOUND": 3, "COUNT": 2, "CS_DEMEAN": 1, "CS_RANK": 1,
        "CS_ZSCORE": 1, "DELAY": 2, "DELTA": 2, "EMA": 2, "IF": 3, "IF_ELSE": 3,
        "LN": 1, "LOG": 1, "POWER": 2, "RANK": 1, "SIGN": 1, "SQRT": 1,
        "TS_ARGMAX": 2, "TS_ARGMIN": 2, "TS_CORR": 3, "TS_COV": 3,
        "TS_COVARIANCE": 3, "TS_DECAY_LINEAR": 2, "TS_DELTA": 2, "TS_MAX": 2,
        "TS_MEAN": 2, "TS_MIN": 2, "TS_PRODUCT": 2, "TS_RANK": 2, "TS_STD": 2,
        "TS_SUM": 2, "ZSCORE": 1,
    }

    BLOCKED_TOKENS = (
        "import ",
        "exec(",
        "eval(",
        "open(",
        "__",
        "os.",
        "sys.",
        "subprocess",
        "lambda ",
    )

    def __init__(
        self,
        *,
        allowed_functions: set[str] | None = None,
        max_expression_length: int = 500,
        max_nesting_depth: int = 10,
    ) -> None:
        self.allowed_functions = allowed_functions or set(self.DEFAULT_ALLOWED_FUNCTIONS)
        self.max_expression_length = int(max_expression_length)
        self.max_nesting_depth = int(max_nesting_depth)

    @staticmethod
    def _normalize(expression: str) -> str:
        return re.sub(r"\s+", " ", expression.strip())

    @staticmethod
    def _max_nesting(expression: str) -> tuple[bool, int]:
        depth = 0
        max_depth = 0
        for ch in expression:
            if ch == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False, max_depth
        return depth == 0, max_depth

    def sanitize(self, expression: str) -> SanitizeResult:
        cleaned = self._normalize(str(expression))
        errors: list[str] = []

        if not cleaned:
            errors.append("Expression is empty")
            return SanitizeResult(valid=False, cleaned=cleaned, errors=errors)

        if len(cleaned) > self.max_expression_length:
            errors.append(
                f"Expression too long: {len(cleaned)} > {self.max_expression_length}"
            )

        lowered = cleaned.lower()
        for token in self.BLOCKED_TOKENS:
            if token.lower() in lowered:
                errors.append(f"Blocked token found: {token}")

        balanced, max_depth = self._max_nesting(cleaned)
        if not balanced:
            errors.append("Unbalanced parentheses")
        if max_depth > self.max_nesting_depth:
            errors.append(
                f"Nesting depth too high: {max_depth} > {self.max_nesting_depth}"
            )

        func_calls = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", cleaned)
        for func in func_calls:
            if func in {"if", "for", "while"}:
                errors.append(f"Blocked language keyword used as function: {func}")
                continue
            if func not in self.allowed_functions and not func.startswith("field_"):
                errors.append(f"Unknown function: {func}")

        errors.extend(self._identifier_errors(cleaned))
        errors.extend(self._arity_errors(cleaned))

        return SanitizeResult(valid=len(errors) == 0, cleaned=cleaned, errors=errors)

    def _identifier_errors(self, cleaned: str) -> list[str]:
        """Reject bare field references, which the evaluator cannot resolve.

        Checking function names alone let `TS_MEAN(close, 10)` through as valid;
        the evaluator then raised "Unknown identifier: close" only once the
        expression reached scoring. A gate that accepts what the next stage
        rejects is worse than no gate, because the failure surfaces far from its
        cause -- an LLM asked for factors without a "$" convention in the prompt
        produced 26 of 26 expressions that passed here and evaluated none.
        """
        errors: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", cleaned):
            name = match.group(0)
            rest = cleaned[match.end():]
            if rest.lstrip().startswith("("):
                continue  # a function call, already checked above
            prev = cleaned[match.start() - 1] if match.start() > 0 else ""
            if prev.isdigit() or prev == ".":
                # the "e" of a float literal such as 1e-8, not an identifier
                continue
            if match.start() > 0 and cleaned[match.start() - 1] == "$":
                if name not in self.KNOWN_FIELDS and name not in seen:
                    seen.add(name)
                    errors.append(f"Unknown field: ${name}")
                continue
            if name in seen:
                continue
            seen.add(name)
            if name in self.KNOWN_FIELDS:
                errors.append(
                    f"Field '{name}' must be written as '${name}'; "
                    "the evaluator resolves fields only through a leading '$'"
                )
            else:
                errors.append(f"Unknown identifier: {name}")
        return errors

    def _arity_errors(self, cleaned: str) -> list[str]:
        """Reject calls with the wrong number of arguments.

        Parsed exactly the way the evaluator parses -- same "$x" -> "field_x"
        rewrite, same ast.parse -- so the two cannot disagree about what the
        expression means. Without this, ZSCORE($return, 10) passed the gate and
        died during scoring with "ZSCORE expects 1 argument(s), got 2", which
        wastes a full evaluation pass over the panel to learn something
        knowable from the text alone.
        """
        transformed = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"field_\1", cleaned)
        try:
            tree = ast.parse(transformed, mode="eval")
        except SyntaxError as exc:
            return [f"Cannot parse expression: {exc.msg}"]

        errors: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id.upper()
            minimum = self.VARIADIC_MIN_ARITY.get(name)
            if minimum is not None:
                if len(node.args) < minimum:
                    errors.append(
                        f"{name} needs at least {minimum} argument(s), got {len(node.args)}"
                    )
                continue
            expected = self.FUNCTION_ARITY.get(name)
            if expected is not None and len(node.args) != expected:
                errors.append(
                    f"{name} expects {expected} argument(s), got {len(node.args)}"
                )
        return errors
