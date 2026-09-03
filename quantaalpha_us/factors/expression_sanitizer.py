from __future__ import annotations

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
