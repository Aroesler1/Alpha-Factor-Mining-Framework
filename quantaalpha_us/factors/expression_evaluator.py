"""Panel evaluator for mined factor expressions.

Turns sanitized expression strings such as

    TS_MEAN($close, 21) / (TS_STD($close, 21) + 1e-8)
    RANK(TS_DELTA($close, 5)) - RANK($volume)

into date x symbol signal panels. This closes the loop the sanitizer only
started: `ExpressionSanitizer` gate-checks strings, this module actually
computes them, and `factor_research` scores them against forward returns.

Design:
- Expressions are parsed with Python's `ast` module after mapping `$name`
  fields to identifiers, then walked against a strict whitelist (arithmetic,
  comparisons, numeric literals, known functions, known fields). Anything
  else raises ExpressionError, so this is a second independent guard behind
  the sanitizer.
- Every node evaluates to a scalar or a pandas DataFrame (dates x symbols).
- Time-series operators use strict windows (min_periods = window) so early
  partially-filled windows produce NaN instead of biased values.

Field panels (see `build_field_panels`):
    $open $high $low $close $adj_close $volume $dollar_volume $return
`$close` maps to adjusted close when available (research convention);
`$return` is the daily percentage change of that panel.
"""

from __future__ import annotations

import ast
import re
import warnings
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


class ExpressionError(ValueError):
    """Raised when an expression cannot be parsed or evaluated safely."""


PanelOrScalar = Any  # pd.DataFrame | float


def build_field_panels(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot long-format daily bars into wide field panels.

    `bars` needs columns: date, symbol, and any of open/high/low/close/
    adj_close/volume/dollar_volume. Output keys are the `$field` names the
    evaluator understands.
    """
    work = bars.assign(
        date=pd.to_datetime(bars["date"], errors="coerce").dt.normalize(),
        symbol=bars["symbol"].astype(str).str.upper(),
    )
    work = work.dropna(subset=["date", "symbol"]).sort_values(["date", "symbol"])

    # Fundamental fields are optional: present only when the caller passed a bar
    # frame augmented by factors.fundamentals. They are already point-in-time
    # (stamped from the Compustat report date, not the period end), so they
    # pivot exactly like a price field with no further alignment.
    fundamental_fields = (
        "roa", "roe", "operating_margin", "leverage", "asset_turnover",
        "book_per_share", "earnings_per_share", "accrual_gap",
    )

    panels: dict[str, pd.DataFrame] = {}
    for col in ("open", "high", "low", "close", "adj_close", "volume",
                "dollar_volume") + fundamental_fields:
        if col in work.columns:
            panel = work.pivot_table(index="date", columns="symbol", values=col, aggfunc="last")
            # parquet commonly carries pandas nullable dtypes (Float64/Int64);
            # numpy ufuncs and np.where raise "boolean value of NA is ambiguous"
            # on those, so pin every panel to plain float64 with NaN missings
            panels[col] = panel.astype("float64")

    # research convention: $close means the adjusted close when available
    if "adj_close" in panels:
        panels["close"] = panels["adj_close"]
    if "close" in panels:
        # fill_method=None: never forward-fill across listing gaps before differencing
        panels["return"] = panels["close"].pct_change(fill_method=None)
    if "dollar_volume" not in panels and {"close", "volume"} <= panels.keys():
        panels["dollar_volume"] = panels["close"] * panels["volume"]
    return panels


def _require_window(value: PanelOrScalar, func: str) -> int:
    if isinstance(value, (int, float)) and float(value).is_integer() and value > 0:
        return int(value)
    raise ExpressionError(f"{func} window must be a positive integer literal, got {value!r}")


def _as_panel(value: PanelOrScalar, like: pd.DataFrame | None = None) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if like is not None:
        return pd.DataFrame(float(value), index=like.index, columns=like.columns)
    raise ExpressionError("Expected a panel operand")


def _rolling(x: PanelOrScalar, w: int) -> pd.core.window.rolling.Rolling:
    if not isinstance(x, pd.DataFrame):
        raise ExpressionError("Time-series operators need a panel operand")
    return x.rolling(w, min_periods=w)


def _ts_decay_linear(x: pd.DataFrame, w: int) -> pd.DataFrame:
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()
    return _rolling(x, w).apply(lambda a: float(np.dot(a, weights)), raw=True)


def _ts_argmax(x: pd.DataFrame, w: int) -> pd.DataFrame:
    # days since the window maximum (0 = today is the max)
    return _rolling(x, w).apply(lambda a: float(len(a) - 1 - np.argmax(a)), raw=True)


def _ts_argmin(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return _rolling(x, w).apply(lambda a: float(len(a) - 1 - np.argmin(a)), raw=True)


def _cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    mean = x.mean(axis=1)
    std = x.std(axis=1).replace(0, np.nan)
    return x.sub(mean, axis=0).div(std, axis=0)


def _safe_log(x: PanelOrScalar) -> PanelOrScalar:
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(np.where(x > 0, np.log(x.where(x > 0)), np.nan), index=x.index, columns=x.columns)
    return float(np.log(x)) if x > 0 else float("nan")


def _if_else(cond: PanelOrScalar, a: PanelOrScalar, b: PanelOrScalar) -> PanelOrScalar:
    if isinstance(cond, pd.DataFrame):
        a_p = _as_panel(a, cond)
        b_p = _as_panel(b, cond)
        return a_p.where(cond.astype(bool) & cond.notna(), b_p).where(cond.notna())
    return a if cond else b


def _pairwise(func: Callable[[pd.DataFrame, pd.DataFrame, int], pd.DataFrame]):
    def wrapper(x: PanelOrScalar, y: PanelOrScalar, w: PanelOrScalar, name: str) -> pd.DataFrame:
        window = _require_window(w, name)
        if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame):
            raise ExpressionError(f"{name} needs two panel operands")
        return func(x, y, window)

    return wrapper


_TS_CORR = _pairwise(lambda x, y, w: x.rolling(w, min_periods=w).corr(y))
_TS_COV = _pairwise(lambda x, y, w: x.rolling(w, min_periods=w).cov(y))


def _elementwise_minmax(op: str, *args: PanelOrScalar) -> PanelOrScalar:
    panels = [a for a in args if isinstance(a, pd.DataFrame)]
    if not panels:
        return min(args) if op == "min" else max(args)
    like = panels[0]
    frames = [_as_panel(a, like) for a in args]
    stacked = np.stack([f.to_numpy(dtype=float) for f in frames])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        out = np.nanmin(stacked, axis=0) if op == "min" else np.nanmax(stacked, axis=0)
    # preserve NaN where every operand is NaN
    all_nan = np.isnan(stacked).all(axis=0)
    out = np.where(all_nan, np.nan, out)
    return pd.DataFrame(out, index=like.index, columns=like.columns)


class ExpressionEvaluator:
    """Evaluate sanitized factor expressions over field panels."""

    def __init__(self, panels: Mapping[str, pd.DataFrame]) -> None:
        if not panels:
            raise ExpressionError("No field panels provided")
        self.panels = {str(k).lower(): v for k, v in panels.items()}

    # ---- public API ------------------------------------------------------

    def evaluate(self, expression: str) -> pd.DataFrame:
        """Evaluate one expression into a dates x symbols panel."""
        transformed = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"field_\1", expression)
        try:
            tree = ast.parse(transformed, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Cannot parse expression: {exc}") from exc
        result = self._eval(tree.body)
        if not isinstance(result, pd.DataFrame):
            raise ExpressionError("Expression reduced to a scalar, not a factor panel")
        return result

    # ---- AST walking -----------------------------------------------------

    def _eval(self, node: ast.AST) -> PanelOrScalar:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ExpressionError(f"Only numeric literals allowed, got {node.value!r}")

        if isinstance(node, ast.Name):
            return self._field(node.id)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval(node.operand)

        if isinstance(node, ast.BinOp):
            left, right = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ExpressionError(f"Operator {type(node.op).__name__} not allowed")

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ExpressionError("Chained comparisons not allowed")
            left, right = self._eval(node.left), self._eval(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Gt):
                return left > right
            if isinstance(op, ast.Lt):
                return left < right
            if isinstance(op, ast.GtE):
                return left >= right
            if isinstance(op, ast.LtE):
                return left <= right
            raise ExpressionError(f"Comparison {type(op).__name__} not allowed")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ExpressionError("Only plain function calls allowed")
            return self._call(node.func.id.upper(), [self._eval(arg) for arg in node.args])

        raise ExpressionError(f"Syntax not allowed: {type(node).__name__}")

    def _field(self, identifier: str) -> pd.DataFrame:
        if not identifier.startswith("field_"):
            raise ExpressionError(f"Unknown identifier: {identifier}")
        name = identifier[len("field_"):].lower()
        if name not in self.panels:
            raise ExpressionError(f"Unknown field: ${name}")
        return self.panels[name]

    # ---- functions -------------------------------------------------------

    def _call(self, name: str, args: list[PanelOrScalar]) -> PanelOrScalar:
        def arity(n: int) -> None:
            if len(args) != n:
                raise ExpressionError(f"{name} expects {n} argument(s), got {len(args)}")

        if name in ("TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN", "TS_SUM", "TS_RANK",
                    "TS_PRODUCT", "TS_ARGMAX", "TS_ARGMIN", "TS_DECAY_LINEAR",
                    "TS_DELTA", "DELTA", "DELAY", "EMA"):
            arity(2)
            x, w = args[0], _require_window(args[1], name)
            if name == "TS_MEAN":
                return _rolling(x, w).mean()
            if name == "TS_STD":
                return _rolling(x, w).std()
            if name == "TS_MAX":
                return _rolling(x, w).max()
            if name == "TS_MIN":
                return _rolling(x, w).min()
            if name == "TS_SUM":
                return _rolling(x, w).sum()
            if name == "TS_RANK":
                return _rolling(x, w).rank(pct=True)
            if name == "TS_PRODUCT":
                return _rolling(x, w).apply(np.prod, raw=True)
            if name == "TS_ARGMAX":
                return _ts_argmax(x, w)
            if name == "TS_ARGMIN":
                return _ts_argmin(x, w)
            if name == "TS_DECAY_LINEAR":
                return _ts_decay_linear(x, w)
            if name in ("TS_DELTA", "DELTA"):
                return x - x.shift(w)
            if name == "DELAY":
                return x.shift(w)
            if name == "EMA":
                return x.ewm(span=w, min_periods=w).mean()

        if name in ("TS_CORR",):
            arity(3)
            return _TS_CORR(args[0], args[1], args[2], name)
        if name in ("TS_COV", "TS_COVARIANCE"):
            arity(3)
            return _TS_COV(args[0], args[1], args[2], name)

        if name in ("RANK", "CS_RANK"):
            arity(1)
            return _as_panel(args[0]).rank(axis=1, pct=True)
        if name in ("ZSCORE", "CS_ZSCORE"):
            arity(1)
            return _cs_zscore(_as_panel(args[0]))
        if name == "CS_DEMEAN":
            arity(1)
            panel = _as_panel(args[0])
            return panel.sub(panel.mean(axis=1), axis=0)

        if name == "ABS":
            arity(1)
            return abs(args[0])
        if name == "SIGN":
            arity(1)
            x = args[0]
            return np.sign(x) if not isinstance(x, pd.DataFrame) else pd.DataFrame(
                np.sign(x.to_numpy(dtype=float)), index=x.index, columns=x.columns).where(x.notna())
        if name in ("LOG", "LN"):
            arity(1)
            return _safe_log(args[0])
        if name == "SQRT":
            arity(1)
            x = args[0]
            if isinstance(x, pd.DataFrame):
                return pd.DataFrame(
                    np.where(x >= 0, np.sqrt(x.where(x >= 0)), np.nan),
                    index=x.index, columns=x.columns)
            return float(np.sqrt(x)) if x >= 0 else float("nan")
        if name == "POWER":
            arity(2)
            return args[0] ** args[1]
        if name == "BOUND":
            arity(3)
            panel = _as_panel(args[0])
            return panel.clip(lower=float(args[1]), upper=float(args[2]))
        if name == "MIN":
            return _elementwise_minmax("min", *args)
        if name == "MAX":
            return _elementwise_minmax("max", *args)
        if name == "COUNT":
            arity(2)
            cond, w = args[0], _require_window(args[1], name)
            if not isinstance(cond, pd.DataFrame):
                raise ExpressionError("COUNT needs a panel condition")
            return cond.astype(float).rolling(w, min_periods=w).sum()
        if name in ("IF", "IF_ELSE"):
            arity(3)
            return _if_else(args[0], args[1], args[2])

        raise ExpressionError(f"Unknown function: {name}")
