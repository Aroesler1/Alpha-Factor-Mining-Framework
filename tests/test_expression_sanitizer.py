import pytest

from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer


def test_valid_expression_passes() -> None:
    sanitizer = ExpressionSanitizer()
    result = sanitizer.sanitize("TS_MEAN($close, 20) / (TS_STD($close, 20) + 1e-8)")
    assert result.valid
    assert result.errors == []


def test_blocked_token_rejected() -> None:
    sanitizer = ExpressionSanitizer()
    result = sanitizer.sanitize("import os; TS_MEAN($close, 20)")
    assert not result.valid
    assert any("Blocked token" in e for e in result.errors)


def test_unknown_function_rejected() -> None:
    sanitizer = ExpressionSanitizer()
    result = sanitizer.sanitize("HACK_FUNC($close, 10)")
    assert not result.valid
    assert any("Unknown function" in e for e in result.errors)


def test_length_rejected() -> None:
    sanitizer = ExpressionSanitizer(max_expression_length=10)
    result = sanitizer.sanitize("TS_MEAN($close, 20)")
    assert not result.valid
    assert any("Expression too long" in e for e in result.errors)


def test_bare_field_rejected_with_actionable_message():
    """A gate that passes what the evaluator rejects is worse than no gate.

    The evaluator resolves fields only through a leading "$", so TS_MEAN(close,
    10) raises "Unknown identifier: close" at scoring time. Checking function
    names alone let it through: an LLM given a prompt with no "$" convention
    produced 26 of 26 expressions that passed this gate and evaluated none.
    """
    result = ExpressionSanitizer().sanitize("TS_MEAN(close, 10)")
    assert not result.valid
    assert any("$close" in e for e in result.errors)


def test_dollar_prefixed_known_fields_accepted():
    s = ExpressionSanitizer()
    assert s.sanitize("TS_MEAN($close, 10)").valid
    assert s.sanitize("ts_mean($adj_close, 10)").valid
    assert s.sanitize("-RANK(TS_CORR(RANK($close), RANK($volume), 5))").valid


def test_unknown_dollar_field_rejected():
    result = ExpressionSanitizer().sanitize("TS_MEAN($vwap, 10)")
    assert not result.valid
    assert any("$vwap" in e for e in result.errors)


def test_scientific_notation_is_not_an_identifier():
    """The `e` of 1e-8 is a float literal, not a bare field reference.

    Guarding divisions with an epsilon is idiomatic here, so a false positive on
    it would reject most well-formed expressions.
    """
    result = ExpressionSanitizer().sanitize("TS_MEAN($close, 21) / (TS_STD($close, 21) + 1e-8)")
    assert result.valid, result.errors
    assert ExpressionSanitizer().sanitize("$close / (1.5e-9 + $volume)").valid


def test_gate_agrees_with_evaluator_on_shipped_factors():
    """Every expression this gate accepts must actually evaluate."""
    import numpy as np
    import pandas as pd
    from quantaalpha_us.factors.expression_evaluator import ExpressionEvaluator

    dates = pd.date_range("2024-01-01", periods=60)
    syms = ["A", "B", "C"]
    rng = np.random.default_rng(0)
    panels = {
        f: pd.DataFrame(rng.random((60, 3)) + 100, index=dates, columns=syms).astype("float64")
        for f in ("open", "high", "low", "close", "adj_close", "volume", "dollar_volume", "return")
    }
    evaluator = ExpressionEvaluator(panels)
    sanitizer = ExpressionSanitizer()
    for expression in (
        "TS_MEAN($close, 21) / (TS_STD($close, 21) + 1e-8)",
        "-RANK(TS_CORR(RANK($close), RANK($volume), 5))",
        "-RANK(TS_MAX($return, 5))",
        "TS_MEAN(close, 10)",
        "ts_mean(adj_close, 10)",
        "TS_MEAN($vwap, 10)",
    ):
        accepted = sanitizer.sanitize(expression).valid
        try:
            evaluator.evaluate(expression)
            evaluates = True
        except Exception:
            evaluates = False
        assert accepted == evaluates, f"gate/evaluator disagree on {expression!r}"


def test_wrong_arity_rejected_at_the_gate():
    """The expression that survived the gate and died mid-scoring.

    ZSCORE takes one argument. `-CS_RANK(ZSCORE($return, 10))` passed the
    sanitizer during a live mining run and raised "ZSCORE expects 1
    argument(s), got 2" only once the evaluator reached it, wasting a full pass
    over a 4.87M-row panel to learn something knowable from the text.
    """
    result = ExpressionSanitizer().sanitize("-CS_RANK(ZSCORE($return, 10))")
    assert not result.valid
    assert any("ZSCORE expects 1" in e for e in result.errors)
    assert ExpressionSanitizer().sanitize("-CS_RANK(ZSCORE($return))").valid


def test_arity_table_matches_the_evaluator():
    """The gate's arity table must not drift from what the evaluator enforces.

    The table is duplicated rather than imported so this module stays free of
    the pandas/numpy dependency the evaluator carries; this test is what keeps
    the copy honest. It calls every function with one argument too many and
    asserts the evaluator objects.
    """
    import numpy as np
    import pandas as pd
    from quantaalpha_us.factors.expression_evaluator import ExpressionEvaluator, ExpressionError

    dates = pd.date_range("2024-01-01", periods=40)
    syms = ["A", "B"]
    rng = np.random.default_rng(0)
    panels = {
        f: pd.DataFrame(rng.random((40, 2)) + 100, index=dates, columns=syms).astype("float64")
        for f in ("open", "high", "low", "close", "adj_close", "volume", "dollar_volume", "return")
    }
    evaluator = ExpressionEvaluator(panels)
    for name, expected in ExpressionSanitizer.FUNCTION_ARITY.items():
        args = ", ".join(["$close"] * (expected + 1))
        with pytest.raises(ExpressionError, match=f"{name} expects {expected} argument"):
            evaluator.evaluate(f"{name}({args})")


def _probe_panels():
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2024-01-01", periods=80)
    syms = ["A", "B", "C", "D"]
    rng = np.random.default_rng(3)
    panels = {
        f: pd.DataFrame(rng.random((80, 4)) + 100, index=dates, columns=syms).astype("float64")
        for f in ("open", "high", "low", "close", "adj_close", "volume", "dollar_volume")
    }
    panels["return"] = panels["close"].pct_change(fill_method=None)
    return panels


def _well_formed_call(name: str, sanitizer: ExpressionSanitizer) -> str | None:
    """Build a syntactically correct call for one function, or None if variadic."""
    if name in ("IF", "IF_ELSE"):
        return f"{name}($close > 100, $volume, 0)"
    if name in sanitizer.VARIADIC_MIN_ARITY:
        return f"{name}($close, $volume)"
    n = sanitizer.FUNCTION_ARITY.get(name)
    if n is None:
        return None
    pair = ("TS_CORR", "TS_COV", "TS_COVARIANCE")
    args = []
    for i in range(n):
        if i == 0:
            args.append("$close")
        elif i == 1 and name in pair:
            args.append("$volume")
        else:
            args.append("5")
    return f"{name}({', '.join(args)})"


def test_every_allowed_function_agrees_between_gate_and_evaluator():
    """The gate must accept an expression if and only if the evaluator runs it.

    Three disagreement classes were found one at a time -- bare field names,
    unknown fields, and wrong arity -- each only after a live mining run wasted
    a full scoring pass on expressions that could never evaluate. This sweeps
    the whole function table instead of waiting for the next one.
    """
    from quantaalpha_us.factors.expression_evaluator import ExpressionEvaluator

    sanitizer = ExpressionSanitizer()
    evaluator = ExpressionEvaluator(_probe_panels())
    disagreements = []
    for name in sorted({f.upper() for f in sanitizer.allowed_functions}):
        expression = _well_formed_call(name, sanitizer)
        assert expression is not None, (
            f"{name} has neither a fixed arity nor a variadic minimum, so the gate "
            "cannot validate its argument count at all"
        )
        accepted = sanitizer.sanitize(expression).valid
        try:
            evaluator.evaluate(expression)
            runs = True
        except Exception:
            runs = False
        if accepted != runs:
            disagreements.append((name, expression, accepted, runs))
    assert not disagreements, f"gate/evaluator disagree: {disagreements}"


def test_variadic_minmax_needs_two_operands():
    """MIN()/MAX() passed the gate and raised a bare ValueError from the builtin."""
    sanitizer = ExpressionSanitizer()
    for expression in ("MIN()", "MAX()", "MIN($close)"):
        result = sanitizer.sanitize(expression)
        assert not result.valid, f"{expression} should be rejected"
        assert any("at least 2" in e for e in result.errors), result.errors
    assert sanitizer.sanitize("MIN($close, $volume)").valid
