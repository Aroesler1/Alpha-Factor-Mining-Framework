"""Tests for the mined-expression evaluator and factor research scoring."""
import numpy as np
import pandas as pd
import pytest

from quantaalpha_us.factors.expression_evaluator import (
    ExpressionError,
    ExpressionEvaluator,
    build_field_panels,
)
from quantaalpha_us.factors.factor_research import (
    forward_open_returns,
    score_expressions,
    select_uncorrelated,
)


def _bars(n_days=120, symbols=("AAA", "BBB", "CCC"), seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for j, sym in enumerate(symbols):
        close = 100.0 * (1 + j) * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n_days)))
        open_px = close * (1 + rng.normal(0, 0.002, n_days))
        rows.append(pd.DataFrame({
            "date": dates,
            "symbol": sym,
            "open": open_px,
            "high": np.maximum(open_px, close) * 1.005,
            "low": np.minimum(open_px, close) * 0.995,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1e5, 1e6, n_days).astype(float),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


def _evaluator(bars=None):
    panels = build_field_panels(bars if bars is not None else _bars())
    return ExpressionEvaluator(panels), panels


def test_ts_mean_matches_pandas_rolling():
    ev, panels = _evaluator()
    out = ev.evaluate("TS_MEAN($close, 5)")
    expected = panels["close"].rolling(5, min_periods=5).mean()
    pd.testing.assert_frame_equal(out, expected)


def test_delta_delay_and_arithmetic():
    ev, panels = _evaluator()
    out = ev.evaluate("TS_DELTA($close, 3) / DELAY($close, 3)")
    close = panels["close"]
    expected = (close - close.shift(3)) / close.shift(3)
    pd.testing.assert_frame_equal(out, expected)


def test_rank_is_cross_sectional():
    ev, _ = _evaluator()
    out = ev.evaluate("RANK($close)")
    row = out.dropna(how="all").iloc[0]
    # three symbols with strictly ordered price levels -> ranks 1/3, 2/3, 1
    assert sorted(row.round(6).tolist()) == [round(1 / 3, 6), round(2 / 3, 6), 1.0]


def test_if_with_comparison():
    ev, panels = _evaluator()
    out = ev.evaluate("IF($close > DELAY($close, 1), 1, -1)")
    up = panels["close"] > panels["close"].shift(1)
    sample = out.iloc[5:]
    assert set(np.unique(sample.to_numpy())) <= {1.0, -1.0}
    expected = np.where(up.iloc[5:], 1.0, -1.0)
    assert (sample.to_numpy() == expected).all()


def test_strict_windows_produce_nan_warmup():
    ev, _ = _evaluator()
    out = ev.evaluate("TS_STD($close, 21)")
    assert out.iloc[:20].isna().all().all()
    assert out.iloc[21:].notna().all().all()


def test_unknown_field_and_function_rejected():
    ev, _ = _evaluator()
    with pytest.raises(ExpressionError):
        ev.evaluate("TS_MEAN($closing_price, 5)")
    with pytest.raises(ExpressionError):
        ev.evaluate("EXEC($close, 5)")


def test_dangerous_syntax_rejected():
    ev, _ = _evaluator()
    for bad in (
        "__import__('os').system('true')",
        "$close.attr",
        "[1, 2, 3]",
        "lambda x: x",
        "'text'",
    ):
        with pytest.raises(ExpressionError):
            ev.evaluate(bad)


def test_scalar_expression_rejected():
    ev, _ = _evaluator()
    with pytest.raises(ExpressionError):
        ev.evaluate("1 + 2")


def test_score_expressions_finds_planted_alpha():
    # plant a signal equal to the future open-to-open return -> IC near 1
    bars = _bars(n_days=200, symbols=tuple(f"S{i:02d}" for i in range(40)))
    panels = build_field_panels(bars)
    fwd = forward_open_returns(panels)

    # inject the future return (plus noise, so daily ICs are high but not
    # degenerate at exactly 1.0) into the volume field
    rng = np.random.default_rng(5)
    leak = fwd + rng.normal(0, fwd.stack().std() * 0.3, size=fwd.shape)
    stacked = leak.stack().rename("volume").reset_index()
    stacked.columns = ["date", "symbol", "leak"]
    bars = bars.merge(stacked, on=["date", "symbol"], how="left")
    bars["volume"] = bars["leak"].fillna(0.0)
    bars = bars.drop(columns=["leak"])
    bars["dollar_volume"] = bars["close"] * bars["volume"]

    report, signals = score_expressions(
        bars, ["RANK($volume)", "RANK(TS_DELTA($close, 5))"], min_cross_section=20)
    frame = report.frame().set_index("expression")

    planted = frame.loc["RANK($volume)"]
    assert planted["mean_ic"] > 0.8
    assert planted["ic_tstat"] > 10

    ordinary = frame.loc["RANK(TS_DELTA($close, 5))"]
    assert abs(ordinary["mean_ic"]) < 0.3


def test_select_uncorrelated_drops_semantic_duplicates():
    bars = _bars(n_days=250, symbols=tuple(f"S{i:02d}" for i in range(40)))
    exprs = [
        "RANK(TS_DELTA($close, 5))",
        "RANK($close / DELAY($close, 5) - 1)",  # same idea, different spelling
        "RANK(TS_STD($close, 21))",
    ]
    report, signals = score_expressions(bars, exprs, min_cross_section=20)
    # force all candidates past the t-stat gate so only dedup is under test
    selected = select_uncorrelated(report, signals, min_abs_tstat=0.0, max_abs_corr=0.7)
    assert len(selected) == 2
    assert "RANK(TS_STD($close, 21))" in selected


def test_invalid_expression_reported_not_raised():
    bars = _bars()
    report, signals = score_expressions(bars, ["import os", "TS_MEAN($close, 5)"])
    frame = report.frame()
    assert isinstance(frame.iloc[0]["error"], str) and frame.iloc[0]["error"]
    assert pd.isna(frame.iloc[1]["error"])


def test_checked_in_candidate_file_all_sanitize_and_evaluate():
    from pathlib import Path

    from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer

    path = Path(__file__).resolve().parent.parent / "configs" / "mined_factors_claude_2026-08.txt"
    candidates = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(candidates) >= 15

    sanitizer = ExpressionSanitizer()
    bars = _bars(n_days=140, symbols=tuple(f"S{i:02d}" for i in range(8)))
    ev, _ = _evaluator(bars)
    for expr in candidates:
        result = sanitizer.sanitize(expr)
        assert result.valid, f"sanitizer rejected: {expr}: {result.errors}"
        panel = ev.evaluate(result.cleaned)
        assert panel.notna().any().any(), f"all-NaN signal: {expr}"
