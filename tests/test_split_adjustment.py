"""Splits and ticker changes must not move factor values or labels.

CRSP carries raw open/high/low next to a split-adjusted close. A panel that mixes
the two scales makes every intraday ratio and every open-to-open label jump by
the split ratio, so these fixtures hold the economic price flat through a split
and require every derived quantity to stay flat too.
"""

import numpy as np
import pandas as pd
import pytest

from quantaalpha_us.factors.expression_evaluator import ExpressionEvaluator, build_field_panels
from quantaalpha_us.factors.factor_research import forward_open_returns
from quantaalpha_us.factors.ic_panel import CUMULATIVE, LAGGED, forward_returns_at


def _split_bars():
    # 2:1 split on the third day; the economic price is 50 throughout
    dates = pd.bdate_range("2020-01-01", periods=6)
    raw = [100.0, 100.0, 50.0, 50.0, 50.0, 50.0]
    factor = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    return pd.DataFrame({
        "date": dates, "symbol": "X", "permno": 1,
        "open": raw, "high": [r * 1.02 for r in raw], "low": [r * 0.98 for r in raw],
        "close": raw, "adj_close": [r / f for r, f in zip(raw, factor)],
        "volume": [1_000.0, 1_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0],
    })


def test_open_to_open_label_ignores_a_split():
    panels = build_field_panels(_split_bars())
    labels = forward_open_returns(panels).iloc[:, 0]
    assert labels.dropna().abs().max() == pytest.approx(0.0)


@pytest.mark.parametrize("kind", [LAGGED, CUMULATIVE])
@pytest.mark.parametrize("horizon", [1, 2])
def test_horizon_labels_ignore_a_split(kind, horizon):
    panels = build_field_panels(_split_bars())
    labels = forward_returns_at(panels, horizon, kind=kind).iloc[:, 0]
    assert labels.dropna().abs().max() == pytest.approx(0.0)


def test_intraday_fields_share_the_close_scale():
    evaluator = ExpressionEvaluator(build_field_panels(_split_bars()))
    gap = evaluator.evaluate("($close - $open) / $open").iloc[:, 0]
    assert gap.abs().max() == pytest.approx(0.0)
    width = evaluator.evaluate("($high - $low) / $close").iloc[:, 0]
    assert width.to_numpy() == pytest.approx(np.full(len(width), 0.04))


def test_dollar_volume_uses_raw_price_and_shares():
    panels = build_field_panels(_split_bars())
    assert panels["dollar_volume"].iloc[:, 0].tolist() == pytest.approx([100_000.0] * 6)


def test_ticker_change_keeps_one_series_and_reuse_keeps_two():
    dates = pd.bdate_range("2020-01-01", periods=6)
    renamed = pd.DataFrame({"date": dates, "permno": 1, "symbol": ["OLD"] * 3 + ["NEW"] * 3,
                            "close": np.arange(6.0) + 10, "adj_close": np.arange(6.0) + 10})
    reused = pd.DataFrame({"date": dates, "permno": [2] * 3 + [3] * 3, "symbol": "SAME",
                           "close": [5.0, 5.0, 5.0, 500.0, 500.0, 500.0],
                           "adj_close": [5.0, 5.0, 5.0, 500.0, 500.0, 500.0]})
    panels = build_field_panels(pd.concat([renamed, reused]))
    assert list(panels["close"].columns) == [1, 2, 3]
    evaluator = ExpressionEvaluator(panels)
    mean5 = evaluator.evaluate("TS_MEAN($close, 5)")
    assert mean5[1].iloc[-1] == pytest.approx(13.0)
    # two different companies that shared a ticker are never one return series
    assert panels["return"][3].dropna().abs().max() == pytest.approx(0.0)
    assert panels["return"][2].dropna().abs().max() == pytest.approx(0.0)


def test_symbol_key_is_still_available_when_permno_is_missing():
    bars = _split_bars().drop(columns=["permno"])
    panels = build_field_panels(bars)
    assert list(panels["close"].columns) == ["X"]
