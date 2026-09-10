"""The Alpha101 baseline must actually be a baseline.

If a transcribed alpha fails to sanitize or evaluate it drops silently out of
the report, which would quietly shrink the published-alpha comparison set and
weaken the memorization test.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantaalpha_us.factors.expression_evaluator import (
    ExpressionEvaluator,
    build_field_panels,
)
from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer

ALPHA101 = Path(__file__).resolve().parents[1] / "configs" / "alpha101_us.txt"


def _expressions() -> list[str]:
    return [line.strip() for line in ALPHA101.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def test_file_exists_and_is_non_trivial():
    assert ALPHA101.exists()
    assert len(_expressions()) >= 40


def test_every_transcribed_alpha_sanitizes():
    sanitizer = ExpressionSanitizer()
    for expr in _expressions():
        result = sanitizer.sanitize(expr)
        assert result.valid, f"{expr[:70]} -> {result.errors}"


def test_transcribed_plus_dropped_accounts_for_all_101():
    """Every one of the 101 is either transcribed or has a recorded reason."""
    dropped = sum(1 for line in ALPHA101.read_text(encoding="utf-8").splitlines()
                  if "DROPPED" in line)
    assert len(_expressions()) + dropped == 101


def test_every_transcribed_alpha_evaluates(synthetic_bars):
    evaluator = ExpressionEvaluator(build_field_panels(synthetic_bars))
    for expr in _expressions():
        signal = evaluator.evaluate(expr)
        assert isinstance(signal, pd.DataFrame)


@pytest.fixture
def synthetic_bars() -> pd.DataFrame:
    """A panel long enough for the 250-day windows some alphas use."""
    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=400)
    symbols = [f"S{i:02d}" for i in range(40)]
    rows = []
    for sym in symbols:
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        rows.append(pd.DataFrame({
            "date": dates,
            "symbol": sym,
            "close": price,
            "adj_close": price,
            "open": price * (1 + rng.normal(0, 0.002, len(dates))),
            "high": price * (1 + abs(rng.normal(0, 0.004, len(dates)))),
            "low": price * (1 - abs(rng.normal(0, 0.004, len(dates)))),
            "volume": rng.integers(1e5, 1e7, len(dates)).astype(float),
        }))
    bars = pd.concat(rows, ignore_index=True)
    bars["dollar_volume"] = bars["close"] * bars["volume"]
    return bars


# ---- published numbering -------------------------------------------------


def test_the_reconstructed_numbering_covers_1_to_101_exactly_once():
    """The structural report and the loop's feedback cite alphas by number, so a
    drifted cursor would attribute a paraphrase to the wrong published alpha."""
    from quantaalpha_us.factors import alpha101

    kept = alpha101.load()
    dropped = alpha101.dropped_numbers()
    numbers = sorted([a.number for a in kept] + dropped)
    assert len(kept) == 50
    assert len(dropped) == 51
    assert numbers == list(range(1, 102))


def test_known_alphas_land_on_their_published_numbers():
    from quantaalpha_us.factors import alpha101

    by_number = {a.number: a.expression for a in alpha101.load()}
    # alpha006 is -1 * correlation(open, volume, 10) in the paper
    assert by_number[6] == "-1 * TS_CORR($open, $volume, 10)"
    # alpha003 is -1 * correlation(rank(open), rank(volume), 10)
    assert by_number[3] == "-1 * TS_CORR(CS_RANK($open), CS_RANK($volume), 10)"
    # alpha005 is dropped for vwap, so 5 must be absent
    assert 5 not in by_number
