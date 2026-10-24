"""Cached-IC scripts must not let a training-window label end after the cut."""

import numpy as np
import pandas as pd
import pytest

from quantaalpha_us.factors.label_clock import rows_known_by


def _ic_frame():
    dates = pd.bdate_range("2015-12-01", periods=40)
    frame = pd.DataFrame({"a": 0.01, "b": 0.02}, index=dates)
    return frame


@pytest.mark.parametrize("horizon", [1, 5])
def test_rows_known_by_drops_the_last_h_plus_one_signal_dates(horizon):
    frame = _ic_frame()
    cut = frame.index[29]
    kept = rows_known_by(frame, cut, horizon)
    assert kept.index.max() == frame.index[29 - horizon - 1]
    assert len(kept) == 30 - horizon - 1


def test_publication_decay_pre_window_ignores_labels_after_the_cut():
    from scripts.sp500_alpha101_decay import summarise

    frame = _ic_frame()
    cut = frame.index[29]
    # the two signal dates just before the cut carry outcomes from after it
    poisoned = frame.copy()
    poisoned.loc[frame.index[28:30]] = 5.0
    rows, summary = summarise("x", poisoned, cut, lag=1, draws=20, block=2, seed=0)
    assert summary["mean_ic_pre"] == pytest.approx(0.015)
    assert summary["pre_days"] == 28
    assert np.isfinite(rows["mean_ic_post"]).all()
