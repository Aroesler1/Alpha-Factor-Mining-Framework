"""Keep outcome prices inside the window used to select a factor."""

from __future__ import annotations

import pandas as pd


def label_end_dates(index: pd.DatetimeIndex, horizon: int = 1) -> pd.Series:
    """Both open-return conventions end at session T+h+1, not signal date T."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("label calendar must be unique and increasing")
    return pd.Series(index, index=index).shift(-(horizon + 1))


def labels_known_by(index: pd.DatetimeIndex, cutoff, horizon: int = 1) -> pd.Series:
    """False also covers outcomes extending beyond the available calendar."""
    return label_end_dates(index, horizon).le(pd.Timestamp(cutoff))
