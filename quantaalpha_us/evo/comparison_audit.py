"""Adjust the published comparison family without evaluating any factor."""

import numpy as np
import pandas as pd


def adjust_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or frame["comparison"].duplicated().any():
        raise ValueError("Expected a nonempty family of distinct comparisons")
    p = frame["p"].to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Every family member needs a finite p-value in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = frame.copy()
    result["holm_p"] = adjusted
    result["family_size"] = len(p)
    result["reject_holm_5pct"] = adjusted <= 0.05
    result["evidence_status"] = "historical_labels_not_boundary_purged"
    return result
