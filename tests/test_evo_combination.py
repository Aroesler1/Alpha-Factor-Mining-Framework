"""A combined factor signal must not reward a name for missing a factor."""

import numpy as np
import pandas as pd
import pytest

from quantaalpha_us.evo.scoring import equal_weight_rank_average


class _StubScorer:
    def __init__(self, signals):
        self.signals = signals

    def evaluate(self, expression):
        return self.signals[expression]


def _signals():
    dates = pd.bdate_range("2020-01-01", periods=3)
    names = [f"s{i}" for i in range(10)]
    # A carries no information (one value for every name); B has the opposite
    # sign and only covers s5..s9
    a = pd.DataFrame(1.0, index=dates, columns=names)
    b = pd.DataFrame(np.nan, index=dates, columns=names)
    b.loc[:, names[5:]] = np.arange(5.0)
    return {"A": a, "B": b}


def test_missing_negative_sign_factor_is_neutral():
    combined = equal_weight_rank_average(_StubScorer(_signals()), ["A", "B"], [1, -1])
    day = combined.iloc[0]
    # names without B get no boost from its absence
    assert day[[f"s{i}" for i in range(5)]].to_numpy() == pytest.approx(np.zeros(5))
    # the best name is the one B ranks lowest, as its negative sign says
    assert day.idxmax() == "s5"


def test_name_covered_by_no_factor_stays_missing():
    signals = _signals()
    signals["A"].loc[:, "s0"] = np.nan
    combined = equal_weight_rank_average(_StubScorer(signals), ["A", "B"], [1, -1])
    assert combined["s0"].isna().all()


def test_rescore_replaces_stored_fitness_and_keeps_membership():
    from types import SimpleNamespace

    from quantaalpha_us.evo.archive import ArchiveEntry
    from scripts.sp500_evo_final_table import rescore_entries

    class _Scorer:
        def evaluate(self, expression):
            if expression == "BROKEN":
                raise ValueError("does not evaluate")
            return pd.DataFrame([[1.0]])

        def metrics(self, expression, signal):
            return SimpleNamespace(fitness=1.5, to_dict=lambda: {"tstat": 2.0, "sign": -1})

    def entry(expr):
        return ArchiveEntry(id=expr, expression=expr, fitness=9.0, niche=("fast", "mixed", "low"),
                            round=1, island="x", operator="explore", parent_ids=(), rationale="")

    out, failed = rescore_entries([entry("A"), entry("BROKEN")], _Scorer())
    assert failed == ["BROKEN"]
    assert [(e.expression, e.fitness, e.metrics["sign"]) for e in out] == [("A", 1.5, -1)]
    assert out[0].niche == ("fast", "mixed", "low")
