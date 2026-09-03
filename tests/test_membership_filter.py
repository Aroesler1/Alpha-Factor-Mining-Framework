"""The scoring universe must be point-in-time index membership, not everything
that was ever a member."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "sp500_score_mined_factors", ROOT / "scripts" / "sp500_score_mined_factors.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
apply_membership_filter = _mod.apply_membership_filter


def _bars():
    """Three names; permno 3 is never an index member."""
    rows = []
    for date in pd.date_range("2024-01-01", periods=3):
        for permno in (1, 2, 3):
            rows.append({"date": date, "symbol": f"S{permno}", "permno": permno,
                         "close": 100.0 + permno})
    return pd.DataFrame(rows)


def _membership(tmp_path, rows):
    path = tmp_path / "membership.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


def test_non_members_are_dropped_on_the_dates_they_are_not_members(tmp_path):
    """A name joining the index midway must not be scored before it joins.

    The surplus names in the real panel are future members as often as deleted
    ones, so this is a lookahead guard, not only a tidiness one.
    """
    dates = pd.date_range("2024-01-01", periods=3)
    rows = [{"date": d, "permno": 1, "active": True} for d in dates]
    rows += [{"date": d, "permno": 2, "active": True} for d in dates[1:]]  # joins day 2
    out = apply_membership_filter(_bars(), _membership(tmp_path, rows))

    assert set(out.permno.unique()) == {1, 2}, "permno 3 was never a member"
    day1 = out[out.date == dates[0]]
    assert set(day1.permno) == {1}, "permno 2 was scored before it joined the index"
    assert len(out) == 5


def test_inactive_rows_are_not_treated_as_membership(tmp_path):
    dates = pd.date_range("2024-01-01", periods=3)
    rows = [{"date": d, "permno": 1, "active": True} for d in dates]
    rows += [{"date": d, "permno": 2, "active": False} for d in dates]
    out = apply_membership_filter(_bars(), _membership(tmp_path, rows))
    assert set(out.permno.unique()) == {1}


def test_missing_membership_file_is_a_hard_error(tmp_path):
    """Silently skipping the filter is how the contaminated universe survived."""
    with pytest.raises(SystemExit, match="Membership file not found"):
        apply_membership_filter(_bars(), str(tmp_path / "absent.parquet"))


def test_none_scores_the_raw_panel_unchanged(capsys):
    out = apply_membership_filter(_bars(), "none")
    assert len(out) == 9
    assert "WARNING" in capsys.readouterr().out


def test_join_is_on_permno_not_symbol(tmp_path):
    """Ticker reuse must not reintroduce the identity ambiguity."""
    bars = _bars()
    bars.loc[bars.permno == 3, "symbol"] = "S1"  # a reused ticker
    dates = pd.date_range("2024-01-01", periods=3)
    rows = [{"date": d, "permno": 1, "active": True} for d in dates]
    out = apply_membership_filter(bars, _membership(tmp_path, rows))
    assert set(out.permno.unique()) == {1}, "matched on symbol, letting permno 3 in"
