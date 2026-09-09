"""Regression fixtures for outcome clocks, family inference and evaluation locks."""

import json

import numpy as np
import pandas as pd
import pytest

from quantaalpha_us.evo.comparison_audit import adjust_comparisons
from quantaalpha_us.evo.config import EvoConfig, Windows
from quantaalpha_us.evo.evaluation_protocol import claim_evaluation, write_exclusive
from quantaalpha_us.evo.scoring import PanelScorer
from quantaalpha_us.factors.label_clock import label_end_dates, labels_known_by
from quantaalpha_us.factors.factor_research import score_expressions
from tests.synthetic_panel import planted_bars_with_fundamentals


@pytest.mark.parametrize("horizon", [1, 5, 63])
def test_boundary_counts_use_outcome_end(horizon):
    dates = pd.bdate_range("2013-01-02", periods=200)
    allowed = labels_known_by(dates, dates[99], horizon)
    assert allowed.iloc[:100].sum() == 100 - horizon - 1
    assert not allowed.iloc[100:].any()


@pytest.mark.parametrize("dates", [pd.DatetimeIndex(["2020-01-02", "2020-01-01"]),
                                  pd.DatetimeIndex(["2020-01-01", "2020-01-01"])])
def test_bad_calendar_refuses_silent_shift(dates):
    with pytest.raises(ValueError):
        label_end_dates(dates)


def test_future_prices_cannot_change_fit_labels_or_half_life_dates():
    dates = pd.bdate_range("2010-01-04", periods=260)
    data = pd.DataFrame(np.arange(260 * 4).reshape(260, 4) + 100., index=dates)
    panels = {name: data.copy() for name in ("open", "close", "adj_close", "high", "low", "volume")}
    config = EvoConfig(windows=Windows(fit_start=str(dates[0].date()), fit_end=str(dates[129].date()),
        validation_start=str(dates[130].date()), validation_end=str(dates[199].date()),
        holdout_start=str(dates[200].date()), holdout_end=str(dates[-1].date()),
        cheap_start=str(dates[80].date())))
    first = PanelScorer(pd.DataFrame(), config, panels=panels)
    changed = {key: value.copy() for key, value in panels.items()}
    for value in changed.values():
        value.loc[dates[130:]] *= np.arange(130, dtype=float)[:, None] + 2
    second = PanelScorer(pd.DataFrame(), config, panels=changed)
    for horizon in first._fwd_fit:
        pd.testing.assert_frame_equal(first._fwd_fit[horizon], second._fwd_fit[horizon])
        assert first._fwd_half_life[horizon].notna().all().all()
    assert first._fwd_validation.iloc[-2:].isna().all().all()
    assert config.to_dict()["label_policy"] == "outcome-end-v1"


def test_one_shot_fit_does_not_use_future_answers():
    bars = planted_bars_with_fundamentals(n_days=180, n_symbols=35)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(bars.date).unique()))
    chosen = dates[:130]
    changed = bars.copy()
    later = pd.to_datetime(changed.date) > chosen[-1]
    changed.loc[later, "open"] *= 10
    a, _ = score_expressions(bars, ["$close"], ic_dates=chosen)
    b, _ = score_expressions(changed, ["$close"], ic_dates=chosen)
    assert a.scores[0].mean_ic == pytest.approx(b.scores[0].mean_ic)
    assert a.scores[0].ic_days <= 128


def test_holm_is_monotone_and_restores_original_order():
    frame = pd.DataFrame({"comparison": list("abcde"), "p": [.399, .047, .5815, .1525, .122]})
    result = adjust_comparisons(frame)
    assert result.holm_p.tolist() == pytest.approx([.798, .235, .798, .488, .488])
    assert not result.reject_holm_5pct.any()


@pytest.mark.parametrize("values", [[np.nan], [-.1], [1.1]])
def test_unknown_family_member_cannot_be_dropped(values):
    with pytest.raises(ValueError):
        adjust_comparisons(pd.DataFrame({"comparison": ["a"], "p": values}))


def test_evaluation_freeze_and_consumption_are_exclusive(tmp_path):
    manifest, record = tmp_path / "manifest.json", tmp_path / "consumed.json"
    snapshot = {"files": {"archive": "abc"}, "settings": {"top_n": 20}}
    write_exclusive(manifest, snapshot)
    with pytest.raises(ValueError, match="differ"):
        claim_evaluation(manifest, record, {**snapshot, "settings": {"top_n": 10}})
    assert not record.exists()
    claim_evaluation(manifest, record, snapshot)
    assert json.loads(record.read_text())["status"] == "consumed_before_read"
    with pytest.raises(ValueError, match="consumed"):
        claim_evaluation(manifest, record, snapshot)
    with pytest.raises(FileExistsError):
        write_exclusive(manifest, snapshot)


def test_completed_comparisons_reproduce_without_bars():
    from scripts.sp500_evo_audit import build
    result = build()
    assert len(result) == 5
    assert result.holm_p.min() == pytest.approx(.235)


def test_fundamentals_provenance_report_preserves_the_unrecoverable_vintage():
    report = pd.read_csv("reports/fundamentals_provenance_audit.csv")
    values = report.set_index("item")["value"].astype(str)
    assert int(values["local_keys_sampled"]) == 32
    assert int(values["comp.fundq_exact_matches"]) == 32
    assert int(values["comp_snapshot.wrds_csq_unrestated_exact_matches"]) == 26
    assert values["extract_vintage"] == "not recoverable"


def test_consumed_historical_entrypoint_refuses_before_price_read(monkeypatch):
    from scripts import sp500_evo_final_table as script
    monkeypatch.setattr("sys.argv", ["sp500_evo_final_table.py"])
    def forbidden(*args, **kwargs):
        raise AssertionError("historical holdout was read")
    monkeypatch.setattr(script.pd, "read_parquet", forbidden)
    with pytest.raises(SystemExit, match="already consumed"):
        script.main()
