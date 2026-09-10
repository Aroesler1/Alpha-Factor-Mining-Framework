"""Offline reconstruction tests for sanitized fundamentals provenance evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import sp500_fundamentals_provenance as provenance
from scripts.sp500_fundamentals_provenance import (
    MATCH_FIELDS,
    TABLE_SPECS,
    _read_local,
    build_report,
    build_sanitized_evidence,
    check_artifacts,
    deterministic_indices,
    select_local_rows,
    sha256_file,
    write_artifacts,
)


def _local_frame(n_rows: int) -> pd.DataFrame:
    sequence = np.arange(n_rows)
    datadates = [
        pd.Timestamp("2000-03-31") + pd.DateOffset(months=3 * int(value))
        for value in sequence
    ]
    return pd.DataFrame(
        {
            "gvkey": [f"{value + 1:06d}" for value in sequence],
            "datadate": datadates,
            "rdq": [value + pd.Timedelta(days=31) for value in datadates],
            "atq": sequence.astype(float) + 100.25,
            "ltq": sequence.astype(float) + 25.5,
            "revtq": sequence.astype(float) * 2.0,
            "niq": sequence.astype(float) / 10.0,
        }
    )


def _candidate_cache(path, sampled: pd.DataFrame) -> None:
    candidates = []
    for index, spec in enumerate(TABLE_SPECS):
        schema, table = spec.source_table.split(".", 1)
        candidates.append({"table_schema": schema, "table_name": table})
        candidate = sampled.copy()
        if spec.source_table.endswith("wrds_csq_pit"):
            stale = candidate.copy()
            stale["niq"] += 1.0
            candidate = pd.concat([stale, candidate], ignore_index=True)
        if spec.source_table.endswith("wrds_csq_unrestated"):
            candidate.loc[candidate.index[:6], "niq"] += 1.0
        candidate.to_csv(
            path / f"alpha_fundq_fingerprint_{index:02d}.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.DataFrame(
            [
                {
                    "max_datadate": "2030-12-31",
                    "max_rdq": "2031-02-01",
                }
            ]
        ).to_csv(
            path / f"alpha_fundq_profile_{index:02d}.csv.gz",
            index=False,
            compression="gzip",
        )
    pd.DataFrame(candidates).to_csv(path / "fundq_candidates.csv", index=False)


def test_linspace_sampling_is_deterministic_and_endpoint_inclusive():
    local = _local_frame(97)
    first, first_indices = select_local_rows(local)
    second, second_indices = select_local_rows(local.copy())
    expected = np.linspace(0, len(local) - 1, 32).astype(int)

    np.testing.assert_array_equal(first_indices, expected)
    np.testing.assert_array_equal(second_indices, expected)
    pd.testing.assert_frame_equal(first, second)
    assert first_indices[0] == 0
    assert first_indices[-1] == len(local) - 1
    assert len(np.unique(first_indices)) == len(first_indices)
    np.testing.assert_array_equal(deterministic_indices(9), np.arange(9))


def test_external_cache_build_reconstructs_report_from_sample_rows(tmp_path):
    local_path = tmp_path / "local.parquet"
    cache_path = tmp_path / "cache"
    output_path = tmp_path / "out"
    cache_path.mkdir()
    local = _local_frame(97)
    local.to_parquet(local_path, index=False)
    loaded = _read_local(local_path)
    sampled, _ = select_local_rows(loaded)
    _candidate_cache(cache_path, sampled)

    evidence, metadata = build_sanitized_evidence(
        loaded,
        cache_path,
        input_kind="external_local_parquet",
        local_input_name=local_path.name,
        local_input_sha256=sha256_file(local_path),
    )
    write_artifacts(output_path, evidence, metadata)
    rebuilt = check_artifacts(
        output_path / "fundamentals_provenance_samples.csv",
        output_path / "fundamentals_provenance_metadata.json",
        output_path / "fundamentals_provenance_audit.csv",
    )

    values = rebuilt.set_index("item")["value"].astype(str)
    for spec in TABLE_SPECS:
        assert int(values[spec.report_item]) == int(evidence[spec.match_column].sum())
    assert int(values["local_keys_sampled"]) == len(evidence)
    assert set(evidence.columns).isdisjoint({"gvkey", "datadate", *MATCH_FIELDS})
    assert metadata["selection"]["input_rows"] == len(local)
    assert metadata["selection"]["sample_rows"] == len(evidence)

    report_path = output_path / "fundamentals_provenance_audit.csv"
    changed = pd.read_csv(report_path, dtype=str)
    changed.loc[changed["item"] == TABLE_SPECS[0].report_item, "value"] = "0"
    changed.to_csv(report_path, index=False)
    with pytest.raises(ValueError, match="does not reconstruct"):
        check_artifacts(
            output_path / "fundamentals_provenance_samples.csv",
            output_path / "fundamentals_provenance_metadata.json",
            report_path,
        )


def test_committed_checker_needs_no_external_inputs(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("check mode attempted to read an external input")

    monkeypatch.setattr(provenance, "_read_local", forbidden)
    monkeypatch.setattr(provenance, "_read_candidate_cache", forbidden)
    monkeypatch.setattr("sys.argv", ["sp500_fundamentals_provenance.py", "--check"])
    assert provenance.main() == 0
    assert "No vendor or holdout data was read" in capsys.readouterr().out

    rebuilt = check_artifacts()
    evidence = pd.read_csv("reports/fundamentals_provenance_samples.csv")
    metadata = json.loads(
        Path("reports/fundamentals_provenance_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    pd.testing.assert_frame_equal(
        rebuilt.reset_index(drop=True),
        build_report(evidence, metadata).reset_index(drop=True),
    )


def test_matching_row_hash_does_not_require_writable_numpy_views():
    """CI NumPy wheels can mark pandas boolean views read-only."""
    sampled = _local_frame(8)
    candidate = provenance._normalise_candidate(sampled)
    matched, digest = provenance._matching_row_hash(sampled.iloc[3], candidate)
    assert matched is True
    assert provenance.HASH_RE.fullmatch(digest)

    unrestated = sampled.copy()
    unrestated.loc[unrestated.index[:3], "niq"] += 1.0
    unrestated_candidate = provenance._normalise_candidate(unrestated)
    missed, empty = provenance._matching_row_hash(
        sampled.iloc[0], unrestated_candidate
    )
    assert missed is False
    assert empty == ""
