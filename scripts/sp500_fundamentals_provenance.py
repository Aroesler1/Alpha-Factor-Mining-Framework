#!/usr/bin/env python3
"""Build or verify sanitized fundamentals provenance evidence offline.

Generation reads a local fundamentals parquet (or an already sampled local
cache) and cached candidate extracts produced outside this repository. It
writes only hashes, match booleans, deterministic selection metadata, and the
aggregate audit report. ``--check`` reads only those committed sanitized
artifacts; it never reads vendor data, the price panel, or the consumed holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "reports/fundamentals_provenance_samples.csv"
DEFAULT_METADATA = ROOT / "reports/fundamentals_provenance_metadata.json"
DEFAULT_REPORT = ROOT / "reports/fundamentals_provenance_audit.csv"

MAX_SAMPLES = 32
KEY_FIELDS = ("gvkey", "datadate")
MATCH_FIELDS = ("rdq", "atq", "ltq", "revtq", "niq")
REQUIRED_FIELDS = KEY_FIELDS + MATCH_FIELDS
SELECTION_ALGORITHM = (
    "numpy.linspace(0, n_rows - 1, min(32, n_rows)).astype(int)"
)
PRODUCT_FAMILY = "Compustat North America quarterly fundamentals"
SOURCE_LIMITATION = "not uniquely recoverable"
VINTAGE_LIMITATION = "not recoverable"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TableSpec:
    source_table: str
    column_prefix: str
    report_item: str
    interpretation: str

    @property
    def match_column(self) -> str:
        return f"{self.column_prefix}_match"

    @property
    def row_hash_column(self) -> str:
        return f"{self.column_prefix}_row_sha256"


TABLE_SPECS = (
    TableSpec(
        "comp.fundq",
        "comp_fundq",
        "comp.fundq_exact_matches",
        "All sampled values match the currently entitled ordinary quarterly table",
    ),
    TableSpec(
        "comp_snapshot.wrds_csq_restated",
        "comp_snapshot_restated",
        "comp_snapshot.wrds_csq_restated_exact_matches",
        "All sampled values also match the current restated Snapshot view",
    ),
    TableSpec(
        "comp_snapshot.wrds_csq_pit",
        "comp_snapshot_pit",
        "comp_snapshot.wrds_csq_pit_exact_matches",
        "Each sampled value appears among revisions in the point-in-time history",
    ),
    TableSpec(
        "comp_snapshot.wrds_csq_unrestated",
        "comp_snapshot_unrestated",
        "comp_snapshot.wrds_csq_unrestated_exact_matches",
        "Six sampled local values do not match the current unrevised view",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_indices(n_rows: int, max_samples: int = MAX_SAMPLES) -> np.ndarray:
    """Return the historical endpoint-inclusive deterministic sample indices."""
    if n_rows <= 0:
        raise ValueError("local fundamentals input has no eligible rows")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    count = min(max_samples, n_rows)
    return np.linspace(0, n_rows - 1, count).astype(int)


def select_local_rows(
    frame: pd.DataFrame, max_samples: int = MAX_SAMPLES
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop unusable keys and apply deterministic sampling without reordering."""
    missing = sorted(set(REQUIRED_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"local fundamentals input is missing columns: {missing}")
    eligible = frame.loc[:, REQUIRED_FIELDS].dropna(subset=list(KEY_FIELDS))
    eligible = eligible.reset_index(drop=True)
    indices = deterministic_indices(len(eligible), max_samples)
    sampled = eligible.iloc[indices].reset_index(drop=True)
    if sampled.duplicated(list(KEY_FIELDS)).any():
        raise ValueError("deterministic sample contains duplicate fundamentals keys")
    return sampled, indices


def _normalise_identifier(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.lstrip("0")
    return text or "0"


def _normalise_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def _normalise_number(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    number = float(numeric)
    if number == 0:
        number = 0.0
    return format(number, ".17g")


def _canonical_row(row: pd.Series) -> dict[str, str]:
    return {
        "gvkey": _normalise_identifier(row["gvkey"]),
        "datadate": _normalise_date(row["datadate"]),
        "rdq": _normalise_date(row["rdq"]),
        "atq": _normalise_number(row["atq"]),
        "ltq": _normalise_number(row["ltq"]),
        "revtq": _normalise_number(row["revtq"]),
        "niq": _normalise_number(row["niq"]),
    }


def _normalise_candidate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"candidate extract is missing columns: {missing}")
    raw = frame.loc[:, REQUIRED_FIELDS].copy()
    normalised = pd.DataFrame(
        {
            "_gvkey": raw["gvkey"].map(_normalise_identifier),
            "_datadate": raw["datadate"].map(_normalise_date),
            "_rdq": raw["rdq"].map(_normalise_date),
            **{
                f"_{field}": pd.to_numeric(raw[field], errors="coerce")
                for field in MATCH_FIELDS[1:]
            },
        },
        index=raw.index,
    )
    return pd.concat([raw, normalised], axis=1)


def _matching_row_hash(local: pd.Series, candidate: pd.DataFrame) -> tuple[bool, str]:
    key = (_normalise_identifier(local["gvkey"]), _normalise_date(local["datadate"]))
    possible = candidate[
        (candidate["_gvkey"] == key[0]) & (candidate["_datadate"] == key[1])
    ]
    if possible.empty:
        return False, ""

    # Copy explicitly: pandas/NumPy may return a read-only view, and in-place
    # &= then fails on some CI wheels with "output array is read-only".
    same = np.array(
        possible["_rdq"].eq(_normalise_date(local["rdq"])),
        dtype=bool,
        copy=True,
    )
    for field in MATCH_FIELDS[1:]:
        left = pd.to_numeric(pd.Series([local[field]]), errors="coerce").iloc[0]
        right = np.asarray(possible[f"_{field}"], dtype=float)
        close = np.isclose(
            right,
            float(left) if not pd.isna(left) else np.nan,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
        same = np.logical_and(same, close)
    matches = possible.loc[same, REQUIRED_FIELDS]
    if matches.empty:
        return False, ""
    hashes = [_sha256_json(_canonical_row(row)) for _, row in matches.iterrows()]
    return True, min(hashes)


def _read_candidate_cache(
    cache_dir: Path,
) -> tuple[dict[TableSpec, pd.DataFrame], list[dict[str, Any]]]:
    candidates_path = cache_dir / "fundq_candidates.csv"
    candidates = pd.read_csv(candidates_path)
    candidates = candidates.assign(
        source_table=(
            candidates["table_schema"].astype(str)
            + "."
            + candidates["table_name"].astype(str)
        )
    )
    frames: dict[TableSpec, pd.DataFrame] = {}
    metadata: list[dict[str, Any]] = []
    for spec in TABLE_SPECS:
        rows = candidates.index[candidates["source_table"] == spec.source_table]
        if len(rows) != 1:
            raise ValueError(
                f"expected one cached candidate for {spec.source_table}, found {len(rows)}"
            )
        index = int(rows[0])
        extract = cache_dir / f"alpha_fundq_fingerprint_{index:02d}.csv.gz"
        profile_path = cache_dir / f"alpha_fundq_profile_{index:02d}.csv.gz"
        if not extract.exists() or not profile_path.exists():
            raise FileNotFoundError(
                f"cached candidate or profile is missing for {spec.source_table}"
            )
        frames[spec] = _normalise_candidate(pd.read_csv(extract))
        profile = pd.read_csv(profile_path).iloc[0]
        metadata.append(
            {
                "source_table": spec.source_table,
                "match_column": spec.match_column,
                "row_hash_column": spec.row_hash_column,
                "extract_file": extract.name,
                "extract_sha256": sha256_file(extract),
                "profile_file": profile_path.name,
                "profile_sha256": sha256_file(profile_path),
                "current_max_datadate": _normalise_date(profile["max_datadate"]),
                "current_max_rdq": _normalise_date(profile["max_rdq"]),
            }
        )
    return frames, metadata


def _date_max(frame: pd.DataFrame, field: str) -> str:
    values = pd.to_datetime(frame[field], errors="coerce")
    if not values.notna().any():
        raise ValueError(f"local fundamentals input has no usable {field} values")
    return values.max().date().isoformat()


def build_sanitized_evidence(
    local: pd.DataFrame,
    candidate_cache: Path,
    *,
    input_kind: str,
    local_input_name: str,
    local_input_sha256: str,
    preserved_local_profile: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build per-sample hashes and booleans from external local-only inputs."""
    sampled, indices = select_local_rows(local)
    candidates, table_metadata = _read_candidate_cache(candidate_cache)
    denominator = max(len(sampled) - 1, 0)
    records: list[dict[str, Any]] = []
    for ordinal, (_, local_row) in enumerate(sampled.iterrows()):
        record: dict[str, Any] = {
            "sample_ordinal": ordinal,
            "selection_input_index": int(indices[ordinal]),
            "selection_fraction": f"{ordinal}/{denominator}",
            "local_row_sha256": _sha256_json(_canonical_row(local_row)),
        }
        for spec in TABLE_SPECS:
            matched, row_hash = _matching_row_hash(local_row, candidates[spec])
            record[spec.match_column] = bool(matched)
            record[spec.row_hash_column] = row_hash
        record["evidence_row_sha256"] = _sha256_json(record)
        records.append(record)

    evidence = pd.DataFrame(records)
    cached_preselection = input_kind == "approved_preselected_cache"
    if cached_preselection and preserved_local_profile is None:
        raise ValueError(
            "a cached preselected local sample requires its preserved aggregate profile"
        )
    local_profile = preserved_local_profile or {
        "latest_datadate": _date_max(local, "datadate"),
        "latest_rdq": _date_max(local, "rdq"),
        "profile_source": "external local parquet",
    }
    metadata = {
        "schema_version": 1,
        "product_family": PRODUCT_FAMILY,
        "historical_source_table": SOURCE_LIMITATION,
        "historical_extract_vintage": VINTAGE_LIMITATION,
        "limitations": [
            (
                "Current aliases and revision histories reproduce the sampled values, "
                "so the exact historical source alias is not uniquely recoverable."
            ),
            (
                "The historical parquet has no query manifest or vintage sidecar, "
                "so its extraction vintage is not recoverable."
            ),
            (
                "The approved cached local extract contains only the selected sample; "
                "the original full row count and source indices are not recoverable."
                if cached_preselection
                else "Hashes establish equality to the supplied local files, not a vendor vintage."
            ),
        ],
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "max_samples": MAX_SAMPLES,
            "input_kind": input_kind,
            "input_rows": int(len(local.dropna(subset=list(KEY_FIELDS)))),
            "sample_rows": int(len(sampled)),
            "historical_full_input_rows": (
                "not recoverable from the approved preselected cache"
                if cached_preselection
                else int(len(local.dropna(subset=list(KEY_FIELDS))))
            ),
            "historical_source_indices": (
                "not recoverable from the approved preselected cache"
                if cached_preselection
                else "recorded as selection_input_index in the sample evidence"
            ),
        },
        "local_input": {
            "file": local_input_name,
            "sha256": local_input_sha256,
            **local_profile,
        },
        "table_evidence": table_metadata,
    }
    validate_sanitized_evidence(evidence, metadata)
    return evidence, metadata


def _table_metadata(metadata: dict[str, Any], source_table: str) -> dict[str, Any]:
    rows = [
        row
        for row in metadata["table_evidence"]
        if row["source_table"] == source_table
    ]
    if len(rows) != 1:
        raise ValueError(f"metadata must contain one profile for {source_table}")
    return rows[0]


def _match_interpretation(spec: TableSpec, count: int, total: int) -> str:
    if count == total:
        return spec.interpretation
    if spec.source_table.endswith("wrds_csq_unrestated"):
        missing = total - count
        labels = {
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six",
            7: "Seven",
            8: "Eight",
            9: "Nine",
            10: "Ten",
        }
        label = labels.get(missing, str(missing))
        return f"{label} sampled local values do not match the current unrevised view"
    return f"{count} of {total} sampled values match {spec.source_table}"


def build_report(
    evidence: pd.DataFrame, metadata: dict[str, Any]
) -> pd.DataFrame:
    """Reconstruct the aggregate historical report from sanitized evidence."""
    evidence = _coerce_evidence(evidence)
    validate_sanitized_evidence(evidence, metadata)
    local_profile = metadata["local_input"]
    ordinary_profile = _table_metadata(metadata, "comp.fundq")
    rows = [
        {
            "evidence": "local_parquet",
            "item": "latest_datadate",
            "value": local_profile["latest_datadate"],
            "interpretation": (
                "Latest fiscal period present in the preserved local quarterly file"
            ),
        },
        {
            "evidence": "local_parquet",
            "item": "latest_rdq",
            "value": local_profile["latest_rdq"],
            "interpretation": (
                "Latest report date present in the preserved local quarterly file"
            ),
        },
        {
            "evidence": "approved_wrds_fingerprint",
            "item": "local_keys_sampled",
            "value": str(len(evidence)),
            "interpretation": "Deterministic keys sampled across the local file",
        },
    ]
    for spec in TABLE_SPECS:
        count = int(evidence[spec.match_column].sum())
        rows.append(
            {
                "evidence": "approved_wrds_fingerprint",
                "item": spec.report_item,
                "value": str(count),
                "interpretation": _match_interpretation(spec, count, len(evidence)),
            }
        )
    rows.extend(
        [
            {
                "evidence": "approved_wrds_profile",
                "item": "comp.fundq_current_max_datadate",
                "value": ordinary_profile["current_max_datadate"],
                "interpretation": (
                    "The current entitled table extends beyond the local extract"
                ),
            },
            {
                "evidence": "approved_wrds_profile",
                "item": "comp.fundq_current_max_rdq",
                "value": ordinary_profile["current_max_rdq"],
                "interpretation": (
                    "The current entitled table extends beyond the local extract"
                ),
            },
            {
                "evidence": "conclusion",
                "item": "product_family",
                "value": metadata["product_family"],
                "interpretation": "Fields and fingerprint identify the product family",
            },
            {
                "evidence": "conclusion",
                "item": "source_table",
                "value": metadata["historical_source_table"],
                "interpretation": (
                    "Current aliases and revision histories reproduce the same sampled values"
                ),
            },
            {
                "evidence": "conclusion",
                "item": "extract_vintage",
                "value": metadata["historical_extract_vintage"],
                "interpretation": (
                    "No query manifest or vintage metadata exists in the historical parquet"
                ),
            },
        ]
    )
    return pd.DataFrame(rows, columns=["evidence", "item", "value", "interpretation"])


def _coerce_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    out = evidence.copy()
    updates: dict[str, pd.Series] = {}
    for field in ("sample_ordinal", "selection_input_index"):
        updates[field] = pd.to_numeric(out[field], errors="raise").astype(int)
    string_fields = [
        "selection_fraction",
        "local_row_sha256",
        "evidence_row_sha256",
        *[spec.row_hash_column for spec in TABLE_SPECS],
    ]
    for field in string_fields:
        updates[field] = out[field].fillna("").astype(str)
    for spec in TABLE_SPECS:
        values = out[spec.match_column]
        if values.dtype != bool:
            mapped = values.astype(str).str.lower().map({"true": True, "false": False})
            if mapped.isna().any():
                raise ValueError(f"{spec.match_column} contains non-boolean values")
            updates[spec.match_column] = mapped.astype(bool)
    return out.assign(**updates)


def _metadata_strings(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _metadata_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _metadata_strings(item)
    elif isinstance(value, str):
        yield value


def validate_sanitized_evidence(
    evidence: pd.DataFrame, metadata: dict[str, Any]
) -> None:
    expected_columns = [
        "sample_ordinal",
        "selection_input_index",
        "selection_fraction",
        "local_row_sha256",
        *[
            column
            for spec in TABLE_SPECS
            for column in (spec.match_column, spec.row_hash_column)
        ],
        "evidence_row_sha256",
    ]
    if list(evidence.columns) != expected_columns:
        raise ValueError("sanitized sample evidence columns are incomplete or reordered")
    forbidden_columns = set(REQUIRED_FIELDS) & set(evidence.columns)
    if forbidden_columns:
        raise ValueError(f"licensed row fields leaked into evidence: {forbidden_columns}")
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported fundamentals provenance evidence schema")
    if metadata.get("product_family") != PRODUCT_FAMILY:
        raise ValueError("product family is missing or changed")
    if metadata.get("historical_source_table") != SOURCE_LIMITATION:
        raise ValueError("historical source limitation is missing or changed")
    if metadata.get("historical_extract_vintage") != VINTAGE_LIMITATION:
        raise ValueError("historical vintage limitation is missing or changed")
    if not metadata.get("limitations"):
        raise ValueError("provenance limitations must be explicit")
    for text in _metadata_strings(metadata):
        if text.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", text):
            raise ValueError("absolute paths are forbidden in sanitized metadata")
        if "select " in text.lower():
            raise ValueError("query text is forbidden in sanitized metadata")

    evidence = _coerce_evidence(evidence)
    selection = metadata["selection"]
    if selection["algorithm"] != SELECTION_ALGORITHM:
        raise ValueError("deterministic selection algorithm is missing or changed")
    if int(selection["max_samples"]) != MAX_SAMPLES:
        raise ValueError("deterministic sample cap is missing or changed")
    if int(selection["sample_rows"]) != len(evidence):
        raise ValueError("sample row count differs from metadata")
    expected_indices = deterministic_indices(int(selection["input_rows"]))
    if evidence["selection_input_index"].tolist() != expected_indices.tolist():
        raise ValueError("selection indices do not reproduce from metadata")
    ordinals = list(range(len(evidence)))
    if evidence["sample_ordinal"].tolist() != ordinals:
        raise ValueError("sample ordinals are not contiguous")
    denominator = max(len(evidence) - 1, 0)
    expected_fractions = [f"{ordinal}/{denominator}" for ordinal in ordinals]
    if evidence["selection_fraction"].tolist() != expected_fractions:
        raise ValueError("selection fractions do not reproduce")
    if evidence["local_row_sha256"].duplicated().any():
        raise ValueError("local row hashes must be unique")

    for row in evidence.to_dict(orient="records"):
        for field in ("local_row_sha256",):
            if not HASH_RE.fullmatch(row[field]):
                raise ValueError(f"invalid hash in {field}")
        for spec in TABLE_SPECS:
            row_hash = row[spec.row_hash_column]
            if bool(row[spec.match_column]) != bool(row_hash):
                raise ValueError(
                    f"{spec.row_hash_column} must be present exactly when the row matches"
                )
            if row_hash and not HASH_RE.fullmatch(row_hash):
                raise ValueError(f"invalid hash in {spec.row_hash_column}")
        recorded = row.pop("evidence_row_sha256")
        if not HASH_RE.fullmatch(recorded) or recorded != _sha256_json(row):
            raise ValueError("sanitized evidence row hash does not verify")

    expected_tables = {spec.source_table for spec in TABLE_SPECS}
    actual_tables = {row["source_table"] for row in metadata["table_evidence"]}
    if actual_tables != expected_tables:
        raise ValueError("table evidence is incomplete or contains unexpected sources")
    for row in metadata["table_evidence"]:
        for field in ("extract_sha256", "profile_sha256"):
            if not HASH_RE.fullmatch(str(row[field])):
                raise ValueError(f"invalid table input hash in {field}")
    local_input = metadata["local_input"]
    if not HASH_RE.fullmatch(str(local_input["sha256"])):
        raise ValueError("invalid local input hash")
    if local_input.get("profile_source") == "preserved sanitized aggregate report":
        if not HASH_RE.fullmatch(str(local_input.get("profile_report_sha256", ""))):
            raise ValueError("invalid preserved aggregate report hash")


def _frame_csv(frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def write_artifacts(
    output_dir: Path, evidence: pd.DataFrame, metadata: dict[str, Any]
) -> None:
    report = build_report(evidence, metadata)
    _atomic_write(output_dir / DEFAULT_EVIDENCE.name, _frame_csv(evidence))
    _atomic_write(
        output_dir / DEFAULT_METADATA.name,
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(output_dir / DEFAULT_REPORT.name, _frame_csv(report))


def check_artifacts(
    evidence_path: Path = DEFAULT_EVIDENCE,
    metadata_path: Path = DEFAULT_METADATA,
    report_path: Path = DEFAULT_REPORT,
) -> pd.DataFrame:
    evidence = _coerce_evidence(pd.read_csv(evidence_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rebuilt = build_report(evidence, metadata)
    expected_text = _frame_csv(rebuilt)
    actual_text = report_path.read_text(encoding="utf-8")
    if actual_text != expected_text:
        raise ValueError(
            "fundamentals provenance report does not reconstruct from sanitized evidence"
        )
    return rebuilt


def _read_local(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=list(REQUIRED_FIELDS))
    return pd.read_csv(path, usecols=list(REQUIRED_FIELDS))


def _preserved_local_profile(report_path: Path) -> dict[str, str]:
    report = pd.read_csv(report_path, dtype=str)
    rows = report.loc[
        report["item"].isin({"latest_datadate", "latest_rdq"}), ["item", "value"]
    ]
    values = rows.set_index("item")["value"].to_dict()
    if set(values) != {"latest_datadate", "latest_rdq"}:
        raise ValueError("preserved report lacks the full local-file date profile")
    for field, value in values.items():
        if _normalise_date(value) != value:
            raise ValueError(f"preserved report contains invalid {field}")
    return {
        **values,
        "profile_source": "preserved sanitized aggregate report",
        "profile_report_file": report_path.name,
        "profile_report_sha256": sha256_file(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--candidate-cache", type=Path)
    local = parser.add_mutually_exclusive_group()
    local.add_argument("--local-parquet", type=Path)
    local.add_argument("--cached-local-sample", type=Path)
    parser.add_argument("--preserved-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    generation_args = (
        args.candidate_cache,
        args.local_parquet,
        args.cached_local_sample,
        args.preserved_report,
        args.output_dir,
    )
    if args.check:
        if any(value is not None for value in generation_args):
            parser.error("--check accepts no generation inputs")
        report = check_artifacts()
        matches = report.loc[
            report["item"].str.endswith("_exact_matches"), "value"
        ].astype(int)
        print(
            f"Verified {len(report)} provenance rows from {MAX_SAMPLES} sanitized "
            f"samples ({matches.min()}-{matches.max()} matches). No vendor or "
            "holdout data was read."
        )
        return 0

    if args.candidate_cache is None or args.output_dir is None:
        parser.error("generation requires --candidate-cache and --output-dir")
    local_path = args.local_parquet or args.cached_local_sample
    if local_path is None:
        parser.error("generation requires --local-parquet or --cached-local-sample")
    if args.cached_local_sample is not None and args.preserved_report is None:
        parser.error("--cached-local-sample requires --preserved-report")
    if args.local_parquet is not None and args.preserved_report is not None:
        parser.error("--preserved-report is only valid with --cached-local-sample")
    input_kind = (
        "external_local_parquet"
        if args.local_parquet is not None
        else "approved_preselected_cache"
    )
    local_frame = _read_local(local_path)
    evidence, metadata = build_sanitized_evidence(
        local_frame,
        args.candidate_cache,
        input_kind=input_kind,
        local_input_name=local_path.name,
        local_input_sha256=sha256_file(local_path),
        preserved_local_profile=(
            _preserved_local_profile(args.preserved_report)
            if args.preserved_report is not None
            else None
        ),
    )
    write_artifacts(args.output_dir, evidence, metadata)
    print(
        f"Wrote {len(evidence)} sanitized samples and reconstructed report to "
        f"{args.output_dir}. No network, model, or holdout data was used."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
