"""Freeze evaluation inputs, then record consumption before reading outcomes.

These files are audit controls, not protection against a user deleting files.
A failed evaluation consumes its window too; recovery requires an explicit
review of the saved record rather than silently trying different settings.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_snapshot(files: dict[str, Path], settings: dict) -> dict:
    return {"schema": 1, "settings": settings,
            "files": {name: file_hash(path) for name, path in sorted(files.items())}}


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def claim_evaluation(manifest: Path, record: Path, actual: dict) -> None:
    if record.exists():
        raise ValueError("Evaluation already consumed. Use published derived tables; do not retune and rescore.")
    if not manifest.is_file():
        raise ValueError("Freeze the complete evaluation manifest before reading the holdout.")
    expected = json.loads(manifest.read_text())
    if expected != actual:
        raise ValueError("Evaluation inputs or settings differ from the frozen manifest.")
    write_exclusive(record, {"status": "consumed_before_read",
                            "manifest_sha256": file_hash(manifest),
                            "consumed_at_utc": datetime.now(timezone.utc).isoformat()})
