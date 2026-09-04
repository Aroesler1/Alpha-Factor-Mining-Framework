"""Recover the alpha NUMBER of each transcribed formula in configs/alpha101_us.txt.

The file is a flat list of expressions with the 51 dropped alphas recorded in
place as `# alphaNNN: DROPPED -- reason`. That is the right format for a
candidate file and a useless one for anything that needs to say WHICH published
alpha a mined factor resembles: "shares 7 nodes with `-1 * TS_DELTA(...)`" is
not a citation, and pasting the formula back to a model that is being tested for
paraphrasing is worse than useless.

So this reconstructs the numbering from the file's own structure -- section
headers give the range, drop comments consume a number, expressions take the
next one -- and a test asserts the reconstruction covers 1..101 exactly once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# quantaalpha_us/factors/ -> quantaalpha_us/ -> the repo root, where configs/ lives
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "configs" / "alpha101_us.txt"

_SECTION = re.compile(r"^#\s*---\s*(\d+)-(\d+)\s*-+\s*$")
_DROPPED = re.compile(r"^#\s*alpha(\d+):\s*DROPPED", re.IGNORECASE)


@dataclass(frozen=True)
class Alpha:
    number: int
    name: str
    expression: str


def load(path: Path | str = DEFAULT_PATH) -> list[Alpha]:
    """The transcribed alphas, in file order, with their published numbers."""
    cursor: int | None = None
    out: list[Alpha] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        section = _SECTION.match(line)
        if section:
            cursor = int(section.group(1))
            continue
        dropped = _DROPPED.match(line)
        if dropped:
            number = int(dropped.group(1))
            # the drop comment names its own alpha, so it is also the check that
            # the cursor has not drifted
            if cursor is not None and cursor != number:
                raise ValueError(
                    f"alpha numbering drifted at {line[:60]!r}: expected alpha{cursor:03d}"
                )
            cursor = number + 1
            continue
        if line.startswith("#"):
            continue
        if cursor is None:
            raise ValueError("expression found before any section header")
        out.append(Alpha(number=cursor, name=f"alpha{cursor:03d}", expression=line))
        cursor += 1
    return out


def dropped_numbers(path: Path | str = DEFAULT_PATH) -> list[int]:
    return [int(_DROPPED.match(line.strip()).group(1))
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if _DROPPED.match(line.strip())]


def names_by_expression(path: Path | str = DEFAULT_PATH) -> dict[str, str]:
    return {a.expression: a.name for a in load(path)}
