#!/usr/bin/env python3
"""Compare this repository's code and prompts against an upstream QuantaAlpha checkout.

Used for the re-check in docs/PROVENANCE.md. Reads text only and never imports
or runs anything from the upstream tree.

    git clone --depth 1 https://github.com/QuantaAlpha/QuantaAlpha /tmp/qa_upstream
    python scripts/provenance_upstream_compare.py --upstream /tmp/qa_upstream

For every Python file under quantaalpha_us/, scripts/ and tests/, and every
prompt template under configs/prompts/, it reports the best line-sequence
similarity against any upstream file (difflib ratio after normalisation), the
longest run of consecutive shared normalised lines against that best match, and
the non-trivial lines that also occur anywhere upstream.

Normalisation: comments, docstrings and blank lines dropped, whitespace
collapsed, `quantaalpha_us` rewritten to `quantaalpha` so a package rename
cannot hide a match. Imports, decorators and other short boilerplate lines are
excluded from the shared-line count.
"""
from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# lines too generic to count as shared code
TRIVIAL = re.compile(
    r"^(from __future__ import annotations|import \w+( as \w+)?|from [\w.]+ import [\w, ()*]+|"
    r"@dataclass.*|\)|\]|\}|\(|return|pass|else:|try:|finally:|.{0,24})$"
)
UPSTREAM_TEXT = {".yaml", ".yml", ".txt", ".md", ".j2", ".jinja", ".jinja2"}


def normalised_lines(path: Path) -> list[str]:
    out, in_doc = [], False
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if path.suffix == ".py":
            if line.startswith(('"""', "'''")):
                quote = line[:3]
                # a one-line docstring opens and closes on the same line
                if not (len(line) > 3 and line.endswith(quote)):
                    in_doc = not in_doc
                continue
            if in_doc or line.startswith("#"):
                continue
            line = re.sub(r"\s+#.*$", "", line)
        line = re.sub(r"\s+", " ", line).replace("quantaalpha_us", "quantaalpha")
        if line:
            out.append(line)
    return out


def collect(root: Path, parts: list[str], suffixes: set[str]) -> list[Path]:
    found: list[Path] = []
    for part in parts:
        base = root / part
        if base.exists():
            found += [p for p in base.rglob("*") if p.is_file() and p.suffix in suffixes
                      and "node_modules" not in p.parts and ".git" not in p.parts]
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", required=True, help="path to an upstream checkout")
    parser.add_argument("--show", type=int, default=15, help="rows to print, highest first")
    args = parser.parse_args()
    upstream = Path(args.upstream)

    local = (collect(ROOT, ["quantaalpha_us", "scripts", "tests"], {".py"})
             + collect(ROOT, ["configs/prompts"], {".txt"}))
    up_py = collect(upstream, ["."], {".py"})
    up_text = collect(upstream, ["."], UPSTREAM_TEXT)
    norm = {p: normalised_lines(p) for p in up_py + up_text}
    upstream_lines = {line for lines in norm.values() for line in lines
                      if not TRIVIAL.match(line)}

    rows = []
    for path in local:
        lines = normalised_lines(path)
        pool = up_py if path.suffix == ".py" else up_text
        best, best_path = 0.0, None
        for other in pool:
            if not lines or not norm[other]:
                continue
            matcher = difflib.SequenceMatcher(None, lines, norm[other], autojunk=False)
            if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
                continue
            ratio = matcher.ratio()
            if ratio > best:
                best, best_path = ratio, other
        run = 0
        if best_path is not None:
            matcher = difflib.SequenceMatcher(None, lines, norm[best_path], autojunk=False)
            run = matcher.find_longest_match(0, len(lines), 0, len(norm[best_path])).size
        shared = [line for line in lines if not TRIVIAL.match(line) and line in upstream_lines]
        rows.append((best, path.relative_to(ROOT), best_path, len(lines), run, shared))

    rows.sort(key=lambda r: -r[0])
    print(f"local files: {len(local)}  upstream python: {len(up_py)}  upstream text: {len(up_text)}")
    print(f"max similarity: {rows[0][0]:.3f} ({rows[0][1]})")
    print(f"longest shared run of normalised lines: {max(r[4] for r in rows)}")
    print(f"files with any shared non-trivial line: {sum(1 for r in rows if r[5])}\n")
    for best, path, other, nloc, run, shared in rows[:args.show]:
        target = other.relative_to(upstream) if other else "-"
        print(f"{best:.3f}  {str(path):58s} nloc={nloc:4d} run={run:2d}  best={target}")
        for line in shared[:3]:
            print(f"         shared: {line[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
