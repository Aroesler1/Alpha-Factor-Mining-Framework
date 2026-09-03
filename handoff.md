# Handoff

## Current goal

Make the repo's relationship to its upstream, QuantaAlpha, accurate and visible.

## Verified state (2026-09-03)

- `README.md` has a `## Provenance` section near the top: upstream repo, paper
  (arXiv:2602.07085), license, kept-vs-rebuilt breakdown, IC framing, and the
  upstream BibTeX under `### Citation`.
- `docs/PROVENANCE.md` holds the file-level table.
- Committed and pushed as `f1def2c` on `main`.
- `pytest tests/ -q` → **81 passed** (docs-only change; same as the pre-change baseline).

## How the provenance table was produced

All 27 `.py` files in `quantaalpha_us/` compared against all 159 `.py` files in the
upstream package tree (4,293 pairs). Both sides normalised: comments and blank lines
stripped, whitespace collapsed, package names rewritten so naming alone could not
depress the score. Similarity = `difflib.SequenceMatcher` ratio.

Result: **max similarity 0.173 anywhere**; 27 original, 0 adapted, 0 shared. The four
files sharing a basename with upstream share only imports, `@dataclass`, walk-forward
window field names, and three risk-config defaults.

## Known risks / caveats

- **The upstream baseline was not pristine.** `../QuantaAlpha_CN` is a local working
  copy carrying an earlier rebrand (39 files with substituted project names) and has no
  git history. Name substitutions were normalised away, and the finding is one of
  *absence* of overlap, which a rebrand cannot manufacture — but the table has not been
  checked against an untouched upstream checkout. `docs/PROVENANCE.md` discloses this
  and gives the re-verification command.
- **`../QuantaAlpha_CN` itself is an attribution problem, and was left untouched** (out
  of scope for this task). Its `README.md` presents upstream's work under a different
  project name with a different maintainer, and its BibTeX block has the paper title
  replaced while keeping upstream's authors and arXiv id. Its `pyproject.toml` names a
  different author and homepage. If that tree was ever published, it needs fixing.
- **The MIT license claim is unverified offline.** No `LICENSE` file exists in the local
  upstream copy; the MIT classifier in its `pyproject.toml` is in a file that was
  rebranded, so it is not independent evidence. Confirm against upstream directly.
- **This repo has no `LICENSE` file** of its own.
- Stale strings elsewhere in `README.md`: it says the repo is maintained at
  `Aroesler1/LLMStrat` (actual remote is `Aroesler1/Alpha-Factor-Mining-Framework`) and
  refers to a `QuantaAlpha_US` directory.

## Paper facts (from the local PDF, primary source)

Upstream headline: **IC 0.1501** on CSI 300 (GPT-5.2), ARR 27.75%, MDD 7.98%. Zero-shot
transfer of CSI-300-mined factors to the S&P 500 is reported as **successful** — ~137%
cumulative excess return over the 2022-01-01–2025-12-26 window (Figure 1) — but the paper
publishes **no S&P 500 IC**. This repo's own ICs are 0.003–0.011.

## Next action

Re-run the comparison against a pristine upstream clone and, if the numbers hold, drop
the baseline caveat from `docs/PROVENANCE.md`. Separately, decide what to do about
`../QuantaAlpha_CN`.
