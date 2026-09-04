"""The four proposal operators and the reflection call: prompts in, candidates out.

Each operator is a template in `configs/prompts/evo/` plus a JSON schema, so the
model's reply is a typed object rather than prose to be regex'd. Templates use
`<<KEY>>` placeholders rather than `{}` or `$` -- the DSL block is full of both
`$close` and JSON braces, and a formatter that treats either as syntax will
either crash or silently mangle a prompt.

Batching. Each operator sends ONE call per island per round carrying its whole
batch, because the schema takes a list. That is four calls per island per round
plus one reflection, so a three-island run costs about 15 calls a round and
about 120 for a full eight-round arm -- the difference between a run that fits
in a subscription and one that does not.

Ordering discipline for replay: everything a prompt contains is derived from the
archive, the population and the memory in a deterministic order. A dict iterated
in insertion order somewhere here would make two runs with the same seed produce
different prompts, and the replay test would catch it as a mismatch rather than
as a bug -- which is why it is a test.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from quantaalpha_us.evo.config import EvoConfig
from quantaalpha_us.factors import ast_tools as at

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "configs" / "prompts" / "evo"

EXPLORE, MUTATE, CROSSOVER, SIMPLIFY, REFLECT = (
    "explore", "mutate", "crossover", "simplify", "reflect"
)
PROPOSAL_OPERATORS = (EXPLORE, MUTATE, CROSSOVER, SIMPLIFY)

CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "rationale": {"type": "string"},
                    "parent_ids": {"type": "array", "items": {"type": "string"}},
                    "changed_node": {"type": "string"},
                },
                "required": ["expression", "rationale", "parent_ids", "changed_node"],
            },
        }
    },
    "required": ["candidates"],
}

REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"rules": {"type": "array", "items": {"type": "string"}}},
    "required": ["rules"],
}


@dataclass
class OperatorRequest:
    """One call: what to send, what shape to expect back, and what it was for."""

    operator: str
    island: str
    round: int
    prompt: str
    schema: dict
    n: int
    parents: tuple[str, ...] = ()
    parent_pairs: tuple[tuple[str, str], ...] = ()

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:16]


@dataclass
class Proposal:
    """One candidate as the model returned it, before any gate has looked at it."""

    expression: str
    rationale: str = ""
    parent_ids: tuple[str, ...] = ()
    changed_node: str = ""
    operator: str = ""
    island: str = ""
    round: int = 0

    def to_dict(self) -> dict:
        return {
            "expression": self.expression, "rationale": self.rationale,
            "parent_ids": list(self.parent_ids), "changed_node": self.changed_node,
            "operator": self.operator, "island": self.island, "round": self.round,
        }


@dataclass
class PromptLibrary:
    """The templates, read once."""

    directory: Path = PROMPT_DIR
    _cache: dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> str:
        if name not in self._cache:
            self._cache[name] = (self.directory / f"{name}.txt").read_text(encoding="utf-8")
        return self._cache[name]


def fill(template: str, values: dict[str, Any]) -> str:
    """`<<KEY>>` substitution. Unfilled keys are an error, not a silent blank."""
    out = template
    for key, value in values.items():
        out = out.replace(f"<<{key}>>", str(value))
    leftover = [seg.split(">>")[0] for seg in out.split("<<")[1:] if ">>" in seg]
    if leftover:
        raise KeyError(f"unfilled prompt placeholders: {sorted(set(leftover))}")
    return out


def render_dsl(config: EvoConfig, library: PromptLibrary) -> str:
    """The shared preamble, with the gate limits and the fit window filled in.

    The window years come from the config rather than being written into the
    template, so a run configured on a shorter window cannot advertise a longer
    one -- and so the prompt audit can assert "no year later than the configured
    fit end" instead of hard-coding 2013 in two places.
    """
    gates = config.gates
    return fill(library.get("_dsl"), {
        "FIT_START_YEAR": config.windows.fit_start[:4],
        "FIT_END_YEAR": config.windows.fit_end[:4],
        "MAX_SYMBOL_LENGTH": gates.max_symbol_length,
        "MAX_BASE_FIELDS": gates.max_base_fields,
        "MAX_FREE_CONSTANTS": gates.max_free_constants,
        "MAX_DEPTH": gates.max_depth,
        "MAX_SHARED_SUBTREE": gates.max_shared_subtree,
        "MIN_COVERAGE": f"{gates.min_coverage:.0%}",
        "MAX_ABS_CORR": f"{gates.max_abs_corr:.2f}",
    })


# --------------------------------------------------------------------------
# the ban list
# --------------------------------------------------------------------------


def alpha101_ban_list(expressions: Sequence[str], *, min_size: int = 3,
                      min_alphas: int = 2, top: int = 12) -> list[str]:
    """Sub-expressions the published set leans on, as text to avoid.

    Counted by how many DISTINCT alphas contain each canonical subtree, not by
    total occurrences: a pattern repeated five times inside one formula is that
    formula's habit, while one appearing in five formulas is the literature's.
    """
    holders: Counter[str] = Counter()
    representative: dict[str, at.Node] = {}
    sizes: dict[str, int] = {}
    for expr in expressions:
        try:
            tree = at.parse(expr)
        except at.ParseError:
            continue
        seen: set[str] = set()
        for sub in tree:
            size = at.size(sub)
            if size < min_size:
                continue
            key = at.subtree_hash(sub)
            representative.setdefault(key, sub)
            sizes[key] = size
            seen.add(key)
        holders.update(seen)
    ranked = sorted(
        (k for k, c in holders.items() if c >= min_alphas),
        key=lambda k: (-holders[k], -sizes[k], at.canonical_text(representative[k])),
    )
    return [
        f"  {at.to_expression(representative[k])}   (in {holders[k]} published alphas)"
        for k in ranked[:top]
    ]


# --------------------------------------------------------------------------
# request builders
# --------------------------------------------------------------------------


def build_explore(config: EvoConfig, library: PromptLibrary, *, island: str,
                  direction: str, round_index: int, niche_summary: Sequence[dict],
                  ban_list: Sequence[str], memory_block: str,
                  n: int | None = None) -> OperatorRequest:
    n = n if n is not None else config.schedule.explore_n
    rows = ["  niche                                   count  best fitness  capacity"]
    for row in niche_summary:
        best = "-" if row["best_fitness"] is None else f"{row['best_fitness']:.3f}"
        rows.append(f"  {row['niche']:<38}{row['count']:>6}{best:>14}{row['capacity']:>10}")
    prompt = fill(library.get("explore"), {
        "DSL": render_dsl(config, library),
        "N": n,
        "DIRECTION": direction,
        "NICHE_SUMMARY": "\n".join(rows),
        "BAN_LIST": "\n".join(ban_list) or "  (none yet)",
        "MEMORY": memory_block,
    })
    return OperatorRequest(EXPLORE, island, round_index, prompt, CANDIDATE_SCHEMA, n)


def build_mutate(config: EvoConfig, library: PromptLibrary, *, island: str,
                 round_index: int, parent_blocks: Sequence[tuple[str, str]],
                 memory_block: str) -> OperatorRequest:
    """`parent_blocks` is (parent expression, rendered feedback block).

    The guidance paragraph tracks `feedback_mode`. In the ablation arm the
    parents carry a fitness scalar and nothing else, so telling the model to
    "read the diagnostics" would be instructions for information it does not
    have -- which would make the ablation measure prompt confusion rather than
    the absence of feedback.
    """
    if config.feedback_mode == "scalar":
        guidance = (
            "Each parent is shown with its fitness and nothing else. Fitness is a "
            "single scalar in which higher is better; it already accounts for "
            "significance, year-to-year stability, complexity and turnover. You are "
            "not given per-year, coverage, turnover or horizon diagnostics in this "
            "run, so choose the component to change from the expression itself."
        )
    else:
        guidance = (
            "Read the diagnostics: a factor whose IC is positive in most years and "
            "negative in two has a regime problem, one with high turnover is paying "
            "costs it does not earn back, and one whose IC half-life is short is "
            "being asked to predict too far ahead. Change the component responsible "
            "for the specific weakness you name."
        )
    body = "\n\n".join(block for _, block in parent_blocks)
    prompt = fill(library.get("mutate"), {
        "DSL": render_dsl(config, library),
        "N": len(parent_blocks),
        "MAX_EDITS": config.gates.max_mutation_edits,
        "GUIDANCE": guidance,
        "PARENTS": body,
        "MEMORY": memory_block,
    })
    return OperatorRequest(MUTATE, island, round_index, prompt, CANDIDATE_SCHEMA,
                           len(parent_blocks),
                           parents=tuple(expr for expr, _ in parent_blocks))


def build_crossover(config: EvoConfig, library: PromptLibrary, *, island: str,
                    round_index: int,
                    pairs: Sequence[tuple[str, str, str, str]]) -> OperatorRequest:
    """`pairs` is (left id, left expression, right id, right expression)."""
    body = []
    for i, (lid, left, rid, right) in enumerate(pairs, start=1):
        body.append(f"pair {i}\n  A [{lid}]  {left}\n  B [{rid}]  {right}")
    prompt = fill(library.get("crossover"), {
        "DSL": render_dsl(config, library),
        "N": len(pairs),
        "MIN_SHARED": config.gates.min_crossover_shared,
        "PAIRS": "\n\n".join(body),
    })
    return OperatorRequest(CROSSOVER, island, round_index, prompt, CANDIDATE_SCHEMA,
                           len(pairs),
                           parent_pairs=tuple((left, right) for _, left, _, right in pairs))


def build_simplify(config: EvoConfig, library: PromptLibrary, *, island: str,
                   round_index: int,
                   parent_blocks: Sequence[tuple[str, str]]) -> OperatorRequest:
    prompt = fill(library.get("simplify"), {
        "DSL": render_dsl(config, library),
        "N": len(parent_blocks),
        "MIN_IC_SHARE": f"{config.gates.simplify_min_ic_share:.0%}",
        "PARENTS": "\n\n".join(block for _, block in parent_blocks),
    })
    return OperatorRequest(SIMPLIFY, island, round_index, prompt, CANDIDATE_SCHEMA,
                           len(parent_blocks),
                           parents=tuple(expr for expr, _ in parent_blocks))


def build_reflection(config: EvoConfig, library: PromptLibrary, *, island: str,
                     round_index: int, accepted: Sequence[str],
                     rejected: Sequence[str]) -> OperatorRequest:
    prompt = fill(library.get("reflect"), {
        "FIT_START_YEAR": config.windows.fit_start[:4],
        "FIT_END_YEAR": config.windows.fit_end[:4],
        "ACCEPTED": "\n".join(accepted) or "  (nothing was accepted this round)",
        "REJECTED": "\n".join(rejected) or "  (nothing was rejected this round)",
        "MAX_RULES": config.schedule.memory_ask_rules,
    })
    return OperatorRequest(REFLECT, island, round_index, prompt, REFLECTION_SCHEMA,
                           config.schedule.memory_ask_rules)


# --------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------


def parse_proposals(payload: Any, request: OperatorRequest) -> list[Proposal]:
    """Typed candidates from a reply. Malformed items are dropped, not guessed at."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return []
    items = payload.get("candidates")
    if not isinstance(items, list):
        return []
    out: list[Proposal] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        expression = item.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            continue
        parents = item.get("parent_ids")
        if not isinstance(parents, list):
            parents = []
        out.append(Proposal(
            expression=expression.strip(),
            rationale=str(item.get("rationale", "")).strip(),
            parent_ids=tuple(str(p) for p in parents),
            changed_node=str(item.get("changed_node", "")).strip(),
            operator=request.operator, island=request.island, round=request.round,
        ))
    return out


def parse_rules(payload: Any, limit: int) -> list[str]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []
    out = []
    for rule in rules[:limit]:
        text = " ".join(str(rule).split())
        if text:
            out.append(text)
    return out
