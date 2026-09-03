"""Tests for the random-grammar null.

The null is only a null if it draws from the same grammar the LLM sets are
constrained to. These tests pin that: every sample passes the sanitizer, arity
is respected, the signature map cannot drift away from the sanitizer's tables,
and the same seed gives the same draw.
"""
from __future__ import annotations

import ast
import re

import pytest

from quantaalpha_us.factors.expression_sanitizer import ExpressionSanitizer
from quantaalpha_us.factors.random_expressions import (
    PRICE_FIELDS,
    SIGNATURES,
    GrammarSampler,
    expression_structure,
    sample_random_set,
    structure_profile,
)

REFERENCE = [
    "TS_DELTA($close, 21) / (TS_STD($close, 21) + 1e-8)",
    "TS_STD($return, 21)",
    "CS_RANK(TS_DELTA($close, 21)) * CS_RANK(TS_MEAN($dollar_volume, 21))",
    "-TS_CORR($return, $volume, 10)",
    "($high - $low) / ($close + 1e-8)",
    "TS_MEAN(CS_RANK(TS_CORR($close, $volume, 5)), 10)",
]


def _sampler(seed: int = 0) -> GrammarSampler:
    return GrammarSampler(seed=seed, profile=structure_profile(REFERENCE))


def test_every_sample_passes_the_sanitizer():
    sanitizer = ExpressionSanitizer()
    sampler = _sampler()
    for _ in range(300):
        result = sanitizer.sanitize(sampler.sample())
        assert result.valid, result.errors


def test_arity_is_respected_for_every_call():
    """Parsed the way the sanitizer parses, so this checks the real thing."""
    sampler = _sampler(1)
    for _ in range(300):
        expr = sampler.sample()
        tree = ast.parse(re.sub(r"\$([A-Za-z_]\w*)", r"field_\1", expr), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id.upper()
            minimum = ExpressionSanitizer.VARIADIC_MIN_ARITY.get(name)
            if minimum is not None:
                assert len(node.args) >= minimum, expr
            else:
                assert len(node.args) == ExpressionSanitizer.FUNCTION_ARITY[name], expr


def test_signature_map_covers_the_sanitizer_exactly():
    """Adding a function to the sanitizer without describing it here must fail.

    Otherwise the null quietly stops sampling part of the grammar the LLM is
    allowed to use, and the comparison tilts without anyone noticing.
    """
    table = set(ExpressionSanitizer.FUNCTION_ARITY) | set(ExpressionSanitizer.VARIADIC_MIN_ARITY)
    assert set(SIGNATURES) == table


def test_fields_are_a_subset_of_known_fields():
    assert set(PRICE_FIELDS) <= ExpressionSanitizer.KNOWN_FIELDS


def test_unknown_field_is_rejected():
    with pytest.raises(ValueError):
        GrammarSampler(seed=0, profile=structure_profile(REFERENCE), fields=("not_a_field",))


def test_time_series_windows_are_positive_integer_literals():
    """The evaluator requires a literal window; a panel or float there is fatal."""
    sampler = _sampler(2)
    windowed = {n for n, sig in SIGNATURES.items() if sig.endswith("W")}
    for _ in range(300):
        expr = sampler.sample()
        tree = ast.parse(re.sub(r"\$([A-Za-z_]\w*)", r"field_\1", expr), mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.func.id.upper() in windowed:
                window = node.args[-1]
                assert isinstance(window, ast.Constant), ast.dump(window)
                assert isinstance(window.value, int) and window.value > 0


def test_same_seed_gives_the_same_draw():
    a = sample_random_set(seed=7, n=25, reference=REFERENCE)
    b = sample_random_set(seed=7, n=25, reference=REFERENCE)
    assert a == b


def test_different_seeds_differ():
    a = sample_random_set(seed=1, n=25, reference=REFERENCE)
    b = sample_random_set(seed=2, n=25, reference=REFERENCE)
    assert a != b


def test_samples_are_distinct():
    drawn = sample_random_set(seed=3, n=40, reference=REFERENCE)
    assert len(set(drawn)) == 40


def test_structure_matches_the_reference_profile():
    """The null must be a search of comparable size and shape, not a strawman."""
    profile = structure_profile(REFERENCE)
    drawn = sample_random_set(seed=4, n=60, reference=REFERENCE)
    calls = [expression_structure(e)[0] for e in drawn]
    depths = [expression_structure(e)[1] for e in drawn]
    assert min(calls) >= min(profile.call_counts)
    assert max(calls) <= max(profile.call_counts)
    # mean call count within one call of the reference set's
    assert abs(sum(calls) / len(calls) - profile.mean_calls) <= 1.0
    assert max(depths) <= ExpressionSanitizer().max_nesting_depth


def test_nesting_stays_under_the_sanitizer_cap():
    sanitizer = ExpressionSanitizer()
    sampler = _sampler(5)
    for _ in range(200):
        _, depth = sanitizer._max_nesting(sampler.sample())
        assert depth <= sanitizer.max_nesting_depth


def test_profile_of_empty_set_is_an_error():
    with pytest.raises(ValueError):
        structure_profile([])
