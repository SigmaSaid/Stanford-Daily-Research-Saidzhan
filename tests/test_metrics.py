"""
test_metrics.py — Unit tests for linguistic metrics.
"""

import pytest
from src.linguistic_metrics import mattr, hapax_ratio, func_word_ratio
from src.utils import safe_div


def test_safe_div():
    assert safe_div(10, 2) == 5.0
    assert safe_div(10, 0, default=0) == 0
    assert safe_div(10, 0, default=-1) == -1


def test_mattr_basic():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    result = mattr(tokens, window=3)
    assert 0 < result <= 1.0


def test_mattr_short_text():
    tokens = ["hello", "world"]
    result = mattr(tokens, window=5)  # window > len
    assert result == 1.0  # 2 unique / 2 total


def test_mattr_repetitive():
    tokens = ["word"] * 100
    result = mattr(tokens, window=10)
    # 1 unique / 10 per window. MATTR averages 91 overlapping windows, and 1/10
    # is not exactly representable in IEEE-754, so compare with a tolerance
    # rather than exact equality.
    assert result == pytest.approx(0.1)


def test_hapax_ratio():
    tokens = ["a", "b", "c", "c", "d", "d", "d"]
    # Unique: a, b, c, d (4 types)
    # Hapax: a, b (2 words appearing once)
    # Ratio: 2/4 = 0.5
    result = hapax_ratio(tokens)
    assert result == 0.5


def test_hapax_ratio_all_unique():
    tokens = ["a", "b", "c", "d"]
    result = hapax_ratio(tokens)
    assert result == 1.0


def test_hapax_ratio_no_hapax():
    tokens = ["a", "a", "b", "b"]
    result = hapax_ratio(tokens)
    assert result == 0.0


def test_func_word_ratio():
    tokens = ["the", "cat", "is", "on", "the", "mat"]
    # Function words: the, is, on, the (4 out of 6)
    result = func_word_ratio(tokens)
    assert 0.6 < result < 0.7


def test_func_word_ratio_no_function_words():
    tokens = ["apple", "banana", "cherry"]
    result = func_word_ratio(tokens)
    assert result == 0.0


def test_func_word_ratio_all_function_words():
    tokens = ["the", "a", "is", "on"]
    result = func_word_ratio(tokens)
    assert result == 1.0


def test_mattr_empty():
    tokens = []
    result = mattr(tokens, window=10)
    assert result == 0.0


# ─── Regression: spaCy Token attribute vs method ──────────────────────────────

def test_compute_metrics_runs_on_real_text():
    """
    Regression test for `TypeError: 'int' object is not callable`.

    compute_metrics() used `token.lower()`, but on a spaCy Token `.lower` is an
    int hash ID, not a method. The string form is `.lower_`. This crashed on the
    first article of any real run.
    """
    from src.linguistic_metrics import compute_metrics

    text = (
        "The University announced a new policy this week. Students responded "
        "with mixed reactions across campus. Café conversations turned heated, "
        "and many argued the decision was premature."
    )
    result = compute_metrics(text)

    assert result, "compute_metrics returned empty dict"
    for key in ("mattr", "ttr", "hapax_ratio", "func_word_ratio", "sentence_count"):
        assert key in result, f"missing metric: {key}"

    assert 0.0 < result["mattr"] <= 1.0
    assert 0.0 < result["ttr"] <= 1.0
    assert result["sentence_count"] >= 3
    assert result["token_count"] > 0


def test_compute_metrics_lowercases_tokens():
    """Tokens must be case-folded, so 'The' and 'the' collapse to one type."""
    from src.linguistic_metrics import compute_metrics

    upper = compute_metrics("The cat sat. THE CAT SAT. the cat sat.")
    # 3 unique types (the, cat, sat) out of 9 tokens if lowercasing works.
    assert upper["unique_word_count"] == 3
