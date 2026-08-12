"""
test_utils.py — Unit tests for shared utilities.

Includes a regression test for iso_to_year_quarter(), which previously used
broken slicing arithmetic (fmt[:10] truncated the format string itself) and
raised ValueError on the full ISO datetime strings the WordPress API returns.
"""

import pytest

from src.utils import iso_to_year_quarter, safe_div, sha256_text, chunks


# ─── iso_to_year_quarter ──────────────────────────────────────────────────────

def test_iso_to_year_quarter_wordpress_datetime():
    """WordPress /wp/v2/posts returns 'YYYY-MM-DDTHH:MM:SS'. This used to raise."""
    assert iso_to_year_quarter("2023-05-01T10:23:00") == (2023, 2)


def test_iso_to_year_quarter_date_only():
    assert iso_to_year_quarter("2023-05-01") == (2023, 2)


@pytest.mark.parametrize(
    "iso_date,expected",
    [
        ("2015-01-01T00:00:00", (2015, 1)),
        ("2019-03-31T23:59:59", (2019, 1)),
        ("2020-04-01T12:00:00", (2020, 2)),
        ("2022-06-30T08:15:00", (2022, 2)),
        ("2023-07-01T09:00:00", (2023, 3)),
        ("2024-09-30T17:45:00", (2024, 3)),
        ("2025-10-01T06:30:00", (2025, 4)),
        ("2026-08-10T23:59:59", (2026, 3)),
    ],
)
def test_iso_to_year_quarter_boundaries(iso_date, expected):
    """Every quarter boundary across the study window 2015-2026."""
    assert iso_to_year_quarter(iso_date) == expected


def test_iso_to_year_quarter_rejects_garbage():
    with pytest.raises(ValueError):
        iso_to_year_quarter("not-a-date")
    with pytest.raises(ValueError):
        iso_to_year_quarter("")


# ─── misc helpers ─────────────────────────────────────────────────────────────

def test_sha256_text_is_stable_and_distinct():
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("world")
    assert len(sha256_text("hello")) == 64


def test_sha256_text_handles_unicode():
    # Corpus contains accented characters; must not raise.
    assert len(sha256_text("Café résumé naïve")) == 64


def test_chunks():
    assert list(chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunks([], 3)) == []


def test_safe_div_zero_denominator():
    assert safe_div(1, 0) == 0.0
    assert safe_div(1, 0, default=-1) == -1
