"""
test_quality.py — Unit tests for data quality checks.
"""

import pytest
import pandas as pd
from src.quality import check_duplicates, check_missing


def test_check_missing():
    df = pd.DataFrame({
        "article_id": [1, 2, 3, 4],
        "title": ["A", "", "C", None],
        "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        "text": ["text1", "text2", "", "text4"],
        "author_name": ["Author1", "Author2", "Author3", "Author4"],
        "year": [2020, 2020, 2020, 2020],
        "article_type": ["opinion", "news", "opinion", "news"],
    })

    result = check_missing(df)
    assert result["missing_title"] == 2  # empty string + None
    assert result["missing_text"] == 1   # empty string
    assert result["missing_date"] == 0


def test_check_duplicates_by_id():
    df = pd.DataFrame({
        "article_id": [1, 1, 2, 3],
        "url": ["url1", "url1", "url2", "url3"],
        "text_hash": ["h1", "h1", "h2", "h3"],
        "title": ["T1", "T1", "T2", "T3"],
        "text": ["text1", "text1", "text2", "text3"],
    })

    clean, removed = check_duplicates(df)
    assert len(clean) == 3
    assert len(removed) == 1
    assert removed.iloc[0]["removal_reason"] == "duplicate_article_id"


def test_check_duplicates_by_url():
    df = pd.DataFrame({
        "article_id": [1, 2, 3],
        "url": ["url1", "url1", "url2"],
        "text_hash": ["h1", "h2", "h3"],
        "title": ["T1", "T2", "T3"],
        "text": ["text1", "text2", "text3"],
    })

    clean, removed = check_duplicates(df)
    assert len(clean) == 2  # One duplicate URL removed
    assert len(removed) == 1


def test_check_duplicates_by_text_hash():
    df = pd.DataFrame({
        "article_id": [1, 2, 3],
        "url": ["url1", "url2", "url3"],
        "text_hash": ["h1", "h1", "h2"],
        "title": ["T1", "T2", "T3"],
        "text": ["same text", "same text", "different"],
    })

    clean, removed = check_duplicates(df)
    assert len(clean) == 2  # One duplicate text removed
    assert len(removed) == 1


def test_check_duplicates_no_duplicates():
    df = pd.DataFrame({
        "article_id": [1, 2, 3],
        "url": ["url1", "url2", "url3"],
        "text_hash": ["h1", "h2", "h3"],
        "title": ["T1", "T2", "T3"],
        "text": ["text1", "text2", "text3"],
    })

    clean, removed = check_duplicates(df)
    assert len(clean) == 3
    assert len(removed) == 0
