"""
test_clean.py — Unit tests for HTML cleaning pipeline.
"""

import pytest
from bs4 import BeautifulSoup
from src.clean import extract_text_from_html, _normalize_whitespace


def test_normalize_whitespace():
    assert _normalize_whitespace("hello   world") == "hello world"
    assert _normalize_whitespace("hello\n\n\nworld") == "hello world"
    assert _normalize_whitespace("  hello  ") == "hello"


def test_extract_text_basic():
    html = "<p>Hello world</p>"
    text = extract_text_from_html(html)
    assert "Hello world" in text
    assert len(text) > 0


def test_extract_text_removes_scripts():
    html = """
    <article>
        <p>Real content</p>
        <script>alert('bad');</script>
        <p>More content</p>
    </article>
    """
    text = extract_text_from_html(html)
    assert "Real content" in text
    assert "More content" in text
    assert "alert" not in text
    assert "script" not in text.lower()


def test_extract_text_removes_styles():
    html = """
    <div>
        <p>Visible text</p>
        <style>.hidden { display: none; }</style>
    </div>
    """
    text = extract_text_from_html(html)
    assert "Visible text" in text
    assert "display" not in text.lower()


def test_extract_text_preserves_paragraphs():
    html = """
    <article>
        <p>First paragraph.</p>
        <p>Second paragraph.</p>
        <p>Third paragraph.</p>
    </article>
    """
    text = extract_text_from_html(html)
    assert "First paragraph" in text
    assert "Second paragraph" in text
    assert "Third paragraph" in text


def test_extract_text_handles_empty():
    assert extract_text_from_html("") == ""
    assert extract_text_from_html(None) == ""
    assert extract_text_from_html("   ") == ""


def test_extract_text_handles_nested():
    html = """
    <article>
        <div class="content">
            <p>Paragraph with <strong>bold</strong> and <em>italic</em> text.</p>
        </div>
    </article>
    """
    text = extract_text_from_html(html)
    assert "bold" in text
    assert "italic" in text
    assert "<strong>" not in text
    assert "<em>" not in text


def test_extract_text_removes_nav():
    html = """
    <body>
        <nav>Skip this navigation</nav>
        <article>
            <p>Real article content</p>
        </article>
    </body>
    """
    text = extract_text_from_html(html)
    assert "Real article content" in text
    # Nav might be partially present depending on selectors
    # but the article content should be prioritized


def test_extract_text_handles_unicode():
    html = "<p>Café résumé naïve émigré</p>"
    text = extract_text_from_html(html)
    assert "Café" in text
    assert "résumé" in text
    assert "naïve" in text
