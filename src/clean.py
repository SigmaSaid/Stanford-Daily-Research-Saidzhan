"""
clean.py — HTML → plain text cleaning pipeline for Stanford Daily articles.

Reads: data/raw/articles_raw.jsonl
Writes: data/processed/articles.csv

Design principles:
  - Preserves raw HTML (never modifies JSONL)
  - Uses actual TSD HTML structure (discovered by inspecting the site)
  - Removes: scripts, styles, nav, ads, social embeds, captions, author bios,
              recommendation blocks, sidebar content
  - Preserves: article paragraphs, quotations, headings
  - Reproducible: every run with the same input produces the same output
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment

from src.config import (
    RAW_JSONL,
    PROCESSED_CSV,
    CATEGORY_CACHE,
    MIN_WORD_COUNT,
    MAX_WORD_COUNT,
)
from src.utils import load_jsonl, get_logger, safe_div, sha256_text

logger = get_logger(__name__)

# ─── Selectors to REMOVE entirely ─────────────────────────────────────────────
# Based on inspection of stanforddaily.com WordPress HTML structure.
# Tag names or CSS-selector strings.
REMOVE_TAGS = [
    "script", "style", "noscript", "iframe",
    "nav", "header", "footer",
    "aside",
]

REMOVE_SELECTORS = [
    # WordPress / TSD structural chrome
    ".widget", ".sidebar", ".site-header", ".site-footer",
    ".navigation", ".nav-links", ".post-navigation",
    # Social / sharing
    ".sharedaddy", ".sd-block", ".jetpack-sharing-buttons",
    "[class*='share']", "[class*='social']",
    # Ads / promos
    ".ad", ".advertisement", "[class*='banner']",
    # Author bio blocks
    ".author-bio", ".author-box", ".author-description",
    "[class*='author-info']",
    # Related / recommended
    ".related-posts", "[class*='recommended']", "[class*='more-stories']",
    # Subscription / newsletter prompts
    "[class*='subscribe']", "[class*='newsletter']",
    # Image captions (optional — comment out to preserve)
    ".wp-caption-text", "figcaption",
    # Comment section
    "#comments", ".comments-area",
    # Breadcrumbs
    ".breadcrumbs", ".breadcrumb",
    # Tags / categories meta at bottom
    ".entry-footer", ".post-tags", ".cat-links",
    # "Read more" / pagination within article
    ".more-link", ".page-links",
    # Any element with aria-label='advertisement'
    "[aria-label='advertisement']",
]

# ─── Content selectors (try in priority order) ────────────────────────────────
CONTENT_SELECTORS = [
    ".entry-content",           # Standard WordPress
    ".post-content",
    ".article-content",
    "[itemprop='articleBody']",
    "article .content",
    "article",
    "main",
]


# ─── HTML cleaner ─────────────────────────────────────────────────────────────

def extract_text_from_html(html: str) -> str:
    """
    Given raw HTML (from WP REST API 'content.rendered' or a full page),
    return clean plain text suitable for linguistic analysis.
    """
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove structural noise tags
    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove by CSS selector
    for selector in REMOVE_SELECTORS:
        try:
            for tag in soup.select(selector):
                tag.decompose()
        except Exception:
            pass

    # Try to isolate the article body
    body = None
    for selector in CONTENT_SELECTORS:
        body = soup.select_one(selector)
        if body:
            break

    target = body if body else soup

    # Extract text: paragraph by paragraph to preserve sentence boundaries
    paragraphs = []
    for elem in target.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]):
        text = elem.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)

    # If no paragraphs found, fall back to full get_text
    if not paragraphs:
        raw_text = target.get_text(" ", strip=True)
    else:
        raw_text = " ".join(paragraphs)

    return _normalize_whitespace(raw_text)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─── Category resolution ──────────────────────────────────────────────────────

def load_category_map() -> dict[int, str]:
    """Return {id: slug} from cached categories.json."""
    if not CATEGORY_CACHE.exists():
        logger.warning("No category cache found at %s", CATEGORY_CACHE)
        return {}
    import json
    with CATEGORY_CACHE.open("r", encoding="utf-8") as fh:
        cats = json.load(fh)
    return {c["id"]: c.get("slug", "") for c in cats}


def resolve_article_type(category_names: list[str], category_slugs: list[str]) -> str:
    """
    Map article categories to a coarse article_type label.
    Uses both human-readable names and slugs.
    """
    all_labels = {s.lower() for s in category_names + category_slugs}

    if any(k in all_labels for k in ["letters", "letter", "community"]):
        return "letter"
    if any(k in all_labels for k in ["editorial", "editorials"]):
        return "editorial"
    if any(k in all_labels for k in ["op-ed", "op-eds", "oped"]):
        return "op-ed"
    if any(k in all_labels for k in ["column", "columnists", "columnist"]):
        return "column"
    if any(k in all_labels for k in ["opinion", "opinions"]):
        return "opinion"
    if any(k in all_labels for k in ["news", "university", "local", "national", "world"]):
        return "news"
    return "other"


# ─── Row builder ─────────────────────────────────────────────────────────────

def build_row(record: dict, cat_map: dict[int, str]) -> dict | None:
    """
    Convert a raw JSONL record into a processed row.
    Returns None if the article should be skipped.
    """
    pid   = record.get("id")
    title = (record.get("title") or "").strip()
    url   = record.get("url", "")
    date  = (record.get("date") or "")[:10]  # YYYY-MM-DD

    # Basic date validation
    if not date or len(date) < 10:
        logger.debug("Skipping %s — invalid date %r", pid, date)
        return None

    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        logger.debug("Skipping %s — cannot parse date %r", pid, date)
        return None

    # Clean text
    html  = record.get("content_html", "")
    text  = extract_text_from_html(html)
    words = text.split()
    wc    = len(words)

    if wc < MIN_WORD_COUNT:
        logger.debug("Skipping %s — too short (%d words)", pid, wc)
        return None
    if wc > MAX_WORD_COUNT:
        logger.debug("Skipping %s — too long (%d words)", pid, wc)
        return None

    # Resolve category slugs
    cat_ids   = record.get("category_ids", [])
    cat_slugs = [cat_map.get(cid, "") for cid in cat_ids if cid in cat_map]
    cat_names = record.get("category_names") or []

    # Prefer stored names; fall back to slugs
    display_cats = cat_names if cat_names else cat_slugs
    category     = ", ".join(display_cats) if display_cats else record.get("corpus", "unknown")

    article_type = resolve_article_type(cat_names, cat_slugs)

    year    = dt.year
    month   = dt.month
    quarter = (month - 1) // 3 + 1

    text_hash = sha256_text(text)

    return {
        "article_id":    pid,
        "title":         title,
        "date":          date,
        "year":          year,
        "month":         month,
        "quarter":       quarter,
        "author_id":     record.get("author_id"),
        "author_name":   (record.get("author_name") or "").strip(),
        "category":      category,
        "article_type":  article_type,
        "corpus":        record.get("corpus", "unknown"),
        "url":           url,
        "text":          text,
        "word_count":    wc,
        "char_count":    len(text),
        "text_hash":     text_hash,
        "collection_method": record.get("collection_method", ""),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_cleaning() -> pd.DataFrame:
    """
    Load raw JSONL, clean each article, and save processed CSV.
    Returns the resulting DataFrame.
    """
    records = load_jsonl(RAW_JSONL)
    logger.info("Loaded %d raw records", len(records))

    if not records:
        logger.error("No raw records found. Run collect.py first.")
        sys.exit(1)

    cat_map = load_category_map()

    rows = []
    skipped = 0
    for rec in records:
        row = build_row(rec, cat_map)
        if row is None:
            skipped += 1
        else:
            rows.append(row)

    logger.info("Cleaned %d articles; skipped %d", len(rows), skipped)

    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False, encoding="utf-8")
    logger.info("Saved processed CSV → %s  (%d rows)", PROCESSED_CSV, len(df))

    return df


def inspect_samples(df: pd.DataFrame, n: int = 5) -> None:
    """
    Print a sample of articles from key years for manual QA.
    This is a human-readable check — not automated.
    """
    years_of_interest = [2015, 2020, 2022, 2023, 2025, 2026]
    print("\n" + "=" * 70)
    print("MANUAL INSPECTION SAMPLE")
    print("=" * 70)

    for yr in years_of_interest:
        subset = df[df["year"] == yr]
        if subset.empty:
            print(f"\n--- {yr}: NO ARTICLES ---")
            continue

        sample = subset.sample(min(n, len(subset)), random_state=42)
        print(f"\n--- {yr} ({len(subset)} articles total) ---")
        for _, row in sample.iterrows():
            print(f"  Title   : {row['title'][:80]}")
            print(f"  Date    : {row['date']}")
            print(f"  Author  : {row['author_name']}")
            print(f"  Category: {row['category']}")
            print(f"  Words   : {row['word_count']}")
            print(f"  Preview : {row['text'][:300]} …")
            print()


if __name__ == "__main__":
    df = run_cleaning()
    inspect_samples(df)
