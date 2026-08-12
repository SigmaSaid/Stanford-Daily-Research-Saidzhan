"""
metadata.py — Discover and cache category/author metadata from the Stanford Daily API.

Run this module once before collect.py so that category IDs are known.
Results are cached to data/raw/categories.json.
"""

import json
from pathlib import Path
from typing import Optional

from src.config import (
    CATS_ENDPOINT,
    CATEGORY_CACHE,
    OPINION_CATEGORY_NAMES,
    NEWS_CATEGORY_NAMES,
)
from src.utils import build_session, polite_get, get_logger

logger = get_logger(__name__)


def fetch_all_categories(session) -> list[dict]:
    """Fetch every category page from the WP REST API and return the full list."""
    cats = []
    page = 1
    while True:
        resp = polite_get(
            session, CATS_ENDPOINT,
            params={"per_page": 100, "page": page},
            logger=logger,
        )
        if resp is None:
            logger.warning("Category fetch failed on page %d", page)
            break
        batch = resp.json()
        if not batch:
            break
        cats.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return cats


def load_or_fetch_categories(force: bool = False) -> list[dict]:
    """
    Return cached categories, or fetch them from the API if the cache is absent
    or force=True.
    """
    if CATEGORY_CACHE.exists() and not force:
        with CATEGORY_CACHE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded %d categories from cache", len(data))
        return data

    logger.info("Fetching categories from API …")
    session = build_session()
    cats = fetch_all_categories(session)
    logger.info("Fetched %d categories total", len(cats))

    CATEGORY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CATEGORY_CACHE.open("w", encoding="utf-8") as fh:
        json.dump(cats, fh, ensure_ascii=False, indent=2)
    logger.info("Saved category cache → %s", CATEGORY_CACHE)
    return cats


def find_category_ids(
    categories: list[dict],
    target_names: list[str],
) -> list[int]:
    """
    Return all category IDs whose slug or name (case-insensitive) matches any
    string in target_names.
    """
    target_set = {n.lower() for n in target_names}
    matched = []
    for cat in categories:
        slug = (cat.get("slug") or "").lower()
        name = (cat.get("name") or "").lower()
        if slug in target_set or name in target_set:
            matched.append(cat["id"])
            logger.info("  Matched category: id=%d  slug=%r  name=%r",
                        cat["id"], cat.get("slug"), cat.get("name"))
    return matched


def build_category_map(categories: list[dict]) -> dict[int, str]:
    """Return {id: slug} mapping for all categories."""
    return {c["id"]: c.get("slug", "") for c in categories}


def discover_targets() -> dict:
    """
    Convenience function: fetch/load categories and return a dict with
    opinion_ids, news_ids, and a full id→slug map.
    """
    cats = load_or_fetch_categories()
    opinion_ids = find_category_ids(cats, OPINION_CATEGORY_NAMES)
    news_ids    = find_category_ids(cats, NEWS_CATEGORY_NAMES)
    id_map      = build_category_map(cats)

    logger.info("Opinion category IDs: %s", opinion_ids)
    logger.info("News category IDs:    %s", news_ids)

    return {
        "categories":   cats,
        "opinion_ids":  opinion_ids,
        "news_ids":     news_ids,
        "id_map":       id_map,
    }


if __name__ == "__main__":
    result = discover_targets()
    print(f"\nOpinion IDs : {result['opinion_ids']}")
    print(f"News IDs    : {result['news_ids']}")
    print(f"Total cats  : {len(result['categories'])}")
