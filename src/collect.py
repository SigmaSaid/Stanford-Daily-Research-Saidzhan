"""
collect.py — Multi-method scraper for The Stanford Daily.

Collection hierarchy:
  Level 1: WordPress REST API  (primary — fastest, most structured)
  Level 2: Category archive pages  (fallback if API blocked/incomplete)
  Level 3: Individual article HTML  (always used to enrich content when needed)

Features:
  - Resumable: skips already-collected article IDs
  - Rate-limited with configurable delays
  - Retries on transient HTTP errors
  - Full HTML preserved in raw JSONL
  - Structured logging
  - Date-filtered to START_DATE … END_DATE

Run from the project root (modules use absolute `src.` imports, so they must
be invoked with -m; `python src/collect.py` will raise ModuleNotFoundError):
  python -m src.collect                    # collect both Opinions + News
  python -m src.collect --corpus opinions  # opinions only
  python -m src.collect --corpus news      # news only
  python -m src.collect --reset-cache      # ignore existing data (re-fetch)
"""

import argparse
import json
import re
import sys
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.config import (
    BASE_URL,
    POSTS_ENDPOINT,
    START_DATE,
    END_DATE,
    API_PER_PAGE,
    MAX_API_PAGES,
    RAW_JSONL,
    FAILED_LOG,
    OPINION_CATEGORY_NAMES,
    NEWS_CATEGORY_NAMES,
    REQUEST_TIMEOUT_SECONDS,
)
from src.metadata import discover_targets
from src.utils import (
    build_session,
    polite_get,
    append_jsonl,
    load_existing_ids,
    get_logger,
)

logger = get_logger(__name__)

# ISO-8601 boundaries for WordPress ?after= / ?before= params
_START_ISO = f"{START_DATE}T00:00:00"
_END_ISO   = f"{END_DATE}T23:59:59"


# ─── Level 1: WordPress REST API ─────────────────────────────────────────────

def fetch_posts_api(
    session,
    category_ids: list[int],
    existing_ids: set[int],
    corpus_label: str,
) -> int:
    """
    Paginate through the WP REST API and save every post that:
      - falls within START_DATE … END_DATE
      - belongs to one of category_ids
      - is not already saved

    Returns the number of newly saved articles.
    """
    saved = 0

    for page in range(1, MAX_API_PAGES + 1):
        params = {
            "per_page": API_PER_PAGE,
            "page":     page,
            "after":    _START_ISO,
            "before":   _END_ISO,
            "orderby":  "date",
            "order":    "asc",
            "_fields":  (
                "id,date,modified,link,title,content,excerpt,"
                "author,categories,tags,status"
            ),
        }
        if category_ids:
            params["categories"] = ",".join(str(i) for i in category_ids)

        resp = polite_get(session, POSTS_ENDPOINT, params=params, logger=logger)
        if resp is None:
            logger.warning("[%s] API request failed — page %d", corpus_label, page)
            break

        # X-WP-TotalPages header tells us the real page count
        total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
        logger.info("[%s] API page %d/%d", corpus_label, page, total_pages)

        posts = resp.json()
        if not posts:
            logger.info("[%s] Empty response on page %d — stopping", corpus_label, page)
            break

        for post in posts:
            pid = post.get("id")
            if pid in existing_ids:
                continue

            record = _api_post_to_record(post, corpus_label)
            append_jsonl(RAW_JSONL, record)
            existing_ids.add(pid)
            saved += 1

        if page >= total_pages:
            break

    logger.info("[%s] API collection complete — %d new articles saved", corpus_label, saved)
    return saved


def _api_post_to_record(post: dict, corpus_label: str) -> dict:
    """Convert a raw WP REST API post dict to our storage schema."""
    content_rendered = (post.get("content") or {}).get("rendered", "")
    title_rendered   = (post.get("title")   or {}).get("rendered", "")
    excerpt_rendered = (post.get("excerpt") or {}).get("rendered", "")

    # Categories come as a list of IDs in the REST API
    cat_ids  = post.get("categories", [])
    tag_ids  = post.get("tags", [])
    author_id = post.get("author")

    return {
        "id":            post.get("id"),
        "title":         _strip_html(title_rendered),
        "date":          post.get("date", ""),
        "modified":      post.get("modified", ""),
        "url":           post.get("link", ""),
        "author_id":     author_id,
        "author_name":   None,   # Enriched later via /wp/v2/users/{id}
        "category_ids":  cat_ids,
        "category_names": [],    # Resolved in clean.py using categories.json
        "tag_ids":       tag_ids,
        "content_html":  content_rendered,
        "excerpt_html":  excerpt_rendered,
        "corpus":        corpus_label,
        "source":        "stanforddaily.com",
        "collection_method": "wp_rest_api",
    }


# ─── Level 2: Archive-page fallback ──────────────────────────────────────────

def fetch_posts_archive(
    session,
    category_slug: str,
    existing_ids: set[int],
    corpus_label: str,
) -> int:
    """
    Walk paginated category archive pages and collect article URLs, then
    fetch each individual page to extract content.
    Returns number of new articles saved.
    """
    saved = 0
    page  = 1
    base_archive = f"{BASE_URL}/category/{category_slug}/"

    while True:
        url = base_archive if page == 1 else f"{base_archive}page/{page}/"
        resp = polite_get(session, url, logger=logger)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "lxml")

        # Extract article links from archive page
        article_links = _extract_archive_links(soup)
        if not article_links:
            logger.info("[%s] No article links found on page %d — stopping", corpus_label, page)
            break

        logger.info("[%s] Archive page %d — %d links", corpus_label, page, len(article_links))

        for href in article_links:
            # Attempt to parse date from URL slug (TSD uses YYYY/MM/DD/slug)
            url_date = _date_from_url(href)
            if url_date and not _in_date_range(url_date):
                continue

            # Build a synthetic ID from URL for dedup purposes
            url_id = _url_to_int_id(href)
            if url_id in existing_ids:
                continue

            record = fetch_single_article(session, href, corpus_label)
            if record is None:
                _log_failed(href, "fetch returned None")
                continue

            # Re-check date from content
            if record.get("date"):
                if not _in_date_range(record["date"][:10]):
                    continue

            append_jsonl(RAW_JSONL, record)
            existing_ids.add(url_id)
            saved += 1

        page += 1
        # Safety: don't crawl indefinitely
        if page > 500:
            logger.warning("[%s] Hit 500-page archive safety limit", corpus_label)
            break

    logger.info("[%s] Archive fallback complete — %d new articles saved", corpus_label, saved)
    return saved


def _extract_archive_links(soup: BeautifulSoup) -> list[str]:
    """Extract article URLs from a TSD category archive page."""
    links = []
    seen  = set()

    def _accept(raw_href: str) -> str | None:
        """Normalise an href to an absolute stanforddaily.com article URL."""
        if not raw_href:
            return None
        # WordPress themes emit either absolute or root-relative URLs.
        # urljoin handles both; without it, relative hrefs were silently dropped.
        absolute = urljoin(BASE_URL, raw_href)
        if "stanforddaily.com" not in absolute:
            return None
        return absolute

    # Common WordPress archive patterns
    for selector in [
        "h2.entry-title a",
        "h3.entry-title a",
        "article a[rel='bookmark']",
        ".post-title a",
        "h2 a[href*='stanforddaily.com']",
    ]:
        for tag in soup.select(selector):
            href = _accept(tag.get("href", ""))
            if href and href not in seen:
                links.append(href)
                seen.add(href)
        if links:
            return links

    # Generic fallback: all internal article-like links
    for tag in soup.find_all("a", href=True):
        href = _accept(tag["href"])
        if href and re.search(r"stanforddaily\.com/\d{4}/\d{2}/\d{2}/", href):
            if href not in seen:
                links.append(href)
                seen.add(href)

    return links


# ─── Level 3: Individual article fetch ───────────────────────────────────────

def fetch_single_article(session, url: str, corpus_label: str) -> dict | None:
    """
    Fetch one article page and return a raw record.
    The HTML is preserved; cleaning happens in clean.py.
    """
    resp = polite_get(session, url, logger=logger, timeout=REQUEST_TIMEOUT_SECONDS)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Title
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title:
        title = og_title.get("content", "")
    elif soup.find("h1", class_=re.compile(r"entry-title|post-title", re.I)):
        title = soup.find("h1", class_=re.compile(r"entry-title|post-title", re.I)).get_text(" ", strip=True)
    elif soup.title:
        title = soup.title.string or ""

    # Date
    date_str = ""
    time_tag = soup.find("time", class_=re.compile(r"entry-date|published|post-date", re.I))
    if time_tag:
        date_str = time_tag.get("datetime", "") or time_tag.get_text(strip=True)
    else:
        # Try meta tags
        for prop in ["article:published_time", "og:updated_time", "datePublished"]:
            meta = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
            if meta:
                date_str = meta.get("content", "")
                break
        if not date_str:
            date_str = _date_from_url(url) or ""

    # Author
    author_name = ""
    author_tag = soup.find(class_=re.compile(r"author|byline", re.I))
    if author_tag:
        a = author_tag.find("a")
        author_name = (a or author_tag).get_text(" ", strip=True)

    # Category
    cat_names: list[str] = []
    for cat_link in soup.select(".cat-links a, .post-categories a, [rel='category tag']"):
        cat_names.append(cat_link.get_text(strip=True))

    # Full page HTML for content extraction in clean.py
    article_html = str(soup.find("article") or soup.find("main") or soup)

    url_int_id = _url_to_int_id(url)

    return {
        "id":            url_int_id,
        "title":         title,
        "date":          date_str,
        "modified":      "",
        "url":           url,
        "author_id":     None,
        "author_name":   author_name,
        "category_ids":  [],
        "category_names": cat_names,
        "tag_ids":       [],
        "content_html":  article_html,
        "excerpt_html":  "",
        "corpus":        corpus_label,
        "source":        "stanforddaily.com",
        "collection_method": "html_fallback",
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """Quick utility: strip HTML tags and decode entities."""
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(" ", strip=True)


def _date_from_url(url: str) -> str | None:
    """Extract YYYY-MM-DD from a TSD URL like /2019/03/15/slug/."""
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _in_date_range(date_str: str) -> bool:
    """Return True if date_str (YYYY-MM-DD) falls within START_DATE…END_DATE."""
    return START_DATE <= date_str[:10] <= END_DATE


def _url_to_int_id(url: str) -> int:
    """Derive a stable integer ID from a URL hash (for HTML-fallback articles)."""
    import hashlib
    return int(hashlib.md5(url.encode()).hexdigest(), 16) % (10**10)


def _log_failed(url: str, reason: str) -> None:
    """Append a failed URL to the failed log."""
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"url": url, "reason": reason}) + "\n")


# ─── Author enrichment ───────────────────────────────────────────────────────

def enrich_author_names(session, records: list[dict]) -> dict[int, str]:
    """
    Fetch display names for author IDs found in records.
    Returns {author_id: display_name}.

    The cache is persisted to disk so that resuming a collection does not
    re-request every author from the API on each run.
    """
    from src.config import API_BASE, AUTHOR_CACHE

    cache: dict[int, str] = {}
    if AUTHOR_CACHE.exists():
        try:
            with AUTHOR_CACHE.open("r", encoding="utf-8") as fh:
                cache = {int(k): v for k, v in json.load(fh).items()}
            logger.info("Loaded %d cached author names", len(cache))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Author cache unreadable — refetching")
            cache = {}

    unknown_ids = sorted(
        {r["author_id"] for r in records
         if r.get("author_id") and r["author_id"] not in cache}
    )
    logger.info("Author names to fetch: %d", len(unknown_ids))

    # Many WordPress sites (including The Stanford Daily) restrict
    # /wp/v2/users/{id} to authenticated callers and return 401 for every
    # request. Without a circuit breaker this loop grinds through thousands of
    # guaranteed failures. Give up early and continue with author_id only.
    AUTH_FAIL_CODES = {401, 403}
    MAX_CONSECUTIVE_FAILS = 10
    consecutive_fails = 0
    aborted = False

    for aid in unknown_ids:
        url = f"{API_BASE}/users/{aid}"
        resp = polite_get(session, url, logger=logger)

        if resp is not None and resp.status_code == 200:
            cache[aid] = resp.json().get("name", "")
            consecutive_fails = 0
            continue

        cache[aid] = ""
        status = resp.status_code if resp is not None else None
        if status in AUTH_FAIL_CODES or status is None:
            consecutive_fails += 1
        else:
            consecutive_fails = 0

        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            logger.warning(
                "Aborting author enrichment after %d consecutive failures "
                "(last status: %s). The users endpoint appears to require "
                "authentication. Articles keep their numeric author_id, which "
                "is sufficient for author-concentration controls; author_name "
                "will be empty.",
                consecutive_fails, status,
            )
            aborted = True
            break

    if aborted:
        logger.warning("Author names unavailable — continuing with author_id only.")

    AUTHOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with AUTHOR_CACHE.open("w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in cache.items()}, fh,
                  ensure_ascii=False, indent=2)
    logger.info("Saved author cache → %s", AUTHOR_CACHE)

    return cache


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_collection(corpus: str = "both", reset_cache: bool = False) -> None:
    """
    Main collection routine.

    corpus: "opinions" | "news" | "both"
    """
    if reset_cache and RAW_JSONL.exists():
        logger.warning("--reset-cache set: removing existing JSONL")
        RAW_JSONL.unlink()

    # Discover category IDs dynamically
    meta   = discover_targets()
    id_map = meta["id_map"]

    targets = []
    if corpus in ("opinions", "both"):
        targets.append(("opinions", meta["opinion_ids"]))
    if corpus in ("news", "both"):
        targets.append(("news", meta["news_ids"]))

    if not targets:
        logger.error("No targets configured. Check OPINION_CATEGORY_NAMES / NEWS_CATEGORY_NAMES in config.py")
        sys.exit(1)

    session     = build_session()
    existing_ids = load_existing_ids(RAW_JSONL)
    logger.info("Existing articles in JSONL: %d", len(existing_ids))

    total_new = 0

    for corpus_label, cat_ids in targets:
        if not cat_ids:
            logger.warning("[%s] No category IDs found — trying archive fallback", corpus_label)
            # Use first matching category slug as archive URL
            slug_candidates = (OPINION_CATEGORY_NAMES if corpus_label == "opinions"
                               else NEWS_CATEGORY_NAMES)
            for slug in slug_candidates:
                n = fetch_posts_archive(session, slug, existing_ids, corpus_label)
                total_new += n
                if n > 0:
                    break
        else:
            logger.info("[%s] Using category IDs: %s", corpus_label, cat_ids)
            n = fetch_posts_api(session, cat_ids, existing_ids, corpus_label)
            total_new += n

            # If the API returned suspiciously few results, run archive as supplement
            if n < 10:
                logger.info("[%s] API returned few results — supplementing with archive", corpus_label)
                for cat_id in cat_ids:
                    slug = id_map.get(cat_id, "")
                    if slug:
                        n2 = fetch_posts_archive(session, slug, existing_ids, corpus_label)
                        total_new += n2

    # Enrich author names.
    #
    # Author IDs are read by streaming the file, and the rewrite streams line by
    # line into a temp file that is then atomically swapped in. A 12-year corpus
    # with full HTML preserved can be several GB, so the previous approach of
    # holding every record in memory at once was not safe at real-data scale.
    logger.info("Enriching author names …")

    author_ids: set[int] = set()
    with RAW_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("author_id"):
                author_ids.add(rec["author_id"])

    author_cache = enrich_author_names(
        session, [{"author_id": aid} for aid in author_ids]
    )

    tmp_path = RAW_JSONL.with_suffix(".jsonl.tmp")
    total = 0
    with RAW_JSONL.open("r", encoding="utf-8") as src_fh, \
         tmp_path.open("w", encoding="utf-8") as dst_fh:
        for line in src_fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line during rewrite")
                continue
            aid = rec.get("author_id")
            if aid and not rec.get("author_name") and aid in author_cache:
                rec["author_name"] = author_cache[aid]
            dst_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1

    tmp_path.replace(RAW_JSONL)

    logger.info("Collection finished. Total articles in JSONL: %d  (new this run: %d)",
                total, total_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape The Stanford Daily")
    parser.add_argument(
        "--corpus", choices=["opinions", "news", "both"], default="both",
        help="Which corpus to collect (default: both)",
    )
    parser.add_argument(
        "--reset-cache", action="store_true",
        help="Ignore existing JSONL and re-collect from scratch",
    )
    args = parser.parse_args()
    run_collection(corpus=args.corpus, reset_cache=args.reset_cache)