"""
quality.py — Data quality checks and reporting.

Reads: data/processed/articles.csv
Writes:
  reports/data_quality_report.csv
  reports/data_quality_report.md
  data/processed/articles_deduplicated.csv
  data/processed/dedup_removed.csv
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_CSV,
    DEDUPED_CSV,
    DEDUP_REMOVED_CSV,
    REPORTS_DIR,
    MIN_WORD_COUNT,
    MAX_WORD_COUNT,
)
from src.utils import get_logger, sha256_text

logger = get_logger(__name__)

QC_CSV = REPORTS_DIR / "data_quality_report.csv"
QC_MD  = REPORTS_DIR / "data_quality_report.md"


# ─── Checks ───────────────────────────────────────────────────────────────────

def check_missing(df: pd.DataFrame) -> dict:
    cols = ["title", "date", "author_name", "text", "article_type", "year"]
    missing = {}
    for col in cols:
        if col in df.columns:
            n = df[col].isna().sum() + (df[col] == "").sum()
            missing[f"missing_{col}"] = int(n)
    return missing


def check_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify duplicates by (a) article_id, (b) url, (c) text_hash.
    Returns (clean_df, removed_df).
    Logs everything removed before dropping.
    """
    original_len = len(df)
    removed_parts = []

    # (a) Duplicate article_id
    dup_id = df[df.duplicated("article_id", keep="first")]
    if not dup_id.empty:
        logger.info("Removing %d duplicate article_id rows", len(dup_id))
        removed_parts.append(dup_id.assign(removal_reason="duplicate_article_id"))
    df = df.drop_duplicates("article_id", keep="first")

    # (b) Duplicate URL
    dup_url = df[df.duplicated("url", keep="first")]
    if not dup_url.empty:
        logger.info("Removing %d duplicate URL rows", len(dup_url))
        removed_parts.append(dup_url.assign(removal_reason="duplicate_url"))
    df = df.drop_duplicates("url", keep="first")

    # (c) Duplicate text hash (identical cleaned text)
    dup_text = df[df.duplicated("text_hash", keep="first")]
    if not dup_text.empty:
        logger.info("Removing %d exact-text-duplicate rows", len(dup_text))
        removed_parts.append(dup_text.assign(removal_reason="duplicate_text_hash"))
    df = df.drop_duplicates("text_hash", keep="first")

    removed = pd.concat(removed_parts, ignore_index=True) if removed_parts else pd.DataFrame()
    logger.info("Deduplication: %d → %d rows (%d removed)",
                original_len, len(df), original_len - len(df))
    return df, removed


def check_lengths(df: pd.DataFrame) -> dict:
    return {
        "articles_below_min_words": int((df["word_count"] < MIN_WORD_COUNT).sum()),
        "articles_above_max_words": int((df["word_count"] > MAX_WORD_COUNT).sum()),
        "articles_empty_text":      int((df["text"].isna() | (df["text"] == "")).sum()),
        "median_word_count":        float(df["word_count"].median()),
        "mean_word_count":          float(df["word_count"].mean()),
    }


def yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-year article counts, authors, and word-count stats."""
    rows = []
    for yr in sorted(df["year"].dropna().unique()):
        sub  = df[df["year"] == yr]

        # Author *names* come from /wp/v2/users/{id}, which many WordPress
        # installations (including The Stanford Daily) restrict to
        # authenticated callers. When that endpoint returns 401 the name column
        # is empty and counting it yields a misleading 0. The numeric author_id
        # is always present in the posts payload, so distinct-author counts are
        # based on it.
        n_author_ids = int(sub["author_id"].nunique()) if "author_id" in sub else 0
        n_author_names = (
            int(sub["author_name"].replace("", pd.NA).nunique())
            if "author_name" in sub else 0
        )

        rows.append({
            "year":                int(yr),
            "n_articles":          len(sub),
            "n_opinions":          int((sub["corpus"] == "opinions").sum()),
            "n_news":              int((sub["corpus"] == "news").sum()),
            "n_unique_author_ids": n_author_ids,
            "n_resolved_author_names": n_author_names,
            "mean_word_count":     round(sub["word_count"].mean(), 1),
            "median_word_count":   round(sub["word_count"].median(), 1),
            "total_words":         int(sub["word_count"].sum()),
        })
    return pd.DataFrame(rows)


def category_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return article counts by category/corpus."""
    return (df.groupby(["corpus", "article_type"])
              .size()
              .reset_index(name="n_articles")
              .sort_values(["corpus", "n_articles"], ascending=[True, False]))


def author_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return top authors by article count."""
    return (df.groupby("author_name")
              .agg(n_articles=("article_id", "count"),
                   mean_words=("word_count", "mean"),
                   years_active=("year", lambda x: f"{int(x.min())}–{int(x.max())}"))
              .sort_values("n_articles", ascending=False)
              .reset_index()
              .head(50))


# ─── Report generator ─────────────────────────────────────────────────────────

def run_quality_checks(df: pd.DataFrame) -> dict:
    """Run all checks and return a summary dict."""
    summary = {"total_articles": len(df)}
    summary.update(check_missing(df))
    summary.update(check_lengths(df))
    return summary


def write_reports(df: pd.DataFrame, summary: dict, ystats: pd.DataFrame) -> None:
    """Write CSV and Markdown quality reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(QC_CSV, index=False)
    logger.info("Saved quality CSV → %s", QC_CSV)

    # Markdown
    lines = [
        "# Data Quality Report\n",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        "\n## Summary\n",
        f"| Metric | Value |",
        f"| --- | --- |",
    ]
    for k, v in summary.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "\n## Articles by Year\n",
        ystats.to_markdown(index=False),
        "\n\n## Notes\n",
        "- 2026 is a partial year (YTD). Do not compare directly to full years.",
        "- 'opinions' corpus = Opinions/Op-Ed/Editorial/Column/Letter as tagged by TSD.",
        "- 'news' corpus = News/University/Local/National/World categories.",
        "- Word count filters: min={}, max={}.".format(MIN_WORD_COUNT, MAX_WORD_COUNT),
    ]

    with QC_MD.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Saved quality Markdown → %s", QC_MD)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_quality_pipeline() -> pd.DataFrame:
    """Load, check, deduplicate, save."""
    if not PROCESSED_CSV.exists():
        logger.error("No processed CSV found. Run clean.py first.")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_CSV, low_memory=False)
    logger.info("Loaded %d articles from %s", len(df), PROCESSED_CSV)

    # Ensure text_hash exists
    if "text_hash" not in df.columns:
        df["text_hash"] = df["text"].fillna("").apply(sha256_text)

    # Run checks on full (pre-dedup) set
    summary = run_quality_checks(df)

    # Deduplicate
    df_clean, removed = check_duplicates(df)

    # Yearly corpus statistics are computed on the DEDUPLICATED corpus, because
    # that is the corpus every downstream module actually analyses. Computing
    # them pre-dedup would over-count articles in Figure 01.
    ystats = yearly_stats(df_clean)
    summary["articles_after_dedup"] = len(df_clean)
    summary["articles_removed_dedup"] = len(removed)

    # Save
    df_clean.to_csv(DEDUPED_CSV, index=False, encoding="utf-8")
    logger.info("Saved deduplicated CSV → %s  (%d rows)", DEDUPED_CSV, len(df_clean))

    if not removed.empty:
        DEDUP_REMOVED_CSV.parent.mkdir(parents=True, exist_ok=True)
        removed.to_csv(DEDUP_REMOVED_CSV, index=False, encoding="utf-8")
        logger.info("Saved removed-duplicates log → %s", DEDUP_REMOVED_CSV)

    # Category / author stats
    cstats = category_stats(df_clean)
    astats = author_stats(df_clean)
    from src.config import TABLES_DIR
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cstats.to_csv(TABLES_DIR / "category_stats.csv", index=False)
    astats.to_csv(TABLES_DIR / "author_stats.csv",   index=False)

    # Canonical yearly corpus table. Consumed by visualize.py for
    # figures 01_articles_by_year and 02_word_count_trends.
    ystats_path = TABLES_DIR / "yearly_corpus_statistics.csv"
    ystats.sort_values("year").to_csv(ystats_path, index=False)
    logger.info("Saved yearly corpus statistics → %s", ystats_path)

    write_reports(df_clean, summary, ystats)

    # Print quick overview
    print("\n=== Quality Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nYearly breakdown:")
    print(ystats.to_string(index=False))

    return df_clean


if __name__ == "__main__":
    run_quality_pipeline()
