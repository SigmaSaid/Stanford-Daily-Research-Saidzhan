"""
ngrams.py — Bigram and trigram extraction and trend analysis.

Reads:  data/processed/articles_deduplicated.csv
Writes:
  reports/tables/ngrams_by_year.csv
  reports/tables/top_ngram_changes.csv
"""

import sys
import re
from collections import Counter

import pandas as pd

from src.config import (
    DEDUPED_CSV,
    TABLES_DIR,
    NGRAM_SIZES,
    TOP_N_NGRAMS,
)
from src.utils import get_logger, safe_div
from src.vocabulary import STOPWORDS

logger = get_logger(__name__)


def tokenize_clean(text: str) -> list[str]:
    """Lowercase alpha tokens only, no stopword removal (preserve phrase context)."""
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def extract_ngrams(tokens: list[str], n: int) -> list[str]:
    """Return all n-grams as space-joined strings."""
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def filter_ngram(ngram: str, n: int) -> bool:
    """
    Return True if the ngram is worth keeping.
    Rules:
      - No ngram that is entirely stopwords
      - No ngram starting or ending with a stopword (for bigrams/trigrams)
      - Must contain at least one alpha token longer than 3 chars
    """
    parts = ngram.split()
    if all(p in STOPWORDS for p in parts):
        return False
    if parts[0] in STOPWORDS or parts[-1] in STOPWORDS:
        return False
    if not any(len(p) > 3 and p not in STOPWORDS for p in parts):
        return False
    return True


def ngrams_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-million-word n-gram frequencies for each year.
    Returns long-format: year | ngram | n | count | total_ngrams | freq_per_million
    """
    rows = []

    for yr in sorted(df["year"].dropna().unique()):
        sub = df[df["year"] == yr]

        # Collect all tokens for this year
        year_tokens: list[str] = []
        for text in sub["text"].dropna():
            year_tokens.extend(tokenize_clean(text))

        total_tokens = len(year_tokens)
        if total_tokens < 100:
            continue

        for n in NGRAM_SIZES:
            all_ngrams = extract_ngrams(year_tokens, n)
            total_ng = len(all_ngrams)
            freq = Counter(ng for ng in all_ngrams if filter_ngram(ng, n))

            for phrase, count in freq.most_common(TOP_N_NGRAMS):
                rows.append({
                    "year":            int(yr),
                    "ngram":           phrase,
                    "n":               n,
                    "count":           count,
                    "total_ngrams":    total_ng,
                    "freq_per_million": round(count / total_ng * 1_000_000, 2),
                })

    return pd.DataFrame(rows)


def top_ngram_changes(ngram_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find n-grams with the largest frequency shift from 2015-2019 to 2023-2026.
    Computed separately for bigrams and trigrams.
    """
    results = []
    for n in NGRAM_SIZES:
        sub = ngram_df[ngram_df["n"] == n]
        pre  = sub[sub["year"].between(2015, 2019)]
        post = sub[sub["year"].between(2023, 2026)]

        pre_avg  = pre.groupby("ngram")["freq_per_million"].mean().rename("pre_freq")
        post_avg = post.groupby("ngram")["freq_per_million"].mean().rename("post_freq")

        merged = pd.concat([pre_avg, post_avg], axis=1).fillna(0)
        merged["absolute_change"] = merged["post_freq"] - merged["pre_freq"]
        merged["relative_change"] = merged.apply(
            lambda r: safe_div(r["post_freq"] - r["pre_freq"], r["pre_freq"] + 0.01) * 100,
            axis=1,
        )
        merged["n"] = n
        merged.reset_index(inplace=True)
        results.append(merged)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return combined.sort_values("absolute_change", ascending=False)


def run_ngram_analysis() -> None:
    if not DEDUPED_CSV.exists():
        logger.error("Deduplicated CSV not found. Run quality.py first.")
        sys.exit(1)

    df = pd.read_csv(DEDUPED_CSV, low_memory=False)
    logger.info("Loaded %d articles for n-gram analysis", len(df))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting n-grams by year (n=%s) …", NGRAM_SIZES)
    ng_df = ngrams_by_year(df)
    ng_df.to_csv(TABLES_DIR / "ngrams_by_year.csv", index=False)
    logger.info("Saved ngrams_by_year.csv  (%d rows)", len(ng_df))

    logger.info("Computing top n-gram changes …")
    changes = top_ngram_changes(ng_df)
    changes.to_csv(TABLES_DIR / "top_ngram_changes.csv", index=False)
    logger.info("Saved top_ngram_changes.csv")

    # Print preview
    for n in NGRAM_SIZES:
        label = "Bigrams" if n == 2 else "Trigrams"
        sub = changes[changes["n"] == n]
        print(f"\nTop 15 INCREASING {label} (2015-2019 → 2023-2026):")
        print(sub.head(15)[["ngram","pre_freq","post_freq","absolute_change"]].to_string(index=False))
        print(f"\nTop 15 DECREASING {label}:")
        print(sub.tail(15)[["ngram","pre_freq","post_freq","absolute_change"]].to_string(index=False))


if __name__ == "__main__":
    run_ngram_analysis()
