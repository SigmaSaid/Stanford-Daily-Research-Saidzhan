"""
vocabulary.py — Word frequency, TF-IDF, and AI-associated vocabulary analysis.

Reads:  data/processed/articles_deduplicated.csv
Writes:
  reports/tables/word_freq_by_year.csv
  reports/tables/tfidf_by_year.csv
  reports/tables/ai_associated_vocabulary.csv
  reports/tables/top_vocabulary_changes.csv
"""

import sys
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DEDUPED_CSV,
    TABLES_DIR,
    AI_EXPLORATORY_VOCAB,
    TOP_N_WORDS,
    PERIODS,
)
from src.utils import get_logger, safe_div

logger = get_logger(__name__)

# Standard English stopwords (extended)
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","through","during","is","are","was",
    "were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","i","you","he",
    "she","it","we","they","me","him","her","us","them","my","your","his",
    "its","our","their","this","that","these","those","who","which","what",
    "where","when","how","why","whom","not","no","nor","so","yet","as",
    "if","then","because","while","although","though","unless","since",
    "also","just","more","most","than","very","too","only","even","still",
    "after","before","over","under","same","other","such","each","own",
    "s","t","re","ve","ll","d","m","stanford","daily",
}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Simple whitespace + punctuation tokenizer; optionally removes stopwords."""
    import re
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# ─── Word frequency by year ───────────────────────────────────────────────────

def word_freq_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-million-word frequency for every non-stopword token, per year.
    Returns a long-format DataFrame: year | word | freq_per_million | count | total_words
    """
    rows = []
    for yr in sorted(df["year"].dropna().unique()):
        sub = df[df["year"] == yr]
        all_tokens: list[str] = []
        for text in sub["text"].dropna():
            all_tokens.extend(tokenize(text))

        total = len(all_tokens)
        if total == 0:
            continue

        freq = Counter(all_tokens)
        for word, count in freq.most_common(TOP_N_WORDS * 3):
            rows.append({
                "year":            int(yr),
                "word":            word,
                "count":           count,
                "total_words":     total,
                "freq_per_million": round(count / total * 1_000_000, 2),
            })

    return pd.DataFrame(rows)


# ─── TF-IDF by year ───────────────────────────────────────────────────────────

def tfidf_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat each year as a single 'document' and compute TF-IDF.
    Returns: year | term | mean_tfidf | rank
    """
    year_texts: dict[int, str] = {}
    for yr in sorted(df["year"].dropna().unique()):
        sub = df[df["year"] == yr]
        combined = " ".join(sub["text"].dropna().astype(str))
        year_texts[int(yr)] = combined

    years  = sorted(year_texts.keys())
    corpus = [year_texts[y] for y in years]

    vec = TfidfVectorizer(
        max_features=TOP_N_WORDS * 5,
        stop_words=list(STOPWORDS),
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vec.fit_transform(corpus)
    terms = vec.get_feature_names_out()

    rows = []
    for i, yr in enumerate(years):
        scores = tfidf_matrix[i].toarray().flatten()
        ranked = np.argsort(scores)[::-1][:TOP_N_WORDS]
        for rank, idx in enumerate(ranked, 1):
            rows.append({
                "year":       yr,
                "term":       terms[idx],
                "mean_tfidf": round(float(scores[idx]), 6),
                "rank":       rank,
            })

    return pd.DataFrame(rows)


# ─── AI-associated vocabulary ─────────────────────────────────────────────────

def ai_vocab_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Track exploratory AI-associated vocabulary frequency per year per million words.
    This is EXPLORATORY — not a validated AI detector.
    """
    rows = []
    for yr in sorted(df["year"].dropna().unique()):
        sub = df[df["year"] == yr]
        all_tokens: list[str] = []
        for text in sub["text"].dropna():
            all_tokens.extend(tokenize(text, remove_stopwords=False))

        total = len(all_tokens)
        if total == 0:
            continue

        freq = Counter(all_tokens)
        # Deduplicate while preserving order: a repeated entry would otherwise
        # emit two identical rows and double-count in the summed yearly aggregate.
        for word in dict.fromkeys(w.lower() for w in AI_EXPLORATORY_VOCAB):
            count = freq.get(word.lower(), 0)
            rows.append({
                "year":             int(yr),
                "word":             word.lower(),
                "count":            count,
                "total_words":      total,
                "freq_per_million": round(count / total * 1_000_000, 4),
            })

    return pd.DataFrame(rows)


# ─── Vocabulary change ────────────────────────────────────────────────────────

def top_vocabulary_changes(freq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify words with the largest absolute frequency change between
    the pre-AI baseline (2015-2019) and the AI era (2023-2026).
    """
    pre  = freq_df[freq_df["year"].between(2015, 2019)]
    post = freq_df[freq_df["year"].between(2023, 2026)]

    pre_avg  = pre.groupby("word")["freq_per_million"].mean().rename("pre_freq")
    post_avg = post.groupby("word")["freq_per_million"].mean().rename("post_freq")

    merged = pd.concat([pre_avg, post_avg], axis=1).dropna()
    merged["absolute_change"]  = merged["post_freq"] - merged["pre_freq"]
    merged["relative_change"]  = merged.apply(
        lambda r: safe_div(r["post_freq"] - r["pre_freq"], r["pre_freq"]) * 100,
        axis=1,
    )
    merged = merged.sort_values("absolute_change", ascending=False)
    merged.reset_index(inplace=True)
    return merged


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_vocabulary_analysis() -> None:
    if not DEDUPED_CSV.exists():
        logger.error("Deduplicated CSV not found. Run quality.py first.")
        sys.exit(1)

    df = pd.read_csv(DEDUPED_CSV, low_memory=False)
    logger.info("Loaded %d articles for vocabulary analysis", len(df))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Word frequency
    logger.info("Computing word frequencies …")
    wf = word_freq_by_year(df)
    wf.to_csv(TABLES_DIR / "word_freq_by_year.csv", index=False)
    logger.info("Saved word_freq_by_year.csv  (%d rows)", len(wf))

    # TF-IDF
    logger.info("Computing TF-IDF by year …")
    tfidf = tfidf_by_year(df)
    tfidf.to_csv(TABLES_DIR / "tfidf_by_year.csv", index=False)
    logger.info("Saved tfidf_by_year.csv  (%d rows)", len(tfidf))

    # AI-associated vocabulary
    logger.info("Tracking exploratory AI-associated vocabulary …")
    ai_df = ai_vocab_trends(df)
    ai_df.to_csv(TABLES_DIR / "ai_associated_vocabulary.csv", index=False)
    logger.info("Saved ai_associated_vocabulary.csv  (%d rows)", len(ai_df))

    # Top vocabulary changes
    logger.info("Computing top vocabulary changes (2015-2019 → 2023-2026) …")
    changes = top_vocabulary_changes(wf)
    changes.to_csv(TABLES_DIR / "top_vocabulary_changes.csv", index=False)
    logger.info("Saved top_vocabulary_changes.csv")

    # Print preview
    print("\nTop 20 words INCREASING after 2022:")
    print(changes.head(20)[["word","pre_freq","post_freq","absolute_change"]].to_string(index=False))
    print("\nTop 20 words DECREASING after 2022:")
    print(changes.tail(20)[["word","pre_freq","post_freq","absolute_change"]].to_string(index=False))


if __name__ == "__main__":
    run_vocabulary_analysis()
