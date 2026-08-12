"""
topics.py — Topic modeling (NMF) and topic distribution over time.

Reads:  data/processed/articles_deduplicated.csv
Writes:
  reports/tables/topic_distributions.csv
  reports/tables/topic_top_words.csv
"""

import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DEDUPED_CSV,
    TABLES_DIR,
    NUM_TOPICS,
    NMF_MAX_ITER,
    NMF_RANDOM_STATE,
    MAX_TFIDF_FEATURES_TOPICS,
)
from src.utils import get_logger
from src.vocabulary import STOPWORDS

logger = get_logger(__name__)


def fit_nmf(texts: list[str]) -> tuple[NMF, TfidfVectorizer, np.ndarray]:
    """
    Fit a TF-IDF vectorizer and NMF model on the given texts.
    Returns (model, vectorizer, document_topic_matrix).
    """
    logger.info("Fitting TF-IDF for NMF (max_features=%d) …", MAX_TFIDF_FEATURES_TOPICS)
    vec = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES_TOPICS,
        stop_words=list(STOPWORDS),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vec.fit_transform(texts)
    logger.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)

    logger.info("Fitting NMF (n_topics=%d) …", NUM_TOPICS)
    nmf = NMF(
        n_components=NUM_TOPICS,
        max_iter=NMF_MAX_ITER,
        random_state=NMF_RANDOM_STATE,
        init="nndsvda",
    )
    W = nmf.fit_transform(tfidf_matrix)  # (n_docs, n_topics)
    logger.info("NMF reconstruction error: %.4f", nmf.reconstruction_err_)
    return nmf, vec, W


def get_top_words(nmf: NMF, vec: TfidfVectorizer, n_words: int = 12) -> list[list[str]]:
    """Return top n words for each topic."""
    feature_names = vec.get_feature_names_out()
    topics = []
    for comp in nmf.components_:
        top_idx = comp.argsort()[::-1][:n_words]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def topic_distributions_by_year(
    W: np.ndarray,
    years: pd.Series,
) -> pd.DataFrame:
    """
    Compute the mean topic weight for each year.
    Returns wide-format: year | topic_0 | topic_1 | … | topic_N
    """
    df_W = pd.DataFrame(W, columns=[f"topic_{i}" for i in range(W.shape[1])])
    df_W["year"] = years.values

    yearly = df_W.groupby("year").mean().reset_index()
    # Normalise each row so topic weights sum to 1
    topic_cols = [c for c in yearly.columns if c.startswith("topic_")]
    row_sums = yearly[topic_cols].sum(axis=1)
    yearly[topic_cols] = yearly[topic_cols].div(row_sums, axis=0)
    return yearly


def run_topic_analysis() -> None:
    if not DEDUPED_CSV.exists():
        logger.error("Deduplicated CSV not found. Run quality.py first.")
        sys.exit(1)

    df = pd.read_csv(DEDUPED_CSV, low_memory=False)
    logger.info("Loaded %d articles for topic modeling", len(df))

    texts = df["text"].fillna("").astype(str).tolist()
    years = df["year"]

    nmf, vec, W = fit_nmf(texts)
    top_words = get_top_words(nmf, vec)

    # Topic top-words table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tw_rows = []
    for i, words in enumerate(top_words):
        tw_rows.append({
            "topic_id":  i,
            "label":     f"topic_{i}",
            "top_words": ", ".join(words),
        })
    tw_df = pd.DataFrame(tw_rows)
    tw_df.to_csv(TABLES_DIR / "topic_top_words.csv", index=False)
    logger.info("Saved topic_top_words.csv")

    # Print topic descriptions
    print("\n=== NMF Topics ===")
    for row in tw_rows:
        print(f"  {row['label']:12s}: {row['top_words']}")

    # Topic distributions by year
    yearly = topic_distributions_by_year(W, years)
    yearly.to_csv(TABLES_DIR / "topic_distributions.csv", index=False)
    logger.info("Saved topic_distributions.csv")

    print("\n=== Topic Share by Year (top 5 topics) ===")
    topic_cols = [c for c in yearly.columns if c.startswith("topic_")]
    mean_share = yearly[topic_cols].mean().sort_values(ascending=False)
    top5 = mean_share.head(5).index.tolist()
    print(yearly[["year"] + top5].to_string(index=False))


if __name__ == "__main__":
    run_topic_analysis()
