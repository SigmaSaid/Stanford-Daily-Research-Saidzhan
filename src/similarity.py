"""
similarity.py — Semantic similarity analysis using cached embeddings.

Reads:
  data/embeddings/embeddings.npy
  data/embeddings/article_ids.csv
Writes:
  reports/tables/semantic_similarity_stats.csv
  reports/tables/between_year_similarity.csv
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    EMBEDDINGS_NPY,
    EMBEDDING_IDS_CSV,
    TABLES_DIR,
    SIMILARITY_SAMPLE_PER_YEAR,
    SIMILARITY_RANDOM_SEED,
)
from src.utils import get_logger

logger = get_logger(__name__)


def load_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    if not EMBEDDINGS_NPY.exists() or not EMBEDDING_IDS_CSV.exists():
        logger.error("Embeddings not found. Run embeddings.py first.")
        sys.exit(1)
    emb    = np.load(str(EMBEDDINGS_NPY))
    ids_df = pd.read_csv(EMBEDDING_IDS_CSV)
    logger.info("Loaded embeddings: %s, %d articles", emb.shape, len(ids_df))
    return emb, ids_df


def within_year_similarity(
    emb: np.ndarray,
    ids_df: pd.DataFrame,
    sample_n: int = SIMILARITY_SAMPLE_PER_YEAR,
    rng: int = SIMILARITY_RANDOM_SEED,
) -> pd.DataFrame:
    """
    For each year, compute pairwise cosine similarity on a random sample.
    Returns: year | mean_sim | median_sim | std_sim | n_articles | n_pairs
    """
    rows = []
    years = sorted(ids_df["year"].dropna().unique().astype(int))

    for yr in years:
        mask  = (ids_df["year"] == yr).values
        idxs  = np.where(mask)[0]

        if len(idxs) < 2:
            continue

        # Sample if needed
        if len(idxs) > sample_n:
            rng_state = np.random.RandomState(rng)
            idxs = rng_state.choice(idxs, size=sample_n, replace=False)

        year_emb = emb[idxs]  # shape: (n, dim)

        # Pairwise cosine similarity (embeddings already L2-normalised → dot product)
        sim_matrix = year_emb @ year_emb.T
        # Extract upper triangle (excluding diagonal)
        n = len(idxs)
        upper = sim_matrix[np.triu_indices(n, k=1)]

        rows.append({
            "year":        yr,
            "mean_sim":    round(float(np.mean(upper)), 5),
            "median_sim":  round(float(np.median(upper)), 5),
            "std_sim":     round(float(np.std(upper)), 5),
            "p10_sim":     round(float(np.percentile(upper, 10)), 5),
            "p90_sim":     round(float(np.percentile(upper, 90)), 5),
            "n_articles":  len(idxs),
            "n_pairs":     len(upper),
        })
        logger.info("Year %d: mean_sim=%.4f  n=%d", yr, rows[-1]["mean_sim"], len(idxs))

    return pd.DataFrame(rows)


def between_year_similarity(
    emb: np.ndarray,
    ids_df: pd.DataFrame,
    sample_n: int = 150,
    rng: int = SIMILARITY_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Compute mean cosine similarity *between* every pair of years.
    Returns: year_a | year_b | mean_cross_sim | n_pairs
    """
    years = sorted(ids_df["year"].dropna().unique().astype(int))
    rng_state = np.random.RandomState(rng)
    rows = []

    for i, ya in enumerate(years):
        for yb in years[i+1:]:
            mask_a = (ids_df["year"] == ya).values
            mask_b = (ids_df["year"] == yb).values

            idxs_a = np.where(mask_a)[0]
            idxs_b = np.where(mask_b)[0]

            if len(idxs_a) == 0 or len(idxs_b) == 0:
                continue

            # Sample
            sa = min(len(idxs_a), sample_n)
            sb = min(len(idxs_b), sample_n)
            idxs_a_s = rng_state.choice(idxs_a, size=sa, replace=False)
            idxs_b_s = rng_state.choice(idxs_b, size=sb, replace=False)

            ea = emb[idxs_a_s]
            eb = emb[idxs_b_s]

            # Cross similarity: (sa, sb)
            cross = ea @ eb.T
            mean_cs = float(np.mean(cross))

            rows.append({
                "year_a":        ya,
                "year_b":        yb,
                "mean_cross_sim": round(mean_cs, 5),
                "n_pairs":       sa * sb,
            })

    return pd.DataFrame(rows)


def run_similarity_analysis() -> None:
    emb, ids_df = load_embeddings()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Computing within-year similarity …")
    within = within_year_similarity(emb, ids_df)
    within.to_csv(TABLES_DIR / "semantic_similarity_stats.csv", index=False)
    logger.info("Saved semantic_similarity_stats.csv")

    logger.info("Computing between-year similarity …")
    between = between_year_similarity(emb, ids_df)
    between.to_csv(TABLES_DIR / "between_year_similarity.csv", index=False)
    logger.info("Saved between_year_similarity.csv")

    print("\nWithin-year semantic similarity:")
    print(within[["year","mean_sim","median_sim","std_sim","n_articles"]].to_string(index=False))

    print("\nSample between-year similarity (recent pairs):")
    recent = between[between["year_a"] >= 2018].tail(15)
    print(recent[["year_a","year_b","mean_cross_sim","n_pairs"]].to_string(index=False))


if __name__ == "__main__":
    run_similarity_analysis()
