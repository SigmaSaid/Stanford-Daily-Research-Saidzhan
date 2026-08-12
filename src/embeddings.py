"""
embeddings.py — Sentence-transformer embedding generation and caching.

Reads:  data/processed/articles_deduplicated.csv
Writes:
  data/embeddings/embeddings.npy
  data/embeddings/article_ids.csv

Embeddings are generated once and cached. Re-running this module
skips articles that are already embedded.
"""

import sys

import numpy as np
import pandas as pd

from src.config import (
    DEDUPED_CSV,
    EMBEDDINGS_DIR,
    EMBEDDINGS_NPY,
    EMBEDDING_IDS_CSV,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    MAX_EMBED_ARTICLES,
    RANDOM_SEED,
)
from src.utils import get_logger

logger = get_logger(__name__)


def load_model():
    """Lazy-load the sentence-transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        return SentenceTransformer(EMBEDDING_MODEL)
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)


def load_existing_embeddings() -> tuple[np.ndarray | None, set[str]]:
    """
    Load cached embeddings and their article ID index.
    Returns (embeddings_array_or_None, set_of_embedded_article_ids).
    """
    if EMBEDDINGS_NPY.exists() and EMBEDDING_IDS_CSV.exists():
        emb = np.load(str(EMBEDDINGS_NPY))
        ids_df = pd.read_csv(EMBEDDING_IDS_CSV, dtype={"article_id": str})
        embedded_ids = set(ids_df["article_id"].astype(str))
        logger.info("Loaded %d cached embeddings", len(embedded_ids))
        return emb, embedded_ids
    return None, set()


def run_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate embeddings for all articles not yet cached.
    Returns (full_embeddings_array, article_ids_dataframe).
    """
    if not DEDUPED_CSV.exists():
        logger.error("Deduplicated CSV not found. Run quality.py first.")
        sys.exit(1)

    df = pd.read_csv(DEDUPED_CSV, low_memory=False)
    df["article_id"] = df["article_id"].astype(str)
    logger.info("Total articles available: %d", len(df))

    # Optionally subsample
    if MAX_EMBED_ARTICLES and len(df) > MAX_EMBED_ARTICLES:
        df = df.sample(MAX_EMBED_ARTICLES, random_state=RANDOM_SEED)
        logger.info("Subsampled to %d articles (MAX_EMBED_ARTICLES limit)", MAX_EMBED_ARTICLES)

    # Check cache
    existing_emb, embedded_ids = load_existing_embeddings()
    new_df = df[~df["article_id"].isin(embedded_ids)].copy()
    logger.info("Articles needing embeddings: %d", len(new_df))

    if new_df.empty:
        logger.info("All articles already embedded — nothing to do.")
        ids_df = pd.read_csv(EMBEDDING_IDS_CSV)
        return existing_emb, ids_df

    # Load model
    model = load_model()

    # Prepare texts (truncate to 512 tokens worth of chars as a safety measure)
    texts = new_df["text"].fillna("").astype(str).str[:4096].tolist()

    logger.info("Generating embeddings in batches of %d …", EMBEDDING_BATCH_SIZE)
    new_embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize for cosine similarity via dot product
    )
    logger.info("Embeddings shape: %s", new_embeddings.shape)

    # Merge with existing cache
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if existing_emb is not None and len(existing_emb) > 0:
        all_embeddings = np.vstack([existing_emb, new_embeddings])
        old_ids_df = pd.read_csv(EMBEDDING_IDS_CSV, dtype={"article_id": str})
        new_ids_df = new_df[["article_id","year","corpus","article_type"]].copy()
        new_ids_df["article_id"] = new_ids_df["article_id"].astype(str)
        all_ids_df = pd.concat([old_ids_df, new_ids_df], ignore_index=True)
    else:
        all_embeddings = new_embeddings
        all_ids_df = new_df[["article_id","year","corpus","article_type"]].copy()
        all_ids_df["article_id"] = all_ids_df["article_id"].astype(str)

    # Save
    np.save(str(EMBEDDINGS_NPY), all_embeddings)
    all_ids_df.to_csv(EMBEDDING_IDS_CSV, index=False)
    logger.info("Saved embeddings → %s  (%d total)", EMBEDDINGS_NPY, len(all_ids_df))
    logger.info("Saved article ID map → %s", EMBEDDING_IDS_CSV)

    return all_embeddings, all_ids_df


if __name__ == "__main__":
    emb, ids = run_embeddings()
    print(f"\nEmbedding matrix shape : {emb.shape}")
    print(f"Articles indexed       : {len(ids)}")
    print(f"Years covered          : {sorted(ids['year'].dropna().unique().astype(int).tolist())}")
