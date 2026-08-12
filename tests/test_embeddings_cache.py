"""
test_embeddings_cache.py — Tests for the embedding cache/persistence logic.

IMPORTANT: these tests deliberately DO NOT download the real
`all-MiniLM-L6-v2` model. They substitute a stub encoder that returns
L2-normalised vectors of the correct dimensionality (384), so that the
project's own caching, merging, persistence and similarity-integration code
paths are exercised without network access.

What these tests DO verify:
  - embeddings + article_ids are persisted to disk
  - a second run is a no-op (does not re-encode already-cached articles)
  - newly added articles are appended, not recomputed from scratch
  - embeddings rows stay aligned with article_ids rows
  - similarity.py can consume the artefacts embeddings.py produces

What they do NOT verify (requires network — see audit report):
  - that the real SentenceTransformer model downloads and loads
  - real semantic quality of the vectors
  - GPU/MPS device selection
"""

import numpy as np
import pandas as pd
import pytest

from src import embeddings as emb_mod
from src.config import EMBEDDING_BATCH_SIZE

DIM = 384  # all-MiniLM-L6-v2 output dimensionality


class _StubEncoder:
    """Deterministic stand-in for SentenceTransformer."""

    def __init__(self):
        self.encode_calls = []

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=False):
        self.encode_calls.append(len(texts))
        rng = np.random.RandomState(0)
        vecs = rng.rand(len(texts), DIM).astype(np.float32)
        if normalize_embeddings:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect all embedding IO into a temp dir so real artefacts are untouched."""
    deduped = tmp_path / "articles_deduplicated.csv"
    npy = tmp_path / "embeddings.npy"
    ids = tmp_path / "article_ids.csv"

    monkeypatch.setattr(emb_mod, "DEDUPED_CSV", deduped)
    monkeypatch.setattr(emb_mod, "EMBEDDINGS_NPY", npy)
    monkeypatch.setattr(emb_mod, "EMBEDDING_IDS_CSV", ids)
    monkeypatch.setattr(emb_mod, "EMBEDDINGS_DIR", tmp_path)
    monkeypatch.setattr(emb_mod, "MAX_EMBED_ARTICLES", None)
    return {"deduped": deduped, "npy": npy, "ids": ids}


def _write_corpus(path, n, start_id=1, year=2020):
    pd.DataFrame({
        "article_id": [start_id + i for i in range(n)],
        "year": [year] * n,
        "corpus": ["opinions"] * n,
        "article_type": ["opinion"] * n,
        "text": [f"Article number {start_id + i} about campus policy." for i in range(n)],
    }).to_csv(path, index=False)


def test_embeddings_persisted_and_aligned(isolated_paths, monkeypatch):
    stub = _StubEncoder()
    monkeypatch.setattr(emb_mod, "load_model", lambda: stub)
    _write_corpus(isolated_paths["deduped"], 10)

    arr, ids_df = emb_mod.run_embeddings()

    assert isolated_paths["npy"].exists(), "embeddings.npy was not written"
    assert isolated_paths["ids"].exists(), "article_ids.csv was not written"
    assert arr.shape == (10, DIM)
    assert len(ids_df) == 10, "ids table must stay row-aligned with embeddings"
    assert stub.encode_calls == [10]

    # Vectors must be L2-normalised, since similarity.py uses a raw dot product
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "embeddings must be L2-normalised"


def test_second_run_does_not_recompute(isolated_paths, monkeypatch):
    stub = _StubEncoder()
    monkeypatch.setattr(emb_mod, "load_model", lambda: stub)
    _write_corpus(isolated_paths["deduped"], 10)

    emb_mod.run_embeddings()
    assert stub.encode_calls == [10]

    # Re-run with identical corpus: must be a complete no-op.
    emb_mod.run_embeddings()
    assert stub.encode_calls == [10], "cache miss — articles were re-encoded"


def test_incremental_articles_are_appended(isolated_paths, monkeypatch):
    stub = _StubEncoder()
    monkeypatch.setattr(emb_mod, "load_model", lambda: stub)

    _write_corpus(isolated_paths["deduped"], 10)
    emb_mod.run_embeddings()

    # Corpus grows by 5 new articles.
    _write_corpus(isolated_paths["deduped"], 15)
    arr, ids_df = emb_mod.run_embeddings()

    assert stub.encode_calls == [10, 5], "should encode only the 5 new articles"
    assert arr.shape == (15, DIM)
    assert len(ids_df) == 15
    assert ids_df["article_id"].duplicated().sum() == 0


def test_similarity_consumes_embeddings_output(isolated_paths, monkeypatch):
    """Integration: similarity.py must be able to read embeddings.py artefacts."""
    from src import similarity as sim_mod

    stub = _StubEncoder()
    monkeypatch.setattr(emb_mod, "load_model", lambda: stub)

    # Two years so within-year and between-year both have work to do.
    rows = []
    for i in range(20):
        rows.append({
            "article_id": 100 + i,
            "year": 2018 if i < 10 else 2024,
            "corpus": "opinions",
            "article_type": "opinion",
            "text": f"Text {i}",
        })
    pd.DataFrame(rows).to_csv(isolated_paths["deduped"], index=False)

    arr, ids_df = emb_mod.run_embeddings()
    ids_df = pd.read_csv(isolated_paths["ids"])

    within = sim_mod.within_year_similarity(arr, ids_df, sample_n=10)
    assert set(within["year"]) == {2018, 2024}
    assert (within["n_pairs"] == 45).all()          # C(10,2)
    assert ((within["mean_sim"] >= -1) & (within["mean_sim"] <= 1)).all()

    between = sim_mod.between_year_similarity(arr, ids_df, sample_n=10)
    assert len(between) == 1
    assert between.iloc[0]["year_a"] == 2018
    assert between.iloc[0]["year_b"] == 2024
