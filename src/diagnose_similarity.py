"""
diagnose_similarity.py — Decompose within-year semantic similarity by group.

The pipeline's semantic_similarity_stats.csv samples a fixed number of articles
per year from the POOLED corpus. Because the Opinions/News mix changes across
the study window, and because news reporting is internally more homogeneous
than opinion writing, a rising pooled similarity curve can be produced entirely
by the changing mix — with no convergence in anyone's writing.

This module recomputes within-year mean pairwise cosine similarity separately
for:

    all        — pooled, comparable to the pipeline output
    news       — news reporting only
    opinions   — all opinion articles
    student    — student-authored opinion articles only
    guest      — guest/institutional opinion submissions only

and tests each series for a monotonic trend. If 'all' trends but 'student' does
not, apparent convergence is a composition effect.

Requires embeddings, so run after the main pipeline:
  python -m src.diagnose_similarity

Writes:
  reports/tables/diag_similarity_by_group.csv
  reports/tables/diag_similarity_trends.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    EMBEDDINGS_NPY, EMBEDDING_IDS_CSV, DEDUPED_CSV, TABLES_DIR, RANDOM_SEED,
)
from src.diagnose_authorship import flag_guest_content
from src.utils import get_logger

logger = get_logger(__name__)

MAX_PER_YEAR = 200   # cap keeps cost bounded; mean pairwise similarity is
                     # unbiased with respect to sample size
MIN_PER_YEAR = 10    # below this the estimate is too noisy to report
N_BOOT = 400


def mean_pairwise_similarity(X: np.ndarray) -> float:
    """
    Mean off-diagonal cosine similarity.

    Embeddings are L2-normalised by embeddings.py, so the dot product is the
    cosine. Computing the full Gram matrix and subtracting the diagonal is
    exact and far faster than looping over pairs.
    """
    m = X.shape[0]
    if m < 2:
        return float("nan")
    S = X @ X.T
    return float((S.sum() - np.trace(S)) / (m * (m - 1)))


def _boot_ci(X: np.ndarray, rng, n_boot=N_BOOT, alpha=0.05):
    m = X.shape[0]
    if m < MIN_PER_YEAR:
        return None, None
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, m, m)
        vals[i] = mean_pairwise_similarity(X[idx])
    return (round(float(np.percentile(vals, 100 * alpha / 2)), 5),
            round(float(np.percentile(vals, 100 * (1 - alpha / 2))), 5))


def run_similarity_diagnostics() -> None:
    if not EMBEDDINGS_NPY.exists() or not EMBEDDING_IDS_CSV.exists():
        logger.error("Embeddings not found. Run `python -m src.embeddings` first.")
        sys.exit(1)

    emb = np.load(EMBEDDINGS_NPY)
    ids = pd.read_csv(EMBEDDING_IDS_CSV)
    if len(ids) != emb.shape[0]:
        logger.error("Row mismatch: %d embeddings vs %d ids", emb.shape[0], len(ids))
        sys.exit(1)

    # Verify normalisation; if vectors are not unit length the dot product is
    # not the cosine and every number below would be wrong.
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        logger.warning("Embeddings are not unit-norm (mean %.4f) — normalising.",
                       float(norms.mean()))
        emb = emb / norms[:, None]

    corpus = pd.read_csv(DEDUPED_CSV, low_memory=False)
    keep = [c for c in ["article_id", "author_id", "author_name", "title",
                        "word_count", "article_type", "corpus", "year"]
            if c in corpus.columns]
    ids["article_id"] = ids["article_id"].astype(str)
    corpus["article_id"] = corpus["article_id"].astype(str)
    meta = ids.merge(corpus[keep], on="article_id", how="left",
                     suffixes=("", "_c"))
    meta["row"] = np.arange(len(meta))

    # Flag guest content among opinion articles
    op = meta[meta["corpus"] == "opinions"].copy()
    if not op.empty and "title" in op.columns:
        op = flag_guest_content(op)
        guest_rows = set(op[op["is_guest"]]["row"])
    else:
        guest_rows = set()
    meta["is_guest"] = meta["row"].isin(guest_rows)

    groups = {
        "all":      lambda d: d,
        "news":     lambda d: d[d["corpus"] == "news"],
        "opinions": lambda d: d[d["corpus"] == "opinions"],
        "student":  lambda d: d[(d["corpus"] == "opinions") & (~d["is_guest"])],
        "guest":    lambda d: d[(d["corpus"] == "opinions") & (d["is_guest"])],
    }

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for year in sorted(meta["year"].dropna().unique()):
        ymeta = meta[meta["year"] == year]
        for gname, sel in groups.items():
            sub = sel(ymeta)
            n_avail = len(sub)
            if n_avail < MIN_PER_YEAR:
                rows.append({"year": int(year), "group": gname,
                             "n_available": n_avail, "n_used": 0,
                             "mean_sim": None, "ci_low": None, "ci_high": None})
                continue
            if n_avail > MAX_PER_YEAR:
                sub = sub.sample(MAX_PER_YEAR, random_state=RANDOM_SEED)
            X = emb[sub["row"].values]
            lo, hi = _boot_ci(X, rng)
            rows.append({
                "year": int(year), "group": gname,
                "n_available": n_avail, "n_used": X.shape[0],
                "mean_sim": round(mean_pairwise_similarity(X), 5),
                "ci_low": lo, "ci_high": hi,
            })

    out = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES_DIR / "diag_similarity_by_group.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("WITHIN-YEAR SEMANTIC SIMILARITY, BY GROUP")
    print("=" * 78)
    piv = out.pivot(index="year", columns="group", values="mean_sim")
    order = [g for g in ["all", "news", "opinions", "student", "guest"]
             if g in piv.columns]
    print(piv[order].to_string())

    print("\nArticles used per year:")
    pivn = out.pivot(index="year", columns="group", values="n_used")
    print(pivn[order].to_string())

    # ── trend tests ───────────────────────────────────────────────────────────
    trows = []
    for gname in order:
        s = out[(out["group"] == gname) & out["mean_sim"].notna()]
        if len(s) < 5:
            continue
        rho, p_rho = stats.spearmanr(s["year"], s["mean_sim"])
        slope, intercept, r, p_lin, se = stats.linregress(s["year"], s["mean_sim"])
        first, last = s.iloc[0], s.iloc[-1]
        trows.append({
            "group": gname,
            "n_years": len(s),
            "first_year": int(first["year"]),
            "last_year": int(last["year"]),
            "sim_first": first["mean_sim"],
            "sim_last": last["mean_sim"],
            "pct_change": round((last["mean_sim"] / first["mean_sim"] - 1) * 100, 1)
                          if first["mean_sim"] else None,
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p_rho), 5),
            "linear_slope": round(float(slope), 6),
            "linear_p": round(float(p_lin), 5),
            "peak_year": int(s.loc[s["mean_sim"].idxmax(), "year"]),
        })
    trends = pd.DataFrame(trows)
    trends.to_csv(TABLES_DIR / "diag_similarity_trends.csv", index=False)

    print("\n" + "=" * 78)
    print("MONOTONIC TREND TESTS")
    print("=" * 78)
    print(trends.to_string(index=False))

    print("\nHow to read this:")
    print("  * 'all' trending while 'student' does not  -> composition effect,")
    print("    not convergence in student writing.")
    print("  * 'news' more similar than 'opinions' is expected: news follows a")
    print("    house style. A rising news SHARE therefore raises 'all' on its own.")
    print("  * peak_year matters. A rise that reverses is not a contraction.")
    print("\nCAUTION: similarity is computed on article openings only, because")
    print("the embedding model truncates long inputs. Later years rest on few")
    print("student articles — read mean_sim against n_used.")


if __name__ == "__main__":
    run_similarity_diagnostics()
