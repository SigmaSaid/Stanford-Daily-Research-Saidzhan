"""
diagnose_robustness.py — Robustness checks for the Opinions-corpus findings.

diagnose_confounds.py established that the pre/post differences survive inside
each corpus and are larger in Opinions than in News. Three alternative
explanations for that Opinions result remain untested:

  1. GENRE MIX — 'opinions' bundles letters, editorials, op-eds and columns.
     If the surviving ~100 pieces/year are a different mix than 2015's ~450,
     a within-corpus composition shift reproduces the same effects.

  2. AUTHOR CONCENTRATION — with ~100 opinion pieces/year, a few prolific
     columnists can drive a corpus-level result. Article-level tests also
     pseudo-replicate: 40 pieces by one author are not 40 independent
     observations.

  3. LENGTH — root_ttr (unique / sqrt(n)) is mathematically length-sensitive,
     and it is one of the largest effects. If opinion pieces changed length,
     part of the 'lexical diversity' change is arithmetic.

Run from the project root, after the main pipeline:
  python -m src.diagnose_robustness

Writes:
  reports/tables/diag_genre_mix.csv
  reports/tables/diag_genre_stratified_tests.csv
  reports/tables/diag_author_concentration.csv
  reports/tables/diag_author_level_tests.csv
  reports/tables/diag_length_trends.csv
  reports/tables/diag_length_stratified_tests.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DEDUPED_CSV, TABLES_DIR
from src.statistics import bootstrap_effect_size_ci
from src.utils import get_logger

logger = get_logger(__name__)

PRE_YEARS = (2015, 2019)
POST_YEARS = (2024, 2026)

# Metrics that carried the headline Opinions effects.
FOCUS_METRICS = [
    "mattr", "root_ttr", "func_word_ratio",
    "avg_sentence_len", "flesch_reading_ease",
]

MIN_GROUP = 20  # minimum n per side before a comparison is attempted


# ─── shared helper ────────────────────────────────────────────────────────────

def _effect(a: pd.Series, b: pd.Series) -> dict:
    """Mann-Whitney U + rank-biserial effect size + bootstrap CI."""
    a = a.dropna()
    b = b.dropna()
    if len(a) < MIN_GROUP or len(b) < MIN_GROUP:
        return {"n_pre": len(a), "n_post": len(b), "median_pre": None,
                "median_post": None, "effect_size": None,
                "effect_ci_low": None, "effect_ci_high": None,
                "p_value": None}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - (2 * u) / (len(a) * len(b))
    lo, hi = bootstrap_effect_size_ci(a.values, b.values, n_boot=1000)
    return {
        "n_pre": len(a), "n_post": len(b),
        "median_pre": round(float(a.median()), 4),
        "median_post": round(float(b.median()), 4),
        "effect_size": round(r, 4),
        "effect_ci_low": lo, "effect_ci_high": hi,
        "p_value": float(p),
    }


def _split(frame: pd.DataFrame, metric: str):
    return (frame[frame["year"].between(*PRE_YEARS)][metric],
            frame[frame["year"].between(*POST_YEARS)][metric])


# ─── 1. Genre mix ─────────────────────────────────────────────────────────────

def genre_mix(op: pd.DataFrame) -> pd.DataFrame:
    """article_type composition of the Opinions corpus, per year."""
    tab = (op.groupby(["year", "article_type"]).size()
             .unstack(fill_value=0).sort_index())
    shares = tab.div(tab.sum(axis=1), axis=0).round(3)
    shares.columns = [f"share_{c}" for c in shares.columns]
    out = pd.concat([tab, shares], axis=1)
    out.insert(0, "n_total", tab.sum(axis=1))
    return out.reset_index()


def genre_stratified_tests(op: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run pre/post inside each article_type.

    If an effect holds within genres, it is not produced by the genre mix.
    """
    rows = []
    for gtype in sorted(op["article_type"].dropna().unique()):
        sub = op[op["article_type"] == gtype]
        for metric in FOCUS_METRICS:
            if metric not in sub.columns:
                continue
            a, b = _split(sub, metric)
            res = _effect(a, b)
            if res["effect_size"] is None:
                continue
            rows.append({"article_type": gtype, "metric": metric, **res})
    return pd.DataFrame(rows)


# ─── 2. Author concentration ──────────────────────────────────────────────────

def author_concentration(op: pd.DataFrame) -> pd.DataFrame:
    """Per-year concentration of the Opinions corpus across author_ids."""
    rows = []
    for yr in sorted(op["year"].dropna().unique()):
        sub = op[op["year"] == yr]
        counts = sub["author_id"].dropna().value_counts()
        n = int(counts.sum())
        if n == 0:
            continue
        shares = counts / n
        # Herfindahl-Hirschman index: 1/n_authors = perfectly even, 1 = one author
        hhi = float((shares ** 2).sum())
        rows.append({
            "year": int(yr),
            "n_articles_with_author": n,
            "n_unique_authors": int(counts.size),
            "articles_per_author": round(n / counts.size, 2),
            "top1_share": round(float(shares.iloc[0]), 4),
            "top5_share": round(float(shares.head(5).sum()), 4),
            "top10_share": round(float(shares.head(10).sum()), 4),
            "hhi": round(hhi, 4),
            "effective_n_authors": round(1 / hhi, 1) if hhi else None,
        })
    return pd.DataFrame(rows)


def author_level_tests(op: pd.DataFrame) -> pd.DataFrame:
    """
    Two re-tests that remove prolific-author leverage:

    'author_mean'  — unit of analysis is the AUTHOR, not the article. Each
                     author contributes one mean per period, so a columnist
                     with 40 pieces counts once. This removes pseudo-replication.
    'drop_top10'   — article-level test with the 10 most prolific authors of
                     each period removed entirely.
    """
    rows = []
    pre_all = op[op["year"].between(*PRE_YEARS)]
    post_all = op[op["year"].between(*POST_YEARS)]

    top_pre = pre_all["author_id"].value_counts().head(10).index
    top_post = post_all["author_id"].value_counts().head(10).index

    for metric in FOCUS_METRICS:
        if metric not in op.columns:
            continue

        # (a) author as unit of analysis
        a = pre_all.groupby("author_id")[metric].mean()
        b = post_all.groupby("author_id")[metric].mean()
        res = _effect(a, b)
        rows.append({"approach": "author_mean", "metric": metric, **res})

        # (b) drop the 10 most prolific authors in each period
        a2 = pre_all[~pre_all["author_id"].isin(top_pre)][metric]
        b2 = post_all[~post_all["author_id"].isin(top_post)][metric]
        res2 = _effect(a2, b2)
        rows.append({"approach": "drop_top10", "metric": metric, **res2})

    return pd.DataFrame(rows)


# ─── 3. Length confound ───────────────────────────────────────────────────────

def length_trends(op: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Word-count trend per year, plus each metric's sensitivity to length."""
    rows = []
    for label, frame in [("opinions", op), ("news", news)]:
        for yr in sorted(frame["year"].dropna().unique()):
            sub = frame[frame["year"] == yr]
            rows.append({
                "corpus": label,
                "year": int(yr),
                "n": len(sub),
                "median_word_count": round(float(sub["word_count"].median()), 1),
                "mean_word_count": round(float(sub["word_count"].mean()), 1),
            })
    trend = pd.DataFrame(rows)

    print("\n  Spearman correlation of each metric with word_count (Opinions):")
    for metric in FOCUS_METRICS:
        if metric not in op.columns:
            continue
        d = op[[metric, "word_count"]].dropna()
        rho, p = stats.spearmanr(d["word_count"], d[metric])
        flag = "  <-- LENGTH-SENSITIVE" if abs(rho) > 0.3 else ""
        print(f"    {metric:22s} rho = {rho:+.3f}  (p = {p:.2e}){flag}")

    return trend


def length_stratified_tests(op: pd.DataFrame) -> pd.DataFrame:
    """
    Coarsened exact matching on article length.

    Articles are binned into word-count quintiles defined on the pre-period
    distribution, and the pre/post effect is computed WITHIN each bin. Because
    length is held approximately constant inside a bin, an effect that persists
    across bins cannot be an artefact of articles getting longer or shorter.
    """
    pre = op[op["year"].between(*PRE_YEARS)]
    post = op[op["year"].between(*POST_YEARS)]
    if len(pre) < MIN_GROUP or len(post) < MIN_GROUP:
        return pd.DataFrame()

    edges = np.unique(np.percentile(pre["word_count"].dropna(),
                                    [0, 20, 40, 60, 80, 100]))
    edges[0], edges[-1] = -np.inf, np.inf

    rows = []
    for metric in FOCUS_METRICS:
        if metric not in op.columns:
            continue
        weighted_num, weighted_den = 0.0, 0.0
        for i in range(len(edges) - 1):
            lo_e, hi_e = edges[i], edges[i + 1]
            a = pre[(pre["word_count"] > lo_e) & (pre["word_count"] <= hi_e)][metric]
            b = post[(post["word_count"] > lo_e) & (post["word_count"] <= hi_e)][metric]
            res = _effect(a, b)
            label = (f"{'' if np.isinf(lo_e) else int(lo_e)}"
                     f"-{'' if np.isinf(hi_e) else int(hi_e)} words")
            rows.append({"metric": metric, "length_bin": label, **res})
            if res["effect_size"] is not None:
                w = min(res["n_pre"], res["n_post"])
                weighted_num += res["effect_size"] * w
                weighted_den += w
        if weighted_den:
            rows.append({"metric": metric, "length_bin": "POOLED (weighted)",
                         "n_pre": None, "n_post": None,
                         "median_pre": None, "median_post": None,
                         "effect_size": round(weighted_num / weighted_den, 4),
                         "effect_ci_low": None, "effect_ci_high": None,
                         "p_value": None})
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_robustness() -> None:
    ling_path = TABLES_DIR / "linguistic_metrics.csv"
    if not ling_path.exists() or not DEDUPED_CSV.exists():
        logger.error("Run the main pipeline first (missing metrics or corpus).")
        sys.exit(1)

    metrics = pd.read_csv(ling_path, low_memory=False)
    corpus = pd.read_csv(DEDUPED_CSV, low_memory=False)

    # author_id lives in the corpus file, not the metrics table
    metrics = metrics.merge(
        corpus[["article_id", "author_id"]], on="article_id", how="left"
    )

    op = metrics[metrics["corpus"] == "opinions"].copy()
    news = metrics[metrics["corpus"] == "news"].copy()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    # ── 1. genre ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("1. GENRE MIX INSIDE THE OPINIONS CORPUS")
    print("=" * 80)
    gm = genre_mix(op)
    gm.to_csv(TABLES_DIR / "diag_genre_mix.csv", index=False)
    share_cols = ["year", "n_total"] + [c for c in gm.columns if c.startswith("share_")]
    print(gm[share_cols].to_string(index=False))

    gs = genre_stratified_tests(op)
    gs.to_csv(TABLES_DIR / "diag_genre_stratified_tests.csv", index=False)
    print("\n--- pre/post effects WITHIN each genre ---")
    if gs.empty:
        print("  No genre had >= %d articles on both sides." % MIN_GROUP)
    else:
        print(gs[["article_type", "metric", "n_pre", "n_post",
                  "effect_size", "effect_ci_low", "effect_ci_high",
                  "p_value"]].to_string(index=False))
    print("\n  Effects that hold within genres are NOT caused by the genre mix.")

    # ── 2. authors ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("2. AUTHOR CONCENTRATION (Opinions)")
    print("=" * 80)
    ac = author_concentration(op)
    ac.to_csv(TABLES_DIR / "diag_author_concentration.csv", index=False)
    print(ac.to_string(index=False))
    print("\n  effective_n_authors = 1/HHI. If it falls sharply, later years rest")
    print("  on fewer distinct voices and article-level tests over-count them.")

    al = author_level_tests(op)
    al.to_csv(TABLES_DIR / "diag_author_level_tests.csv", index=False)
    print("\n--- re-tests with prolific-author leverage removed ---")
    print(al[["approach", "metric", "n_pre", "n_post", "median_pre", "median_post",
              "effect_size", "effect_ci_low", "effect_ci_high",
              "p_value"]].to_string(index=False))
    print("\n  'author_mean' treats each AUTHOR as one observation.")
    print("  If effects shrink toward zero there, they were driven by prolific writers.")

    # ── 3. length ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("3. ARTICLE-LENGTH CONFOUND")
    print("=" * 80)
    lt = length_trends(op, news)
    lt.to_csv(TABLES_DIR / "diag_length_trends.csv", index=False)
    print("\n  Median word count by year:")
    piv = lt.pivot(index="year", columns="corpus", values="median_word_count")
    print(piv.to_string())

    ls = length_stratified_tests(op)
    ls.to_csv(TABLES_DIR / "diag_length_stratified_tests.csv", index=False)
    print("\n--- pre/post effects WITHIN length bins (coarsened exact matching) ---")
    if ls.empty:
        print("  Not enough data to stratify by length.")
    else:
        print(ls[["metric", "length_bin", "n_pre", "n_post",
                  "effect_size", "effect_ci_low", "effect_ci_high"]].to_string(index=False))
    print("\n  Compare each POOLED (weighted) value against the unstratified effect")
    print("  from diagnose_confounds.py. A large drop means the effect was length.")

    print("\n" + "=" * 80)
    print("Saved 6 diagnostic tables to reports/tables/ (prefix diag_)")
    print("=" * 80)
    print("\nREMINDER: all results are temporal associations. Surviving these")
    print("checks rules out specific artefacts; it does not establish a cause.")


if __name__ == "__main__":
    run_robustness()
