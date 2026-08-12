"""
diagnose_confounds.py — Confound diagnostics for the corpus-composition problem.

The pooled tests in statistics.py mix the Opinions and News corpora. Because the
Opinions share of the corpus falls sharply over the study window, a pooled
pre/post difference can be produced entirely by composition change, with no
linguistic change in either corpus. This module separates the two.

It answers three questions:

  1. COMPOSITION — how did the Opinions/News mix change per year?
  2. WITHIN-CORPUS — do the pre/post differences survive when Opinions and News
     are tested separately? (News acts as the control corpus.)
  3. CHANGE-POINT ROBUSTNESS — statistics.py only searches 2021-2024, so it is
     guaranteed to return a breakpoint in that window. Here the search is
     widened across the whole series to test whether 2021-2024 is special.

Run from the project root, after the main pipeline:
  python -m src.diagnose_confounds

Writes:
  reports/tables/diag_composition.csv
  reports/tables/diag_within_corpus_tests.csv
  reports/tables/diag_ai_vocab_by_corpus.csv
  reports/tables/diag_changepoint_widened.csv
"""

import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.config import (
    DEDUPED_CSV,
    TABLES_DIR,
    ALPHA,
    MULTIPLE_CORRECTION,
    AI_EXPLORATORY_VOCAB,
)
from src.statistics import bootstrap_effect_size_ci, fit_segmented_regression
from src.vocabulary import tokenize
from src.utils import get_logger

logger = get_logger(__name__)

PRE_YEARS = (2015, 2019)
POST_YEARS = (2024, 2026)

TEST_METRICS = [
    "mattr", "ttr", "root_ttr", "hapax_ratio", "func_word_ratio",
    "avg_sentence_len", "med_sentence_len",
    "flesch_reading_ease", "flesch_kincaid_grade",
]


# ─── 1. Composition ───────────────────────────────────────────────────────────

def composition_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in sorted(df["year"].dropna().unique()):
        sub = df[df["year"] == yr]
        n_op = int((sub["corpus"] == "opinions").sum())
        n_nw = int((sub["corpus"] == "news").sum())
        total = n_op + n_nw
        rows.append({
            "year": int(yr),
            "n_opinions": n_op,
            "n_news": n_nw,
            "n_total": total,
            "opinions_share": round(n_op / total, 4) if total else None,
        })
    return pd.DataFrame(rows)


# ─── 2. Within-corpus pre/post tests ──────────────────────────────────────────

def within_corpus_tests(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Run the pre/post comparison separately inside each corpus."""
    rows = []
    for corpus in ["opinions", "news"]:
        sub = metrics_df[metrics_df["corpus"] == corpus]
        for metric in TEST_METRICS:
            if metric not in sub.columns:
                continue
            a = sub[sub["year"].between(*PRE_YEARS)][metric].dropna()
            b = sub[sub["year"].between(*POST_YEARS)][metric].dropna()

            if len(a) < 5 or len(b) < 5:
                rows.append({"corpus": corpus, "metric": metric,
                             "n_pre": len(a), "n_post": len(b),
                             "median_pre": None, "median_post": None,
                             "p_value": None, "effect_size": None,
                             "effect_ci_low": None, "effect_ci_high": None})
                continue

            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            r = 1 - (2 * u) / (len(a) * len(b))
            lo, hi = bootstrap_effect_size_ci(a.values, b.values)

            rows.append({
                "corpus": corpus,
                "metric": metric,
                "n_pre": len(a),
                "n_post": len(b),
                "median_pre": round(float(a.median()), 4),
                "median_post": round(float(b.median()), 4),
                "p_value": float(p),
                "effect_size": round(r, 4),
                "effect_ci_low": lo,
                "effect_ci_high": hi,
            })

    out = pd.DataFrame(rows)
    valid = out["p_value"].notna()
    if valid.any():
        corr = multipletests(out.loc[valid, "p_value"],
                             alpha=ALPHA, method=MULTIPLE_CORRECTION)
        out.loc[valid, "p_corrected"] = corr[1]
        out["significant"] = out["p_corrected"] < ALPHA
    else:
        out["p_corrected"] = None
        out["significant"] = False
    return out


# ─── 3. AI-associated vocabulary, split by corpus ─────────────────────────────

def ai_vocab_by_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Per-million frequency of the exploratory vocabulary, per year AND corpus."""
    vocab = set(w.lower() for w in AI_EXPLORATORY_VOCAB)
    rows = []
    for corpus in ["opinions", "news"]:
        sub_c = df[df["corpus"] == corpus]
        for yr in sorted(sub_c["year"].dropna().unique()):
            sub = sub_c[sub_c["year"] == yr]
            tokens = []
            for text in sub["text"].dropna():
                tokens.extend(tokenize(str(text), remove_stopwords=False))
            total = len(tokens)
            if total == 0:
                continue
            freq = Counter(tokens)
            hits = sum(freq.get(w, 0) for w in vocab)
            rows.append({
                "corpus": corpus,
                "year": int(yr),
                "n_articles": len(sub),
                "total_words": total,
                "ai_vocab_hits": hits,
                "freq_per_million": round(hits / total * 1_000_000, 2),
            })
    return pd.DataFrame(rows)


# ─── 4. Widened change-point search ───────────────────────────────────────────

def widened_changepoint(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Search every feasible breakpoint across the full series, per corpus.

    If the best breakpoint for a metric lands outside 2021-2024, or if several
    candidate years fit almost equally well, then the 'AI era' breakpoint
    reported by statistics.py is an artefact of its restricted search window.
    """
    rows = []
    for corpus in ["opinions", "news"]:
        sub = metrics_df[metrics_df["corpus"] == corpus]
        for metric in TEST_METRICS:
            if metric not in sub.columns:
                continue
            yearly = sub.groupby("year")[metric].median().reset_index().dropna()
            if len(yearly) < 6:
                continue

            years = yearly["year"].values.astype(float)
            values = yearly[metric].values.astype(float)

            slope, intercept, r_lin, p_lin, _ = stats.linregress(years, values)
            r2_linear = float(r_lin ** 2) if np.isfinite(r_lin) else np.nan

            candidates = []
            lo_bp, hi_bp = int(years.min()) + 2, int(years.max()) - 1
            for bp in range(lo_bp, hi_bp + 1):
                fit = fit_segmented_regression(years, values, bp)
                if fit.get("valid"):
                    candidates.append((bp, fit["r2"]))

            if not candidates:
                continue

            candidates.sort(key=lambda t: -t[1])
            best_bp, best_r2 = candidates[0]
            runner_bp, runner_r2 = (candidates[1] if len(candidates) > 1
                                    else (None, None))

            rows.append({
                "corpus": corpus,
                "metric": metric,
                "best_breakpoint": best_bp,
                "best_r2": round(best_r2, 4),
                "runner_up_breakpoint": runner_bp,
                "runner_up_r2": round(runner_r2, 4) if runner_r2 is not None else None,
                "r2_gap_to_runner_up": (round(best_r2 - runner_r2, 4)
                                        if runner_r2 is not None else None),
                "r2_linear": round(r2_linear, 4) if np.isfinite(r2_linear) else None,
                "search_window": f"{lo_bp}-{hi_bp}",
                "in_2021_2024": 2021 <= best_bp <= 2024,
            })
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_diagnostics() -> None:
    if not DEDUPED_CSV.exists():
        logger.error("Deduplicated CSV not found. Run the pipeline first.")
        sys.exit(1)

    ling_path = TABLES_DIR / "linguistic_metrics.csv"
    if not ling_path.exists():
        logger.error("linguistic_metrics.csv not found. Run the pipeline first.")
        sys.exit(1)

    df = pd.read_csv(DEDUPED_CSV, low_memory=False)
    metrics_df = pd.read_csv(ling_path, low_memory=False)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 200)

    # 1 ─ composition
    comp = composition_table(df)
    comp.to_csv(TABLES_DIR / "diag_composition.csv", index=False)
    print("\n" + "=" * 78)
    print("1. CORPUS COMPOSITION BY YEAR")
    print("=" * 78)
    print(comp.to_string(index=False))
    first, last = comp.iloc[0], comp.iloc[-1]
    print(f"\nOpinions share: {first['opinions_share']:.1%} ({first['year']}) "
          f"-> {last['opinions_share']:.1%} ({last['year']})")
    print("If this shifted substantially, POOLED pre/post tests are confounded.")

    # 2 ─ within-corpus tests
    wc = within_corpus_tests(metrics_df)
    wc.to_csv(TABLES_DIR / "diag_within_corpus_tests.csv", index=False)
    print("\n" + "=" * 78)
    print("2. PRE/POST TESTS *WITHIN* EACH CORPUS  (News = control)")
    print("=" * 78)
    for corpus in ["opinions", "news"]:
        s = wc[wc["corpus"] == corpus]
        print(f"\n--- {corpus.upper()} ---")
        print(s[["metric", "n_pre", "n_post", "median_pre", "median_post",
                 "effect_size", "effect_ci_low", "effect_ci_high",
                 "p_corrected", "significant"]].to_string(index=False))

    print("\nHow to read this:")
    print("  * Effect in Opinions but NOT News -> specific to opinion writing.")
    print("  * Effect in BOTH -> general editorial/topical drift, not AI-specific.")
    print("  * Effect ONLY when pooled -> composition artefact.")

    # 3 ─ AI vocabulary by corpus
    ai = ai_vocab_by_corpus(df)
    ai.to_csv(TABLES_DIR / "diag_ai_vocab_by_corpus.csv", index=False)
    print("\n" + "=" * 78)
    print("3. EXPLORATORY AI-ASSOCIATED VOCABULARY, BY CORPUS (per million words)")
    print("=" * 78)
    pivot = ai.pivot(index="year", columns="corpus", values="freq_per_million")
    if {"opinions", "news"}.issubset(pivot.columns):
        pivot["opinions_minus_news"] = (pivot["opinions"] - pivot["news"]).round(1)
    print(pivot.to_string())
    print("\nIf both columns rise together, the rise is NOT specific to opinion writing.")

    # 4 ─ widened change-point
    cp = widened_changepoint(metrics_df)
    cp.to_csv(TABLES_DIR / "diag_changepoint_widened.csv", index=False)
    print("\n" + "=" * 78)
    print("4. CHANGE-POINT SEARCH, WINDOW WIDENED TO THE FULL SERIES")
    print("=" * 78)
    print(cp[["corpus", "metric", "best_breakpoint", "best_r2",
              "runner_up_breakpoint", "r2_gap_to_runner_up",
              "r2_linear", "in_2021_2024"]].to_string(index=False))
    if not cp.empty:
        share = cp["in_2021_2024"].mean()
        print(f"\nBreakpoints landing in 2021-2024: {share:.0%}")
        print("A small r2_gap_to_runner_up means the breakpoint year is NOT well")
        print("identified — several years fit about equally well.")

    print("\n" + "=" * 78)
    print("Saved: diag_composition.csv, diag_within_corpus_tests.csv,")
    print("       diag_ai_vocab_by_corpus.csv, diag_changepoint_widened.csv")
    print("=" * 78)
    print("\nREMINDER: these are temporal associations. Nothing here identifies")
    print("a cause, and the exploratory vocabulary list is not an AI detector.")


if __name__ == "__main__":
    run_diagnostics()
