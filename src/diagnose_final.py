"""
diagnose_final.py — The last check before writing up.

diagnose_authorship.py showed that guest submissions were SUPPRESSING the
effect, and that student-only estimates are larger than the all-Opinions ones
(e.g. MATTR 0.372 -> 0.518). Two questions remain before that number can be
reported as a finding about student writing:

  1. IS IT A FEW COLUMNISTS? The student-only post period is ~162 articles and
     author concentration is high. Article-level tests pseudo-replicate: 30
     pieces by one columnist are not 30 independent observations. Here each
     AUTHOR contributes one value, and the top writers are also dropped.

  2. WHEN DID IT CHANGE? This is the question the whole study is about. The
     year-by-year student-only trajectory, plus an unrestricted change-point
     search, shows whether the change is a step near generative-AI adoption or
     a gradual trend that predates it.

Run from the project root, after diagnose_authorship.py:
  python -m src.diagnose_final

Writes:
  reports/tables/diag_final_author_level.csv
  reports/tables/diag_final_student_yearly.csv
  reports/tables/diag_final_student_changepoint.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DEDUPED_CSV, TABLES_DIR
from src.statistics import bootstrap_effect_size_ci, fit_segmented_regression
from src.diagnose_authorship import flag_guest_content
from src.utils import get_logger

logger = get_logger(__name__)

PRE_YEARS = (2015, 2019)
POST_YEARS = (2024, 2026)
MIN_GROUP = 15

# root_ttr is excluded: rho = 0.74 with word count makes it length-driven.
FOCUS_METRICS = ["mattr", "func_word_ratio", "avg_sentence_len",
                 "flesch_reading_ease"]


def _effect(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.dropna(), b.dropna()
    if len(a) < MIN_GROUP or len(b) < MIN_GROUP:
        return {"n_pre": len(a), "n_post": len(b), "median_pre": None,
                "median_post": None, "effect_size": None,
                "effect_ci_low": None, "effect_ci_high": None, "p_value": None}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - (2 * u) / (len(a) * len(b))
    lo, hi = bootstrap_effect_size_ci(a.values, b.values, n_boot=2000)
    return {"n_pre": len(a), "n_post": len(b),
            "median_pre": round(float(a.median()), 4),
            "median_post": round(float(b.median()), 4),
            "effect_size": round(r, 4), "effect_ci_low": lo,
            "effect_ci_high": hi, "p_value": float(p)}


def author_robust_tests(stu: pd.DataFrame) -> pd.DataFrame:
    """Student-only effects with prolific-writer leverage progressively removed."""
    pre = stu[stu["year"].between(*PRE_YEARS)]
    post = stu[stu["year"].between(*POST_YEARS)]

    rows = []
    for metric in FOCUS_METRICS:
        if metric not in stu.columns:
            continue

        rows.append({"approach": "1_article_level", "metric": metric,
                     **_effect(pre[metric], post[metric])})

        # each author counts once
        rows.append({"approach": "2_author_as_unit", "metric": metric,
                     **_effect(pre.groupby("author_id")[metric].mean(),
                               post.groupby("author_id")[metric].mean())})

        # drop the 5 most prolific student writers in each period
        t_pre = pre["author_id"].value_counts().head(5).index
        t_post = post["author_id"].value_counts().head(5).index
        rows.append({"approach": "3_drop_top5_authors", "metric": metric,
                     **_effect(pre[~pre["author_id"].isin(t_pre)][metric],
                               post[~post["author_id"].isin(t_post)][metric])})
    return pd.DataFrame(rows)


def student_author_concentration(stu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in sorted(stu["year"].dropna().unique()):
        sub = stu[stu["year"] == yr]
        counts = sub["author_id"].dropna().value_counts()
        if counts.empty:
            continue
        shares = counts / counts.sum()
        hhi = float((shares ** 2).sum())
        rows.append({"year": int(yr), "n_student_articles": int(counts.sum()),
                     "n_authors": int(counts.size),
                     "top1_share": round(float(shares.iloc[0]), 3),
                     "effective_n_authors": round(1 / hhi, 1) if hhi else None})
    return pd.DataFrame(rows)


def student_yearly(stu: pd.DataFrame) -> pd.DataFrame:
    """Year-by-year medians on student-authored articles only."""
    rows = []
    for yr in sorted(stu["year"].dropna().unique()):
        sub = stu[stu["year"] == yr]
        row = {"year": int(yr), "n": len(sub)}
        for metric in FOCUS_METRICS:
            if metric in sub.columns:
                row[metric] = round(float(sub[metric].median()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def student_changepoint(yearly: pd.DataFrame) -> pd.DataFrame:
    """
    Unrestricted change-point search on the student-only series.

    Reports the linear baseline alongside, because a segmented model always
    fits at least as well as a straight line and will therefore always return
    some breakpoint.
    """
    rows = []
    years = yearly["year"].values.astype(float)
    for metric in FOCUS_METRICS:
        if metric not in yearly.columns:
            continue
        vals = yearly[metric].values.astype(float)
        ok = ~np.isnan(vals)
        if ok.sum() < 6:
            continue
        yy, vv = years[ok], vals[ok]

        slope, _, r_lin, p_lin, _ = stats.linregress(yy, vv)
        cands = []
        for bp in range(int(yy.min()) + 2, int(yy.max())):
            fit = fit_segmented_regression(yy, vv, bp)
            if fit.get("valid"):
                cands.append((bp, fit["r2"], fit["slope_before"], fit["slope_after"]))
        if not cands:
            continue
        cands.sort(key=lambda t: -t[1])
        bp, r2, sb, sa = cands[0]
        rows.append({
            "metric": metric,
            "best_breakpoint": bp,
            "best_r2": round(r2, 4),
            "runner_up": cands[1][0] if len(cands) > 1 else None,
            "r2_gap": round(r2 - cands[1][1], 4) if len(cands) > 1 else None,
            "r2_linear": round(float(r_lin ** 2), 4),
            "linear_p": round(float(p_lin), 5),
            "slope_before": sb, "slope_after": sa,
        })
    return pd.DataFrame(rows)


def run_final() -> None:
    ling = TABLES_DIR / "linguistic_metrics.csv"
    if not ling.exists() or not DEDUPED_CSV.exists():
        logger.error("Run the main pipeline first.")
        sys.exit(1)

    metrics = pd.read_csv(ling, low_memory=False)
    corpus = pd.read_csv(DEDUPED_CSV, low_memory=False)
    keep = [c for c in ["article_id", "author_id", "author_name", "title"]
            if c in corpus.columns]
    metrics = metrics.merge(corpus[keep], on="article_id", how="left")

    op = flag_guest_content(metrics[metrics["corpus"] == "opinions"].copy())
    stu = op[~op["is_guest"]].copy()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    print("\n" + "=" * 80)
    print("STUDENT-AUTHORED OPINION ARTICLES ONLY  (guest content removed)")
    print("=" * 80)
    print(f"Total student articles: {len(stu)}")

    conc = student_author_concentration(stu)
    print("\n--- author concentration among student writers ---")
    print(conc.to_string(index=False))

    # ── 1 ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("1. IS THE EFFECT DRIVEN BY A FEW COLUMNISTS?")
    print("=" * 80)
    ar = author_robust_tests(stu)
    ar.to_csv(TABLES_DIR / "diag_final_author_level.csv", index=False)
    for metric in FOCUS_METRICS:
        sub = ar[ar["metric"] == metric]
        if sub.empty:
            continue
        print(f"\n--- {metric} ---")
        print(sub[["approach", "n_pre", "n_post", "median_pre", "median_post",
                   "effect_size", "effect_ci_low", "effect_ci_high",
                   "p_value"]].to_string(index=False))
    print("\n  If '2_author_as_unit' stays close to '1_article_level', the change")
    print("  is broad across writers. If it collapses, a few columnists drove it.")

    # ── 2 ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("2. WHEN DID IT CHANGE?  (student-only, year by year)")
    print("=" * 80)
    yearly = student_yearly(stu)
    yearly.to_csv(TABLES_DIR / "diag_final_student_yearly.csv", index=False)
    print(yearly.to_string(index=False))

    cp = student_changepoint(yearly)
    cp.to_csv(TABLES_DIR / "diag_final_student_changepoint.csv", index=False)
    print("\n--- unrestricted change-point search ---")
    print(cp.to_string(index=False))
    print("\n  A HIGH r2_linear means a gradual trend across the whole window.")
    print("  A breakpoint is only meaningful if r2_gap is large AND it clearly")
    print("  beats the linear baseline. Breakpoints before 2022 cannot be")
    print("  attributed to generative AI.")

    print("\n" + "=" * 80)
    print("Saved 3 tables (prefix diag_final_)")
    print("=" * 80)
    print("\nNOTE: yearly student n may be small in later years. Read the")
    print("trajectory alongside the n column before drawing conclusions.")


if __name__ == "__main__":
    run_final()
