"""
diagnose_authorship.py — Separate student writing from guest submissions.

Inspection of the Opinions corpus revealed that the single largest byline in
the post-2024 period is a shared account used for guest submissions
("From the Community"), carrying pieces by alumni, parents, faculty and
outside professionals. Those authors are not students.

Since the research question concerns STUDENT writing, and since guest
contributors would plausibly write more formally than undergraduates, a
change in the share of guest submissions could produce the entire measured
pre/post effect without any student writing changing at all.

This module answers three questions:

  1. PREVALENCE — what share of each year's Opinions corpus is guest content,
     and did that share change across the study window?

  2. MECHANISM — within a SINGLE period, do guest pieces differ linguistically
     from student pieces? If they differ in the same direction as the pre/post
     effect, composition is a sufficient explanation for it.

  3. CORRECTED ESTIMATE — re-run the pre/post comparison on student-authored
     articles only. Whatever survives here is the defensible finding.

Run from the project root, after the main pipeline:
  python -m src.diagnose_authorship

Writes:
  reports/tables/diag_guest_share_by_year.csv
  reports/tables/diag_guest_vs_student.csv
  reports/tables/diag_student_only_tests.csv
"""

import sys

import pandas as pd
from scipy import stats

from src.config import DEDUPED_CSV, TABLES_DIR
from src.statistics import bootstrap_effect_size_ci
from src.utils import get_logger

logger = get_logger(__name__)

PRE_YEARS = (2015, 2019)
POST_YEARS = (2024, 2026)
MIN_GROUP = 20

FOCUS_METRICS = [
    "mattr", "func_word_ratio", "avg_sentence_len",
    "flesch_reading_ease", "root_ttr",
]

# Title prefixes used by The Stanford Daily for non-student / institutional
# content. Matched case-insensitively against the start of the title.
GUEST_TITLE_PATTERNS = [
    "from the community",
    "letter to the editor",
    "letter from",
    "op-ed:",
    "guest column",
]

# Bylines that are institutional rather than individual.
INSTITUTIONAL_NAME_PATTERNS = [
    "editorial board", "the daily", "staff", "community", "opinions desk",
]


def _effect(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.dropna(), b.dropna()
    if len(a) < MIN_GROUP or len(b) < MIN_GROUP:
        return {"n_a": len(a), "n_b": len(b), "median_a": None, "median_b": None,
                "effect_size": None, "effect_ci_low": None,
                "effect_ci_high": None, "p_value": None}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - (2 * u) / (len(a) * len(b))
    lo, hi = bootstrap_effect_size_ci(a.values, b.values, n_boot=1000)
    return {"n_a": len(a), "n_b": len(b),
            "median_a": round(float(a.median()), 4),
            "median_b": round(float(b.median()), 4),
            "effect_size": round(r, 4),
            "effect_ci_low": lo, "effect_ci_high": hi,
            "p_value": float(p)}


def flag_guest_content(op: pd.DataFrame) -> pd.DataFrame:
    """
    Mark each opinion article as guest/institutional or student-authored.

    Two independent signals are used, because neither is complete on its own:
      - the title prefix (TSD labels guest content explicitly)
      - a shared byline account carrying a large share of the corpus
    """
    op = op.copy()
    title_l = op["title"].fillna("").str.lower().str.strip()

    by_title = title_l.str.startswith(tuple(GUEST_TITLE_PATTERNS))

    name_l = op.get("author_name")
    if name_l is not None:
        name_l = name_l.fillna("").str.lower()
        by_name = name_l.apply(
            lambda s: any(p in s for p in INSTITUTIONAL_NAME_PATTERNS) if s else False
        )
    else:
        by_name = pd.Series(False, index=op.index)

    # A byline carrying >15% of the whole Opinions corpus is institutional,
    # not an individual student writer.
    shares = op["author_id"].value_counts(normalize=True)
    bulk_ids = set(shares[shares > 0.15].index)
    by_volume = op["author_id"].isin(bulk_ids)

    op["is_guest"] = by_title | by_name | by_volume
    op["guest_signal"] = (
        by_title.map({True: "title", False: ""})
        + by_name.map({True: "|name", False: ""})
        + by_volume.map({True: "|volume", False: ""})
    ).str.strip("|")
    return op


def guest_share_by_year(op: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in sorted(op["year"].dropna().unique()):
        sub = op[op["year"] == yr]
        n_guest = int(sub["is_guest"].sum())
        rows.append({
            "year": int(yr),
            "n_total": len(sub),
            "n_guest": n_guest,
            "n_student": len(sub) - n_guest,
            "guest_share": round(n_guest / len(sub), 4) if len(sub) else None,
        })
    return pd.DataFrame(rows)


def guest_vs_student(op: pd.DataFrame) -> pd.DataFrame:
    """
    Compare guest vs student pieces WITHIN the same period.

    Holding the period fixed removes any time trend, so a difference here is
    a pure authorship effect.
    """
    rows = []
    for label, yrs in [("pre_2015_2019", PRE_YEARS), ("post_2024_2026", POST_YEARS)]:
        window = op[op["year"].between(*yrs)]
        for metric in FOCUS_METRICS:
            if metric not in window.columns:
                continue
            student = window[~window["is_guest"]][metric]
            guest = window[window["is_guest"]][metric]
            res = _effect(student, guest)
            rows.append({"period": label, "metric": metric,
                         "comparison": "student_vs_guest", **res})
    return pd.DataFrame(rows)


def student_only_tests(op: pd.DataFrame) -> pd.DataFrame:
    """The corrected pre/post estimate: student-authored articles only."""
    students = op[~op["is_guest"]]
    rows = []
    for metric in FOCUS_METRICS:
        if metric not in students.columns:
            continue
        a = students[students["year"].between(*PRE_YEARS)][metric]
        b = students[students["year"].between(*POST_YEARS)][metric]
        rows.append({"metric": metric, "subset": "student_only", **_effect(a, b)})
    return pd.DataFrame(rows)


def run_authorship_diagnostics() -> None:
    ling_path = TABLES_DIR / "linguistic_metrics.csv"
    if not ling_path.exists() or not DEDUPED_CSV.exists():
        logger.error("Run the main pipeline first.")
        sys.exit(1)

    metrics = pd.read_csv(ling_path, low_memory=False)
    corpus = pd.read_csv(DEDUPED_CSV, low_memory=False)

    keep = [c for c in ["article_id", "author_id", "author_name", "title", "url"]
            if c in corpus.columns]
    metrics = metrics.merge(corpus[keep], on="article_id", how="left")

    op = metrics[metrics["corpus"] == "opinions"].copy()
    if op.empty:
        logger.error("No opinions articles found.")
        sys.exit(1)

    op = flag_guest_content(op)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    # ── 1 ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("1. GUEST / INSTITUTIONAL CONTENT AS A SHARE OF THE OPINIONS CORPUS")
    print("=" * 80)
    gs = guest_share_by_year(op)
    gs.to_csv(TABLES_DIR / "diag_guest_share_by_year.csv", index=False)
    print(gs.to_string(index=False))

    pre_share = op[op["year"].between(*PRE_YEARS)]["is_guest"].mean()
    post_share = op[op["year"].between(*POST_YEARS)]["is_guest"].mean()
    print(f"\nGuest share  pre ({PRE_YEARS[0]}-{PRE_YEARS[1]}): {pre_share:.1%}")
    print(f"Guest share post ({POST_YEARS[0]}-{POST_YEARS[1]}): {post_share:.1%}")
    print(f"Change: {post_share - pre_share:+.1%}")

    print("\nDetection signals used:")
    print(op["guest_signal"].replace("", "student").value_counts().to_string())

    # ── 2 ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("2. DO GUEST PIECES DIFFER FROM STUDENT PIECES *WITHIN* A PERIOD?")
    print("=" * 80)
    gv = guest_vs_student(op)
    gv.to_csv(TABLES_DIR / "diag_guest_vs_student.csv", index=False)
    print(gv[["period", "metric", "n_a", "n_b", "median_a", "median_b",
              "effect_size", "effect_ci_low", "effect_ci_high",
              "p_value"]].to_string(index=False))
    print("\n  n_a = student articles, n_b = guest articles, same period.")
    print("  A non-zero effect here is a pure AUTHORSHIP difference with no time")
    print("  trend involved. If it points the same way as the pre/post effect,")
    print("  then the shift in guest share can account for that effect.")

    # ── 3 ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("3. CORRECTED PRE/POST ESTIMATE — STUDENT-AUTHORED ARTICLES ONLY")
    print("=" * 80)
    so = student_only_tests(op)
    so.to_csv(TABLES_DIR / "diag_student_only_tests.csv", index=False)
    print(so[["metric", "n_a", "n_b", "median_a", "median_b",
              "effect_size", "effect_ci_low", "effect_ci_high",
              "p_value"]].to_string(index=False))
    print("\n  Compare against the all-Opinions effects from diagnose_confounds.py:")
    print("    mattr 0.372 | func_word_ratio -0.464 | avg_sentence_len -0.160")
    print("    flesch_reading_ease -0.261 | root_ttr 0.393")
    print("\n  Effects that shrink toward zero were driven by guest content.")
    print("  Effects that persist are genuine changes in STUDENT writing.")

    print("\n" + "=" * 80)
    print("Saved: diag_guest_share_by_year.csv, diag_guest_vs_student.csv,")
    print("       diag_student_only_tests.csv")
    print("=" * 80)
    print("\nCAUTION: guest detection is heuristic. Verify the flagged counts")
    print("against the printed signal breakdown before relying on the corrected")
    print("estimate, and report the detection rule in the methods section.")


if __name__ == "__main__":
    run_authorship_diagnostics()
