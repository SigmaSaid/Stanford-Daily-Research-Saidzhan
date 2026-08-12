"""
figures_article.py — The two publication figures the article needs.

The figures produced by visualize.py are computed on the POOLED opinion corpus,
mixing student writers with guest and institutional contributors. The article's
central claims are about student writers specifically, so illustrating them
with pooled charts would show a different quantity than the sentence beside it
describes — exactly the conflation the analysis exists to expose.

These two read the student/guest split directly:

  FIG_02_student_mattr_by_year.png
      Median lexical diversity of student-authored opinion articles, with the
      sample size printed above each point and the fitted change point marked.

  FIG_03_similarity_student_vs_guest.png
      Within-year semantic similarity, students versus guest contributors,
      with the pooled series shown for comparison.

Run from the project root, after diagnose_final.py and diagnose_similarity.py:
  python -m src.figures_article

Writes both PNGs to reports/figures/.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import TABLES_DIR, FIGURES_DIR, FIGURE_DPI
from src.utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", context="talk", palette="muted")
plt.rcParams["figure.dpi"] = FIGURE_DPI
plt.rcParams["savefig.dpi"] = FIGURE_DPI

CHATGPT_LINE = 2022.9   # public release, late November 2022
STUDENT_C = "#8c1515"   # cardinal
GUEST_C = "#b7791f"
POOLED_C = "#9aa0a8"


def _save(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved figure → %s", path)
    return path


# ─── Figure 2 ─────────────────────────────────────────────────────────────────

def fig_student_mattr() -> bool:
    src = TABLES_DIR / "diag_final_student_yearly.csv"
    if not src.exists():
        logger.error("Missing %s — run `python -m src.diagnose_final` first.", src.name)
        return False

    df = pd.read_csv(src).dropna(subset=["mattr"]).sort_values("year")
    if df.empty:
        logger.error("No student-only rows to plot.")
        return False

    # Break point, if the change-point table is available
    bp = None
    cp_path = TABLES_DIR / "diag_final_student_changepoint.csv"
    if cp_path.exists():
        cp = pd.read_csv(cp_path)
        row = cp[cp["metric"] == "mattr"]
        if not row.empty and "best_breakpoint" in row.columns:
            bp = int(row.iloc[0]["best_breakpoint"])

    fig, ax = plt.subplots(figsize=(12, 6.4))

    ax.plot(df["year"], df["mattr"], marker="o", markersize=9, linewidth=2.6,
            color=STUDENT_C, zorder=3, label="Student-authored opinion articles")

    # Sample size above each point: the reader must see how thin later years are
    span = df["mattr"].max() - df["mattr"].min()
    for _, r in df.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["year"], r["mattr"]),
                    textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=10.5, color="#5c6169", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=0.6, alpha=.88))

    ax.axvline(CHATGPT_LINE, color="#2b6cb0", linestyle="--", linewidth=2,
               alpha=.75, zorder=1)
    # Anchored to the top of the axes, left of the line: the 2023 data point
    # sits at the series minimum and its n-label occupies the space below.
    ax.annotate("ChatGPT released\nNov 2022",
                xy=(CHATGPT_LINE, 1.0), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(-10, -14),
                fontsize=11, color="#2b6cb0", ha="right", va="top")

    if bp is not None:
        ax.axvspan(bp - .5, df["year"].max() + .5, color=STUDENT_C, alpha=.055,
                   zorder=0)
        ax.annotate(f"fitted change point: {bp}",
                    xy=(bp, 1.0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(10, -14),
                    fontsize=11, color=STUDENT_C, ha="left", va="top")

    ax.set_xlabel("Year")
    ax.set_ylabel("Lexical diversity (MATTR, median)")
    ax.set_title("Lexical diversity of student-authored opinion writing",
                 pad=32, loc="left")
    ax.set_xticks(df["year"].astype(int))
    ax.set_xticklabels(df["year"].astype(int), rotation=0)
    ax.set_ylim(df["mattr"].min() - span * .35, df["mattr"].max() + span * .40)
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc="upper left", frameon=True, fontsize=11.5)

    fig.text(0.0, -0.035,
             "Guest and institutional submissions excluded. Higher values mean "
             "a wider vocabulary within an article.\nMATTR uses a moving window "
             "and is not inflated by article length. Later years rest on small "
             "samples — read each point against its n.",
             fontsize=10.5, color="#5c6169", ha="left")

    _save("FIG_02_student_mattr_by_year")
    return True


# ─── Figure 3 ─────────────────────────────────────────────────────────────────

def fig_similarity_split() -> bool:
    src = TABLES_DIR / "diag_similarity_by_group.csv"
    if not src.exists():
        logger.error("Missing %s — run `python -m src.diagnose_similarity` first.",
                     src.name)
        return False

    df = pd.read_csv(src)
    piv = df.pivot(index="year", columns="group", values="mean_sim")
    nused = df.pivot(index="year", columns="group", values="n_used")
    if "student" not in piv.columns:
        logger.error("No 'student' group in similarity table.")
        return False

    trends = None
    tp = TABLES_DIR / "diag_similarity_trends.csv"
    if tp.exists():
        trends = pd.read_csv(tp).set_index("group")

    def _lab(group, pretty):
        """Append the trend statistic to a legend label, when available."""
        if trends is None or group not in trends.index:
            return pretty
        r = trends.loc[group]
        rho, p = r.get("spearman_rho"), r.get("spearman_p")
        if pd.isna(rho) or pd.isna(p):
            return pretty
        p_txt = "p<0.01" if p < 0.01 else f"p={p:.2f}"
        return f"{pretty}  (\u03c1={rho:.2f}, {p_txt})"

    fig, ax = plt.subplots(figsize=(12, 6.4))
    yrs = piv.index.astype(int)

    if "all" in piv.columns:
        ax.plot(yrs, piv["all"], marker="s", markersize=6, linewidth=1.8,
                color=POOLED_C, linestyle=":", zorder=2,
                label=_lab("all", "All articles (pooled)"))
    if "guest" in piv.columns:
        ax.plot(yrs, piv["guest"], marker="^", markersize=8, linewidth=2.2,
                color=GUEST_C, zorder=3,
                label=_lab("guest", "Guest / community submissions"))
    ax.plot(yrs, piv["student"], marker="o", markersize=9, linewidth=2.8,
            color=STUDENT_C, zorder=4,
            label=_lab("student", "Student-authored opinion"))

    ax.axvline(CHATGPT_LINE, color="#2b6cb0", linestyle="--", linewidth=1.8,
               alpha=.6, zorder=1)

    lo = float(np.nanmin(piv[[c for c in ["all", "guest", "student"]
                              if c in piv.columns]].values))
    hi = float(np.nanmax(piv[[c for c in ["all", "guest", "student"]
                              if c in piv.columns]].values))
    pad = (hi - lo) * .18

    if "student" in nused.columns:
        for y in yrs:
            n = nused.loc[y, "student"]
            if pd.notna(n) and n > 0:
                ax.annotate(f"{int(n)}", (y, piv.loc[y, "student"]),
                            textcoords="offset points", xytext=(0, -19),
                            ha="center", fontsize=9.5, color=STUDENT_C,
                            zorder=6, bbox=dict(facecolor="white",
                            edgecolor="none", pad=0.5, alpha=.85))

    ax.set_xlabel("Year")
    ax.set_ylabel("Within-year semantic similarity")
    ax.set_title("Are opinion articles becoming more alike?", pad=22, loc="left")
    ax.set_xticks(yrs)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc="upper left", frameon=True, fontsize=11)

    fig.text(0.0, -0.045,
             "Higher values mean that year's articles resemble one another more "
             "closely. Small red figures are the number of student\narticles "
             "compared in that year. Student writing converges; guest "
             "submissions do not. Similarity is measured on\narticle openings, "
             "because the embedding model truncates long inputs.",
             fontsize=10.5, color="#5c6169", ha="left")

    _save("FIG_03_similarity_student_vs_guest")
    return True


def main() -> None:
    ok2 = fig_student_mattr()
    ok3 = fig_similarity_split()

    made = [n for n, ok in [("FIG_02_student_mattr_by_year.png", ok2),
                            ("FIG_03_similarity_student_vs_guest.png", ok3)] if ok]
    print("\nFigures written to reports/figures/:")
    for m in made:
        print(f"  {m}")
    if len(made) < 2:
        print("\nSome figures were skipped — see the errors above.")
        sys.exit(1)
    print("\nBoth use the student/guest split, so they match the article's claims.")
    print("Do not substitute 03_mattr_by_year.png or 11_semantic_similarity.png:")
    print("those pool students with guest contributors.")


if __name__ == "__main__":
    main()
