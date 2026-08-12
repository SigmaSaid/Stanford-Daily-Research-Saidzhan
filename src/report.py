"""
report.py — Assemble reports/research_report.md from the real result tables.

This module does NOT compute anything new and does NOT invent any numbers. It
reads the CSVs produced by the pipeline and the diagnostic modules and lays
them out in a structure suitable for writing over.

Interpretation is left to the author. Sections that require human judgement are
marked with >>> WRITE: prompts.

Run from the project root, after:
  python -m src.run_pipeline
  python -m src.diagnose_confounds
  python -m src.diagnose_robustness
  python -m src.diagnose_authorship
  python -m src.diagnose_final

Then:
  python -m src.report

Writes:
  reports/research_report.md
"""

import sys
from datetime import date

import pandas as pd

from src.config import (
    REPORTS_DIR, TABLES_DIR, START_DATE, END_DATE,
    MIN_WORD_COUNT, MAX_WORD_COUNT, MATTR_WINDOW,
    EMBEDDING_MODEL, SPACY_MODEL, ALPHA, MULTIPLE_CORRECTION,
    AI_EXPLORATORY_VOCAB,
)
from src.utils import get_logger

logger = get_logger(__name__)

OUT = REPORTS_DIR / "research_report.md"


def _load(name: str):
    """Load a table if present; return None and warn otherwise."""
    p = TABLES_DIR / name
    if not p.exists():
        logger.warning("Missing table (section will be marked): %s", name)
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:
        logger.warning("Could not read %s: %s", name, exc)
        return None


P_COLUMNS = {"p_value", "p_corrected", "linear_p", "p_uncorrected"}


def _fmt_p(v) -> str:
    """
    Render a p-value for publication.

    Rounding tiny p-values to a fixed number of decimals prints '0', which is
    both wrong and a common reviewer complaint: a p-value is never exactly
    zero. Very small values are reported as an inequality instead.
    """
    if pd.isna(v):
        return ""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v < 1e-16:
        return "< 1e-16"
    if v < 0.001:
        return "< 0.001"
    return f"{v:.3f}"


def _md(df, cols=None, floatfmt=4):
    """Render a DataFrame as a markdown table, or a placeholder if absent."""
    if df is None or df.empty:
        return "_Table not available — run the corresponding module._\n"
    d = df[cols] if cols else df
    d = d.copy()

    # Columns that are structurally empty in this study and would read as a
    # bug if printed. Author display names are never resolvable because the
    # publication's users endpoint requires authentication.
    ALWAYS_DROP_IF_EMPTY = {"n_resolved_author_names", "n_unique_authors"}

    def _is_empty_column(col) -> bool:
        """True if a column carries no information: all NaN, 0, or blank."""
        if col.isna().all():
            return True
        nonnull = col.dropna()
        if nonnull.empty:
            return True
        if pd.api.types.is_numeric_dtype(col):
            return bool((nonnull == 0).all())
        # Object dtype may still hold numbers read as strings.
        coerced = pd.to_numeric(nonnull, errors="coerce")
        if coerced.notna().all():
            return bool((coerced == 0).all())
        return bool((nonnull.astype(str).str.strip() == "").all())

    for c in list(d.columns):
        if c in ("year", "n", "n_articles"):
            continue
        if _is_empty_column(d[c]) or (
            c in ALWAYS_DROP_IF_EMPTY and _is_empty_column(d[c])
        ):
            d = d.drop(columns=[c])

    for c in d.columns:
        if c in P_COLUMNS:
            d[c] = d[c].map(_fmt_p)

    for c in d.select_dtypes("float").columns:
        d[c] = d[c].round(floatfmt)

    try:
        return d.to_markdown(index=False) + "\n"
    except ImportError:
        return "```\n" + d.to_string(index=False) + "\n```\n"


def build_report() -> str:
    yearly = _load("yearly_corpus_statistics.csv")
    pooled = _load("statistical_results.csv")
    within = _load("diag_within_corpus_tests.csv")
    aivoc = _load("diag_ai_vocab_by_corpus.csv")
    guest = _load("diag_guest_share_by_year.csv")
    gvs = _load("diag_guest_vs_student.csv")
    author = _load("diag_final_author_level.csv")
    stu_yr = _load("diag_final_student_yearly.csv")
    stu_cp = _load("diag_final_student_changepoint.csv")
    conc = _load("diag_author_concentration.csv")
    lenstrat = _load("diag_length_stratified_tests.csv")
    genre = _load("diag_genre_mix.csv")

    L = []
    A = L.append

    # ── header ────────────────────────────────────────────────────────────────
    A("# The AI Vocabulary Shift: Linguistic Change in The Stanford Daily, "
      "2015–2026\n")
    A(f"_Generated {date.today().isoformat()} from pipeline output. "
      "All figures below are computed from the collected corpus._\n")

    A("> **Status:** draft skeleton. Numbers are real; interpretation is not "
      "written. Sections marked `>>> WRITE:` require the author.\n")

    # ── corpus ────────────────────────────────────────────────────────────────
    A("\n## 1. Corpus\n")
    if yearly is not None:
        total = int(yearly["n_articles"].sum())
        A(f"Articles analysed: **N = {total:,}**, "
          f"{START_DATE} to {END_DATE}, from The Stanford Daily "
          "WordPress REST API (`/wp-json/wp/v2/posts`).\n")
        A("\n### 1.1 Articles by year\n")
        A(_md(yearly, floatfmt=1))
        A("\n**Notes.** 2026 is partial (through "
          f"{END_DATE}) and is not comparable to full years without "
          "adjustment. Distinct-author counts use the numeric `author_id` "
          "supplied with each post; author display names could not be "
          "retrieved because the publication's `/wp/v2/users/` endpoint "
          "requires authentication (HTTP 401), so authors are pseudonymous "
          "throughout.\n")
    else:
        A("_Corpus table unavailable._\n")

    if genre is not None:
        A("\n### 1.2 Genre composition of the Opinions corpus\n")
        share_cols = ["year", "n_total"] + [c for c in genre.columns
                                            if c.startswith("share_")]
        A(_md(genre, [c for c in share_cols if c in genre.columns], floatfmt=3))
        A("\n>>> WRITE: note whether subcategory tagging changed over the "
          "window, and whether that reflects editorial practice or metadata "
          "practice.\n")

    # ── methods ───────────────────────────────────────────────────────────────
    A("\n## 2. Methods\n")
    A(f"- **Text extraction:** HTML → plain text; scripts, navigation, "
      f"advertising, captions, author bios and related-post blocks removed.\n")
    A(f"- **Filters:** articles retained at {MIN_WORD_COUNT}–{MAX_WORD_COUNT} "
      "words. Deduplicated by article ID, URL and SHA-256 of cleaned text.\n")
    A(f"- **Linguistic metrics:** spaCy `{SPACY_MODEL}`. MATTR uses a "
      f"{MATTR_WINDOW}-token moving window and is length-robust; simple TTR "
      "and root TTR are length-sensitive and reported for reference only.\n")
    A(f"- **Embeddings:** `{EMBEDDING_MODEL}`.\n")
    A(f"- **Tests:** Mann–Whitney U with rank-biserial correlation as effect "
      f"size and percentile bootstrap 95% CIs; {MULTIPLE_CORRECTION} "
      f"correction at α = {ALPHA}.\n")
    A("- **Change-points:** segmented regression, reported against a "
      "single-line baseline. A segmented model always fits at least as well "
      "as a straight line, so a breakpoint is treated as meaningful only when "
      "it clearly beats the linear fit and is well separated from the "
      "runner-up year.\n")
    A("- **Control corpus:** the paper's own news reporting, collected and "
      "processed identically.\n")

    A("\n### 2.1 Periodisation\n")
    A("| Period | Years | Role |\n|---|---|---|\n"
      "| Baseline | 2015–2019 | pre-generative-AI |\n"
      "| Transition | 2020–2023 | excluded from the primary contrast |\n"
      "| Recent | 2024–2026 | post-widespread-availability |\n")

    A("\n### 2.2 Authorship classification\n")
    A("Opinion articles were classified as student-authored or guest/"
      "institutional using three signals: an explicit title prefix (e.g. "
      "'From the Community'), an institutional byline, and any single byline "
      "account holding more than 15% of the Opinions corpus.\n")
    A("\n>>> WRITE: state the exact counts flagged by each signal and "
      "acknowledge that this rule is heuristic.\n")

    # ── results ───────────────────────────────────────────────────────────────
    A("\n## 3. Results\n")

    A("\n### 3.1 Pooled corpus (Opinions + News)\n")
    A("Reported for completeness. These estimates are **confounded** by the "
      "changing Opinions/News mix and should not be interpreted on their own.\n\n")
    A(_md(pooled, [c for c in ["metric", "n_a", "n_b", "median_a", "median_b",
                               "effect_size", "effect_ci_low", "effect_ci_high",
                               "p_corrected"]
                   if pooled is not None and c in pooled.columns]))

    A("\n### 3.2 Within each corpus (News as control)\n")
    A(_md(within, [c for c in ["corpus", "metric", "n_pre", "n_post",
                               "median_pre", "median_post", "effect_size",
                               "effect_ci_low", "effect_ci_high",
                               "p_corrected"]
                   if within is not None and c in within.columns]))
    A("\n>>> WRITE: compare Opinions against News for each metric. Note where "
      "confidence intervals do not overlap, and where News shows no effect.\n")

    A("\n### 3.3 Authorship composition\n")
    A(_md(guest, floatfmt=3))
    A("\n#### Guest vs student pieces within the same period\n")
    A("Holding the period fixed removes any time trend, so differences here "
      "are purely attributable to authorship.\n\n")
    A(_md(gvs, [c for c in ["period", "metric", "n_a", "n_b", "median_a",
                            "median_b", "effect_size", "effect_ci_low",
                            "effect_ci_high", "p_value"]
                if gvs is not None and c in gvs.columns]))
    A("\n>>> WRITE: state the direction of the authorship difference and "
      "whether the change in guest share inflates or suppresses the "
      "pre/post contrast.\n")

    A("\n### 3.4 Student-authored articles only\n")
    A("Primary estimates. Each row applies a progressively stricter control "
      "for prolific-author leverage: `2_author_as_unit` gives every author a "
      "single value, removing pseudo-replication.\n\n")
    A(_md(author, [c for c in ["approach", "metric", "n_pre", "n_post",
                               "median_pre", "median_post", "effect_size",
                               "effect_ci_low", "effect_ci_high", "p_value"]
                   if author is not None and c in author.columns]))

    A("\n#### Author concentration (Opinions)\n")
    A(_md(conc, floatfmt=3))
    A("\n>>> WRITE: report how the effective number of authors changed, and "
      "treat this as a limitation on how far the later years generalise.\n")

    A("\n### 3.5 Timing\n")
    A("Year-by-year medians, student-authored articles only:\n\n")
    A(_md(stu_yr))
    A("\n#### Unrestricted change-point search\n")
    A(_md(stu_cp))
    A("\n>>> WRITE: for each metric, state whether the breakpoint precedes or "
      "follows late 2022, how well separated it is from the runner-up year, "
      "and whether it beats the linear baseline. Metrics that inflect before "
      "2022 cannot be attributed to generative AI.\n")

    A("\n### 3.6 Article length\n")
    A("Pre/post effects computed within word-count strata (coarsened exact "
      "matching), so length is approximately held constant:\n\n")
    A(_md(lenstrat, [c for c in ["metric", "length_bin", "n_pre", "n_post",
                                 "effect_size", "effect_ci_low",
                                 "effect_ci_high"]
                     if lenstrat is not None and c in lenstrat.columns]))

    A("\n### 3.7 Exploratory AI-associated vocabulary\n")
    A(f"A list of {len(set(w.lower() for w in AI_EXPLORATORY_VOCAB))} terms "
      "drawn from public commentary about LLM writing style. **This is not a "
      "validated AI detector**, and results are hypothesis-generating only.\n\n")
    if aivoc is not None and {"year", "corpus", "freq_per_million"}.issubset(aivoc.columns):
        piv = aivoc.pivot(index="year", columns="corpus",
                          values="freq_per_million")
        if {"opinions", "news"}.issubset(piv.columns):
            piv["opinions_minus_news"] = (piv["opinions"] - piv["news"]).round(1)
        A(_md(piv.reset_index(), floatfmt=1))
    else:
        A(_md(aivoc))
    A("\n>>> WRITE: state whether any rise is specific to opinion writing. If "
      "News rises equally, the pattern is not opinion-specific. Note the "
      "gap between corpora over time.\n")

    # ── interpretation ────────────────────────────────────────────────────────
    A("\n## 4. Interpretation\n")
    A(">>> WRITE: This section is deliberately empty.\n\n"
      "Constraints to respect:\n\n"
      "1. This is an **observational** study. Report temporal association; do "
      "not claim causation.\n"
      "2. Do not treat 2022 as evidence of AI influence by itself. Any "
      "coinciding change in corpus size, authorship or editorial policy is an "
      "equally consistent explanation.\n"
      "3. Statistical significance is not importance. Report n, effect size "
      "and CI together; large samples make trivial differences significant.\n"
      "4. Where metrics disagree on timing, say so rather than reporting only "
      "the ones that align.\n"
      "5. The vocabulary list is exploratory and cannot establish that any "
      "text was AI-generated.\n")

    # ── limitations ───────────────────────────────────────────────────────────
    A("\n## 5. Limitations\n")
    A("- **No causal identification.** Temporal coincidence is not causation, "
      "and no AI-usage measure exists for these authors.\n")
    A("- **Confounded timing.** Any contraction of the opinion section "
      "coincides with generative-AI availability; selection into who kept "
      "writing cannot be separated from an AI effect in this design.\n")
    A("- **Author names unavailable.** The WordPress users endpoint returns "
      "HTTP 401, so analysis uses numeric `author_id` only. Individual "
      "authors cannot be verified as students.\n")
    A("- **Heuristic authorship classification.** Guest content is identified "
      "by title prefix, byline and volume, not by verified affiliation.\n")
    A("- **Small recent samples.** Student-authored opinion counts in later "
      "years are low; yearly medians there are unstable.\n")
    A(f"- **2026 is partial** (through {END_DATE}).\n")
    A("- **Embedding truncation.** `all-MiniLM-L6-v2` encodes at most 256 "
      "word-pieces, so similarity results describe article openings rather "
      "than whole articles.\n")
    A("- **Length-sensitive metrics.** Root TTR scales with article length "
      "and is not used for inference.\n")
    A("- **Archive coverage.** Only articles exposed by the site's API are "
      "included; category tagging practices changed over the window.\n")

    # ── repro ─────────────────────────────────────────────────────────────────
    A("\n## 6. Reproducibility\n")
    A("```bash\n"
      "pip install -r requirements.txt\n"
      "python -m spacy download en_core_web_sm\n"
      "python -m pytest tests/ -v\n"
      "python -m src.run_pipeline\n"
      "python -m src.diagnose_confounds\n"
      "python -m src.diagnose_robustness\n"
      "python -m src.diagnose_authorship\n"
      "python -m src.diagnose_final\n"
      "python -m src.report\n"
      "```\n")
    A("\nAll parameters are centralised in `src/config.py`. Random seeds are "
      "fixed. Figures are in `reports/figures/`; tables in `reports/tables/`.\n")

    return "\n".join(L)


def main() -> None:
    if not TABLES_DIR.exists():
        logger.error("No tables directory. Run the pipeline first.")
        sys.exit(1)

    text = build_report()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")

    logger.info("Wrote %s (%d characters)", OUT, len(text))
    print(f"\nWrote {OUT}")
    print(f"Sections needing your input: {text.count('>>> WRITE:')}")
    print("\nEvery number is read from reports/tables/. Nothing is invented.")
    print("Section 4 (Interpretation) is intentionally empty.")


if __name__ == "__main__":
    main()
