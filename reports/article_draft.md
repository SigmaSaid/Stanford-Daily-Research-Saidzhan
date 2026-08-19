# Methods and Sources

**"When We Went Looking for AI in Student Writing, We Found Something Else"**
By Saidzhan and Afshinkhan · The Stanford Daily, Opinions

This note documents the data, methods, and limitations behind the article. It
accompanies the piece rather than duplicating it — for the findings themselves,
read the article.

## Corpus

11,242 articles from The Stanford Daily, published 2015-01-01 to 2026-08-10,
retrieved via the publication's public WordPress REST API
(`/wp-json/wp/v2/posts`). Two sections were collected: Opinions (the subject)
and News (the control), processed identically. Articles were stripped of
markup, navigation, advertising, and author bios; filtered to 50–15,000 words;
and deduplicated by article ID, URL, and a SHA-256 hash of the cleaned text.

## Authorship classification

Opinion articles were classified as student-authored or guest/institutional
using three signals: an explicit title prefix ("From the Community," "Letter
to the Editor"), an institutional byline, or any single byline account holding
more than 15% of the Opinions corpus. This rule is heuristic, not verified
ground truth — the publication's `/wp/v2/users/` endpoint requires
authentication (HTTP 401), so author display names could not be retrieved.
Authors are identified by numeric ID only throughout.

## Measures

- **Lexical diversity**: MATTR (moving-average type–token ratio), which is
  robust to article length, unlike simple type–token ratio.
- **Readability**: Flesch Reading Ease.
- **Semantic similarity**: sentence embeddings
  (`sentence-transformers/all-MiniLM-L6-v2`), with within-year mean pairwise
  cosine similarity as the clustering measure. The model truncates at ~256
  word-pieces, so similarity reflects article openings, not full articles.
- **AI-associated vocabulary**: a list of 59 terms drawn from public
  commentary about LLM writing style. This is exploratory only and is **not**
  a validated AI-text detector.

## Statistics

Mann–Whitney U tests comparing 2015–2019 against 2024–2026, with rank-biserial
correlation as the effect size and percentile bootstrap 95% confidence
intervals. Benjamini–Hochberg FDR correction across metrics (α = 0.05).
Change-points from segmented regression, always reported against a
single-line baseline, since a segmented model will otherwise appear to find a
breakpoint whether or not one exists.

## Robustness checks

Every headline result was re-tested: within each corpus separately, within
article genre, within word-count strata, with each author counted once
rather than once per article, and with the five most prolific writers in
each period removed.

## Limitations

- **Observational, not causal.** Temporal coincidence is not causation, and
  no measure of AI use exists for these authors.
- **Confounded timing.** The Opinions section's contraction coincides with
  generative AI's availability; selection into who kept writing cannot be
  separated from an AI effect in this design.
- **Heuristic authorship classification.** Guest content is identified by
  title prefix, byline, and volume — not verified affiliation.
- **Small recent samples.** Student-authored opinion counts in later years
  are low; yearly medians there are unstable.
- **2026 is partial** (through 2026-08-10).
- **Archive coverage.** Only articles exposed by the site's API are
  included; category-tagging practices may have changed over the window.

## Sources

1. **The piece that prompted this one.** T. Mui, "From the Community | What
   15 years of Daily opinion pieces reveal about diversity," *The Stanford
   Daily*, November 5, 2025.
2. **This repository.** The full data-collection and analysis pipeline,
   every diagnostic table, and all figures referenced in the article are in
   this repo, so any number in the piece can be traced back to its source
   table.

## Reproducibility

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m pytest tests/ -v
python -m src.run_pipeline
python -m src.diagnose_confounds
python -m src.diagnose_robustness
python -m src.diagnose_authorship
python -m src.diagnose_final
python -m src.report
```

All parameters are centralized in `src/config.py`. Random seeds are fixed.
Figures are in `reports/figures/`; tables are in `reports/tables/`.
