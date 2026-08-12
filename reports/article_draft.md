# DRAFT FOR THE STANFORD DAILY — OPINIONS

> **How to use this file.** Paste the ARTICLE section into a Google Doc and
> share it with the Opinions editors. Everything below the article — methods
> note, sources, checklist — is for you and the editors, not for publication,
> though editors often want the methods note attached.
>
> **Every number in this draft is taken from your own result tables.** Nothing
> is estimated or invented. Before you submit, re-verify each one against
> `reports/tables/` using the verification list at the end.

---

## SUBMISSION BLOCK

**Title:** When I went looking for AI in student writing, I found something else

**Alternative titles:**
- The Daily's opinion section lost 79% of its student writers. Here's what that did to the writing.
- 11,242 articles later: what changed in student opinion writing, and what didn't

**Slug:** `what-11000-daily-articles-reveal-about-ai-and-student-writing`

**Author:** [YOUR FULL NAME]

**Author bio (one or two sentences, runs at the foot of the piece):**
[YOUR NAME] is a [year / affiliation] who works on [field]. The code and data
for this analysis are available at [REPOSITORY URL].

**Section:** Opinions

**Word count:** ~1,250

**Figures:** 3 (see placement markers in the text)

---

# ARTICLE

## When I went looking for AI in student writing, I found something else

In November, a piece in this section reported that Stanford students' published
opinions have been converging — that even as the language of diversity became
more common, the range of ideas expressed narrowed. I read it and wondered
about a different question. That analysis ended in 2024. ChatGPT arrived in
November 2022. If student writing was changing, was a new tool part of the
reason?

So I collected everything. Using the Daily's public API, I downloaded 11,242
articles published between January 2015 and August 2026 — every opinion piece
and every news story I could reach. Then I did something the earlier analysis
didn't: I kept the news section as a control. If a change appears in opinion
writing but not in the news reporting produced by the same paper under the same
editors, it is telling you something about opinion writers. If it appears in
both, it is telling you something about the paper.

I expected to find either a clear AI signature or nothing at all. Instead, the
first thing I found had nothing to do with writing.

**In 2015, this section published 448 opinion pieces. In 2024, it published 92.**

> **[FIGURE 1 — `reports/figures/01_articles_by_year.png`]**
> *Caption:* Articles published per year, 2015–2026. The opinion section
> contracted far more sharply than news reporting. 2026 covers January through
> August only.

The section didn't just shrink. It changed hands. Sorting articles by who wrote
them, I found that submissions from outside the student body — published under
the "From the Community" byline — grew from under a third of the section in the
2015–2019 period to more than half in 2024–2026. In 2023, a single byline
accounted for 51 of the section's 105 opinion articles.

Counting distinct writers tells the same story. Adjusting for how unevenly
articles are distributed across authors, the section carried the equivalent of
about 20 independent voices a year in 2015. By 2024 and 2025 that figure was
under four.

I should be clear about something. The article that prompted this one ran under
that same "From the Community" byline, written by a high school senior. That is
not a criticism of it — it is a careful piece of work, and it is the reason I
started digging. It is also, precisely, the finding.

### The writing did change — but you can only see it once you separate the writers

Here is where the authorship problem stops being a curiosity and starts
mattering.

When I measured lexical diversity — roughly, how much a writer varies their
vocabulary rather than repeating the same words — the whole opinion section
showed a moderate increase between 2015–2019 and 2024–2026. But guest
contributors and students write differently, and the mix had shifted. So I
separated them.

Among student-authored articles alone, the increase was substantially
**larger** than in the section as a whole. The guest submissions had been
pulling the average down and masking the student signal. Alongside that,
students' sentences got shorter and their prose got harder to read by standard
readability measures.

> **[FIGURE 2 — `reports/figures/FIG_02_student_mattr_by_year.png`]**
> *Caption:* Median lexical diversity (MATTR) of student-authored opinion
> articles, 2015–2026. The measure is flat for eight years and then rises from
> 2024. Later years rest on small samples — 28 student articles in 2023, 32 in
> 2024. Guest and institutional submissions are excluded.

The change is not the work of a few prolific columnists. When I re-ran the
comparison counting each author once, rather than counting a writer with 30
published pieces 30 times, the effect held. When I dropped the five most
prolific writers in each period entirely, it held again. This is a broad shift
across student writers, not a handful of loud voices.

And it is specific to opinion writing. The same measures applied to the news
section over the same years moved in the same direction but far more weakly —
and for sentence length and readability, the news section did not move at all.

### Students are also writing more like each other

The earlier analysis in this section found that opinion articles have grown
more similar to one another over time. Using a different method — comparing
articles as points in a semantic space and measuring how tightly clustered each
year's writing is — I find the same thing, and I can now say something about
who it applies to.

The convergence is a student phenomenon, and it has a shape worth describing
precisely. Student-authored articles grew markedly more similar to one another
between 2015 and 2023 — a rise of roughly two-thirds on this measure — and then
moved back apart over the two years since. Across the full twelve years the
trend is upward and unlikely to be chance, but the most recent direction is
downward, and the 2023 peak rests on only 28 student articles. I would not lean
on that peak.

What is steadier is the contrast. Guest submissions show no reliable trend at
all; if anything they became slightly more varied. They are also, in every
single year, more similar to one another than student pieces are — community
submissions cluster on a narrow band of civic and policy topics. The pooled
measure, which mixes the two populations, produces a weaker and less reliable
signal than looking at students alone.

> **[FIGURE 3 — `reports/figures/FIG_03_similarity_student_vs_guest.png`]**
> *Caption:* Within-year semantic similarity of opinion articles, students
> versus guest contributors, with the pooled series for comparison. Higher
> values mean that year's articles resemble one another more closely. Student
> writing converges through 2023 and then moves back apart; guest submissions
> show no reliable trend. Small figures give the number of student articles
> compared in each year.

So students are drawing on more varied vocabulary within each article, while
their articles collectively cover a narrower range of ground. Those two things
sound contradictory and are not. A wider word stock in service of a narrower
set of arguments is exactly what you would expect from a section that has
fewer, more similar people writing in it.

### Why I can't blame AI

This is the part I want to be careful about, because it would be easy to write
the other article.

Four things stop me.

**The timing doesn't line up cleanly.** Lexical diversity does break sharply in
2023, the first full year after ChatGPT's release, and sentence length turns in
2022. But two other measures — function-word usage and readability — begin
shifting in 2017 and 2019 respectively, years before generative AI was
available to anyone. A single cause should not produce four different starting
dates.

**The vocabulary evidence isn't there.** I tracked a list of words widely
associated with AI-generated prose — "delve," "underscore," "multifaceted,"
"landscape" and dozens more. If students were leaning on these tools, that
vocabulary should rise in opinion writing specifically. It doesn't. Opinion
writing used those words *more* in 2019 than in 2024, and 2023 is the lowest
value in the entire series. Where the words do rise, in 2024 and 2025, they
rise in news reporting just as much.

**The section changed at the same time.** Student opinion volume fell by roughly
four-fifths across the window, and the number of distinct student voices fell
with it. Whoever kept writing after 2022 is not a random sample of who was
writing in 2015. A change in *who writes* and a change in *how they write*
arrived together, and no amount of statistics on this dataset can pull them
apart.

**I can't see AI use.** Nobody can, from text alone. There is no reliable
detector, and I did not build one. What I have is a description of what
published writing looks like, not evidence about how it was produced.

### What I think this is actually about

The most defensible reading is the least dramatic one. The Daily's opinion
section became smaller and more concentrated. Fewer students write in it, they
write more like one another, and a growing share of what appears there comes
from outside the student body entirely. Something did change in student prose
over this period. Whether a chatbot had anything to do with it, this dataset
cannot say.

That should matter to anyone who reads claims about AI and student writing. The
strongest apparent effect in my whole analysis — a change in function-word usage
— turns out on inspection to have begun in 2017. If I had started my dataset in
2020, as many studies of this kind do, I would have found a clean story and
believed it.

**For the Daily:** the concentration is measurable and worth watching.
Publishing periodic counts of how many distinct students write for the section
would create useful accountability, and distinguishing student bylines from
community submissions in the archive would make analyses like this one far more
reliable.

**For students:** the barrier to writing here is lower than the numbers suggest.
Fewer than a dozen of you filled most of this section last year.

**For anyone studying AI's effect on writing:** control for who is writing.
In this corpus, ignoring authorship did not just add noise — it hid the effect
entirely.

The question I started with was what tools students are using. The more useful
question turned out to be who is still writing at all.

---

# METHODS NOTE (for editors; not for publication)

**Corpus.** 11,242 articles from The Stanford Daily, published 2015-01-01 to
2026-08-10, retrieved via the publication's public WordPress REST API. Articles
were stripped of markup, navigation, advertising and author bios; filtered to
50–15,000 words; and deduplicated by article ID, URL and text hash. Two
sections were collected: Opinions (the subject) and News (the control).

**Authorship classification.** Opinion articles were classified as
student-authored or guest/institutional using three signals: an explicit title
prefix ("From the Community," "Letter to the Editor"), an institutional byline,
and any single byline account holding more than 15% of the opinion corpus. The
rule is heuristic and is reported as such. Author display names were not
retrievable — the publication's user endpoint requires authentication — so
authors are identified by numeric ID only.

**Measures.** Lexical diversity uses MATTR (moving-average type–token ratio),
which unlike simple type–token ratio is robust to article length. Readability
uses Flesch Reading Ease. Semantic similarity uses sentence embeddings, with
within-year mean pairwise cosine similarity as the clustering measure. Root
type–token ratio was computed but excluded from all conclusions because it
correlates strongly with article length.

**Statistics.** Mann–Whitney U tests comparing 2015–2019 against 2024–2026, with
rank-biserial correlation as the effect size and percentile bootstrap 95%
confidence intervals. Benjamini–Hochberg correction across metrics. Change
points from segmented regression, always reported against a single-line
baseline, since a segmented model always fits at least as well as a straight
line and will otherwise appear to find a breakpoint whether or not one exists.

**Robustness.** Every headline result was re-tested (a) within each corpus
separately, (b) within article genre, (c) within article-length strata, (d)
with each author counted once rather than once per article, and (e) with the
five most prolific writers in each period removed.

**Limitations.** Observational; no causal claim. No measure of AI use exists for
these authors. The contraction of the section is confounded in time with AI
availability. Later years rest on small samples. 2026 is partial. Embeddings
truncate long articles, so similarity reflects article openings.

**Reproducibility.** Full pipeline and all result tables at [REPOSITORY URL].

---

# THREE SOURCES

**1. The prior analysis this piece responds to.**
Mui, T. "From the Community | What 15 years of Daily opinion pieces reveal
about diversity." The Stanford Daily, November 5, 2025.
https://stanforddaily.com/2025/11/05/from-the-community-more-diversity-rhetoric-fewer-diverse-ideas-what-15-years-of-daily-opinion-pieces-reveal/
→ Cited in the opening and in the convergence section. Verified: you have the
   article.

**2. The dataset and analysis code.**
Your own corpus of 11,242 Stanford Daily articles (2015–2026), with full
pipeline, diagnostics and result tables.
→ ACTION REQUIRED: publish the repository and insert the URL. A data piece
   without accessible data is much weaker, and editors may ask.

**3. An interview with a Stanford Daily Opinions editor.**
→ ACTION REQUIRED, AND DO THIS FIRST. Ask:
   - Why did opinion volume decline after 2020?
   - Did submission, solicitation or columnist policy change?
   - Why has "From the Community" grown as a share of the section?
   - Did category tagging practice change around 2019–2020?
   An editor's answer speaks directly to your largest confound and would let
   you convert a limitation into reporting. This is the single highest-value
   thing you can add.

*A note on citations: if you decide to cite published research on AI writing
markers as a fourth source, find and read it yourself. Do not use a citation
you have not personally verified.*

---

# CHECKLIST BEFORE YOU SUBMIT

**Charts — DONE.** All three figures exist and use the correct populations:
- [x] Figure 1 — `01_articles_by_year.png`
- [x] Figure 2 — `FIG_02_student_mattr_by_year.png` (student-only)
- [x] Figure 3 — `FIG_03_similarity_student_vs_guest.png` (student vs. guest)

Regenerate any time with `python -m src.figures_article`. Do NOT substitute
`03_mattr_by_year.png` or `11_semantic_similarity.png` — those pool students
with guest contributors and would illustrate a different quantity than the
sentences beside them.

**Numbers to re-verify against your tables.**
- [ ] 11,242 articles → `yearly_corpus_statistics.csv`, sum of `n_articles`
- [ ] 448 (2015) and 92 (2024) opinion articles → same table
- [ ] Guest share under a third → over half → `diag_guest_share_by_year.csv`
- [ ] 51 of 105 in 2023 → author_id 38 count for 2023
- [ ] ~20 effective voices (2015) → under 4 (2024–25) →
      `diag_author_concentration.csv`
- [ ] 28 student articles in 2023, 32 in 2024 → `diag_final_student_yearly.csv`
- [ ] Student effect larger than section-wide → `diag_final_author_level.csv`
      vs. `diag_within_corpus_tests.csv`
- [ ] Effect survives author-as-unit and drop-top-5 → `diag_final_author_level.csv`
- [ ] News weaker; no movement in sentence length or readability →
      `diag_within_corpus_tests.csv`
- [ ] Convergence rises to 2023 then reverses; net +24% → `diag_similarity_by_group.csv`
- [ ] No reliable guest trend → `diag_similarity_trends.csv`
- [ ] Breakpoints 2023, 2022, 2017, 2019 → `diag_final_student_changepoint.csv`
- [ ] Opinion AI-vocabulary higher in 2019 than 2024; 2023 lowest →
      `diag_ai_vocab_by_corpus.csv`

**Editorial.**
- [ ] Author name and bio filled in
- [ ] Repository URL inserted (two places)
- [ ] Editor interview obtained and worked into the section on why
- [ ] Read the whole piece aloud once
- [ ] Confirm no statistical jargon survives in the article body
      (no p-values, no effect sizes, no "rank-biserial")
- [ ] Share the Google Doc with editors

**A word on the review process.** DE and ME feedback, and several rounds of
revision, are normal. Expect to be asked to cut. The methods note exists so
that the article body can stay readable while the rigor remains checkable.
