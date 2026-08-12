"""
build_site.py — Build a static research website from the real result tables.

Reads reports/tables/*.csv and reports/figures/*.png and writes a
self-contained site to site/. No numbers are hardcoded: every value shown is
read from the pipeline output, so the site cannot drift from the analysis.

Data is inlined as JSON, so site/index.html opens directly from disk with no
server. Chart.js is loaded from a CDN; without a connection the tables and
figures still render.

Run from the project root, after the pipeline and all diagnostics:
  python -m src.build_site

Writes:
  site/index.html
  site/figures/*.png
  site/data/*.csv   (download copies)
"""

import json
import shutil
import sys

import pandas as pd

from src.config import (
    PROJECT_ROOT, TABLES_DIR, FIGURES_DIR, REPORTS_DIR,
    START_DATE, END_DATE, EMBEDDING_MODEL, SPACY_MODEL,
)
from src.utils import get_logger

logger = get_logger(__name__)

SITE_DIR = PROJECT_ROOT / "site"


def _load(name):
    p = TABLES_DIR / name
    if not p.exists():
        logger.warning("Missing table: %s", name)
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:
        logger.warning("Unreadable %s: %s", name, exc)
        return None


def _records(df, cols=None):
    """DataFrame -> list of dicts with NaN converted to None for valid JSON."""
    if df is None or df.empty:
        return []
    d = df[[c for c in cols if c in df.columns]] if cols else df
    return json.loads(d.to_json(orient="records"))


def _html_table(df, cols=None, caption=""):
    if df is None or df.empty:
        return f'<p class="missing">Table not available.</p>'
    d = df[[c for c in cols if c in df.columns]].copy() if cols else df.copy()
    for c in d.select_dtypes("float").columns:
        d[c] = d[c].round(4)
    head = "".join(f"<th>{c}</th>" for c in d.columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{'' if pd.isna(v) else v}</td>" for v in row
        ) + "</tr>"
        for row in d.itertuples(index=False)
    )
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="tw"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build() -> None:
    if not TABLES_DIR.exists():
        logger.error("No tables found. Run the pipeline first.")
        sys.exit(1)

    yearly = _load("yearly_corpus_statistics.csv")
    within = _load("diag_within_corpus_tests.csv")
    guest = _load("diag_guest_share_by_year.csv")
    author = _load("diag_final_author_level.csv")
    stu_yr = _load("diag_final_student_yearly.csv")
    stu_cp = _load("diag_final_student_changepoint.csv")
    conc = _load("diag_author_concentration.csv")
    aivoc = _load("diag_ai_vocab_by_corpus.csv")

    total_n = int(yearly["n_articles"].sum()) if yearly is not None else 0

    # ── headline numbers, read from the tables (never hardcoded) ──────────────
    def _pick(df, **eq):
        if df is None or df.empty:
            return None
        m = pd.Series(True, index=df.index)
        for k, v in eq.items():
            if k not in df.columns:
                return None
            m &= df[k] == v
        sub = df[m]
        return None if sub.empty else sub.iloc[0].to_dict()

    hero = _pick(author, approach="2_author_as_unit", metric="mattr")
    hero_art = _pick(author, approach="1_article_level", metric="mattr")
    news_mattr = _pick(within, corpus="news", metric="mattr")
    op_mattr = _pick(within, corpus="opinions", metric="mattr")

    def _fmt(d, key, nd=3):
        if not d or key not in d or pd.isna(d[key]):
            return "—"
        return f"{float(d[key]):.{nd}f}"

    # ── chart payloads ────────────────────────────────────────────────────────
    payload = {
        "corpus": _records(yearly, ["year", "n_opinions", "n_news"]),
        "student": _records(stu_yr, ["year", "n", "mattr", "func_word_ratio",
                                     "avg_sentence_len", "flesch_reading_ease"]),
        "effects": _records(within, ["corpus", "metric", "effect_size",
                                     "effect_ci_low", "effect_ci_high"]),
        "guest": _records(guest, ["year", "guest_share", "n_student", "n_guest"]),
        "conc": _records(conc, ["year", "effective_n_authors", "n_unique_authors"]),
        "aivoc": _records(aivoc, ["year", "corpus", "freq_per_million"]),
    }

    figures = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []

    # ── copy assets ───────────────────────────────────────────────────────────
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "figures").mkdir(exist_ok=True)
    (SITE_DIR / "data").mkdir(exist_ok=True)
    for f in figures:
        shutil.copy2(f, SITE_DIR / "figures" / f.name)
    n_csv = 0
    for c in sorted(TABLES_DIR.glob("*.csv")):
        shutil.copy2(c, SITE_DIR / "data" / c.name)
        n_csv += 1

    fig_cards = "".join(
        f'<figure><img src="figures/{f.name}" alt="{f.stem}" loading="lazy">'
        f'<figcaption>{f.stem.replace("_", " ")}</figcaption></figure>'
        for f in figures
    ) or '<p class="missing">No figures found.</p>'

    dl_links = "".join(
        f'<li><a href="data/{c.name}" download>{c.name}</a></li>'
        for c in sorted(TABLES_DIR.glob("*.csv"))
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Linguistic Change in The Stanford Daily, 2015–2026</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{{--ink:#16171a;--mut:#5c6169;--line:#e2e4e8;--bg:#fbfbfc;--acc:#8c1515;--acc2:#2b6cb0;--ok:#276749}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
header{{background:#fff;border-bottom:1px solid var(--line);padding:56px 24px 40px}}
.wrap{{max-width:1080px;margin:0 auto}}
h1{{font-family:Georgia,"Times New Roman",serif;font-size:2.3rem;line-height:1.2;margin:0 0 12px;letter-spacing:-.01em}}
.sub{{color:var(--mut);font-size:1.05rem;margin:0 0 22px;max-width:70ch}}
.pill{{display:inline-block;background:#f3f4f6;border:1px solid var(--line);
 border-radius:999px;padding:4px 13px;font-size:.8rem;color:var(--mut);margin:0 6px 6px 0}}
nav{{position:sticky;top:0;background:rgba(255,255,255,.96);backdrop-filter:blur(8px);
 border-bottom:1px solid var(--line);z-index:20}}
nav .wrap{{display:flex;gap:22px;overflow-x:auto;padding:13px 24px}}
nav a{{color:var(--mut);text-decoration:none;font-size:.88rem;white-space:nowrap}}
nav a:hover{{color:var(--acc)}}
main{{padding:0 24px 90px}}
section{{padding:52px 0;border-bottom:1px solid var(--line)}}
h2{{font-family:Georgia,serif;font-size:1.55rem;margin:0 0 8px}}
h3{{font-size:1.02rem;margin:30px 0 10px;color:var(--mut);font-weight:600;
 text-transform:uppercase;letter-spacing:.06em}}
p{{max-width:74ch}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:14px;margin:26px 0}}
.stat{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:17px}}
.stat .v{{font-size:1.65rem;font-weight:650;font-variant-numeric:tabular-nums}}
.stat .l{{font-size:.79rem;color:var(--mut);margin-top:3px}}
.chart{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:18px;margin:20px 0}}
.chart h4{{margin:0 0 4px;font-size:.98rem}}
.chart .note{{font-size:.83rem;color:var(--mut);margin:0 0 14px;max-width:74ch}}
.cbox{{position:relative;height:330px}}
.tw{{overflow-x:auto;margin:16px 0;background:#fff;border:1px solid var(--line);border-radius:9px}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 11px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#f7f8f9;font-weight:600;position:sticky;top:0}}
tbody tr:hover{{background:#fafbfc}}
.callout{{border-left:3px solid var(--acc);background:#fff;padding:15px 19px;margin:22px 0;border-radius:0 8px 8px 0}}
.callout.warn{{border-left-color:#b7791f}}
.callout.ok{{border-left-color:var(--ok)}}
.callout h4{{margin:0 0 6px;font-size:.95rem}}
.callout p{{margin:0;font-size:.92rem;color:#3d4149}}
figure{{margin:0;background:#fff;border:1px solid var(--line);border-radius:9px;overflow:hidden}}
figure img{{width:100%;display:block}}
figcaption{{padding:9px 13px;font-size:.8rem;color:var(--mut);border-top:1px solid var(--line);text-transform:capitalize}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px;margin-top:20px}}
ul.dl{{columns:2;font-size:.86rem}} ul.dl a{{color:var(--acc2)}}
.missing{{color:var(--mut);font-style:italic}}
footer{{padding:34px 24px;color:var(--mut);font-size:.85rem}}
code{{background:#f3f4f6;padding:2px 5px;border-radius:4px;font-size:.86em}}
@media(max-width:640px){{h1{{font-size:1.7rem}}ul.dl{{columns:1}}}}
</style></head><body>

<header><div class="wrap">
<h1>Linguistic Change in Student Opinion Writing<br>at The Stanford Daily, 2015–2026</h1>
<p class="sub">A corpus study of {total_n:,} articles testing whether the lexical and
stylistic properties of student writing changed across the period in which
generative AI became widely available.</p>
<span class="pill">{START_DATE} → {END_DATE}</span>
<span class="pill">N = {total_n:,}</span>
<span class="pill">Observational</span>
<span class="pill">News corpus as control</span>
</div></header>

<nav><div class="wrap">
<a href="#findings">Findings</a><a href="#corpus">Corpus</a>
<a href="#timing">Timing</a><a href="#control">Control</a>
<a href="#authorship">Authorship</a><a href="#vocab">Vocabulary</a>
<a href="#figures">Figures</a><a href="#limits">Limitations</a>
<a href="#data">Data</a>
</div></nav>

<main class="wrap">

<section id="findings">
<h2>Findings</h2>
<div class="stats">
  <div class="stat"><div class="v">{total_n:,}</div><div class="l">Articles analysed</div></div>
  <div class="stat"><div class="v">{_fmt(hero,'effect_size')}</div><div class="l">MATTR effect, author-level</div></div>
  <div class="stat"><div class="v">{_fmt(op_mattr,'effect_size')}</div><div class="l">MATTR effect, Opinions</div></div>
  <div class="stat"><div class="v">{_fmt(news_mattr,'effect_size')}</div><div class="l">MATTR effect, News (control)</div></div>
</div>

<div class="callout ok"><h4>What the data supports</h4>
<p>Student-authored opinion writing shows a substantial increase in lexical
diversity and a decrease in function-word density between the 2015–2019 baseline
and 2024–2026. The effect persists when each author counts as a single
observation, so it is not produced by a small number of prolific columnists,
and it is markedly larger than the change in the same newspaper's news
reporting over the same period.</p></div>

<div class="callout warn"><h4>What the data does not support</h4>
<p>No causal attribution to generative AI. The metrics disagree on when the
change began — some inflect years before generative AI was available — the
opinion section contracted sharply over the window so that later years rest on
far fewer writers, and an exploratory list of AI-associated vocabulary shows no
increase specific to opinion writing.</p></div>
</section>

<section id="corpus">
<h2>Corpus</h2>
<p>Collected from the publication's WordPress REST API. Articles were cleaned of
markup, filtered by length, and deduplicated by identifier, URL and text hash.</p>
<div class="chart"><h4>Articles per year by section</h4>
<p class="note">The opinion section contracted far more sharply than news
reporting. 2026 is partial.</p>
<div class="cbox"><canvas id="cCorpus"></canvas></div></div>
{_html_table(yearly)}
</section>

<section id="timing">
<h2>When did it change?</h2>
<p>Year-by-year medians for student-authored opinion articles only, with guest
and institutional submissions removed.</p>
<div class="chart"><h4>Lexical diversity (MATTR), student authors only</h4>
<p class="note">MATTR uses a moving window and is robust to article length,
unlike simple type–token ratio. Later years rest on small samples — read the
trajectory against the article counts in the table below.</p>
<div class="cbox"><canvas id="cMattr"></canvas></div></div>
<div class="chart"><h4>Function-word ratio and readability</h4>
<p class="note">Plotted on independent axes; the two series are not directly
comparable in magnitude.</p>
<div class="cbox"><canvas id="cOther"></canvas></div></div>
{_html_table(stu_yr, caption="Student-authored articles only")}
<h3>Change-point analysis</h3>
<p>Segmented regression against a single-line baseline. A segmented model always
fits at least as well as a straight line, so a breakpoint counts only when it
clearly beats the linear fit and is well separated from the runner-up year.</p>
{_html_table(stu_cp)}
</section>

<section id="control">
<h2>The control corpus</h2>
<p>The same newspaper's news reporting was collected and processed identically.
It provides an internal control for editorial, topical and platform changes that
would affect all published text.</p>
<div class="chart"><h4>Effect sizes with 95% confidence intervals</h4>
<p class="note">Rank-biserial correlation, baseline versus recent period. Bars
show bootstrap confidence intervals. Where the Opinions and News intervals do
not overlap, the difference between corpora is itself reliable.</p>
<div class="cbox" style="height:430px"><canvas id="cEffects"></canvas></div></div>
{_html_table(within)}
</section>

<section id="authorship">
<h2>Who is writing</h2>
<p>The opinion section carries both student writing and guest submissions from
alumni, parents, faculty and outside contributors. Because those groups write
differently, a change in their mix can create or mask an apparent change in
student writing.</p>
<div class="chart"><h4>Guest and institutional share of the opinion section</h4>
<div class="cbox"><canvas id="cGuest"></canvas></div></div>
<div class="chart"><h4>Effective number of distinct authors</h4>
<p class="note">Inverse Herfindahl index. A falling value means the corpus rests
on fewer distinct voices, which limits how far later years generalise.</p>
<div class="cbox"><canvas id="cConc"></canvas></div></div>
<h3>Estimates with author leverage removed</h3>
{_html_table(author)}
</section>

<section id="vocab">
<h2>Exploratory AI-associated vocabulary</h2>
<div class="callout warn"><h4>Not a detector</h4>
<p>This word list is drawn from public commentary about the style of
large-language-model output. It has not been validated against known
AI-generated text and cannot establish that any article was machine-written.
Several terms also occur naturally in articles <em>about</em> artificial
intelligence, which would inflate later years irrespective of authorship.</p></div>
<div class="chart"><h4>Frequency per million words, by section</h4>
<p class="note">If both sections move together, the pattern is not specific to
opinion writing.</p>
<div class="cbox"><canvas id="cVocab"></canvas></div></div>
</section>

<section id="figures">
<h2>Figures</h2>
<div class="grid">{fig_cards}</div>
</section>

<section id="limits">
<h2>Limitations</h2>
<ul>
<li><strong>No causal identification.</strong> This is observational. No measure
of AI use exists for these authors.</li>
<li><strong>Confounded timing.</strong> The contraction of the opinion section
coincides with generative-AI availability. Selection into who continued writing
cannot be separated from an AI effect in this design.</li>
<li><strong>Pseudonymous authors.</strong> The publication's users endpoint
requires authentication, so analysis uses numeric author identifiers.
Individuals cannot be verified as students.</li>
<li><strong>Heuristic authorship classification.</strong> Guest content is
identified by title prefix, byline and volume, not verified affiliation.</li>
<li><strong>Small recent samples.</strong> Student-authored opinion counts are
low in later years, so yearly medians there are unstable.</li>
<li><strong>Length-sensitive metrics.</strong> Root type–token ratio scales with
article length and is reported for reference only, not used for inference.</li>
<li><strong>Embedding truncation.</strong> <code>{EMBEDDING_MODEL}</code> encodes
a limited number of word-pieces, so semantic results describe article openings.</li>
<li><strong>Partial final year.</strong> 2026 runs only to {END_DATE}.</li>
</ul>
</section>

<section id="data">
<h2>Data and reproducibility</h2>
<p>Every figure and table on this page is generated from the pipeline output.
Linguistic metrics use spaCy <code>{SPACY_MODEL}</code>; embeddings use
<code>{EMBEDDING_MODEL}</code>. All {n_csv} result tables:</p>
<ul class="dl">{dl_links}</ul>
</section>

</main>
<footer class="wrap">Generated from pipeline output. Numbers on this page are
read directly from the result tables and are not hand-entered.</footer>

<script>
const D = {json.dumps(payload)};
const MUT='#5c6169', ACC='#8c1515', ACC2='#2b6cb0', LINE='#e2e4e8';
Chart.defaults.font.family='-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif';
Chart.defaults.color=MUT; Chart.defaults.maintainAspectRatio=false;
const g=(a,k)=>a.map(r=>r[k]);
const miss=id=>{{const e=document.getElementById(id); if(!e)return;
 const b=e.closest('.cbox'); if(b){{b.style.height='auto';
 b.innerHTML='<p class="missing">Data not available — run the corresponding diagnostic module.</p>';}}}};
const mk=(id,cfg)=>{{const e=document.getElementById(id);
 if(!e)return; if(!window.Chart){{miss(id);return;}} new Chart(e,cfg);}};

if(D.corpus.length) mk('cCorpus',{{type:'bar',data:{{labels:g(D.corpus,'year'),
 datasets:[{{label:'Opinions',data:g(D.corpus,'n_opinions'),backgroundColor:ACC}},
           {{label:'News',data:g(D.corpus,'n_news'),backgroundColor:'#c8ccd2'}}]}},
 options:{{scales:{{x:{{stacked:true,grid:{{display:false}}}},
  y:{{stacked:true,title:{{display:true,text:'Articles'}},grid:{{color:LINE}}}}}}}}}}); else miss('cCorpus');

if(D.student.length) mk('cMattr',{{type:'line',data:{{labels:g(D.student,'year'),
 datasets:[{{label:'MATTR (median)',data:g(D.student,'mattr'),borderColor:ACC,
  backgroundColor:'rgba(140,21,21,.07)',fill:true,tension:.25,pointRadius:4}}]}},
 options:{{plugins:{{tooltip:{{callbacks:{{afterLabel:c=>'n = '+D.student[c.dataIndex].n}}}}}},
  scales:{{y:{{title:{{display:true,text:'MATTR'}},grid:{{color:LINE}}}},
   x:{{grid:{{display:false}}}}}}}}}}); else miss('cMattr');

if(D.student.length) mk('cOther',{{type:'line',data:{{labels:g(D.student,'year'),
 datasets:[{{label:'Function-word ratio',data:g(D.student,'func_word_ratio'),
   borderColor:ACC2,tension:.25,yAxisID:'y',pointRadius:3}},
  {{label:'Flesch reading ease',data:g(D.student,'flesch_reading_ease'),
   borderColor:'#b7791f',tension:.25,yAxisID:'y1',pointRadius:3}}]}},
 options:{{scales:{{y:{{position:'left',title:{{display:true,text:'Function-word ratio'}},grid:{{color:LINE}}}},
  y1:{{position:'right',title:{{display:true,text:'Flesch reading ease'}},grid:{{display:false}}}},
  x:{{grid:{{display:false}}}}}}}}}}); else miss('cOther');

if(D.effects.length){{
 const ms=[...new Set(D.effects.map(r=>r.metric))];
 const ds=['opinions','news'].map((c,i)=>({{label:c==='opinions'?'Opinions':'News (control)',
  data:ms.map(m=>{{const r=D.effects.find(x=>x.corpus===c&&x.metric===m);
   return r?[r.effect_ci_low,r.effect_ci_high]:null;}}),
  backgroundColor:i?'#c8ccd2':ACC,borderWidth:0,borderSkipped:false}}));
 mk('cEffects',{{type:'bar',data:{{labels:ms,datasets:ds}},
  options:{{indexAxis:'y',plugins:{{tooltip:{{callbacks:{{label:c=>
    c.dataset.label+': ['+c.raw[0]+', '+c.raw[1]+']'}}}}}},
   scales:{{x:{{title:{{display:true,text:'Rank-biserial effect size (95% CI)'}},grid:{{color:LINE}}}},
    y:{{grid:{{display:false}}}}}}}}}});}} else miss('cEffects');

if(D.guest.length) mk('cGuest',{{type:'line',data:{{labels:g(D.guest,'year'),
 datasets:[{{label:'Guest share',data:g(D.guest,'guest_share'),borderColor:'#b7791f',
  backgroundColor:'rgba(183,121,31,.1)',fill:true,tension:.25,pointRadius:3}}]}},
 options:{{scales:{{y:{{min:0,max:1,title:{{display:true,text:'Share of opinion articles'}},
  ticks:{{callback:v=>(v*100)+'%'}},grid:{{color:LINE}}}},x:{{grid:{{display:false}}}}}}}}}}); else miss('cGuest');

if(D.conc.length) mk('cConc',{{type:'line',data:{{labels:g(D.conc,'year'),
 datasets:[{{label:'Effective authors (1/HHI)',data:g(D.conc,'effective_n_authors'),
  borderColor:'#276749',tension:.25,pointRadius:3}}]}},
 options:{{scales:{{y:{{title:{{display:true,text:'Effective authors'}},grid:{{color:LINE}}}},
  x:{{grid:{{display:false}}}}}}}}}}); else miss('cConc');

if(D.aivoc.length){{
 const ys=[...new Set(D.aivoc.map(r=>r.year))].sort();
 const s=c=>ys.map(y=>{{const r=D.aivoc.find(x=>x.year===y&&x.corpus===c);return r?r.freq_per_million:null;}});
 mk('cVocab',{{type:'line',data:{{labels:ys,datasets:[
  {{label:'Opinions',data:s('opinions'),borderColor:ACC,tension:.25,pointRadius:3}},
  {{label:'News',data:s('news'),borderColor:'#9aa0a8',tension:.25,pointRadius:3}}]}},
  options:{{scales:{{y:{{title:{{display:true,text:'Per million words'}},grid:{{color:LINE}}}},
   x:{{grid:{{display:false}}}}}}}}}});}} else miss('cVocab');
</script>
</body></html>"""

    out = SITE_DIR / "index.html"
    out.write_text(html, encoding="utf-8")

    logger.info("Wrote %s", out)
    print(f"\nSite written to: {out}")
    print(f"  figures copied : {len(figures)}")
    print(f"  tables copied  : {n_csv}")
    print("\nOpen site/index.html in a browser. All values are read from")
    print("reports/tables/ — nothing on the page is hand-entered.")


if __name__ == "__main__":
    build()
