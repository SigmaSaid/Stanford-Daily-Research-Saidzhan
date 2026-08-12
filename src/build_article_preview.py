"""
build_article_preview.py — Render the article with its figures embedded.

Copying Markdown into a Google Doc loses the images: relative paths do not
survive the paste. This renders the ARTICLE section of reports/article_draft.md
to a single self-contained HTML file with every figure inlined as base64, so
selecting all in a browser and pasting into a Google Doc carries the text AND
the images across in one step.

Only the ARTICLE section is rendered. The methods note, sources and checklist
stay in the Markdown file, since they are for you and the editors rather than
for the published piece.

Run from the project root:
  python -m src.build_article_preview

Writes:
  reports/article_preview.html
"""

import base64
import html as html_mod
import re
import sys

from src.config import PROJECT_ROOT, REPORTS_DIR
from src.utils import get_logger

logger = get_logger(__name__)

SRC_MD = REPORTS_DIR / "article_draft.md"
OUT = REPORTS_DIR / "article_preview.html"


def _inline(text: str) -> str:
    """Escape HTML, then apply bold and italic markup."""
    t = html_mod.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", t)
    t = re.sub(r"`([^`]+?)`", r"<code>\1</code>", t)
    return t


def _embed_image(rel_path: str) -> str:
    """Return an <img> with the PNG inlined as base64, or a visible warning."""
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        logger.warning("Figure not found: %s", p)
        return (f'<p class="warn">MISSING FIGURE: {html_mod.escape(rel_path)} '
                f'— generate it before submitting.</p>')
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{html_mod.escape(p.stem)}">'


def _render_figure_block(lines: list[str]) -> str:
    """Turn a '> **[FIGURE n — `path`]**' blockquote into a real figure."""
    joined = " ".join(l.lstrip("> ").strip() for l in lines)
    m = re.search(r"`([^`]+\.png)`", joined)
    if not m:
        return f'<blockquote>{_inline(joined)}</blockquote>'
    img = _embed_image(m.group(1))

    cap = ""
    cm = re.search(r"\*Caption:\*\s*(.+)", joined)
    if cm:
        cap = cm.group(1).strip()
        cap = re.split(r"\*Note:", cap)[0].strip()

    return (f'<figure>{img}'
            f'<figcaption>{_inline(cap)}</figcaption></figure>')


def md_to_html(md: str) -> str:
    out, buf, quote = [], [], []

    def flush_para():
        if buf:
            out.append(f"<p>{_inline(' '.join(buf).strip())}</p>")
            buf.clear()

    def flush_quote():
        if quote:
            joined = " ".join(q.lstrip("> ").strip() for q in quote)
            if "[FIGURE" in joined:
                out.append(_render_figure_block(quote))
            else:
                out.append(f"<blockquote>{_inline(joined)}</blockquote>")
            quote.clear()

    for raw in md.splitlines():
        line = raw.rstrip()

        if line.startswith(">"):
            flush_para()
            quote.append(line)
            continue
        flush_quote()

        if not line.strip():
            flush_para()
            continue

        if line.startswith("### "):
            flush_para()
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            flush_para()
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            flush_para()
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        else:
            buf.append(line.strip())

    flush_quote()
    flush_para()
    return "\n".join(out)


def main() -> None:
    if not SRC_MD.exists():
        logger.error("Missing %s", SRC_MD)
        sys.exit(1)

    text = SRC_MD.read_text(encoding="utf-8")

    m = re.search(r"^# ARTICLE\s*$(.*?)^# METHODS NOTE",
                  text, re.S | re.M)
    if not m:
        logger.error("Could not locate the ARTICLE section in %s. "
                     "Expected '# ARTICLE' followed later by '# METHODS NOTE'.",
                     SRC_MD.name)
        sys.exit(1)
    body_md = m.group(1).strip()

    body = md_to_html(body_md)
    n_fig = body.count("<figure>")
    n_missing = body.count("MISSING FIGURE")

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Article preview</title>
<style>
body{{max-width:760px;margin:0 auto;padding:48px 24px 90px;
 font:17px/1.72 Georgia,"Times New Roman",serif;color:#1a1c1f}}
h1{{font-size:2.05rem;line-height:1.22;margin:0 0 28px}}
h2{{font-size:1.5rem;margin:38px 0 14px}}
h3{{font-size:1.16rem;margin:34px 0 10px}}
p{{margin:0 0 19px}}
figure{{margin:32px 0;padding:0}}
figure img{{width:100%;display:block;border:1px solid #e2e4e8;border-radius:4px}}
figcaption{{font-size:.88rem;line-height:1.5;color:#5c6169;margin-top:9px;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
blockquote{{border-left:3px solid #d8dade;margin:22px 0;padding:2px 0 2px 18px;color:#42464c}}
code{{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:.85em;
 font-family:ui-monospace,Menlo,Consolas,monospace}}
strong{{font-weight:700}}
.warn{{background:#fff4e5;border:1px solid #f0c98a;padding:11px 15px;
 border-radius:5px;color:#8a5a00;font-family:sans-serif;font-size:.9rem}}
.banner{{background:#f3f4f6;border:1px solid #e2e4e8;border-radius:6px;
 padding:13px 17px;margin-bottom:34px;font:14px/1.55 -apple-system,
 BlinkMacSystemFont,"Segoe UI",sans-serif;color:#5c6169}}
@media print{{.banner{{display:none}}body{{padding:0}}}}
</style></head><body>
<div class="banner"><strong>Preview.</strong> Select all (Ctrl+A), copy, and
paste into a Google Doc — the figures travel with the text. Fill in the name,
bio and repository URL before sharing with editors. This banner does not
print.</div>
{body}
</body></html>"""

    OUT.write_text(page, encoding="utf-8")
    logger.info("Wrote %s", OUT)

    print(f"\nWrote {OUT}")
    print(f"  figures embedded : {n_fig}")
    if n_missing:
        print(f"  MISSING figures  : {n_missing}  <-- generate these first")
    print("\nOpen it in a browser, select all, copy, paste into a Google Doc.")
    print("Images are inlined as base64, so they survive the paste.")


if __name__ == "__main__":
    main()
