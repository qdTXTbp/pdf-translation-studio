"""Build a side-by-side bilingual reading view from a BabelDOC dual PDF.

A dual PDF places the source page and its translation on one sheet: the left
half is the untouched original, the right half the translated page. That makes
it the only artifact where source and translation are guaranteed to describe
the same region of the same page, which is exactly what paragraph pairing needs.

Line counts differ between the halves (the original breaks lines where the
typesetter did, the translation breaks them where the Chinese text wraps), so
lines are clustered into paragraphs first and paired by vertical overlap.
"""

from __future__ import annotations

import html
import os
import re

import pymupdf

# A vertical gap wider than this many points separates two paragraphs. Body
# leading in these papers runs 9-14pt, so 1.9x the dominant leading is a safe
# break that does not split a paragraph at an ordinary line break.
GAP_FACTOR = 1.9

# Left edges further apart than this many points belong to different columns.
COLUMN_GAP = 18.0

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]")
_LATIN_WORD = re.compile(r"[A-Za-zÀ-ÿ]{2,}")


def _ordered(lines):
    """Put a page half into reading order: columns left to right, lines top to
    bottom inside each column.

    Journal pages are two-column, so sorting purely by y interleaves the two
    columns and fuses unrelated paragraphs. Columns are found by clustering the
    left edges; a gap over COLUMN_GAP means a new column starts. A centered
    title spanning both columns becomes its own group, which reads correctly.
    """
    if not lines:
        return []
    xs = sorted(l["x0"] for l in lines)
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] > COLUMN_GAP:
            groups.append([x])
        else:
            groups[-1].append(x)
    spans = [(g[0], g[-1]) for g in groups]

    def col_of(x):
        best, dist = 0, 1e9
        for i, (lo, hi) in enumerate(spans):
            d = 0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi))
            if d < dist:
                best, dist = i, d
        return best

    out = []
    for i in range(len(spans)):
        col = [l for l in lines if col_of(l["x0"]) == i]
        for l in sorted(col, key=lambda r: r["y0"]):
            l["col"] = i
            out.append(l)
    return out


def _lines(pg, x_lo, x_hi):
    """Text lines whose left edge falls in [x_lo, x_hi), in reading order."""
    out = []
    for blk in pg.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            text = "".join(sp["text"] for sp in ln["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            if not (x_lo <= x0 < x_hi):
                continue
            out.append({"y0": y0, "y1": y1, "x0": x0, "x1": x1, "text": text,
                        "size": max((sp["size"] for sp in ln["spans"]), default=0.0)})
    return _ordered(out)


def _paragraphs(lines):
    """Cluster lines into paragraphs using the vertical rhythm of the page.

    A gap larger than GAP_FACTOR times the most common leading starts a new
    paragraph. The dominant leading is used rather than each adjacent pair so
    a single loose line does not fragment a paragraph.
    """
    if not lines:
        return []
    gaps = []
    for a, b in zip(lines, lines[1:]):
        g = b["y0"] - a["y0"]
        if g > 0.5:
            gaps.append(g)
    if gaps:
        gaps.sort()
        modal = gaps[len(gaps) // 2]
    else:
        modal = 12.0
    if modal <= 0.5:
        modal = 12.0
    limit = max(modal * GAP_FACTOR, modal + 4.0)

    paras = [[lines[0]]]
    for prev, cur in zip(lines, lines[1:]):
        # A column change is always a paragraph break: the two columns are
        # separate text blocks even when their y ranges happen to overlap.
        new_col = cur.get("col", 0) != prev.get("col", 0)
        if new_col or cur["y0"] - prev["y0"] > limit:
            paras.append([cur])
        else:
            paras[-1].append(cur)
    return paras


def _para_record(para):
    return {
        "top": min(l["y0"] for l in para),
        "bottom": max(l["y1"] for l in para),
        "text": " ".join(l["text"] for l in para),
        "size": max(l["size"] for l in para),
    }


def _overlap(a, b):
    lo = max(a["top"], b["top"])
    hi = min(a["bottom"], b["bottom"])
    raw = max(0.0, hi - lo)
    if raw <= 0.0:
        return 0.0
    # Normalise by the shorter of the two. Absolute overlap is useless here:
    # paragraphs are tall, so any two of them on the same page overlap a little
    # and every source would find a "match", hiding real gaps.
    span = min(a["bottom"] - a["top"], b["bottom"] - b["top"])
    return raw / span if span > 0 else 0.0


def _pair(src_paras, tgt_paras):
    """Pair each source paragraph with the translated paragraph it maps to.

    Overlapping vertical extents are the evidence: a translated paragraph sits
    at roughly the height of its source. Greedy by overlap, each target used
    once, and a source with no real overlap keeps only its own text so nothing
    is silently dropped.
    """
    pairs = []
    used = set()
    for i, sp in enumerate(src_paras):
        best, best_ov = None, 0.0
        for j, tp in enumerate(tgt_paras):
            if j in used:
                continue
            ov = _overlap(sp, tp)
            if ov > best_ov:
                best, best_ov = j, ov
        if best is not None and best_ov > 0.34:
            used.add(best)
            pairs.append((sp["text"], tgt_paras[best]["text"],
                          sp["size"], tgt_paras[best]["size"]))
        else:
            pairs.append((sp["text"], "", sp["size"], 0.0))
    left = [tp for j, tp in enumerate(tgt_paras) if j not in used]
    for tp in left:
        pairs.append(("", tp["text"], 0.0, tp["size"]))
    return pairs


def _classify(text):
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    if cjk and latin:
        return "mixed"
    if cjk:
        return "cjk"
    if latin:
        return "latin"
    return "other"


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #f6f6f4; color: #1b1b1b;
  font: 15px/1.75 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: #fffffff2;
  backdrop-filter: blur(8px); border-bottom: 1px solid #dcdcd6; padding: 12px 24px; }
h1 { font-size: 17px; margin: 0 0 4px; font-weight: 650; }
.meta { font-size: 12.5px; color: #6b6b66; }
.meta b { color: #1b1b1b; font-weight: 600; }
.bar { margin-top: 8px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.bar input { flex: 1 1 220px; min-width: 180px; padding: 6px 10px; font-size: 13px;
  border: 1px solid #cfcfc8; border-radius: 6px; background: #fff; color: inherit; }
.bar label { font-size: 12.5px; color: #55554f; display: flex; gap: 5px; align-items: center; }
.page { max-width: 1180px; margin: 0 auto; padding: 20px 24px 60px; }
.pnum { font-size: 12px; letter-spacing: .09em; text-transform: uppercase;
  color: #8a8a82; margin: 26px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #e2e2db; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 0 20px;
  padding: 9px 0; border-bottom: 1px solid #ececE5; }
.pair:last-child { border-bottom: 0; }
.col { min-width: 0; }
.col .tag { font-size: 11px; color: #9a9a92; display: block; margin-bottom: 2px; }
.src { color: #3d3d38; }
.tgt { color: #10141c; font-weight: 480; }
.pair.only-src { background: #fff8e6; }
.pair.only-tgt { background: #f0f6ff; }
.pair.only-src .col.src .tag::after { content: " · 未译出"; color: #b06a00; }
.miss { font-size: 11px; color: #b06a00; }
@media (max-width: 780px) { .pair { grid-template-columns: 1fr; gap: 4px 0; } }
body.dark { background: #16161a; color: #e6e6e2; }
body.dark header { background: #1c1c21f2; border-color: #303038; }
body.dark .src { color: #b9b9b2; }
body.dark .tgt { color: #f2f2ee; }
body.dark .pair { border-color: #2a2a31; }
body.dark .pair.only-src { background: #2b2416; }
body.dark .pair.only-tgt { background: #1a2130; }
body.dark .bar input { background: #22222a; border-color: #383842; }
body.dark .meta b { color: #f2f2ee; }
"""

_JS = """
const q = document.getElementById('q');
const only = document.getElementById('onlyMiss');
const dark = document.getElementById('dark');
function apply() {
  const s = (q.value || '').trim().toLowerCase();
  document.querySelectorAll('.pair').forEach(p => {
    let ok = !s || p.dataset.text.includes(s);
    if (ok && only.checked) ok = p.classList.contains('only-src');
    p.style.display = ok ? '' : 'none';
  });
  document.querySelectorAll('.page').forEach(pg => {
    const vis = Array.from(pg.querySelectorAll('.pair')).some(p => p.style.display !== 'none');
    pg.querySelector('.pnum').style.display = vis ? '' : 'none';
  });
}
q.addEventListener('input', apply);
only.addEventListener('change', apply);
function setDark(on) { document.body.classList.toggle('dark', on); try { localStorage.setItem('cmp-dark', on ? '1' : '0'); } catch (e) {} }
dark.addEventListener('change', () => setDark(dark.checked));
try { if (localStorage.getItem('cmp-dark') === '1') { dark.checked = true; setDark(true); } } catch (e) {}
"""


def build(dual_path: str, out_html: str, title: str = "") -> dict:
    """Write a bilingual HTML view of a dual PDF. Returns a small report."""
    doc = pymupdf.open(dual_path)
    half = doc[0].rect.width / 2.0
    pages_html = []
    para_count = missing = 0
    lang_hits = {"cjk": 0, "latin": 0, "mixed": 0, "other": 0}

    for pno, pg in enumerate(doc):
        src = _paragraphs(_lines(pg, 0.0, half - 2))
        tgt = _paragraphs(_lines(pg, half + 2, pg.rect.width + 1))
        sp = [_para_record(p) for p in src]
        tp = [_para_record(p) for p in tgt]
        pairs = _pair(sp, tp)
        para_count += len(pairs)

        rows = []
        for s, t, ss, ts in pairs:
            if s and not t:
                missing += 1
            if s:
                lang_hits[_classify(s)] += 1
            cls = "pair" + ("" if (s and t) else (" only-src" if s else " only-tgt"))
            tag_s = f"原文 · {ss:.0f}pt" if ss else "原文"
            tag_t = f"译文 · {ts:.0f}pt" if ts else "译文 · 缺失"
            rows.append(
                f'<div class="{cls}" data-text="{html.escape((s + " " + t).lower(), quote=True)}">'
                f'<div class="col src"><span class="tag">{tag_s}</span>{html.escape(s)}</div>'
                f'<div class="col tgt"><span class="tag">{tag_t}</span>{html.escape(t)}</div>'
                f"</div>"
            )
        pages_html.append(
            f'<div class="page"><div class="pnum">第 {pno + 1} 页 / 共 {len(doc)} 页</div>'
            + "".join(rows) + "</div>"
        )
    doc.close()

    name = os.path.basename(dual_path)
    doc_html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title or name)}</title><style>{_CSS}</style></head><body>
<header>
  <h1>{html.escape(title or name)}</h1>
  <div class="meta">逐段对照 · 共 <b>{para_count}</b> 段，其中 <b>{missing}</b> 段未译出
    · 原文以 CJK {lang_hits['cjk']} 段 / 拉丁 {lang_hits['latin']} 段 / 混排 {lang_hits['mixed']} 段</div>
  <div class="bar">
    <input id="q" type="search" placeholder="搜索原文或译文…（支持中英葡等任意文字）">
    <label><input id="onlyMiss" type="checkbox">只看未译出的段落</label>
    <label><input id="dark" type="checkbox">深色</label>
  </div>
</header>
{"".join(pages_html)}
<script>{_JS}</script>
</body></html>
"""
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(doc_html)
    return {"pages": len(pages_html), "paragraphs": para_count, "missing": missing,
            "html": out_html, "langs": lang_hits}
