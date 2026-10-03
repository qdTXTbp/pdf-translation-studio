"""Paragraph-aware reflow for BabelDOC output.

BabelDOC writes one `BT ... Tf x y TD <glyph> Tj ET` block per glyph, with
absolute page coordinates. That makes paragraph bounds reconstructible from
the stream alone: cluster glyphs by y into lines, then cluster lines by
leading and x-overlap into paragraphs. Each paragraph is then scaled to fill
the room actually available to it.

Two ceilings keep the result safe:
  vertical   -- the gap down to the next paragraph (or the bottom margin)
  horizontal -- the room from the paragraph's left edge to the column's right
                edge, which a wider glyph run would otherwise overflow

Scale never drops below 1.0: this fills whitespace, it does not shrink text.
"""
from __future__ import annotations

import collections
import os
import re
import shutil
from pathlib import Path

import pymupdf

# Marks a PDF this module has already re-laid out, so a repeat run is a no-op.
RE_FLOW_MARK = "auto-filled-by-pdfstudio"

# Typical baseline split of an em box: most of the ink sits above the baseline.
ASCENT = 0.78
DESCENT = 0.22

# Growth factors are quantised to this ladder so nearby paragraphs share a size.
SCALE_STEP = 0.05

# A paragraph is only enlarged when the room available allows at least this much.
# Below it the fill is invisible but the font size it creates is not.
MIN_GROW = 1.10

# Left edges further apart than this are different columns, not first-line
# indents. Two-column journals here have a ~240pt gutter while paragraph
# indents run ~20pt, so anything in between separates them cleanly.
GUTTER_GAP = 60.0

# `Tf` may be written as a bare size, or PrefixedName + size.
_BT_BLOCK = re.compile(
    rb"BT\s*(?P<body>.*?)\bET\b",
    re.S,
)
_KV = re.compile(
    rb"/(?P<font>[^\s/]+)\s+(?P<size>\d+(?:\.\d+)?)\s+Tf"
    rb"|(?P<ax>-?\d+(?:\.\d+)?)\s+(?P<ay>-?\d+(?:\.\d+)?)\s+T[dD]"
    rb"|(?P<lead>-?\d+(?:\.\d+)?)\s+TL",
)


def _blocks(stream: bytes):
    """Yield (start, end, font, size, x, y) for every BT..ET run."""
    out = []
    for m in _BT_BLOCK.finditer(stream):
        body = m.group("body")
        font = size = x = y = None
        for kv in _KV.finditer(body):
            if kv.group("font") is not None:
                font = kv.group("font")
                size = float(kv.group("size"))
            elif kv.group("ay") is not None:
                x = float(kv.group("ax"))
                y = float(kv.group("ay"))
        if size is None or y is None:
            continue
        out.append({"span": (m.start(), m.end()), "font": font,
                    "size": size, "x": x, "y": y})
    return out


def _columns(lines, gap=GUTTER_GAP):
    """Split lines into columns by left edge, preserving reading order.

    A two-column journal interleaves its columns vertically, so ordering lines
    by `y` alone alternates between them. The `x0` test that ends a paragraph
    on an outdent then fires at *every* column switch and shatters the text:
    measured 20 "paragraphs" on a page that holds four, each a two-line
    fragment mixing a left-column line with a right-column one. Every fragment
    was then resized on its own, which is how one paragraph ends up rendering
    at two different sizes with two different left margins.
    """
    if not lines:
        return []
    xs = sorted(ln["x0"] for ln in lines)
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] > gap:
            groups.append([x])
        else:
            groups[-1].append(x)
    # A column is identified by the modal left edge of each cluster; assigns
    # every line to the nearest one so a slightly indented first line still
    # lands in its own column rather than starting a new one.
    centers = [sorted(g)[len(g) // 2] for g in groups]
    out = [[] for _ in centers]
    for ln in lines:
        best = min(range(len(centers)), key=lambda i: abs(centers[i] - ln["x0"]))
        out[best].append(ln)
    return [col for col in out if col]


def _group(blocks, page_h):
    """Cluster glyph blocks into lines, then lines into paragraphs."""
    if not blocks:
        return []
    order = sorted(blocks, key=lambda b: (-b["y"], b["x"]))
    lines = []
    for b in order:
        tol = max(2.0, b["size"] * 0.45)
        if lines and abs(lines[-1]["y"] - b["y"]) <= tol:
            lines[-1]["items"].append(b)
            lines[-1]["y"] = sum(i["y"] for i in lines[-1]["items"]) / len(lines[-1]["items"])
        else:
            lines.append({"y": b["y"], "items": [b]})
    for ln in lines:
        xs = [i["x"] for i in ln["items"]]
        ln["size"] = max(i["size"] for i in ln["items"])
        ln["x0"] = min(xs)
        # `x` is the insertion point of a glyph, i.e. its LEFT edge, so the run
        # ends one glyph further right than the last x. CJK glyphs are a full em
        # wide and some run wider still, so 1.2 em is used to stay on the safe
        # side; the estimate only has to keep an upper bound on the run.
        ln["x1"] = max(xs) + ln["size"] * 1.2

    paras = []
    for col in _columns(lines):
        col.sort(key=lambda l: -l["y"])
        paras.append([col[0]])
        for a, b in zip(col, col[1:]):
            lead = a["y"] - b["y"]
            room = max(a["size"], b["size"]) * 1.75
            # A large gap, or a line starting left of the paragraph's indent,
            # ends it. Both tests now only ever compare lines from one column.
            if lead > room or (b["x0"] < a["x0"] - 6):
                paras.append([b])
            else:
                paras[-1].append(b)
    return paras


def _text_layer(doc, page, dual: bool):
    """Return (xref_of_content_stream_with_text, ...) for the translation layer.

    A mono page draws its text directly. A dual page nests it: the page stream
    invokes `fzFrm0`/`fzFrm1`, each of which only invokes `/fullpage Do` in
    turn, so the glyphs live one level further down. Walk in and pick the
    subtree that actually carries text -- for dual that is the last one drawn
    (the translated right half).
    """
    if not dual:
        ids = list(page.get_contents())
        return ids[0] if ids else None

    obj = doc.xref_object(page.xref)
    drawn = []
    for st in page.get_contents():
        s = doc.xref_stream(st)
        if s:
            drawn += [m.decode() for m in re.findall(rb"/(\w+)\s+Do\b", s)]
    if not drawn:
        return None
    name = drawn[-1]
    m = re.search(rf"/{re.escape(name)}\s+(\d+)\s+0\s+R", obj)
    if not m:
        return None
    for x in _collect_forms(doc, int(m.group(1))):
        raw = doc.xref_stream(x)
        if raw and b"BT" in raw:
            return x
    return None


def reflow(path: str, gain: float = 1.0, max_scale: float = 1.2,
           margin: float = 8.0, dual_side: str = "auto",
           min_scale: float = 0.9) -> dict:
    """Fit each paragraph to the slack around it, growing or slightly shrinking.

    Growing fills the whitespace a rebuilt paragraph leaves behind. Shrinking is
    the correct cure for overflow: the engine can leave a long line running a
    few points past the column edge, and nudging the size down beats either
    accepting the clipping or blanking the area with a solid backdrop.

    Returns a report with the paragraph count, how many were resized (`scaled`),
    how many of those shrank (`shrunk`), the largest factor applied, and the
    pages touched.

    The pass is idempotent: it records a marker in the PDF metadata and returns
    early if it has already run. Re-applying it would compound the factors.
    """
    doc = pymupdf.open(path)
    dual = "dual" in os.path.basename(path).lower()
    report = {"paragraphs": 0, "scaled": 0, "shrunk": 0, "max_scale": 1.0,
              "pages": [], "skipped": False}
    meta = doc.metadata or {}
    if RE_FLOW_MARK in (meta.get("keywords") or ""):
        doc.close()
        report["skipped"] = True
        return report

    for page in doc:
        if not page.get_contents():
            continue
        pw, ph = page.rect.width, page.rect.height
        # A dual page places two copies side by side, but the translated half is
        # drawn inside a Form whose matrix shifts it right by pw/2. Glyph
        # coordinates are therefore local to that half and run 0..pw/2, so the
        # usable column is the left half of the local frame.
        right_bound = pw / 2 if dual else pw
        left_bound = 0.0

        # Real obstacles -- vector art and raster images -- are what text must
        # not cover. Measuring them beats guessing from the widest text run:
        # a narrow paragraph beside a figure legitimately has all the room up
        # to the figure, while a full-width one has none. Raster images larger
        # than most of the page are the scan backdrop, not an obstacle.
        obstacles = _obstacles(page, pw, ph)
        if dual:
            # The translated half lives in a Form whose matrix shifts it right
            # by pw/2, so its glyphs use local coordinates 0..pw/2. Shift the
            # obstacle boxes into that same frame or they never match.
            obstacles = [pymupdf.Rect(b.x0 - pw / 2, b.y0, b.x1 - pw / 2, b.y1)
                         for b in obstacles]

        key = _text_layer(doc, page, dual)
        if key is None:
            continue
        raw = doc.xref_stream(key)
        if not raw or b"BT" not in raw:
            continue
        blocks = _blocks(raw)
        paras = _group(blocks, ph)
        if not paras:
            continue

        # Usable right edge. Glyph runs are the only evidence available about
        # where the column ends. Use the *modal* right edge rather than the
        # widest one: a line that already overflows is by definition the widest
        # line on the page, so deriving the limit from it makes the limit agree
        # with the overflow and the pass concludes there is nothing to fix --
        # which is how three overlong lines survived every run. The mode is the
        # body text's true edge, so anything past it is genuinely over.
        ends = collections.Counter(round(ln["x1"]) for p in paras for ln in p)
        modal_right = ends.most_common(1)[0][0] if ends else right_bound - margin
        col_right = min(right_bound - margin, modal_right + 2.0)

        edits = {}
        for pi, para in enumerate(paras):
            report["paragraphs"] += 1
            top = max(ln["y"] for ln in para)
            bot = min(ln["y"] for ln in para)
            n = len(para)
            # Track every line's own width: the widest line is what a scale
            # factor has to keep inside the column.
            line_w = [ln["x1"] - ln["x0"] for ln in para]
            cur_w = max(line_w)
            x0 = min(ln["x0"] for ln in para)

            # Vertical room. A glyph's insertion point sits on the baseline and
            # the ink runs mostly ABOVE it: roughly 0.78 em up, 0.22 em down.
            # What limits growth is therefore the descender of our last line
            # against the ascender of the next paragraph's first line, not the
            # full em box of either -- scoring the full em overstates the need
            # several-fold and shrinks paragraphs that were never too tall.
            nxt = None
            nxt_size = 0.0
            for q in paras:
                qtop = max(ln["y"] for ln in q)
                if qtop < top - 1 and (nxt is None or qtop > nxt):
                    nxt, nxt_size = qtop, max(ln["size"] for ln in q)
            lead = ((top - bot) / (n - 1)) if n > 1 else para[0]["size"] * 1.2
            # Last paragraph on the page: the limit is the bottom margin.
            ceiling = (nxt + ASCENT * nxt_size) if nxt is not None else margin

            # A figure below the column blocks growth just as much as the next
            # paragraph does, and unlike the next paragraph it leaves no tell in
            # the glyph coordinates.
            for b in obstacles:
                if b.x0 > right_bound or b.x1 < left_bound:
                    continue
                if b.x1 > x0 and b.x0 < right_bound and b.y1 < top - 1:
                    if b.y1 > ceiling:
                        ceiling = b.y1 + 2.0

            cur_h = lead * (n - 1) + DESCENT * para[0]["size"]
            v_room = (top - ceiling) * 0.97
            kv = v_room / cur_h if cur_h > 0 else 1.0

            # Room to the right, bounded by the column AND by whatever figure
            # sits at this paragraph's height. This is what makes filling safe:
            # where the page is genuinely blank the limit is the column, and
            # where artwork starts the limit is the artwork.
            seg_right = min(col_right, right_bound - margin)
            y_lo, y_hi = bot - DESCENT * para[0]["size"], top + ASCENT * para[0]["size"]
            for b in obstacles:
                if b.y1 <= y_lo or b.y0 >= y_hi:
                    continue
                if b.x0 > x0 + 1:
                    seg_right = min(seg_right, b.x0 - 2.0)
            kh = (seg_right - x0) / cur_w if cur_w > 0 else 1.0
            # Hard stop short of the page edge. It has to be tighter than the
            # column test because the column edge is derived from glyph
            # insertion points and can understate the ink by a fraction of a
            # glyph; measured overshoot was ~10pt on a 410pt run.
            hard = (right_bound - margin * 2.5 - x0) / cur_w if cur_w > 0 else 1.0
            kh = min(kh, hard)

            # A paragraph is trimmed only when it actually overflows; growth
            # needs a real amount of room to be worth a new font size. In dense
            # two-column journal typesetting the gap between paragraphs is often
            # smaller than the type itself (line pitch 4.5pt against 6pt glyphs,
            # measured), so a loose "this paragraph does not fit" test misfires
            # on ordinary text and would corrupt a layout the engine already got
            # right.
            k = min(kv, kh, max_scale)
            k = k * gain
            # Snap the factor to a coarse ladder. Each paragraph computes its
            # own headroom, so a continuous k hands every paragraph a slightly
            # different size, which reads as ragged. A 0.05 ladder costs a
            # fraction of a point and collapses most of that spread.
            k = round(k / SCALE_STEP) * SCALE_STEP
            # Below this the fill is not worth the price. A 5% nudge fills a
            # sliver of whitespace but mints a brand-new font size: on a
            # two-column paper it turned a document whose dominant size was
            # 10pt (43%) into one with a *second* dominant size, 9.5pt at 23%,
            # because every 9pt paragraph grew. Two sizes a reader cannot tell
            # apart competing for dominance is exactly what "字体大小不协调"
            # looks like. Demand a real step before growing type.
            #
            # Shrinking has no such bar: any overflow must be recovered, and
            # min_scale bounds how far it may go.
            if k >= 1.0:
                if k < MIN_GROW or k > max_scale:
                    continue
            elif k < min_scale:
                continue
            if abs(k - 1.0) < 0.005:
                continue
            report["scaled"] += 1
            if k < 1.0:
                report["shrunk"] += 1
            report["max_scale"] = max(report["max_scale"], k)

            # Re-lay the paragraph from its top baseline down. Glyphs advance
            # with the font size but keep their own `TD` positions, so raising
            # the size alone makes neighbours overlap: the x offsets must grow
            # by the same factor, measured from the paragraph's left edge.
            lead0 = lead
            for li, ln in enumerate(para):
                new_y = top - lead0 * k * li
                for it in ln["items"]:
                    new_x = x0 + (it["x"] - x0) * k
                    edits[it["span"]] = (it["size"] * k, new_x, new_y)
        if not edits:
            continue

        # Rewrite back-to-front so earlier spans stay addressable.
        buf = bytearray(raw)
        for (s, e), (new_size, new_x, new_y) in sorted(edits.items(), reverse=True):
            body = buf[s:e].decode("latin-1")
            body2 = re.sub(r"(\d+(?:\.\d+)?)(\s+Tf)",
                           lambda m: f"{new_size:.2f}{m.group(2)}", body, count=1)
            body2 = re.sub(
                r"(-?\d+(?:\.\d+)?)(\s+)(-?\d+(?:\.\d+)?)(\s+T[dD])",
                lambda m: f"{new_x:.2f} {new_y:.2f}{m.group(4)}",
                body2, count=1)
            buf[s:e] = body2.encode("latin-1")
        doc.update_stream(key, bytes(buf))
        report["pages"].append(page.number + 1)

    # Leave a marker so a second pass over the same file is a no-op.
    meta = dict(doc.metadata or {})
    kw = (meta.get("keywords") or "").strip()
    meta["keywords"] = (kw + ", " + RE_FLOW_MARK).strip(", ")
    try:
        doc.set_metadata(meta)
    except Exception:
        pass

    tmp = str(path) + ".reflow"
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    shutil.move(tmp, path)
    return report


def _obstacles(page, pw, ph):
    """Visual objects text must not be laid over: vector art and raster images.

    The full-page raster is the scan backdrop -- it is the paper itself, not
    artwork sitting on it -- so anything covering most of the page is dropped.
    """
    boxes = []
    area = pw * ph
    try:
        for im in page.get_images(full=True):
            b = page.get_image_bbox(im)
            if not b or b.is_empty or b.width < 2 or b.height < 2:
                continue
            if b.width * b.height > area * 0.55:
                continue
            boxes.append(b)
    except Exception:
        pass
    try:
        for d in page.get_drawings():
            b = d.get("rect")
            if not b or b.width < 2 or b.height < 2:
                continue
            if b.width * b.height > area * 0.55:
                continue
            boxes.append(b)
    except Exception:
        pass
    return boxes


def _collect_forms(doc, xref, seen=None, depth=0):
    """Return xref plus every Form XObject beneath it (images excluded)."""
    if seen is None:
        seen = set()
    if xref in seen or depth > 8:
        return []
    seen.add(xref)
    try:
        obj = doc.xref_object(xref)
    except Exception:
        return []
    if "/Subtype/Image" in obj.replace(" ", "") or "/Subtype /Image" in obj:
        return []
    if "/Form" not in obj and depth > 0:
        return []
    out = [xref]
    res = re.search(r"/Resources\s+(\d+)\s+0\s+R", obj)
    body = doc.xref_object(int(res.group(1))) if res else obj
    for _, x in re.findall(r"/(\w+)\s+(\d+)\s+0\s+R", body):
        xi = int(x)
        if xi == xref:
            continue
        try:
            sub = doc.xref_object(xi)
        except Exception:
            continue
        if "/XObject" in sub or "/Form" in sub:
            out += _collect_forms(doc, xi, seen, depth + 1)
    return out
