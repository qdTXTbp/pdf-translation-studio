"""Restore text that belongs to a figure, which the engine re-flowed as if it
were prose.

The engine treats text inside a figure exactly like body text: it re-lays it to
fit the new language. Measured on a phase diagram that costs real information --
the axis tick `2100` and the label `mol% Al2O3` lost characters, and
`10 20 30 40 50 60` fused into `10 2030 40 50 60` (104 characters in the source
became 103). The reference tool does the same thing, so it is a property of the
rebuild-everything approach rather than a defect in our configuration.

**Why not detect figure regions.** The obvious approach -- find the artwork's
bounding box and copy it back -- was tried and does not work: figure frames,
column rules and the drawings themselves all cluster into boxes that overlap the
prose beside them. Measured on a real paper, the guard against covering prose
refused every candidate (8 figures found, 0 restored), and loosening it would
have painted over translated body text. Artwork and text are intertwined in
coordinate space; they cannot be separated that way.

**What works instead.** Step down a level and repair the individual lines that
are actually wrong. Text inside a figure has a signature that prose does not:

* it is short -- "2100", "mol% Al2O3", "10 20 30" -- while a body line runs the
  full column;
* it is set small, while headings are large;
* it sits immediately next to line work, which headings do not.

Only lines matching all three are restored, and each one is copied back from the
untouched original at its own rectangle. Nothing else on the page is touched, so
a mis-detection costs one line rather than a block. For a dual PDF the original
sits in the left half of the same sheet; for a single-language page the caller
supplies the source document.
"""

from __future__ import annotations

import os
from typing import Any

import pymupdf

# Longer than this and the line is prose, not a figure label.
SHORT_LINE_CHARS = 30
# Figure annotation is set small; headings are not. Anything at or above this is
# left alone so a heading next to a rule is never clobbered.
MAX_FIG_FONT = 12.0
# How close line work has to be for a short line to count as an annotation.
NEAR_ART_PT = 6.0
# A drawing as wide as the column is a rule or a border, not artwork, and every
# heading on the page sits near one.
RULE_WIDTH_RATIO = 0.85
# Objects smaller than this are dots and ticks, too small to anchor a label.
MIN_ART_SIZE = 3.0
# Render scale for the copied-back pixels. 3x is about 216 dpi, enough for the
# 6-8pt annotation this targets; 4x cost roughly twice the file size for no
# visible gain on text.
ZOOM = 3.0
# Channel spread below which a patch is treated as grayscale. Storing those as
# one channel instead of three cut the added file size roughly in half on a
# 15-page paper with 195 restored labels, with no visible difference.
GRAY_TOL = 14

PROTECT_MARK = "figure-text-restored"


def _is_dual(doc: pymupdf.Document) -> bool:
    if not len(doc):
        return False
    r = doc[0].rect
    return r.width > r.height * 1.3


def _art_boxes(page: pymupdf.Page, col_w: float) -> list[pymupdf.Rect]:
    """Line work and images on a page, excluding column-wide rules."""
    out: list[pymupdf.Rect] = []
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if not r:
                continue
            if r.width < MIN_ART_SIZE and r.height < MIN_ART_SIZE:
                continue
            if r.width > col_w * RULE_WIDTH_RATIO:
                continue
            out.append(pymupdf.Rect(r))
    except Exception:  # noqa: BLE001
        pass
    for im in page.get_images(full=True):
        try:
            b = page.get_image_bbox(im)
        except Exception:  # noqa: BLE001
            continue
        if b and not b.is_empty and (b.width > MIN_ART_SIZE or b.height > MIN_ART_SIZE):
            out.append(pymupdf.Rect(b))
    return out


def _near(a: pymupdf.Rect, b: pymupdf.Rect, gap: float) -> bool:
    return (a.x0 - b.x1 < gap and b.x0 - a.x1 < gap
            and a.y0 - b.y1 < gap and b.y0 - a.y1 < gap)


def _figure_lines(page: pymupdf.Page, x_lo: float, x_hi: float,
                  art: list[pymupdf.Rect]) -> list[tuple[pymupdf.Rect, str]]:
    """Short, small lines inside [x_lo, x_hi) that sit next to line work."""
    found: list[tuple[pymupdf.Rect, str]] = []
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            txt = "".join(sp["text"] for sp in ln["spans"]).strip()
            if not txt or not any(c.isalnum() for c in txt):
                continue
            if len(txt) >= SHORT_LINE_CHARS:
                continue
            # Already in the target language, so it is a translation (a table
            # header the engine handled correctly), not source annotation.
            # Measured: without this, one translated Chinese label per paper got
            # painted back to the source language.
            if any("\u4e00" <= c <= "\u9fff" for c in txt):
                continue
            sizes = [sp["size"] for sp in ln["spans"] if sp["text"].strip()]
            if not sizes or max(sizes) >= MAX_FIG_FONT:
                continue
            lb = pymupdf.Rect(ln["bbox"])
            if lb.x0 < x_lo or lb.x1 > x_hi:
                continue
            if any(_near(lb, a, NEAR_ART_PT) for a in art):
                found.append((lb, txt))
    return found


def _grayish(pix: pymupdf.Pixmap) -> bool:
    """True when a patch carries no real colour, so one channel will do.

    Sampled rather than exhaustive: the check runs on every restored patch and
    its only job is to pick a storage format.
    """
    if pix.n < 3:
        return True
    s = pix.samples
    n = pix.n
    step = n * 23
    for i in range(0, len(s) - n, step):
        r, g, b = s[i], s[i + 1], s[i + 2]
        if max(abs(r - g), abs(g - b), abs(r - b)) > GRAY_TOL:
            return False
    return True


def _ocr_texts(pix: pymupdf.Pixmap) -> list[str]:
    """Best-effort OCR of restored regions, for reporting only.

    PaddleOCR is optional. The restore never needs it, so a missing installation
    degrades to "no measurement" rather than a failure.
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:  # noqa: BLE001
        return []
    try:
        import numpy as np  # type: ignore
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 1:
            img = img[:, :, 0]
        elif pix.n >= 3:
            img = img[:, :, :3]
        ocr = _OCR_CACHE.get("m")
        if ocr is None:
            ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
            _OCR_CACHE["m"] = ocr
        res = ocr.ocr(img, cls=False)
        out: list[str] = []
        for group in res or []:
            for item in group or []:
                if len(item) >= 2 and isinstance(item[1], (list, tuple)):
                    out.append(str(item[1][0]))
        return out
    except Exception:  # noqa: BLE001
        return []


_OCR_CACHE: dict[str, Any] = {}


def protect(path: str, source: str | None = None) -> dict[str, Any]:
    """Copy figure-annotation pixels from the original back over the rebuilt page.

    `source` is only needed for a single-language PDF; a dual PDF carries its own
    original in the left half and is therefore self-contained.
    """
    doc = pymupdf.open(path)
    dual = _is_dual(doc)
    report: dict[str, Any] = {
        "pages": [], "candidates": 0, "restored": 0,
        "lines": [], "ocr": [], "skipped": False,
    }
    meta = doc.metadata or {}
    if PROTECT_MARK in (meta.get("keywords") or ""):
        doc.close()
        report["skipped"] = True
        return report

    src: pymupdf.Document | None = None
    if not dual:
        if not source or not os.path.exists(source):
            doc.close()
            report["skipped"] = True
            return report
        src = pymupdf.open(source)

    for pno, page in enumerate(doc):
        pw = page.rect.width
        half = pw / 2 if dual else 0.0
        col_w = half if dual else pw
        art = _art_boxes(page, col_w)
        if not art:
            continue
        if dual:
            targets = [(_figure_lines(page, half, pw, art), page, half)]
        else:
            if src is None or pno >= len(src):
                continue
            src_page = src[pno]
            art_src = _art_boxes(src_page, col_w)
            targets = [(_figure_lines(src_page, 0, pw, art_src), src_page, 0.0)]
        for lines, src_page, offset in targets:
            if not lines:
                continue
            report["pages"].append(pno + 1)
            for lb, txt in lines:
                report["candidates"] += 1
                src_box = pymupdf.Rect(lb.x0 - offset, lb.y0, lb.x1 - offset, lb.y1)
                try:
                    pix = src_page.get_pixmap(
                        clip=src_box, matrix=pymupdf.Matrix(ZOOM, ZOOM), alpha=False)
                    if pix.width < 2 or pix.height < 2:
                        continue
                    if _grayish(pix):
                        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
                    page.insert_image(pymupdf.Rect(lb), pixmap=pix, overlay=True)
                except Exception:  # noqa: BLE001
                    continue
                report["restored"] += 1
                if len(report["lines"]) < 40:
                    report["lines"].append({"page": pno + 1, "text": txt[:40]})
                texts = _ocr_texts(pix)
                if texts and len(report["ocr"]) < 10:
                    report["ocr"].append({"page": pno + 1, "restored": txt[:30],
                                          "ocr": texts[:6]})

    report["pages"] = sorted(set(p for p in report["pages"] if p))
    if report["restored"]:
        try:
            meta = doc.metadata or {}
            kw = (meta.get("keywords") or "").strip()
            meta["keywords"] = (kw + " " + PROTECT_MARK).strip()
            doc.set_metadata(meta)
            doc.save(path, incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
        except Exception:  # noqa: BLE001
            try:
                doc.save(path, incremental=False, garbage=0, deflate=True,
                         encryption=pymupdf.PDF_ENCRYPT_KEEP)
            except Exception:  # noqa: BLE001
                pass
    if src is not None:
        src.close()
    doc.close()
    return report


def summary(rep: dict[str, Any]) -> str:
    if rep.get("skipped"):
        return "已处理过或缺少原始文件，跳过"
    if not rep.get("candidates"):
        return "未发现图内文字"
    return (f"图内文字候选 {rep['candidates']} 行，还原 {rep['restored']} 行"
            f"（页 {','.join(str(p) for p in rep['pages'])}）")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        sys.exit("usage: protect_figures.py <pdf> [source.pdf]")
    p = sys.argv[1]
    s = sys.argv[2] if len(sys.argv) > 2 else None
    rep = protect(p, s)
    print(summary(rep))
    for row in rep.get("lines", [])[:20]:
        print(f"  p{row['page']}: {row['text']!r}")
