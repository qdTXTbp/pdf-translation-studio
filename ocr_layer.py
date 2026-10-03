"""Give a scanned PDF a text layer so the engine can translate it.

A scanned PDF holds one bitmap per page and no text at all. Measured on a
13-page ScanSnap scan: `text_layer_chars = 0`, one full-page image per page, no
vector content. The whole translation chain starts by extracting the text layer,
so there is nothing for it to work with -- not an extraction failure, simply
nothing there.

The fix is to build that layer ourselves:

    scanned page (bitmap only)
      -> OCR: text plus a quad for each line
      -> write the text back with render mode 3 (invisible)
      -> page now has a full-page backdrop AND an invisible text layer

That output shape is exactly what `server.scan_text_layer` already recognises
(`needs_normalize`), so the existing chain takes over from there: the backdrop
mode keeps the scan intact and the translation is laid over it.

**Coordinate mapping.** OCR runs on a rendered pixmap at RENDER_ZOOM, so its
pixels divide straight back into PDF points. Rendering at a fixed zoom rather
than reading the embedded image keeps the arithmetic trivial and independent of
whatever resolution the scanner used.

**Invisible, not absent.** Render mode 3 draws nothing yet keeps the glyphs
extractable, so `get_text()` and the engine can read the page while the scan
stays visually untouched.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import pymupdf

# OCR runs on the page rendered at this scale; divide its pixels by it to get
# PDF points back.
RENDER_ZOOM = 2.0
# Lines shorter than this are noise from the scan.
MIN_LINE_CHARS = 2
# Recognitions below this confidence are dropped rather than written into the
# text layer, where they would later be translated as if they were real.
MIN_CONF = 0.50
# A PDF averaging fewer extractable characters than this per page is a scan.
SCANNED_CHARS_PER_PAGE = 20
# PDF text render mode 3 = neither fill nor stroke.
INVISIBLE = 3

LATIN_FONT = "helv"
CJK_FONT = "china-s"


def _has_cjk(text: str) -> bool:
    return any("\u3400" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" for c in text)


def text_chars(path: str, limit: int = 4000) -> int:
    """Extractable characters in the first few pages, for the scanned test."""
    doc = pymupdf.open(path)
    total = 0
    for pno, page in enumerate(doc):
        if pno >= 3:
            break
        total += len(page.get_text().strip())
        if total > limit:
            break
    doc.close()
    return total


def needs_text_layer(path: str) -> bool:
    """True when a PDF is a scan with nothing to extract."""
    try:
        doc = pymupdf.open(path)
    except Exception:  # noqa: BLE001
        return False
    pages = len(doc)
    if not pages:
        doc.close()
        return False
    probe = min(3, pages)
    chars = 0
    for i in range(probe):
        chars += len(doc[i].get_text().strip())
        if chars > SCANNED_CHARS_PER_PAGE * probe:
            break
    doc.close()
    return chars < SCANNED_CHARS_PER_PAGE * probe


class Backend(Protocol):
    """An OCR engine. Implementations return (quad, text, confidence)."""

    name: str

    def recognize(self, pix: pymupdf.Pixmap) -> list[tuple[list[list[float]], str, float]]:
        ...


class RapidOcrBackend:
    """rapidocr-onnxruntime: ~20 MB, reuses the onnxruntime the engine ships."""

    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        self._engine = RapidOCR()

    def recognize(self, pix: pymupdf.Pixmap):
        import numpy as np  # type: ignore
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 1:
            img = np.repeat(img, 3, axis=2)
        elif pix.n == 4:
            img = img[:, :, :3]
        result, _ = self._engine(img)
        out = []
        for item in result or []:
            try:
                quad, text, score = item[0], item[1], item[2]
            except (IndexError, TypeError):
                continue
            out.append((quad, str(text), float(score)))
        return out


class PaddleVlBackend:
    """PaddleOCR-VL (0.9B VLM): best accuracy, needs a GPU and a model download.

    Kept behind the same interface so upgrading is a one-line change; the
    import is deferred because the package is large and optional.
    """

    name = "paddleocr-vl"

    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "PaddleOCR-VL 未安装。安装：runtime\\python\\python.exe -m pip "
                "install paddleocr paddlepaddle-gpu"
            ) from exc
        self._engine = PaddleOCRVL()

    def recognize(self, pix: pymupdf.Pixmap):
        import numpy as np  # type: ignore
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 1:
            img = np.repeat(img, 3, axis=2)
        elif pix.n == 4:
            img = img[:, :, :3]
        out = []
        try:
            result = self._engine.predict(img)
        except Exception:  # noqa: BLE001
            return out
        for res in result or []:
            boxes = getattr(res, "boxes", None) or (res.get("boxes") if isinstance(res, dict) else None)
            texts = getattr(res, "rec_texts", None) or (res.get("rec_texts") if isinstance(res, dict) else None)
            scores = getattr(res, "rec_scores", None) or (res.get("rec_scores") if isinstance(res, dict) else None)
            if not boxes or not texts:
                continue
            for i, txt in enumerate(texts):
                if i >= len(boxes):
                    break
                box = boxes[i]
                score = float(scores[i]) if scores and i < len(scores) else 0.9
                out.append((box, str(txt), score))
        return out


def _backend(name: str | None = None) -> Backend | None:
    """Pick a backend: the requested one, else PaddleOCR-VL, else rapidocr."""
    order = [name] if name else []
    order += ["paddleocr-vl", "rapidocr"]
    for candidate in order:
        if not candidate:
            continue
        cls = {"paddleocr-vl": PaddleVlBackend, "rapidocr": RapidOcrBackend}.get(candidate)
        if cls is None:
            continue
        try:
            return cls()
        except Exception:  # noqa: BLE001
            continue
    return None


def _quad_rect(quad: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def _merge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join consecutive lines of one column into paragraph-sized blocks.

    This is the difference between a usable and a useless text layer. Writing
    each OCR line as its own text object makes the engine translate line by
    line, and a sentence cut into pieces translates into pieces -- measured on a
    1969 scan: "factorily explained; the recent reviews on the subject, by Boyd
    (1967)," came back as "人满意地得到解释；Boyd（" plus a separate "1967),".
    Merging restores the sentence before the engine ever sees it.

    Lines join when they share a column (left edges within half a character),
    are close vertically (under 1.8 line pitches), and are set at a similar size.
    """
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: (r["rect"][1], r["rect"][0]))
    out: list[dict[str, Any]] = [dict(ordered[0], parts=[ordered[0]["text"]])]
    for r in ordered[1:]:
        prev = out[-1]
        _, px0, _, py1 = prev["rect"]
        x0, y0, _, _ = r["rect"]
        gap = y0 - py1
        pitch = max(prev["height"], r["height"])
        same_col = abs(x0 - prev["rect"][0]) <= max(2.0, prev["height"] * 0.6)
        close = -pitch * 0.6 <= gap <= pitch * 1.8
        same_size = abs(r["size"] - prev["size"]) <= max(1.0, prev["size"] * 0.35)
        if same_col and close and same_size:
            prev["parts"].append(r["text"])
            prev["rect"] = (min(prev["rect"][0], x0), min(prev["rect"][1], y0),
                            max(prev["rect"][2], r["rect"][2]),
                            max(prev["rect"][3], r["rect"][3]))
            prev["height"] = max(prev["height"], r["height"])
            prev["size"] = max(prev["size"], r["size"])
        else:
            out.append(dict(r, parts=[r["text"]]))
    for blk in out:
        blk["text"] = " ".join(blk["parts"])
    return out


def _rows_to_blocks(rows: Any, zoom: float) -> list[dict[str, Any]]:
    """OCR rows (pixel quads) -> paragraph-sized blocks in PDF points."""
    raw: list[dict[str, Any]] = []
    for quad, text, score in rows:
        text = text.strip()
        if len(text) < MIN_LINE_CHARS or score < MIN_CONF:
            continue
        x0, y0, x1, y1 = _quad_rect(quad)
        px0, px1 = x0 / zoom, x1 / zoom
        py0, py1 = y0 / zoom, y1 / zoom
        h = max(1.0, py1 - py0)
        raw.append({"text": text, "rect": (px0, py0, px1, py1),
                    "height": h, "size": h * 0.82})
    return _merge_rows(raw)


def _place_blocks(page: Any, blocks: list[dict[str, Any]]) -> int:
    """Write merged blocks back as invisible text. Returns how many landed."""
    wrote = 0
    for blk in blocks:
        x0, y0, x1, y1 = blk["rect"]
        size = blk["size"]
        text = blk["text"]
        font = CJK_FONT if _has_cjk(text) else LATIN_FONT
        # PyMuPDF works in top-left origin coordinates, the same as the pixmap
        # OCR ran on -- so the box goes straight across, only padded a little:
        # OCR routinely reports a tight box, and a text box that does not fit
        # its content silently writes nothing (measured: 410 blocks, 0 written).
        #
        # Do NOT flip through "page_h - y": PyMuPDF already converts on write,
        # so flipping here mirrors the whole layer vertically. That is exactly
        # what happened before: every line landed at its mirror position, text
        # from the foot of the page was drawn at the top over the figures, and
        # the engine laid its translation out along those mirrored boxes.
        box = pymupdf.Rect(x0 - 1.0, y0 - 1.0,
                           x1 + 3.0, y1 + max(size, 4.0) * 0.6)
        if box.width < 2 or box.height < 2:
            continue
        # The layer is invisible, so shrinking the type costs nothing and
        # guarantees the text lands. Step down until it fits.
        placed = False
        for scale in (1.0, 0.85, 0.7, 0.55, 0.4, 0.28):
            candidate_fonts = (font, CJK_FONT) if font != CJK_FONT else (font,)
            for fname in candidate_fonts:
                try:
                    rc = page.insert_textbox(
                        box, text, fontsize=max(2.0, size * scale),
                        fontname=fname, render_mode=INVISIBLE, align=0)
                except Exception:  # noqa: BLE001
                    continue
                if rc is not None and rc >= 0:
                    placed = True
                    break
            if placed:
                break
        if placed:
            wrote += 1
    return wrote


def build_text_layer(src: str, dst: str, backend_name: str | None = None,
                     on_page: Any = None) -> dict[str, Any]:
    """OCR `src` and write `dst` with an invisible text layer over the scans.

    Returns a report; `ok` is False when no backend is installed, in which case
    the caller should keep its existing "no text layer" message.

    `on_page(done, total)` is called after each page. OCR takes seconds per page
    on CPU, so a caller that shows progress needs to hear about it while the
    work is happening -- not once at the end.
    """
    report: dict[str, Any] = {
        "ok": False, "backend": None, "pages": 0, "lines": 0,
        "skipped": [], "out": dst,
    }
    engine = _backend(backend_name)
    if engine is None:
        report["reason"] = "没有可用的 OCR 后端"
        return report

    report["backend"] = engine.name
    doc = pymupdf.open(src)
    total_pages = len(doc)
    for pno, page in enumerate(doc):
        try:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM),
                                  alpha=False)
        except Exception:  # noqa: BLE001
            report["skipped"].append(pno + 1)
            continue
        try:
            rows = engine.recognize(pix)
        except Exception:  # noqa: BLE001
            report["skipped"].append(pno + 1)
            continue
        wrote = _place_blocks(page, _rows_to_blocks(rows, RENDER_ZOOM))
        report["lines"] += wrote
        if wrote:
            report["pages"] += 1
        if on_page is not None:
            try:
                on_page(pno + 1, total_pages)
            except Exception:  # noqa: BLE001
                pass
    try:
        doc.save(dst, garbage=3, deflate=True)
        report["ok"] = report["lines"] > 0
    except Exception as exc:  # noqa: BLE001
        report["reason"] = f"保存失败: {exc}"
    doc.close()
    return report


# --------------------------------------------------------------------------
# pages whose text layer is not in reading order
# --------------------------------------------------------------------------
# Some publisher PDFs store the text of a page split into many short runs whose
# order in the content stream is not the reading order. The engine groups
# adjacent runs into "paragraphs", so it glues the tail of one line to the head
# of another and sends nonsense to the model; the translation comes back
# scrambled and is then drawn on top of the leftover English.
#
# The signature is measurable: on an 18-page journal paper the three affected
# pages scored 4.2-4.4 mean word length with 34-35 % of "words" two characters or
# shorter, against 5.4-6.0 and 18-22 % on every other page. The gap is wide, so
# the test is deliberately strict -- a normal page with short words must not
# trip it.
SCRAMBLED_MAX_WORD_LEN = 5.0
SCRAMBLED_MIN_SHORT_PCT = 30.0
SCRAMBLED_MIN_WORDS = 80
# One image this much larger than the page makes it a scan, not a text page.
BACKDROP_COVER = 0.6


def page_word_stats(page: Any) -> tuple[int, float, float]:
    """(words, mean word length, percent of 1-2 character words) for a page."""
    words = page.get_text().split()
    if not words:
        return 0, 0.0, 0.0
    mean = sum(len(w) for w in words) / len(words)
    short = sum(1 for w in words if len(w) <= 2) / len(words) * 100.0
    return len(words), mean, short


def _has_backdrop(page: Any) -> bool:
    """True when one image covers most of the page: a scan, not a text page."""
    area = abs(page.rect.width * page.rect.height) or 1.0
    for img in page.get_images(full=True):
        try:
            rects = page.get_image_rects(img[0])
        except Exception:  # noqa: BLE001
            rects = []
        for rect in rects:
            if abs(rect.width * rect.height) / area > BACKDROP_COVER:
                return True
    return False


def scrambled_pages(path: str) -> list[int]:
    """1-based numbers of text pages whose text layer is out of reading order.

    Pages that are already scans are left alone: the OCR path owns those, and
    re-rasterising them would only lose resolution.
    """
    out: list[int] = []
    try:
        doc = pymupdf.open(path)
    except Exception:  # noqa: BLE001
        return out
    try:
        for pno, page in enumerate(doc):
            if _has_backdrop(page):
                continue
            words, mean, short = page_word_stats(page)
            if (words >= SCRAMBLED_MIN_WORDS
                    and mean < SCRAMBLED_MAX_WORD_LEN
                    and short >= SCRAMBLED_MIN_SHORT_PCT):
                out.append(pno + 1)
    except Exception:  # noqa: BLE001
        return out
    finally:
        doc.close()
    return out


def rebuild_pages(src: str, dst: str, pages: list[int],
                  backend_name: str | None = None, on_page: Any = None,
                  dpi: int = 200) -> dict[str, Any]:
    """Give the listed pages a correct text layer, leaving the rest untouched.

    Each affected page is rendered to a bitmap which becomes the page content,
    so the broken layer cannot reach the engine at all. The original glyphs are
    then painted over in white: the page is a bitmap now, so covering them is the
    only way to keep them from showing through the translation. The OCR text goes
    back invisibly -- the same shape the rest of the pipeline already handles.

    dpi is the render scale. 200 keeps ordinary body text crisp when zoomed and
    costs one page-sized image per rebuilt page.
    """
    report: dict[str, Any] = {
        "ok": False, "backend": None, "pages": 0, "lines": 0,
        "skipped": [], "out": dst,
    }
    engine = _backend(backend_name)
    if engine is None:
        report["reason"] = "没有可用的 OCR 后端"
        return report
    report["backend"] = engine.name

    wanted = set(pages)
    zoom = dpi / 72.0
    src_doc = pymupdf.open(src)
    out_doc = pymupdf.open()
    try:
        done = 0
        for pno, page in enumerate(src_doc):
            if (pno + 1) not in wanted:
                out_doc.insert_pdf(src_doc, from_page=pno, to_page=pno)
                continue
            try:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
            except Exception:  # noqa: BLE001
                out_doc.insert_pdf(src_doc, from_page=pno, to_page=pno)
                report["skipped"].append(pno + 1)
                continue
            new_page = out_doc.new_page(width=page.rect.width,
                                        height=page.rect.height)
            new_page.insert_image(new_page.rect, pixmap=pix)
            try:
                rows = engine.recognize(pix)
            except Exception:  # noqa: BLE001
                rows = []
            blocks = _rows_to_blocks(rows, zoom)
            # Paint out only the blocks OCR is about to redraw. Anything OCR
            # missed keeps its original glyphs: no translation will be laid over
            # it, so erasing it would just punch a hole in the page.
            try:
                boxes = [pymupdf.Rect(b[0], b[1], b[2], b[3])
                         for b in page.get_text("blocks") if b[6] == 0]
            except Exception:  # noqa: BLE001
                boxes = []
            covered: set[int] = set()
            for blk in blocks:
                br = pymupdf.Rect(blk["rect"])
                for i, rect in enumerate(boxes):
                    if i in covered:
                        continue
                    hit = rect & br
                    if not (hit.is_valid and hit.get_area() > 0):
                        continue
                    covered.add(i)
                    try:
                        new_page.draw_rect(
                            pymupdf.Rect(rect.x0 - 0.5, rect.y0 - 0.5,
                                         rect.x1 + 0.5, rect.y1 + 0.5),
                            color=None, fill=(1, 1, 1))
                    except Exception:  # noqa: BLE001
                        pass
            wrote = _place_blocks(new_page, blocks)
            report["lines"] += wrote
            if wrote:
                report["pages"] += 1
            done += 1
            if on_page is not None:
                try:
                    on_page(done, max(1, len(wanted)))
                except Exception:  # noqa: BLE001
                    pass
        out_doc.save(dst, garbage=3, deflate=True)
        report["ok"] = report["lines"] > 0
    except Exception as exc:  # noqa: BLE001
        report["reason"] = f"保存失败: {exc}"
    finally:
        out_doc.close()
        src_doc.close()
    return report


def rebuild_report(rep: dict[str, Any]) -> str:
    if not rep.get("ok"):
        return rep.get("reason") or "OCR 重建未产生文字层"
    return (f"OCR 重建（{rep['backend']}）："
            f"{rep['pages']} 页、{rep['lines']} 段文字层已重写"
            + (f"，跳过 {len(rep['skipped'])} 页" if rep.get("skipped") else ""))


def summary(rep: dict[str, Any]) -> str:
    if not rep.get("ok"):
        return rep.get("reason") or "OCR 未产生文字层"
    return (f"OCR（{rep['backend']}）：{rep['pages']} 页、{rep['lines']} 行文字层已写入"
            + (f"，跳过 {len(rep['skipped'])} 页" if rep.get("skipped") else ""))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 3:
        sys.exit("usage: ocr_layer.py <scanned.pdf> <out.pdf> [backend]")
    s, d = sys.argv[1], sys.argv[2]
    b = sys.argv[3] if len(sys.argv) > 3 else None
    print("需要 OCR:", needs_text_layer(s))
    rep = build_text_layer(s, d, b)
    print(summary(rep))
    if rep.get("ok"):
        print("新文件可提取字符:", text_chars(d))
