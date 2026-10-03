"""Post-process typography audit for a rebuilt PDF.

Reads a finished translation and reports the layout defects that a reader
actually notices, in numbers, so a regression is visible without opening the
file:

* overflow   -- a line whose right edge crosses the column edge or the page
* size sprawl -- how concentrated the translated font sizes are. A rebuilt
  paragraph that was nudged a few percent leaves a size nobody else uses, and
  a page carrying two nearly-identical dominant sizes reads as ragged.
* neighbour jumps -- adjacent lines in the same column whose sizes differ by
  more than a point, which is what the eye catches first
* collisions -- translated lines overlapping each other
* covered art -- solid fills painted over figure artwork or images

Only the translated half of a dual PDF is audited; the left half is the
original and is supposed to look exactly like the source.
"""

from __future__ import annotations

import collections
import os
import re
from typing import Any

import pymupdf

# A size is "clean" when it lands on a half point. Anything else came from a
# percentage nudge rather than a deliberate typographic step.
CLEAN_GRID = 0.5
CLEAN_TOL = 0.02

# Two sizes within this distance are visually the same size and must not both
# be dominant on one document.
NEAR_DUPLICATE = 0.35

JUMP_THRESHOLD = 1.0
COLLISION_OVERLAP = 1.5


def _is_dual(doc: pymupdf.Document) -> bool:
    if not len(doc):
        return False
    return doc[0].rect.width > doc[0].rect.height * 1.3


def _clean(size: float) -> bool:
    r = round(size / CLEAN_GRID) * CLEAN_GRID
    return abs(size - r) <= CLEAN_TOL


def _rows(doc: pymupdf.Document) -> list[dict[str, Any]]:
    """Translated lines as {page,y0,y1,x0,x1,size,text}."""
    out = []
    for pno, pg in enumerate(doc):
        half = pg.rect.width / 2 if _is_dual(doc) else 0.0
        for blk in pg.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            for ln in blk["lines"]:
                text = "".join(sp["text"] for sp in ln["spans"]).strip()
                if not text or ln["bbox"][0] < half:
                    continue
                sizes = [sp["size"] for sp in ln["spans"] if sp["text"].strip()]
                if not sizes:
                    continue
                out.append({
                    "page": pno + 1,
                    "y0": ln["bbox"][1], "y1": ln["bbox"][3],
                    "x0": ln["bbox"][0], "x1": ln["bbox"][2],
                    "size": max(sizes), "text": text,
                })
    return out


def audit(path: str) -> dict[str, Any]:
    """Audit one PDF. Returns metrics plus a list of human-readable findings."""
    doc = pymupdf.open(path)
    dual = _is_dual(doc)
    rows = _rows(doc)
    findings: list[dict[str, Any]] = []

    # --- overflow -------------------------------------------------------
    overflow = []
    for pg in doc:
        width = pg.rect.width
        half = width / 2 if dual else 0.0
        limit = width - 2.0
        for blk in pg.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            for ln in blk["lines"]:
                text = "".join(sp["text"] for sp in ln["spans"]).strip()
                if not text or ln["bbox"][0] < half:
                    continue
                if ln["bbox"][2] > limit:
                    overflow.append({
                        "page": pg.number + 1,
                        "over": round(ln["bbox"][2] - limit, 1),
                        "size": round(ln["spans"][0]["size"], 1),
                        "text": text[:40],
                    })
    if overflow:
        worst = max(o["over"] for o in overflow)
        findings.append({
            "kind": "overflow", "level": "warn",
            "text": (f"{len(overflow)} 行文字越过栏边，最多超出 {worst}pt。"
                     f"首处：第 {overflow[0]['page']} 页 "
                     f"{overflow[0]['text']!r}（字号 {overflow[0]['size']}）"),
        })

    # --- size sprawl ----------------------------------------------------
    sizes = collections.Counter()
    for r in rows:
        sizes[round(r["size"], 2)] += 1
    total = sum(sizes.values()) or 1
    dirty = sum(n for s, n in sizes.items() if not _clean(s))
    dominants = [(s, n / total) for s, n in sizes.items() if n / total >= 0.05]
    dominants.sort(key=lambda kv: -kv[1])

    near = []
    for i in range(len(dominants)):
        for j in range(i + 1, len(dominants)):
            if abs(dominants[i][0] - dominants[j][0]) <= NEAR_DUPLICATE:
                near.append((dominants[i], dominants[j]))
    if near:
        (s1, r1), (s2, r2) = near[0]
        findings.append({
            "kind": "size", "level": "warn",
            "text": (f"存在两个几乎相同的字号同时占主导："
                     f"{s1:g}pt（{r1:.1%}）与 {s2:g}pt（{r2:.1%}）。"
                     f"通常是一段被放大了几个百分点造成的"),
        })
    if dirty / total > 0.25:
        findings.append({
            "kind": "size", "level": "info",
            "text": (f"{dirty / total:.0%} 的字号不在 0.5pt 网格上"
                     f"（{len(sizes)} 种字号），属于百分比微调留下的痕迹"),
        })

    # --- neighbour jumps ------------------------------------------------
    jumps = 0
    samples = []
    bypage: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        bypage[r["page"]].append(r)
    for page, rs in bypage.items():
        rs.sort(key=lambda r: (round(r["x0"] / 40.0), r["y0"]))
        for i in range(1, len(rs)):
            prev, cur = rs[i - 1], rs[i]
            if abs(cur["size"] - prev["size"]) <= JUMP_THRESHOLD:
                continue
            if abs(cur["y0"] - prev["y0"]) >= 20 or abs(cur["x0"] - prev["x0"]) >= 30:
                continue
            jumps += 1
            if len(samples) < 3 and len(cur["text"]) > 6:
                samples.append(f"p{page} {prev['size']:g}→{cur['size']:g}")
    if jumps > 40:
        findings.append({
            "kind": "jump", "level": "info",
            "text": (f"同栏相邻行字号相差 1pt 以上的有 {jumps} 处"
                     f"（{'、'.join(samples)}）；上下标与化学式也会计入"),
        })

    # --- collisions -----------------------------------------------------
    collisions = 0
    for page, rs in bypage.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                ox = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
                oy = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
                if ox > COLLISION_OVERLAP and oy > COLLISION_OVERLAP:
                    collisions += 1
    if collisions:
        findings.append({
            "kind": "collision", "level": "info",
            "text": (f"{collisions} 对译文行互相重叠。正文段落的重叠多为"
                     f"上下标换行所致，成片出现才需要处理"),
        })

    # --- covered artwork -------------------------------------------------
    covered = 0
    for pg in doc:
        area = pg.rect.width * pg.rect.height
        fills = [d["rect"] for d in pg.get_drawings()
                 if d.get("fill") and all(c > 0.85 for c in d["fill"])
                 and d["rect"].width > 8 and d["rect"].height > 4]
        art = []
        for im in pg.get_images(full=True):
            try:
                b = pg.get_image_bbox(im)
            except Exception:
                continue
            if b and not b.is_empty and b.width * b.height < area * 0.5:
                art.append(b)
        art.extend(d["rect"] for d in pg.get_drawings()
                   if d.get("fill") and not all(c > 0.85 for c in d["fill"])
                   and d["rect"].width > 12 and d["rect"].height > 12)
        for f in fills:
            for b in art:
                ox = min(f.x1, b.x1) - max(f.x0, b.x0)
                oy = min(f.y1, b.y1) - max(f.y0, b.y0)
                if ox > 3 and oy > 3:
                    covered += 1
                    break
    if covered:
        # Only meaningful when the backdrop mode produced these fills. The
        # source PDF's own white boxes show up here too, so the finding states
        # the condition instead of asserting damage.
        findings.append({
            "kind": "art", "level": "info",
            "text": (f"{covered} 处实底与图元重叠。开启底衬模式时这里就是被盖住的"
                     f"图内刻度与标签；底衬关闭时多为原文件自带的白色图块"),
        })
    report = {
        "file": os.path.basename(path),
        "pages": len(doc),
        "lines": len(rows),
        "sizes": len(sizes),
        "dirty_ratio": round(dirty / total, 3),
        "dominant": [(round(s, 2), round(r, 3)) for s, r in dominants[:4]],
        "overflow": len(overflow),
        "overflow_worst": round(max((o["over"] for o in overflow), default=0.0), 1),
        "jumps": jumps,
        "collisions": collisions,
        "covered_art": covered,
        "findings": findings,
    }
    doc.close()
    return report


def summary(report: dict[str, Any]) -> str:
    """One-line digest for a job log."""
    dom = "、".join(f"{s:g}pt({r:.0%})" for s, r in report["dominant"][:3])
    return (f"共 {report['lines']} 行译文，{report['sizes']} 种字号"
            f"（主档 {dom}），越界 {report['overflow']} 行"
            f"（最多 {report['overflow_worst']}pt），"
            f"非网格字号 {report['dirty_ratio']:.0%}，"
            f"压图 {report['covered_art']} 处")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        sys.exit("usage: layout_check.py <pdf> [pdf ...]")
    for p in sys.argv[1:]:
        rep = audit(p)
        print(f"\n=== {p} ===")
        print("  " + summary(rep))
        for f in rep["findings"]:
            print(f"  [{f['level']}] {f['text']}")
