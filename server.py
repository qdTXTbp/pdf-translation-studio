#!/usr/bin/env python3
"""
Local PDF Translation Studio -- backend.

Engine : BabelDOC 0.6.4  (layout-preserving bilingual / monolingual PDF translation,
                          the same engine behind the reference output)
Backend: DeepSeek API over the OpenAI-compatible protocol.

Standard library only -- no pip installs. Serves a small UI on 127.0.0.1 and
drives the `babeldoc` CLI as a subprocess, streaming stage progress to the
browser over Server-Sent Events.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

# Paragraph reflow (auto-fill) needs PyMuPDF. The engine venv has it; if it is
# ever missing we keep running and simply skip that post-processing step.
try:
    import reflow
except Exception:  # noqa: BLE001
    reflow = None

# The bilingual reading view is built from the dual PDF, also via PyMuPDF.
try:
    import compare
except Exception:  # noqa: BLE001
    compare = None

# Typography audit run after reflow, so a layout regression shows up as numbers
# in the job log instead of only being noticed by eye.
try:
    import layout_check
except Exception:  # noqa: BLE001
    layout_check = None

# Figure annotation (axis ticks, compound names, table cells) is re-flowed by
# the engine as if it were prose; this copies the original pixels back.
try:
    import protect_figures
except Exception:  # noqa: BLE001
    protect_figures = None

# A scanned PDF holds bitmaps only, so the chain has no text to extract. This
# OCRs a copy and writes an invisible text layer, which is the shape the rest of
# the pipeline already expects.
try:
    import ocr_layer
except Exception:  # noqa: BLE001
    ocr_layer = None

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
OUT_DIR = ROOT / "out"
LOG_DIR = ROOT / "logs"
WORK_DIR = ROOT / "work"
INBOX = ROOT / "inbox"
TERMS_DIR = ROOT / "glossary"
CONF_PATH = ROOT / "config.json"

HOST = "127.0.0.1"
PORT = int(os.environ.get("PDFSTUDIO_PORT", "8760"))

# Opening a PDF through an inline Content-Disposition depends on the browser's
# own PDF plugin, which silently does nothing in some builds. Launching a real
# viewer is deterministic.
EDGE_CANDIDATES = [
    os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
    os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
]

for _d in (UI_DIR, OUT_DIR, LOG_DIR, INBOX, TERMS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# --------------------------------------------------------------------------
# babeldoc stage model -- turns its line-oriented output into a real bar.
# Order matches the engine's actual execution order.
# --------------------------------------------------------------------------
STAGE_NAMES = [
    "Parse PDF and Create Intermediate Representation",
    "DetectScannedFile",
    "Parse Page Layout",
    "Parse Paragraphs",
    "Parse Formulas and Styles",
    "Automatic Term Extraction",
    "Translate Paragraphs",
    "Typesetting",
    "Add Fonts",
    "Generate drawing instructions",
    "Subset font",
    "Save PDF",
]
STAGE_WEIGHTS = [6, 2, 10, 4, 4, 10, 42, 10, 4, 2, 3, 3]

# What the browser shows. The engine's labels are English; the UI is Chinese, and
# a caption reading "Parse Page Layout" for a minute tells the user less than
# "版面分析" does.
STAGE_LABELS_ZH = [
    "解析 PDF",
    "检测扫描件",
    "版面分析",
    "解析段落",
    "公式与样式",
    "提取术语",
    "翻译段落",
    "排版",
    "嵌入字体",
    "生成绘图指令",
    "字体子集化",
    "保存 PDF",
]

# the engine emits alternate labels for the same stage
STAGE_ALIASES = {
    "translate": 6,               # label used in --skip-translation mode
    "translate paragraphs": 6,
    "automatic term extraction": 5,
}

# The engine swallows API failures and still exits 0, producing a PDF that
# looks fine but is untranslated. Detect that and surface it loudly.
ERR_RE = re.compile(
    r"Error code: 4\d\d|AuthenticationError|Authentication Fails"
    r"|invalid_request_error|Error translating paragraph"
)
TOK_RE = re.compile(r"(Prompt|Completion) tokens:\s*(\d+)")
# The engine states the real outcome here. A zero token bill is normal when the
# answer came from the local cache, so this -- not the token count -- decides
# whether a run actually translated anything.
DONE_RE = re.compile(
    r"Translation completed\.\s*Total:\s*(\d+),\s*Successful:\s*(\d+)"
)

# A run counts as finished only when nearly every paragraph came back translated.
COMPLETE_RATIO = 0.98
# The bar's share of the work that belongs to the OCR pass on a scanned file.
# Reserved up front so the bar moves while OCR runs instead of sitting at zero.
OCR_BAND = 15.0

# Failures a second run can actually fix. The engine already retries a failed
# paragraph a few times itself, so what is left is either the wire (worth
# another pass) or the model's answer to one specific prompt (identical every
# time -- measured on a 277-paragraph review: three runs, three times the same
# 252/277, ~95 s spent to buy the same 25 fallbacks twice).
# The engine's own progress bars go through a rich Console that buffers when
# stderr is a pipe -- measured, the whole block lands at process exit, so the bar
# cannot follow it. These lines DO stream as they happen, so they drive a coarse
# progress instead: enough to show the run is moving, and never backwards.
COARSE_STEPS = [
    ("Loading ONNX model", "加载版面模型…", 16.0, 1),
    ("Loaded glossary", "加载术语表…", 18.0, 1),
    ("start to translate", "解析版面与段落…", 22.0, 2),
    ("Found first title paragraph", "翻译段落…", 32.0, 5),
    ("Translation completed", "生成 PDF…", 85.0, 6),
]
TRANSIENT_RE = re.compile(
    r"Connection error|ConnectError|APIConnectionError|Connection reset"
    r"|RemoteDisconnected|ProxyError|Read timed out|ReadTimeout|WriteTimeout"
    r"|timed out|Timeout|Rate limit|Too Many Requests|Error code: 429"
    r"|Error code: 5\d\d|Server error|Service Unavailable|Temporary failure",
    re.I,
)
_TOTAL_W = float(sum(STAGE_WEIGHTS))
STAGE_BASE = []
_acc = 0
for _w in STAGE_WEIGHTS:
    STAGE_BASE.append(_acc / _TOTAL_W * 100.0)
    _acc += _w
STAGE_BASE.append(100.0)

STAGE_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9 ,\-/]{3,60}?)\s+\(\d+/\d+\)\s+-+\s+(\d+)/(\d+)"
)

DEFAULT_PARAMS = {
    "lang_in": "auto",
    "lang_out": "zh",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "qps": 4,
    "watermark": False,
    "alternating": False,
    "mono_only": False,
    "dual_only": False,
    "no_terms": True,
    "formula_hint": True,
    "bilingual_view": True,
    "layout_check": True,
    "protect_figures": True,
    "ocr_scanned": True,
    # Some publisher PDFs store a page's text out of reading order; the engine
    # then glues fragments of different lines into one paragraph and the
    # translation comes out scrambled. Rebuild just those pages from OCR.
    "rebuild_scrambled": True,
    # Left empty by default: a role prompt that spells out "formulas must be
    # preserved" measurably backfires on this model (fallback paragraphs 5 -> 7,
    # same-as-input refusals 15 -> 21), because it reinforces the model's own
    # reading of such paragraphs as untranslatable notation.
    "system_prompt": "",
    "auto_glossary": True,
    "skip_scanned": True,
    # OFF by default, despite fixing overflow. The engine gives *every* text
    # object a solid backdrop including the ones inside figures, so the axis
    # labels of a chart get blanked out -- pixel sampling on a phase diagram
    # showed all 16 Y-axis ticks dropping to pure white. That damage is far
    # worse than the overflow it prevents, and overflow is now handled by
    # shrinking the paragraph instead. Image-type PDFs still auto-enable it
    # (see _is_image_pdf), where there is no other way to show the text.
    "ocr_workaround": False,
    "font_family": "",
    "glossary": "",
    "builtin_glossary": True,
    "glossary_files": [],
    "font_scale": 1.0,
    "auto_fill": True,
    "pages": "",
    "fetch_assets": False,
}


def stage_index(name: str) -> int:
    n = re.sub(r"\s+", " ", name.strip()).lower()
    if n in STAGE_ALIASES:
        return STAGE_ALIASES[n]
    for i, s in enumerate(STAGE_NAMES):
        sl = s.lower()
        if n == sl or n.startswith(sl) or sl in n:
            return i
    return -1


# --------------------------------------------------------------------------
# engine discovery
# --------------------------------------------------------------------------
def find_babeldoc() -> str | None:
    """Locate the engine, preferring the runtime bundled next to this file.

    A portable checkout must not depend on whatever happens to be installed on
    the machine, so the project-local venv wins over anything on PATH.
    """
    here = Path(__file__).resolve().parent
    for rel in (("runtime", "python", "Scripts", "babeldoc.exe"),
                ("runtime", "python", "Scripts", "babeldoc"),
                ("runtime", "venv", "Scripts", "babeldoc.exe"),
                ("runtime", "venv", "bin", "babeldoc")):
        cand = here.joinpath(*rel)
        if cand.exists():
            return str(cand)
    for name in ("babeldoc.exe", "babeldoc"):
        p = shutil.which(name)
        if p:
            return p
    home = Path.home()
    for cand in (
        home / "scoop" / "persist" / "uv" / "tools" / "shims" / "babeldoc.exe",
        home / ".local" / "bin" / "babeldoc.exe",
    ):
        if cand.exists():
            return str(cand)
    return None


BABELDOC = find_babeldoc()


def child_env() -> dict:
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    env["COLUMNS"] = "400"
    return env


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def load_config() -> dict:
    cfg = {"api_key": "", "params": dict(DEFAULT_PARAMS)}
    if CONF_PATH.exists():
        try:
            raw = json.loads(CONF_PATH.read_text(encoding="utf-8-sig"))
            if isinstance(raw, dict):
                cfg["api_key"] = raw.get("api_key", "") or ""
                p = raw.get("params") or {}
                if isinstance(p, dict):
                    cfg["params"].update({k: v for k, v in p.items() if k in DEFAULT_PARAMS})
        except Exception:
            pass
    if not cfg["api_key"]:
        cfg["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
    return cfg


def save_config(cfg: dict) -> None:
    CONF_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------
class Job:
    def __init__(self, files: list[str], params: dict):
        self.id = uuid.uuid4().hex[:12]
        self.files = files
        self.params = params
        self.status = "queued"
        self.progress = 0.0
        self.stage = "starting"
        self.events: list[dict] = []
        self.outputs: list[str] = []
        self.compare_html: str | None = None
        self.rc: int | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.proc: subprocess.Popen | None = None
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._tick_stop: threading.Event | None = None
        self._seen_stage = -1
        self.errors: list[str] = []
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.translated: int | None = None
        self.total_paras: int | None = None

    # -- pub/sub ---------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subs.append(q)
        for ev in list(self.events[-500:]):
            q.put(ev)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def emit(self, ev: dict) -> None:
        ev["job"] = self.id
        ev["progress"] = round(self.progress, 1)
        ev["stage"] = self.stage
        ev["status"] = self.status
        # A run can sit in one engine stage for a minute. Showing the clock is
        # what separates "working" from "hung" for whoever is watching.
        ev["elapsed"] = round(time.time() - self.started, 1)
        self.events.append(ev)
        if len(self.events) > 4000:
            del self.events[:2000]
        with self._lock:
            for q in list(self._subs):
                q.put(ev)

    # -- progress --------------------------------------------------------
    def _phase(self, name: str, pct: float | None = None) -> None:
        """Name the step that is running now and show it in the browser.

        The engine's own stage lines only cover the engine. OCR and the
        post-processing passes emit nothing at all, so without this the bar and
        the caption freeze for as long as those take.
        """
        self.stage = name
        if pct is not None:
            self.progress = max(self.progress, pct)
        self.emit({"type": "progress"})

    def _start_ticker(self) -> None:
        """Push a progress frame on a timer until the job ends.

        The UI only moves on an explicit progress event; stage lines alone do
        not redraw the bar. A frame every half second keeps the elapsed state
        visible even while one long subprocess call is blocking.
        """
        self._tick_stop = threading.Event()

        def loop() -> None:
            assert self._tick_stop is not None
            while not self._tick_stop.wait(0.5):
                self.emit({"type": "progress"})

        threading.Thread(target=loop, daemon=True).start()

    def _stop_ticker(self) -> None:
        if self._tick_stop is not None:
            self._tick_stop.set()

    def _consume(self, line: str) -> None:
        if not line.strip():
            return
        for needle, label, pct, idx in COARSE_STEPS:
            if needle in line:
                if idx > self._seen_stage:
                    self._seen_stage = idx
                self._phase(label, pct)
                break
        m = STAGE_RE.search(line)
        if m:
            idx = stage_index(m.group(1))
            if idx >= 0:
                done, total = int(m.group(2)), max(1, int(m.group(3)))
                if idx > self._seen_stage:
                    self._seen_stage = idx
                    self.stage = (STAGE_LABELS_ZH[idx]
                                  if idx < len(STAGE_LABELS_ZH)
                                  else STAGE_NAMES[idx])
                try:
                    frac = min(1.0, done / total)
                except ZeroDivisionError:
                    frac = 0.0
                base = STAGE_BASE[idx] if idx < len(STAGE_BASE) else self.progress
                span = STAGE_BASE[min(idx + 1, len(STAGE_BASE) - 1)] - base
                self.progress = max(self.progress, base + span * frac)
        # babeldoc exits 0 even when every API call failed -- record evidence
        if len(self.errors) < 12 and ERR_RE.search(line):
            self.errors.append(re.sub(r"\s+", " ", line.strip())[:400])
        mt = TOK_RE.search(line)
        if mt:
            if mt.group(1) == "Prompt":
                self.prompt_tokens = int(mt.group(2))
            else:
                self.completion_tokens = int(mt.group(2))
        md = DONE_RE.search(line)
        if md:
            self.total_paras = int(md.group(1))
            self.translated = int(md.group(2))

        self.emit({"type": "log", "text": line})

    # -- lifecycle -------------------------------------------------------
    def _is_image_pdf(self, path: str) -> bool:
        """Detect a scan-image PDF whose text layer is invisible (Tr 3).

        The engine draws the rebuilt text *under* the backdrop image, so such a
        page looks untranslated even though its text layer does hold Chinese.
        The fix is --ocr-workaround: it gives the new text a solid fill instead
        of deleting anything, so figures and layout survive untouched.
        """
        info = scan_text_layer(path)
        if not info.get("needs_normalize"):
            return False
        self.emit({"type": "log",
                   "text": (f"[precheck] {os.path.basename(path)}: 图像型 PDF"
                            f"（{info['backdrops']} 张全页底图 + 隐形文字层），"
                            "启用底衬模式：保留图表与排版，译文覆盖在原文之上")})
        return True

    def _attempt_failures(self, logf: Path, start: int) -> tuple[int, int]:
        """Split the failures of *one* attempt into (transient, deterministic).

        Only the bytes this attempt wrote are read. The log is appended to
        across retries, so scanning the whole file makes every later attempt
        look as broken as the first, and the loop can never decide to stop.

        Two causes hide behind the same "try fallback" line. A wire error is
        worth another pass. A paragraph the model echoes back, or one whose
        answer is not valid JSON, is deterministic: the same prompt yields the
        same answer, so a retry only pays for it again. Bibliography entries and
        figure captions land in the second bucket, where the model is right to
        decline -- translating "Am. Ceram. Soc. Bull." would destroy a citation.
        """
        try:
            with open(logf, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(start)
                text = fh.read()
        except OSError:
            return 0, 0
        header = (text.count("during translation. try fallback")
                  + text.count("Error translating paragraph")
                  + text.count("Error during translation"))
        transient = len(TRANSIENT_RE.findall(text))
        same = text.count("Translation result is the same as input")
        return transient, max(0, header - transient) + same

    def run(self) -> None:
        self._start_ticker()
        try:
            self._run_pipeline()
        finally:
            self._stop_ticker()

    def _run_pipeline(self) -> None:
        self.status = "running"
        params = dict(self.params)
        files = list(self.files)
        user_workaround = bool(params.get("ocr_workaround"))
        rebuilt_pages: list[int] = []

        # A page whose text layer is stored out of reading order makes the engine
        # glue the tail of one line to the head of another, so it translates
        # nonsense and then draws the Chinese back over the leftover English.
        # Give just those pages a fresh text layer from OCR; every other page is
        # copied through untouched, pixel for pixel.
        if ocr_layer and params.get("rebuild_scrambled", True) and files:
            prepared = []
            for f in files:
                try:
                    bad = ocr_layer.scrambled_pages(f)
                except Exception:  # noqa: BLE001
                    bad = []
                if not bad:
                    prepared.append(f)
                    continue
                stem = os.path.splitext(os.path.basename(f))[0]
                WORK_DIR.mkdir(exist_ok=True)
                rebuilt_out = WORK_DIR / f"{stem}.rebuilt.pdf"
                try:
                    fresh = rebuilt_out.stat().st_mtime >= os.path.getmtime(f)
                except OSError:
                    fresh = False
                shown = "、".join(str(n) for n in bad)
                if rebuilt_out.exists() and fresh:
                    self.emit({
                        "type": "log",
                        "text": (f"[重建] 复用 {rebuilt_out.name}"
                                 f"（第 {shown} 页文字层错序，源文件未变）"),
                    })
                    prepared.append(str(rebuilt_out))
                    rebuilt_pages.extend(bad)
                    continue
                self.emit({
                    "type": "log",
                    "text": (f"[重建] 第 {shown} 页的文字层顺序错乱，"
                             "引擎会把不同行的碎片拼成一段。正在 OCR 重建这几页…"),
                })

                def on_page(done: int, total: int) -> None:
                    self._phase(f"OCR 重建错序页（{done}/{total} 页）",
                                OCR_BAND + done / max(1, total) * OCR_BAND)

                try:
                    rep = ocr_layer.rebuild_pages(f, str(rebuilt_out), bad,
                                                  on_page=on_page)
                except Exception as exc:  # noqa: BLE001
                    self.emit({"type": "warn", "text": f"[重建] 失败：{exc}"})
                    prepared.append(f)
                    continue
                self.emit({"type": "log" if rep.get("ok") else "warn",
                           "text": f"[重建] {ocr_layer.rebuild_report(rep)}"})
                if rep.get("ok"):
                    prepared.append(str(rebuilt_out))
                    rebuilt_pages.extend(bad)
                else:
                    prepared.append(f)
            files = prepared
            if rebuilt_pages:
                self.emit({
                    "type": "log",
                    "text": ("[重建] 这几页已换成位图 + 正确顺序的文字层；"
                             "原文已抹白，不会再与译文叠在一起。"),
                })

        # A scan has no text layer at all, so the engine would have nothing to
        # translate and silently emit a copy of the input. OCR a copy first and
        # translate that instead: the result is a full-page backdrop plus an
        # invisible text layer, which is exactly the shape _is_image_pdf()
        # already knows how to handle.
        if ocr_layer and params.get("ocr_scanned", True) and files:
            prepared = []
            for f in files:
                try:
                    scanned = ocr_layer.needs_text_layer(f)
                except Exception:  # noqa: BLE001
                    scanned = False
                if not scanned:
                    prepared.append(f)
                    continue
                stem = os.path.splitext(os.path.basename(f))[0]
                # NOT under out_dir: the retry path clears the PDFs there to
                # discard a half-finished run, which would delete the OCR copy
                # right after producing it (observed: "文件不存在：...ocr.pdf").
                WORK_DIR.mkdir(exist_ok=True)
                ocr_out = WORK_DIR / f"{stem}.ocr.pdf"
                # OCR is the slowest step of a scanned job -- measured ~5 s per
                # page on CPU, one page at a time. A scan does not change, so
                # reuse the text layer built last time instead of rebuilding an
                # identical one. Regenerate only if the source is newer.
                try:
                    fresh = ocr_out.stat().st_mtime >= os.path.getmtime(f)
                except OSError:
                    fresh = False
                if ocr_out.exists() and fresh:
                    self.emit({
                        "type": "log",
                        "text": (f"[OCR] 复用上次的文字层 {ocr_out.name}，"
                                 f"跳过识别（源文件未变）"),
                    })
                    prepared.append(str(ocr_out))
                    continue
                self.emit({
                    "type": "log",
                    "text": f"[OCR] {os.path.basename(f)} 是扫描件（无文字层），正在识别…",
                })

                def on_page(done: int, total: int) -> None:
                    self._phase(f"OCR 识别扫描件（{done}/{total} 页）",
                                done / max(1, total) * OCR_BAND)

                try:
                    rep = ocr_layer.build_text_layer(f, str(ocr_out), on_page=on_page)
                except Exception as exc:  # noqa: BLE001
                    self.emit({"type": "warn", "text": f"[OCR] 失败：{exc}"})
                    prepared.append(f)
                    continue
                self.emit({"type": "log" if rep.get("ok") else "warn",
                           "text": f"[OCR] {ocr_layer.summary(rep)}"})
                if rep.get("ok"):
                    prepared.append(str(ocr_out))
                else:
                    self.emit({
                        "type": "warn",
                        "text": ("[OCR] 未产生文字层，翻译无法进行。装一个 OCR 后端后重试：\n"
                                 "        runtime\\python\\python.exe -m pip install rapidocr-onnxruntime\n"
                                 "        或安装效果更好的 PaddleOCR-VL：\n"
                                 "        runtime\\python\\python.exe -m pip install paddleocr paddlepaddle-gpu"),
                    })
                    prepared.append(f)
            files = prepared

        if not params.get("ocr_workaround") and files:
            if any(self._is_image_pdf(f) for f in files):
                params["ocr_workaround"] = True
        # A rebuilt page is a bitmap with a clean, invisible text layer and the
        # original glyphs already painted out. The engine's backdrop mode keeps
        # the original text and draws it in solid black, which on those pages
        # would print the OCR text over the translation. Never let the rebuild
        # switch it on behind the user's back.
        if rebuilt_pages and not user_workaround:
            params["ocr_workaround"] = False
        argv = build_argv(files, params)
        shown = []
        skip = False
        for a in argv:
            if skip:
                shown.append("<key>")
                skip = False
                continue
            if a == "--openai-api-key":
                skip = True
            shown.append(a)
        self.emit({"type": "status", "text": "running"})
        self.emit({"type": "cmd", "text": "babeldoc " + " ".join(shown)})

        # pre-flight: a PDF with no text layer is a pure scan and cannot be
        # translated, so say so plainly instead of burning API calls on it
        for f in self.files:
            n = text_layer_chars(f)
            if n is not None and n < 40:
                self.emit({
                    "type": "warn",
                    "text": (f"{os.path.basename(f)} 没有可提取的文字层"
                             "（纯扫描图片），翻译引擎无从下手。"
                             "请先用 OCR 工具生成带文字层的 PDF，"
                             "或改用该文献的电子版。"),
                })

        # The engine says nothing until its first stage line, which is seconds
        # away (it loads an ONNX layout model first). Without a caption the bar
        # would read "starting" in English for that whole stretch.
        self._phase("翻译引擎启动中…")
        logf = LOG_DIR / f"{self.id}.log"
        rc = 1
        prev: tuple[int | None, int | None] | None = None
        for attempt in range(3):
            # Reset before each run: _consume fills these from the engine's own
            # summary line, and a stale value from the previous attempt would
            # make the comparison below meaningless.
            self.translated = None
            self.total_paras = None
            try:
                start = logf.stat().st_size if attempt else 0
            except OSError:
                start = 0
            rc = self._attempt(argv, logf, append=attempt > 0)
            result = (self.translated, self.total_paras)
            if rc == 0 and self._ratio() >= COMPLETE_RATIO:
                break
            if attempt >= 2:
                break
            transient, deterministic = self._attempt_failures(logf, start)
            if rc == 0 and result == prev:
                # Same paragraph count and same success count as the previous
                # attempt: the model is answering the same prompts the same way,
                # so one more run would only pay for the same answers again.
                self.emit({
                    "type": "log",
                    "text": (f"[跳过重试] 第 {attempt + 1} 次与上次结果完全一致"
                             f"（{self.translated or 0}/{self.total_paras or '?'} 段），"
                             "重跑不会改变结果，已保留原文。"),
                })
                break
            if rc == 0 and transient == 0:
                # Nothing to recover: every failure is the model's own answer to
                # one specific prompt, and the same prompt yields the same
                # answer. Retrying would resend them and buy them twice more.
                self.emit({
                    "type": "log",
                    "text": (f"[跳过重试] {deterministic} 段是模型对同一提示词的固定回答"
                             "（原样返回、图注或参考文献等），重试必然相同，"
                             "已保留原文，不再重复调用。"),
                })
                break
            # The engine keeps the source text for paragraphs it could not
            # translate and still exits 0, so a bad run looks like success.
            # Drop only this job's half-done PDFs and go again. Older results in
            # the same folder belong to other documents and stay untouched.
            out_dir = Path(self.params.get("out_dir") or OUT_DIR)
            for stale in out_dir.glob("*.pdf"):
                try:
                    if stale.stat().st_mtime >= self.started - 5:
                        stale.unlink()
                except OSError:
                    pass
            left = (self.total_paras or 0) - (self.translated or 0)
            self.emit({
                "type": "warn",
                "text": (f"本次有 {left} 段未翻译成功，保留了原文"
                         f"（共 {self.total_paras or '?'} 段，"
                         f"成功 {self.translated or 0} 段）。"
                         f"检测到 {transient} 处网络/接口异常，正在自动重跑"
                         f"第 {attempt + 2} 次。"),
            })
            prev = result
            time.sleep(3)
        self._finish(rc, rebuilt_pages)

    def _ratio(self) -> float:
        """Fraction of paragraphs the engine reported as translated."""
        if not self.translated:
            return 0.0
        return self.translated / (self.total_paras or self.translated)

    def _attempt(self, argv: list[str], logf: Path, append: bool = False) -> int:
        """Run the engine once and stream its output. Returns its exit code."""
        try:
            self.proc = subprocess.Popen(
                [BABELDOC] + argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=child_env(),
                cwd=str(ROOT),
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:  # noqa: BLE001
            self.status = "error"
            self.emit({"type": "error", "text": f"cannot start babeldoc: {exc}"})
            return 1

        with open(logf, "a" if append else "w", encoding="utf-8") as fh:
            assert self.proc.stdout is not None
            for raw in self.proc.stdout:
                line = ANSI_RE.sub("", raw.rstrip("\r\n"))
                fh.write(line + "\n")
                self._consume(line)
        self.proc.wait()
        return self.proc.returncode or 0

    def _finish(self, rc: int, rebuilt_pages: list[int] | None = None) -> None:
        self.rc = rc
        out_dir = Path(self.params.get("out_dir") or OUT_DIR)
        since = self.started - 5
        try:
            for f in sorted(out_dir.glob("*.pdf")):
                if f.stat().st_mtime >= since:
                    self.outputs.append(str(f))
        except Exception:
            pass
        # A dual built from a rebuilt file has the painted-out pages on the
        # left. Put the user's own file back there, so the reading view still
        # shows the original next to the translation.
        if rc == 0 and rebuilt_pages and self.files:
            for f in list(self.outputs):
                if f.lower().endswith(".pdf") and "dual" in os.path.basename(f).lower():
                    try:
                        if restore_dual_left(f, self.files[0]):
                            self.emit({
                                "type": "log",
                                "text": (f"[重建] {os.path.basename(f)}"
                                         " 左半页已换回原始页面"),
                            })
                    except Exception as exc:  # noqa: BLE001
                        self.emit({"type": "log",
                                   "text": f"[重建] 左半页还原失败: {exc}"})
        scale = float(self.params.get("font_scale") or 1.0)
        self._phase("整理产物", 96.0)
        if rc == 0 and self.outputs and abs(scale - 1.0) > 1e-6:
            self._phase("调整译文字号", 96.5)
            total = 0
            for f in self.outputs:
                try:
                    total += scale_fonts(f, scale)
                except Exception as exc:  # noqa: BLE001
                    self.emit({"type": "log",
                               "text": f"[font] 调整字号失败 {os.path.basename(f)}: {exc}"})
            self.emit({"type": "log",
                       "text": f"[font] 译文字号 ×{scale:g}，共调整 {total} 处"})
        if rc == 0 and self.outputs and reflow and self.params.get("auto_fill", True):
            self._phase("排版自适应填充", 97.0)
            for f in self.outputs:
                try:
                    rep = reflow.reflow(f, gain=1.0, min_scale=0.9)
                    if rep.get("scaled"):
                        grew = rep["scaled"] - rep.get("shrunk", 0)
                        self.emit({
                            "type": "log",
                            "text": (f"[排版] {os.path.basename(f)}: "
                                     f"共 {rep['paragraphs']} 段，"
                                     f"{grew} 段放大、{rep.get('shrunk', 0)} 段收窄"
                                     f"（最大 ×{rep['max_scale']:.2f}）"
                                     f"（页 {','.join(str(p) for p in rep['pages'])}）"),
                        })
                except Exception as exc:  # noqa: BLE001
                    self.emit({"type": "log",
                               "text": f"[排版] 自适应填充失败 {os.path.basename(f)}: {exc}"})
        if rc == 0 and self.outputs and protect_figures and self.params.get("protect_figures", True):
            self._phase("图内文字还原", 98.0)
            # Must run after reflow: that pass moves text, and the restoration
            # reads the current rectangle of each line. A dual PDF carries its
            # own original in the left half, so only a single-language page
            # needs the source document.
            src_pdf = self.files[0] if self.files else None
            for f in self.outputs:
                if not f.lower().endswith(".pdf"):
                    continue
                try:
                    rep = protect_figures.protect(f, src_pdf)
                except Exception as exc:  # noqa: BLE001
                    self.emit({"type": "log",
                               "text": f"[图内文字] 跳过 {os.path.basename(f)}: {exc}"})
                    continue
                if rep.get("restored"):
                    self.emit({"type": "log",
                               "text": (f"[图内文字] {os.path.basename(f)}: "
                                        f"{protect_figures.summary(rep)}")})
        if rc == 0 and self.outputs and layout_check and self.params.get("layout_check", True):
            self._phase("排版体检", 99.0)
            # Audit what was just produced. The metrics make a regression visible
            # in the log: two competing dominant font sizes, lines past the
            # column edge, or solid fills sitting on top of artwork.
            for f in self.outputs:
                if not f.lower().endswith(".pdf"):
                    continue
                try:
                    rep = layout_check.audit(f)
                except Exception as exc:  # noqa: BLE001
                    self.emit({"type": "log",
                               "text": f"[体检] 跳过 {os.path.basename(f)}: {exc}"})
                    continue
                self.emit({
                    "type": "log",
                    "text": f"[体检] {os.path.basename(f)}: {layout_check.summary(rep)}",
                })
                for item in rep["findings"]:
                    self.emit({"type": "warn" if item["level"] == "warn" else "log",
                               "text": f"[体检] {item['text']}"})
        if rc == 0 and self.outputs and compare and self.params.get("bilingual_view", True):
            self._phase("生成逐段对照视图", 99.5)
            # The dual PDF is the only artifact where a source paragraph and its
            # translation sit on the same sheet at the same height, so it is what
            # the side-by-side reader is built from.
            dualf = next((f for f in self.outputs if "dual" in os.path.basename(f).lower()), None)
            if dualf:
                try:
                    target = os.path.splitext(dualf)[0] + ".对照.html"
                    br = compare.build(dualf, target,
                                       title=os.path.basename(dualf).split(".zh.")[0])
                    self.compare_html = target
                    self.emit({
                        "type": "log",
                        "text": (f"[对照] 逐段对照视图已生成：{br['paragraphs']} 段，"
                                 f"其中 {br['missing']} 段未译出（{os.path.basename(target)}）"),
                    })
                except Exception as exc:  # noqa: BLE001
                    self.emit({"type": "log",
                               "text": f"[对照] 生成对照视图失败: {exc}"})
        if rc == 0 and self.outputs:
            self.progress = 100.0
            if self.translated:
                # the engine reported successful paragraphs (fresh or cached).
                # Where a paragraph fails it silently keeps the source text, so
                # a low ratio means a mostly-untranslated document that still
                # exits 0. Judge the ratio, not the exit code. Anything short of
                # complete counts as unfinished: a single fallback paragraph is
                # a visible block of untranslated prose, not a rounding error.
                total = self.total_paras or self.translated
                ratio = self.translated / total if total else 1.0
                if ratio < 1.0:
                    left = total - self.translated
                    self.status = "warn"
                    self.stage = f"{left} 段未翻译（{self.translated}/{total}）"
                    self.emit({
                        "type": "warn",
                        "text": (f"仍有 {left} 段没有翻译成功（{self.translated}/{total}），"
                                 "这些段落保留着原文。已自动重跑多次仍未补全，"
                                 "说明是模型对这段内容持续返回异常响应。"
                                 "把并发 QPS 调低（如 1）或换 --openai-model 后再试；"
                                 "也可以在日志里搜 'try fallback' 定位具体段落。"),
                    })
                else:
                    self.stage = "done"
                    self.status = "done"
            elif self.errors or not self.prompt_tokens:
                # exit code 0 but the translation never actually happened
                self.status = "warn"
                self.stage = "completed with errors"
                if any("401" in e or "uthentication" in e for e in self.errors):
                    msg = ("API 鉴权失败：密钥无效。产出的是未翻译的原文副本，"
                           "请填入有效 Key 并点「保存密钥」后重跑。")
                elif self.prompt_tokens == 0:
                    msg = ("引擎未消耗任何 token，翻译请求没有成功发出，"
                           "产出可能仍是原文，请检查日志。")
                else:
                    msg = "引擎报告了错误但仍产出文件，结果可能不完整，请检查日志。"
                self.emit({"type": "warn", "text": msg})
            else:
                self.stage = "done"
                self.status = "done"
        elif rc == 0:
            self.status = "error"
            self.emit({"type": "error", "text": "engine finished but produced no PDF"})
        else:
            self.status = "error"
        self.finished = time.time()
        self.emit({
            "type": "end",
            "rc": rc,
            "outputs": self.outputs,
            "compare_html": self.compare_html,
            "errors": self.errors[:5],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        })

    def cancel(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def restore_dual_left(dual_path: str, original_path: str) -> bool:
    """Put the real original back on the left half of a side-by-side dual.

    The engine builds the dual from whatever file it was handed. For pages that
    were rebuilt from OCR that is the rebuilt copy, whose original text was
    painted out -- so the reading view's left column comes up blank. Re-compose
    those pages with the user's own file, which still carries the original.
    """
    try:
        import pymupdf
    except Exception:  # noqa: BLE001
        return False
    try:
        dual = pymupdf.open(dual_path)
        orig = pymupdf.open(original_path)
    except Exception:  # noqa: BLE001
        return False
    tmp = dual_path + ".tmp"
    out = pymupdf.open()
    try:
        for i in range(min(len(dual), len(orig))):
            dp = dual[i]
            half = dp.rect.width / 2.0
            height = dp.rect.height
            page = out.new_page(width=dp.rect.width, height=height)
            page.show_pdf_page(pymupdf.Rect(0, 0, half, height), orig, i)
            page.show_pdf_page(pymupdf.Rect(half, 0, 2 * half, height), dual, i,
                               clip=pymupdf.Rect(half, 0, 2 * half, height))
        out.save(tmp, garbage=3, deflate=True)
    except Exception:  # noqa: BLE001
        out.close(); dual.close(); orig.close()
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
    out.close(); dual.close(); orig.close()
    try:
        os.replace(tmp, dual_path)
    except OSError:
        return False
    return True


def detect_language(path: str) -> tuple[str | None, dict]:
    """Best-effort source-language guess from the PDF's own text layer.

    The user works across languages (a Japanese paper, a Portuguese one, English
    ones), so guessing beats a hardcoded default. Degrades gracefully when
    PyMuPDF is missing, leaving the choice to the caller.
    """
    try:
        import pymupdf
    except Exception:
        return None, {}
    try:
        doc = pymupdf.open(path)
        text = "".join(doc[i].get_text() for i in range(min(doc.page_count, 6)))
        doc.close()
    except Exception:
        return None, {}
    if len(text.strip()) < 40:
        return None, {}

    n = max(len(text), 1)
    kana = sum(1 for c in text if 0x3040 <= ord(c) <= 0x30FF)
    han = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    cyr = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)
    stats = {
        "chars": len(text),
        "kana_pct": round(kana * 100.0 / n, 1),
        "han_pct": round(han * 100.0 / n, 1),
        "cyr_pct": round(cyr * 100.0 / n, 1),
    }

    # Japanese carries kana; Chinese does not. Check kana first.
    if kana * 100.0 / n > 2.0:
        return "ja", stats
    if han * 100.0 / n > 15.0:
        return "zh", stats
    if cyr * 100.0 / n > 20.0:
        return "ru", stats

    low = text.lower()

    def hits(words: list[str]) -> int:
        return sum(1 for w in words if w in low)

    scores = {
        "pt": hits(["ção", "ões", " uma ", " dos ", " das ", " não ", " para o ", " do "]),
        "es": hits([" una ", " los ", " las ", " del ", " pero ", " con ", " para el ", "ñ"]),
        "fr": hits([" les ", " des ", " une ", " est ", " dans ", " pour ", " par ", "é "]),
        "de": hits([" der ", " die ", " und ", " des ", " nicht ", " eine ", " mit ", "ß"]),
        "it": hits([" della ", " degli ", " nella ", " sono ", " con ", " più "]),
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= 5:
        return best, stats
    return "en", stats


def text_layer_chars(path: str) -> int | None:
    """Count extractable text characters; None when the file cannot be read.

    A PDF with essentially no text layer is a pure scan: there is nothing for
    the engine to translate, so warn before burning API calls on it.
    """
    try:
        import pymupdf
    except Exception:
        return None
    try:
        doc = pymupdf.open(path)
        txt = "".join(doc[i].get_text() for i in range(min(doc.page_count, 8)))
        doc.close()
    except Exception:
        return None
    return len(txt.strip())


def scan_text_layer(path: str) -> dict:
    """Report whether a PDF is an image-backed scan with a hidden text layer.

    Publisher "scans" are frequently a full-page image plus an *invisible* OCR
    text layer (rendering mode 3). The engine rebuilds the text fine, but the
    image is drawn after it, so the translation ends up hidden behind the scan
    and the page looks untranslated. Those files need the backdrop removed and
    the text made visible first.
    """
    out = {"ok": False, "backdrops": 0, "invisible": 0, "chars": 0,
           "needs_normalize": False}
    try:
        import pymupdf
    except Exception:
        return out
    try:
        doc = pymupdf.open(path)
    except Exception:
        return out
    try:
        chars = backdrops = invisible = 0
        for page in doc:
            chars += len(page.get_text().strip())
            parea = abs(page.rect.width * page.rect.height) or 1.0
            for img in page.get_images(full=True):
                try:
                    rects = page.get_image_rects(img[0])
                except Exception:
                    rects = []
                if any(abs(r.width * r.height) / parea > 0.6 for r in rects):
                    backdrops += 1
                    break
            for x in page.get_contents():
                st = doc.xref_stream(x)
                if st and b"3 Tr" in st:
                    invisible += 1
                    break
        out.update(ok=True, chars=chars, backdrops=backdrops,
                   invisible=invisible, pages=len(doc))
        # Only worth touching when a real text layer exists to expose, and only
        # when the document is *mostly* image pages. A single figure page does
        # not make a text PDF a scan, and switching the whole run into backdrop
        # mode would keep the original text on every other page -- which is the
        # interleaving this is meant to avoid.
        out["needs_normalize"] = bool(
            backdrops and chars > 100 and backdrops * 2 >= max(1, len(doc))
        )
    except Exception:
        pass
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out


_TF_RE = re.compile(rb"(\d+(?:\.\d+)?)\s+Tf\b")


def scale_fonts(path: str, scale: float) -> int:
    """Multiply the translated text's font sizes in a finished PDF by `scale`.

    The engine picks the translated size from the original blocks, which reads
    small on dense papers, and exposes no flag for it. So the sizes are
    rescaled in place. Only the translated layer is touched: for a mono output
    that is the page stream, for a dual output it is the last Form XObject
    (the right-hand translated half) while the left half -- the pristine
    original -- is left alone.
    """
    import pymupdf

    is_dual = "dual" in os.path.basename(path).lower()
    doc = pymupdf.open(path)
    touched = 0

    def rewrite(stream: bytes) -> bytes:
        nonlocal touched

        def repl(m: re.Match) -> bytes:
            nonlocal touched
            touched += 1
            return f"{float(m.group(1)) * scale:.3f}".encode() + b" Tf"

        return _TF_RE.sub(repl, stream)

    def collect(xref: int, seen: set, depth: int = 0) -> list[int]:
        """The Form XObject at `xref` plus every Form XObject nested below it.

        The engine wraps each half of a bilingual page in a container that only
        says `/fullpage Do`; the text lives one level further down, so the whole
        tree has to be walked rather than just the first XObject.
        """
        if xref in seen or depth > 8:
            return []
        seen.add(xref)
        try:
            o = doc.xref_object(xref)
        except Exception:
            return []
        if "/Subtype /Image" in o:
            return []
        out = [xref] if "/Subtype /Form" in o else []
        for m in re.finditer(r"/(\w+)\s+(\d+)\s+0\s+R", o):
            out += collect(int(m.group(2)), seen, depth + 1)
        return out

    for page in doc:
        if is_dual:
            names: list[bytes] = []
            for x in page.get_contents():
                st = doc.xref_stream(x)
                if st:
                    names += re.findall(rb"/(\w+)\s+Do\b", st)
            if not names:
                continue
            # the translated half is the last one drawn
            right = names[-1].decode()
            m = re.search(rf"/{re.escape(right)}\s+(\d+)\s+0\s+R",
                          doc.xref_object(page.xref))
            if not m:
                continue
            targets = collect(int(m.group(1)), set())
        else:
            # a mono page draws the translation straight in its content stream,
            # so take that stream itself and then any Form XObject beneath it
            targets = list(page.get_contents())
            for x in list(targets):
                targets += collect(x, set())
        for xref in dict.fromkeys(targets):
            st = doc.xref_stream(xref)
            if not st:
                continue
            new = rewrite(st)
            if new != st:
                doc.update_stream(xref, new)
    doc.saveIncr()
    doc.close()
    return touched


def _term_sources(csv_path: Path) -> list[str]:
    """First column (source terms) of a glossary CSV, tolerant of encoding."""
    try:
        raw = csv_path.read_bytes()
    except Exception:
        return []
    text = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        return []
    out: list[str] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        head = line.split(",", 1)[0].strip().strip('"')
        if i == 0 and head.lower() == "source":
            continue
        if head:
            out.append(head)
    return out


def pick_glossaries(files: list[str], top_n: int = 3) -> list[Path]:
    """Choose the glossary tables whose terms actually occur in the document.

    Loading all tables makes every batch carry whatever terms happen to appear
    in it, which inflates the prompt and, worse, lets a short term from an
    unrelated field match inside ordinary prose. Scoring each table by how
    often its own terms occur in the source keeps the injected set small and
    on-topic. Returns [] when the text cannot be read, which lets the caller
    keep its previous behaviour.
    """
    text = ""
    try:
        import pymupdf  # noqa: PLC0415
        doc = pymupdf.open(files[0])
        try:
            for page in list(doc)[:6]:
                text += page.get_text()
        finally:
            doc.close()
    except Exception:
        return []
    if len(text) < 200:
        return []
    low = text.lower()
    scored: list[tuple[int, Path]] = []
    for csv_path in sorted(TERMS_DIR.glob("*.csv")):
        hits = 0
        for term in _term_sources(csv_path):
            if len(term) < 2:
                continue
            n = low.count(term.lower())
            if n:
                hits += 1 + (n.bit_length() - 1)  # presence counts, repeats weigh more
        if hits:
            scored.append((hits, csv_path))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return []
    # Keep tables that are within reach of the best one, then cap the count:
    # a table with a couple of stray hits is noise, not a domain match.
    best = scored[0][0]
    keep = [c for s, c in scored if s >= max(2, best * 0.15)][:top_n]
    return keep


def build_argv(files: list[str], p: dict) -> list[str]:
    argv: list[str] = []
    for f in files:
        argv += ["--files", f]
    # "auto" resolves the source language from the PDF's own text layer
    lang_in = str(p.get("lang_in") or "auto")
    if lang_in == "auto":
        lang_in = (detect_language(files[0])[0] if files else None) or "en"
    argv += [
        "--openai",
        "--openai-model", str(p.get("model") or "deepseek-chat"),
        "--openai-base-url", str(p.get("base_url") or "https://api.deepseek.com/v1"),
        "--openai-api-key", str(p.get("api_key") or ""),
        "--lang-in", lang_in,
        "--lang-out", str(p.get("lang_out") or "zh"),
        "--output", str(p.get("out_dir") or OUT_DIR),
        "--qps", str(p.get("qps") or 4),
        "--watermark-output-mode",
        "watermarked" if p.get("watermark") else "no_watermark",
    ]
    if p.get("alternating"):
        argv.append("--use-alternating-pages-dual")
    if p.get("mono_only"):
        argv.append("--no-dual")
    if p.get("dual_only"):
        argv.append("--no-mono")
    if p.get("no_terms"):
        argv.append("--no-auto-extract-glossary")
    # Tells the model that chemical formulas and inline math are placeholders it
    # must carry through untouched. Without it the model tends to decide a
    # paragraph mixing Japanese prose with formulas "needs no translation" and
    # echoes the input, which the engine then scores as a failed paragraph.
    if p.get("formula_hint", True):
        argv.append("--add-formula-placehold-hint")
    # The engine rejects a paragraph whose output is near-identical to its input
    # (edit distance < 5 on 20+ tokens) and falls back to the source text. On
    # papers where Japanese prose is interleaved with formulas the model tends
    # to declare the paragraph technical and echo it, which silently leaves a
    # block of Japanese in the output. This role text tells it to translate the
    # prose while carrying formulas and identifiers through unchanged.
    if p.get("system_prompt"):
        argv += ["--custom-system-prompt", str(p["system_prompt"])]
    # The engine's scan detector misfires on image-backed PDFs that do carry an
    # OCR text layer (a common publisher layout). Skipping it lets those work.
    if p.get("skip_scanned", True):
        argv.append("--skip-scanned-detection")
    # For scan-image PDFs this fills the new text with a solid background, so
    # the translation stays readable over the backdrop without losing figures.
    if p.get("ocr_workaround"):
        argv.append("--ocr-workaround")
    if p.get("font_family"):
        argv += ["--primary-font-family", str(p["font_family"])]
    # Glossaries are plain CSVs (source,target,tgt_lng). `glossary_files` names
    # which bundled tables to load; when it is empty every table in the folder
    # is used. A user-supplied path (or list) is appended on top.
    gl: list[str] = []
    want = p.get("glossary_files")
    if want:
        for n in (want if isinstance(want, (list, tuple)) else [want]):
            cand = TERMS_DIR / Path(str(n)).name
            if cand.is_file():
                gl.append(str(cand))
    elif p.get("builtin_glossary", True):
        # Prefer the few tables this document actually talks about; fall back to
        # the whole set only when the text cannot be scored.
        picked = pick_glossaries(files) if p.get("auto_glossary", True) else []
        if not picked:
            picked = sorted(TERMS_DIR.glob("*.csv"))
        gl += [str(f) for f in picked]
    if p.get("glossary"):
        gl += [x.strip() for x in str(p["glossary"]).split(",") if x.strip()]
    if gl:
        argv += ["--glossary-files", ",".join(dict.fromkeys(gl))]
    if p.get("pages"):
        argv += ["--pages", str(p["pages"])]
    return argv


def start_job(files: list[str], params: dict) -> Job:
    job = Job(files, params)
    with JOBS_LOCK:
        JOBS[job.id] = job
    threading.Thread(target=job.run, daemon=True).start()
    return job


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "PDFStudio/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence per-request noise
        pass

    # -- helpers ---------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # -- GET -------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = u.path
        qs = parse_qs(u.query)

        if path == "/api/stream":
            return self._sse(qs.get("job", [""])[0])

        if path == "/api/state":
            cfg = load_config()
            return self._json({
                "engine": BABELDOC,
                "engine_ok": bool(BABELDOC),
                "root": str(ROOT),
                "out_dir": str(OUT_DIR),
                "api_key_set": bool(cfg["api_key"]),
                "params": cfg["params"],
                "home": str(Path.home()),
                "downloads": str(Path.home() / "Downloads"),
            })

        if path == "/api/jobs":
            with JOBS_LOCK:
                items = [
                    {
                        "id": j.id, "status": j.status, "progress": round(j.progress, 1),
                        "stage": j.stage, "files": j.files, "outputs": j.outputs,
                        "compare_html": j.compare_html,
                        "rc": j.rc, "started": j.started, "finished": j.finished,
                        "errors": j.errors[:3],
                        "prompt_tokens": j.prompt_tokens,
                        "completion_tokens": j.completion_tokens,
                    }
                    for j in JOBS.values()
                ]
            items.sort(key=lambda x: x["started"], reverse=True)
            return self._json({"jobs": items})

        if path == "/api/browse":
            d = (qs.get("dir", [""])[0] or str(Path.home() / "Downloads"))
            return self._browse(d)

        if path == "/api/detect":
            src = qs.get("path", [""])[0]
            if not src or not os.path.isfile(src):
                return self._json({"error": "file not found"}, 404)
            code, stats = detect_language(src)
            return self._json({
                "path": src,
                "lang": code,
                "stats": stats,
                "available": bool(stats),
            })

        if path == "/api/download":
            return self._serve_file(qs.get("path", [""])[0], as_attachment=True)

        if path == "/api/preview":
            return self._serve_file(qs.get("path", [""])[0], as_attachment=False,
                                    inline=True)

        if path == "/api/terms":
            return self._terms_get(qs.get("name", [""])[0])

        if path.startswith("/api/") or path == "/favicon.ico":
            return self._json({"error": "not found"}, 404)

        return self._static(path)

    # -- POST ------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        qs = parse_qs(u.query)

        if u.path == "/api/upload":
            name = Path(qs.get("name", ["upload.pdf"])[0]).name
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if n <= 0:
                return self._json({"error": "empty body"}, 400)
            dest = INBOX / name
            remaining = n
            with open(dest, "wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            return self._json({"path": str(dest), "size": dest.stat().st_size})

        if u.path == "/api/config":
            body = self._read_json()
            cfg = load_config()
            if "api_key" in body:
                cfg["api_key"] = str(body.get("api_key") or "")
            p = body.get("params")
            if isinstance(p, dict):
                for k, v in p.items():
                    if k in DEFAULT_PARAMS:
                        cfg["params"][k] = v
            save_config(cfg)
            return self._json({"ok": True})

        if u.path == "/api/run":
            body = self._read_json()
            files = [str(f) for f in (body.get("files") or [])]
            files = [f for f in files if f and Path(f).exists()]
            if not files:
                return self._json({"error": "no existing input files"}, 400)
            cfg = load_config()
            params = dict(cfg["params"])
            params.update(body.get("params") or {})
            params["api_key"] = body.get("api_key") or cfg["api_key"]
            if not params["api_key"]:
                return self._json({"error": "DeepSeek API key is not set"}, 400)
            params["out_dir"] = str(OUT_DIR)
            job = start_job(files, params)
            return self._json({"job": job.id})

        if u.path == "/api/cancel":
            body = self._read_json()
            with JOBS_LOCK:
                j = JOBS.get(str(body.get("job") or ""))
            if j:
                j.cancel()
                return self._json({"ok": True})
            return self._json({"error": "unknown job"}, 404)

        if u.path == "/api/reveal":
            body = self._read_json()
            target = Path(str(body.get("path") or OUT_DIR))
            try:
                if target.is_file():
                    target = target.parent
                os.startfile(str(target))  # noqa: S606  (Windows shell open)
                return self._json({"ok": True})
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc)}, 500)

        if u.path == "/api/open":
            return self._open_external(str(self._read_json().get("path") or ""))

        if u.path == "/api/terms":
            return self._terms_save(self._read_json())

        return self._json({"error": "not found"}, 404)

    # -- pieces ----------------------------------------------------------
    def _open_external(self, raw: str) -> None:
        """Open a produced PDF in Edge, falling back to the shell handler."""
        p = self._safe(raw)
        if not p:
            return self._json({"error": "file not found"}, 404)
        exe = next((c for c in EDGE_CANDIDATES if c and os.path.isfile(c)), None)
        try:
            if exe:
                subprocess.Popen([exe, str(p)])  # noqa: S603
                return self._json({"ok": True, "via": "edge", "exe": exe})
            os.startfile(str(p))  # noqa: S606
            return self._json({"ok": True, "via": "shell"})
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)}, 500)

    def _terms_get(self, name: str) -> None:
        if not name:
            files = []
            for f in sorted(TERMS_DIR.glob("*.csv")):
                try:
                    body = f.read_text(encoding="utf-8-sig").splitlines()
                    n = len([x for x in body[1:] if x.strip()])
                except Exception:
                    n = 0
                files.append({"name": f.name, "entries": n})
            return self._json({"files": files, "dir": str(TERMS_DIR)})
        f = TERMS_DIR / Path(name).name
        if not f.is_file():
            return self._json({"error": "not found"}, 404)
        rows = []
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8-sig").splitlines()):
                if i == 0 or not line.strip():
                    continue
                parts = line.split(",")
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    rows.append({
                        "source": parts[0].strip(),
                        "target": parts[1].strip(),
                        "tgt_lng": parts[2].strip() if len(parts) > 2 else "zh",
                    })
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)}, 500)
        return self._json({"name": f.name, "entries": rows})

    def _terms_save(self, body: dict) -> None:
        name = Path(str(body.get("name") or "custom-zh.csv")).name
        if not name.lower().endswith(".csv"):
            name += ".csv"
        lines = ["source,target,tgt_lng"]
        for r in body.get("entries") or []:
            s = str(r.get("source") or "").strip().replace(",", " ").replace("\n", " ")
            t = str(r.get("target") or "").strip().replace(",", " ").replace("\n", " ")
            if s and t:
                lines.append(f"{s},{t},zh")
        try:
            (TERMS_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)}, 500)
        return self._json({"ok": True, "name": name, "entries": len(lines) - 1})

    def _sse(self, job_id: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            return self._json({"error": "unknown job"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = job.subscribe()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                payload = json.dumps(ev, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                if ev.get("type") == "end":
                    break
        except Exception:
            pass
        finally:
            job.unsubscribe(q)

    def _browse(self, d: str) -> None:
        try:
            p = Path(d).expanduser()
            if not p.is_dir():
                p = Path.home() / "Downloads"
            if not p.is_dir():
                p = Path.home()
        except Exception:
            p = Path.home()
        dirs, pdfs = [], []
        try:
            for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                if child.name.startswith(".") or child.name.startswith("$"):
                    continue
                try:
                    if child.is_dir():
                        dirs.append({"name": child.name, "path": str(child)})
                    elif child.suffix.lower() == ".pdf":
                        pdfs.append({
                            "name": child.name,
                            "path": str(child),
                            "size": child.stat().st_size,
                            "mtime": child.stat().st_mtime,
                        })
                except (OSError, PermissionError):
                    continue
        except (OSError, PermissionError) as exc:
            return self._json({"error": str(exc)}, 403)
        parent = str(p.parent) if p.parent != p else ""
        return self._json({
            "dir": str(p), "parent": parent,
            "dirs": dirs[:400], "pdfs": pdfs[:400],
        })

    def _safe(self, raw: str) -> Path | None:
        if not raw:
            return None
        try:
            p = Path(raw)
            if p.exists() and p.is_file():
                return p
        except Exception:
            return None
        return None

    def _serve_file(self, raw: str, as_attachment: bool, inline: bool = False) -> None:
        p = self._safe(raw)
        if not p:
            return self._json({"error": "file not found"}, 404)
        if p.suffix.lower() != ".pdf" and p.suffix.lower() not in (".csv", ".log", ".txt", ".html"):
            return self._json({"error": "unsupported file type"}, 403)
        try:
            data = p.read_bytes()
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)}, 500)
        disp = "attachment" if as_attachment else ("inline" if inline else "attachment")
        ctype = {
            ".pdf": "application/pdf",
            ".html": "text/html; charset=utf-8",
            ".csv": "text/csv; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
            ".log": "text/plain; charset=utf-8",
        }.get(p.suffix.lower(), "application/octet-stream")
        # HTTP headers are latin-1, so a non-ASCII name like the Chinese one
        # these outputs carry would raise mid-response. Send an ASCII fallback
        # plus the RFC 5987 encoded form the browser actually prefers.
        ascii_name = p.name.encode("ascii", "replace").decode("ascii").replace('"', "'")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header(
            "Content-Disposition",
            f"{disp}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(p.name)}",
        )
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        if rel.startswith("ui/"):
            rel = rel[3:]
        target = (UI_DIR / rel).resolve()
        try:
            target.relative_to(UI_DIR.resolve())
        except ValueError:
            return self._json({"error": "forbidden"}, 403)
        if not target.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".ico": "image/x-icon",
            ".json": "application/json; charset=utf-8",
        }.get(target.suffix.lower(), "application/octet-stream")
        try:
            if not target.resolve().is_relative_to(UI_DIR.resolve()):
                return self._json({"error": "forbidden"}, 403)
        except Exception:
            pass
        try:
            self._send(200, target.read_bytes(), ctype)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc)}, 500)


def main() -> int:
    open_browser = "--no-browser" not in sys.argv
    if not BABELDOC:
        print("[!] babeldoc not found on PATH.", file=sys.stderr)
        print("    install:  uv tool install --python 3.12 BabelDOC", file=sys.stderr)
    url = f"http://{HOST}:{PORT}/"
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    print(f"engine : {BABELDOC or 'NOT FOUND'}")
    print(f"root   : {ROOT}")
    print(f"ui     : {url}")
    print("press Ctrl+C to stop")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
