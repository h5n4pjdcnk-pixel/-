# 【模块 S-4】独立 OCR 工具 · 核心：引擎抽象 + 逐页兜底
"""PDF 扫描件 OCR 模块（独立版）

把没有文字层的 PDF（扫描件、整页图片）识别成文字。可作库调用，也可用命令行。

核心思路：文字层够用就直接用，不够才渲染成图片走 OCR —— 电子版 PDF 零额外开销。

三种可插拔引擎：
  rapidocr   本地离线，pip install rapidocr-onnxruntime。免费、快、中文准。
  tesseract  本地离线，需本机装 Tesseract 可执行文件（含 chi_sim 语言包）。
  vlm        云端视觉大模型，需设置环境变量 OCR_VLM_API_KEY。
auto 模式按 rapidocr -> tesseract -> vlm 挑第一个可用的。

可调参数都在 ocr_settings.py 里，也可用环境变量覆盖。
"""

import io
import re
import shutil

try:  # 新版 PyMuPDF 用 pymupdf 这个包名，import fitz 会打弃用警告
    import pymupdf as fitz
except ImportError:  # 兼容旧版本
    import fitz

from ocr_settings import CLOUD_VLM, OCR_CONFIG, cloud_configured

# ==========================================================
#  一、文字层是否够用
# ==========================================================

_MEANINGFUL = re.compile(r'[A-Za-z0-9\u4e00-\u9fa5]')


def meaningful_char_count(text):
    """有效字符数（中英文数字），用来判断一页到底有没有内容。"""
    return len(_MEANINGFUL.findall(text or ""))


def needs_ocr(text, min_chars=None):
    """文字层有效字符少于阈值，就认为这页是扫描件、需要 OCR。"""
    if min_chars is None:
        min_chars = int(OCR_CONFIG.get("min_text_chars", 80))
    return meaningful_char_count(text) < min_chars


# ==========================================================
#  二、引擎探测与选择
# ==========================================================

_rapidocr_module = None
_rapidocr_engine = None
_vlm_llm = None
_vlm_calls = 0
_vlm_notice_shown = False

AUTO_ORDER = ("rapidocr", "tesseract", "vlm")

TIPS = {
    "rapidocr": "pip install rapidocr-onnxruntime",
    "tesseract": "安装 Tesseract-OCR 并把安装目录加进 PATH，语言包需含 chi_sim",
    "vlm": "设置环境变量 OCR_VLM_API_KEY（并把 OCR_VLM_API_URL 指向你的服务）",
}


def _load_rapidocr():
    """兼容新旧两种包名：rapidocr_onnxruntime(1.x) / rapidocr(2.x)。"""
    global _rapidocr_module
    if _rapidocr_module is not None:
        return _rapidocr_module
    try:
        import rapidocr_onnxruntime as mod
    except Exception:
        try:
            import rapidocr as mod
        except Exception:
            mod = False
    _rapidocr_module = mod
    return mod


def has_rapidocr():
    return bool(_load_rapidocr())


def has_tesseract():
    return shutil.which("tesseract") is not None


def has_vlm():
    """云端引擎可用 = 已经把密钥通过环境变量提供进来。"""
    return cloud_configured()


def available_engines():
    return {
        "rapidocr": has_rapidocr(),
        "tesseract": has_tesseract(),
        "vlm": has_vlm(),
    }


def format_engines(avail=None):
    avail = avail or available_engines()
    return "，".join("%s=%s" % (k, "可用" if v else "不可用") for k, v in avail.items())


def resolve_engine(settings=None):
    """按 OCR_CONFIG['engine'] 决定实际用哪个引擎；返回 None 表示不做 OCR。"""
    settings = settings or OCR_CONFIG
    wanted = str(settings.get("engine", "auto")).strip().lower()

    if wanted in ("off", "none", "disable", "disabled"):
        return None

    avail = available_engines()

    if wanted != "auto":
        if avail.get(wanted):
            return wanted
        print("⚠️ OCR 引擎 '%s' 当前不可用，回退到自动模式。启用方法：%s"
              % (wanted, TIPS.get(wanted, "")))
        print("   当前引擎状态：%s" % format_engines(avail))

    for name in AUTO_ORDER:
        if avail.get(name):
            return name
    return None


# ==========================================================
#  三、各引擎实现
# ==========================================================

def render_page_image(page, dpi=None):
    """把一页渲染成 PIL 图片。dpi 越高越清晰，也越慢越吃内存。"""
    if dpi is None:
        dpi = int(OCR_CONFIG.get("dpi", 200))
    pix = page.get_pixmap(dpi=int(dpi))
    try:
        return pix.pil_image()
    except Exception:
        from PIL import Image
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _rapidocr_lines(result, y_tolerance=12):
    """把 rapidocr 的框结果按「先上下、后左右」排成阅读顺序。"""
    items = []
    for entry in result or []:
        try:
            box, text = entry[0], entry[1]
        except Exception:
            continue
        if not text:
            continue
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        items.append((min(ys), min(xs), str(text)))
    items.sort(key=lambda it: (round(it[0] / y_tolerance), it[1]))
    return "\n".join(it[2] for it in items)


def _ocr_rapidocr(image):
    global _rapidocr_engine
    mod = _load_rapidocr()
    if not mod:
        return ""

    if _rapidocr_engine is None:
        print("   ⏳ 首次加载 RapidOCR 模型（只需一次）...")
        _rapidocr_engine = mod.RapidOCR()

    import numpy as np
    out = _rapidocr_engine(np.array(image))

    # 2.x 返回对象，含 .txts / .boxes；1.x 返回 (result, elapse)
    if hasattr(out, "txts"):
        txts = list(out.txts or [])
        boxes = getattr(out, "boxes", None)
        if boxes is None:
            return "\n".join(str(t) for t in txts if t)
        return _rapidocr_lines([[b, t] for b, t in zip(boxes, txts)])

    if isinstance(out, tuple):
        out = out[0]
    return _rapidocr_lines(out)


def _ocr_tesseract(page):
    """走 PyMuPDF 内置的 Tesseract 接口（要用本机已装好的 tesseract）。"""
    lang = OCR_CONFIG.get("tesseract_lang", "chi_sim+eng")
    dpi = int(OCR_CONFIG.get("dpi", 200))
    tessdata = OCR_CONFIG.get("tessdata_dir") or None
    try:
        textpage = page.get_textpage_ocr(language=lang, dpi=dpi, full=True, tessdata=tessdata)
        return page.get_text(textpage=textpage) or ""
    except Exception as e:
        print("   ❌ Tesseract OCR 失败：%s" % e)
        print("      检查是否装了 Tesseract、是否在 PATH 里、语言包是否含 chi_sim。")
        return ""


_VLM_PROMPT = """你是 OCR 引擎。请把这张图片里所有可见文字原样转写出来。

【严格指令】
1. 逐字转写，不要翻译、不要总结、不要改写、不要补充原文没有的内容。
2. 按人类阅读顺序输出（从上到下、从左到右）；表格保留结构，用「列1 | 列2」这种行式表达。
3. 数字、单位、化学式、型号、符号要准确；实在辨认不清的字符用 ▢ 占位，不要靠猜。
4. 图注、页眉页脚、页码若可见也一并转写；纯装饰性图形无需描述。
5. 直接输出转写文本，不要任何前言后记，不要 ``` 代码块。
"""


def _get_vlm():
    global _vlm_llm
    if _vlm_llm is None:
        from langchain_openai import ChatOpenAI
        cfg = CLOUD_VLM
        _vlm_llm = ChatOpenAI(
            model=cfg["model_name"],
            openai_api_key=cfg["api_key"],
            openai_api_base=cfg["api_url"],
            temperature=0.0,
            max_retries=2,
        )
    return _vlm_llm


def _ocr_vlm(image):
    import base64

    global _vlm_notice_shown
    if not _vlm_notice_shown:
        _vlm_notice_shown = True
        print("   ⚠️ 没检测到本地 OCR 引擎，改用云端 VLM：每页一次 API 调用（会消耗额度）。")
        print("      想免费离线跑，先装本地引擎：pip install rapidocr-onnxruntime")

    max_side = int(OCR_CONFIG.get("vlm_max_side", 1600))
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content="你是一个只做逐字转写的 OCR 引擎。"),
        HumanMessage(content=[
            {"type": "text", "text": _VLM_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,%s" % b64}},
        ]),
    ]
    try:
        text = _get_vlm().invoke(messages).content
    except Exception as e:
        print("   ❌ 云端 VLM OCR 调用失败：%s" % e)
        return ""
    if isinstance(text, list):  # 少数模型返回分块内容
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    return (text or "").replace("```", "").strip()


# ==========================================================
#  三点五、OCR 结果后处理
#
#  多模态模型转写时有两个典型副作用，会直接影响下游：
#    1. 把整串英文缩写拆成单字母：「BMS」-> 「B M S」、「E10」-> 「E 1 0」。
#       而 document_loader.py 正是按 "BMS"/"PCS"/"SEI" 这类关键词过滤文本块的，
#       拆开之后关键词匹配不上，整页会被当成无关内容丢掉。
#    2. 数字用全角：「2．5」-> 需要还原成「2.5」，否则抽出来的阈值是脏数据。
# ==========================================================

_FULLWIDTH_ALNUM = {}
for _c in list(range(0xFF10, 0xFF1A)) + list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)):
    _FULLWIDTH_ALNUM[chr(_c)] = chr(_c - 0xFEE0)
_FULLWIDTH_ALNUM["．"] = "."   # 全角句点当小数点用
_FULLWIDTH_ALNUM["　"] = " "   # 全角空格
del _c

# 连续两个及以上的「单字符英文/数字」且只用空白分隔 -> 合成一个词
_SINGLE_ALNUM_RUN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z0-9](?:[ \t]+[A-Za-z0-9]){1,})(?![A-Za-z0-9])"
)


def normalize_ocr_text(text):
    """消除 OCR 常见副作用，让结果更接近原文。只用于 OCR 输出，不动原生文字层。"""
    if not text:
        return ""
    text = "".join(_FULLWIDTH_ALNUM.get(ch, ch) for ch in text)
    text = _SINGLE_ALNUM_RUN.sub(lambda m: re.sub(r"[ \t]+", "", m.group(1)), text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ==========================================================
#  四、对外主接口
# ==========================================================

def ocr_page(page, engine=None, dpi=None):
    """对一页做 OCR 并返回文字；engine 为 None 时按配置自动选。"""
    global _vlm_calls
    if engine is None:
        engine = resolve_engine()
    if not engine:
        return ""

    if engine == "tesseract":
        return normalize_ocr_text(_ocr_tesseract(page))

    image = render_page_image(page, dpi)
    if engine == "rapidocr":
        return normalize_ocr_text(_ocr_rapidocr(image))
    if engine == "vlm":
        limit = int(OCR_CONFIG.get("vlm_max_pages", 0) or 0)
        if limit and _vlm_calls >= limit:
            print("   ⚠️ 云端 VLM OCR 已达上限 %d 页，本页跳过（改 OCR_VLM_MAX_PAGES 可调）" % limit)
            return ""
        _vlm_calls += 1
        return normalize_ocr_text(_ocr_vlm(image))
    return ""


def extract_pages_text(pdf_path, engine=None):
    """逐页取文本，文字层不足的页自动 OCR。

    返回 (每页文本列表, 统计信息 dict)。engine 传 None 表示按 config 自动选择。
    """
    chosen = engine if engine is not None else resolve_engine()
    stats = {
        "pages": 0,
        "need_ocr_pages": 0,
        "ocr_pages": 0,
        "engine": chosen or "",
        "mode": str(OCR_CONFIG.get("engine", "auto")).strip().lower(),
        "empty_pages": 0,
        "ocr_chars": 0,
    }
    pages_text = []

    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            stats["pages"] += 1
            native = page.get_text() or ""

            if not needs_ocr(native):
                pages_text.append(native)
                continue

            stats["need_ocr_pages"] += 1
            if chosen is None:
                pages_text.append(native)
                continue

            ocr_text = ocr_page(page, engine=chosen) or ""
            stats["ocr_pages"] += 1
            stats["ocr_chars"] += meaningful_char_count(ocr_text)

            if meaningful_char_count(ocr_text) == 0:
                stats["empty_pages"] += 1
                print("   ⚠️ 第 %d 页 OCR 后仍无有效文字。" % stats["pages"])

            # OCR 结果反而更短，说明原文字层更靠谱，保留原文
            if meaningful_char_count(ocr_text) >= meaningful_char_count(native):
                pages_text.append(ocr_text)
            else:
                pages_text.append(native)
    finally:
        doc.close()

    return pages_text, stats


def format_stats(stats, label=""):
    """把统计信息拼成一行给日志用。"""
    if not stats:
        return ""
    prefix = ("%s " % label) if label else ""
    pages = stats.get("pages", 0)
    need = stats.get("need_ocr_pages", 0)
    done = stats.get("ocr_pages", 0)

    if not need:
        return "%s共 %d 页，文字层完好，未触发 OCR" % (prefix, pages)
    if not done:
        if stats.get("mode") in ("off", "none", "disable", "disabled"):
            return "%s共 %d 页，其中 %d 页没有文字层；OCR 已在 ocr_settings.py 里关闭，已原样跳过" % (
                prefix, pages, need)
        return "%s共 %d 页，其中 %d 页没有文字层，但当前没有可用的 OCR 引擎，已原样跳过" % (
            prefix, pages, need)
    return "%s共 %d 页，其中 %d 页走 OCR（引擎 %s，取回 %d 字）" % (
        prefix, pages, done, stats.get("engine") or "无", stats.get("ocr_chars", 0))
