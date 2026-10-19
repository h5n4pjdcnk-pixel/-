# 【模块 A-1】切分规则：读文档 + 领域关键词过滤 + 定长切块
# --- 路径兜底：无论本文件放在哪个子文件夹，都能 import 到项目根与各子模块 ---
import os as _os, sys as _sys
_PROJ = _os.path.dirname(_os.path.abspath(__file__))
while _PROJ != _os.path.dirname(_PROJ) and not _os.path.exists(_os.path.join(_PROJ, "config.py")):
    _PROJ = _os.path.dirname(_PROJ)
for _n in [""] + sorted(_d for _d in _os.listdir(_PROJ) if _os.path.isdir(_os.path.join(_PROJ, _d))):
    _p = _os.path.join(_PROJ, _n)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# -------------------------------------------------------------------------
import fitz  # noqa: F401  (保留：说明本模块的 PDF 解析底座是 PyMuPDF)
from config import CHUNK_SIZE, CHUNK_OVERLAP
from ocr_utils import extract_pages_text, format_stats


def load_pdf_and_chunk(pdf_path):
    # 逐页取文本，文字层不足的页（扫描件）自动走 OCR 兜底
    pages, ocr_stats = extract_pages_text(pdf_path)
    text = "".join(pages)

    #领域关键词过滤
    keywords = [
        # 运维与故障类
        "电池", "BMS", "电压", "故障", "报警", "PCS", "维护",
        # 电化学机理类
        "电化学", "SEI", "内阻", "热失控", "锂离子", "析锂", "电解液",
        # 规范、安全与操作规程类
        "规范", "标准", "安全", "消防", "应急", "规程", "操作",
        "白皮书", "双链", "预警", "防护", "隔离", "绝缘", "接地"
    ]
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i:i + CHUNK_SIZE]
        if any(k in chunk for k in keywords):
            chunks.append(chunk)

    if ocr_stats.get("need_ocr_pages"):
        print("   🔎 OCR 兜底：" + format_stats(ocr_stats))
        if ocr_stats.get("empty_pages"):
            print("      ⚠️ 有 %d 页 OCR 后仍无有效文字，可能是纯图片页或扫描质量太差。"
                  % ocr_stats["empty_pages"])

    return chunks
