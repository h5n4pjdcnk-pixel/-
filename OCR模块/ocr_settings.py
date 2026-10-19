# 【模块 S-4】独立 OCR 工具 · 参数设置
# -*- coding: utf-8 -*-
"""OCR 模块的全部可调参数。

设计原则：本地引擎零配置即可运行；只有云端引擎才需要密钥。
为便于分享/提交，代码里不写死任何密钥，云端配置一律通过环境变量提供。
"""

import os

# ==========================================================
#  引擎选择
#    rapidocr   本地离线（推荐）：pip install rapidocr-onnxruntime
#    tesseract  本地离线：需本机安装 Tesseract 可执行文件（含 chi_sim 语言包）
#    vlm        云端视觉大模型：需要 OCR_VLM_API_KEY
#    auto       按 rapidocr -> tesseract -> vlm 挑第一个可用的
#    off        关闭 OCR
# ==========================================================
ENGINE = os.getenv("OCR_ENGINE", "auto")

# 一页的有效字符数（中英文数字）少于这个值，就认为该页没有文字层、需要 OCR
MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "80"))

# 渲染分辨率：越高越清晰，也越慢、越占内存（200~300 一般够用）
DPI = int(os.getenv("OCR_DPI", "200"))

# ---- Tesseract（仅 tesseract 引擎需要）----
TESSERACT_LANG = os.getenv("OCR_TESSERACT_LANG", "chi_sim+eng")
TESSDATA_DIR = os.getenv("OCR_TESSDATA_DIR") or None

# ---- 云端 VLM（仅 vlm 引擎需要）----
VLM_MAX_SIDE = int(os.getenv("OCR_VLM_MAX_SIDE", "1600"))   # 图片长边上限，控制 token 消耗
VLM_MAX_PAGES = int(os.getenv("OCR_VLM_MAX_PAGES", "50"))   # 单次运行最多 OCR 多少页，防误烧额度

CLOUD_VLM = {
    "model_name": os.getenv("OCR_VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct"),
    "api_key": os.getenv("OCR_VLM_API_KEY", ""),
    "api_url": os.getenv("OCR_VLM_API_URL", "https://api.siliconflow.cn/v1"),
}

# 供 pdf_ocr.py 读取的集中配置
OCR_CONFIG = {
    "engine": ENGINE,
    "min_text_chars": MIN_TEXT_CHARS,
    "dpi": DPI,
    "tesseract_lang": TESSERACT_LANG,
    "tessdata_dir": TESSDATA_DIR,
    "vlm_max_side": VLM_MAX_SIDE,
    "vlm_max_pages": VLM_MAX_PAGES,
}


def cloud_configured():
    """云端 VLM 是否已配置（密钥够长才算填过）。"""
    return len(CLOUD_VLM.get("api_key", "").strip()) >= 20
