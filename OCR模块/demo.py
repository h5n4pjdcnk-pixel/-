# 【模块 S-4】独立 OCR 工具 · 演示
# -*- coding: utf-8 -*-
"""一键演示：不需要自备素材，脚本自己造一份「扫描件」再 OCR 出来。

流程：
  1) 用 PyMuPDF 生成一份带中文/英文缩写/数字的「电子版」PDF（有文字层）
  2) 把整页栅格化成图片，再包成新 PDF —— 这就是「扫描件」（没有文字层）
  3) 对扫描件跑 OCR，并和电子版原文对比相似度

运行：
    python demo.py
"""

import difflib
import os
import re
import sys

try:  # 新版 PyMuPDF 用 pymupdf 这个包名，import fitz 会打弃用警告
    import pymupdf as fitz
except ImportError:  # 兼容旧版本
    import fitz

from pdf_ocr import (
    available_engines,
    extract_pages_text,
    format_engines,
    format_stats,
    resolve_engine,
)

OUT_DIR = "demo_output"

LINES = [
    ("储能电站运维手册（节选）", 16),
    ("本手册适用于电化学储能电站的日常运维、故障处置与安全合规管理。", 11.5),
    ("1. 电池管理系统 BMS 报警处置", 11.5),
    ("当电池单体电压低于 2.5V 时，BMS 触发深度放电报警并上报 E10 故障，", 11.5),
    ("该报警严重程度为致命，处理方式为立即停机并隔离故障模组。", 11.5),
    ("当电池单体电压高于 3.65V 时，BMS 触发过充报警，严重程度为极高。", 11.5),
    ("电池温度超过 75 摄氏度时，BMS 必须触发热失控预警，并联动消防系统。", 11.5),
    ("PCS 过温告警通常由防尘网堵塞引起，需按维护规程清理风道。", 11.5),
]


def make_digital_pdf(path):
    """生成一份有文字层的电子版 PDF。"""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 80
    for i, (line, size) in enumerate(LINES):
        page.insert_text((50, y), line, fontsize=size, fontname="china-s")
        y += 34 if i == 0 else 30
    doc.save(path)
    doc.close()


def make_scanned_pdf(src_path, dst_path, dpi=150, jpg_quality=80):
    """把电子版整页渲染成一张图片，再包成新 PDF —— 模拟扫描件。

    用 JPEG 而不是 PNG：同样的页面，PNG 会让这个演示 PDF 涨到 11MB，
    JPEG 只要几百 KB，而 OCR 效果几乎没差别。
    """
    src = fitz.open(src_path)
    img = src[0].get_pixmap(dpi=dpi).tobytes("jpg", jpg_quality=jpg_quality)
    src.close()

    out = fitz.open()
    page = out.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(0, 0, 595, 842), stream=img)
    out.save(dst_path)
    out.close()


def normalize(text):
    return re.sub(r"\s+", "", text or "")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    digital = os.path.join(OUT_DIR, "demo_电子版.pdf")
    scanned = os.path.join(OUT_DIR, "demo_扫描件.pdf")

    engine = resolve_engine()
    print("=" * 64)
    print(" PDF 扫描件 OCR 模块 —— 演示")
    print(" 引擎状态：" + format_engines(available_engines()))
    print(" 本次使用：" + (engine or "无（OCR 已关闭或没有可用引擎）"))
    print("=" * 64)

    make_digital_pdf(digital)
    make_scanned_pdf(digital, scanned)
    print("\n[1] 已生成两个 PDF：")
    print("    电子版（有文字层）  : " + digital)
    print("    扫描件（整页图片）  : " + scanned)

    doc = fitz.open(digital)
    original = doc[0].get_text()
    doc.close()
    doc = fitz.open(scanned)
    raw_text = doc[0].get_text()
    doc.close()

    print("\n[2] 直接用 get_text() 读文字层：")
    print("    电子版读到 %d 个有效字符" % len(normalize(original)))
    print("    扫描件读到 %d 个有效字符   <-- 「扫描件什么都抽不出来」的根因" % len(normalize(raw_text)))

    print("\n[3] 用本模块的 extract_pages_text() 处理扫描件：")
    pages, stats = extract_pages_text(scanned)
    ocr_text = pages[0] if pages else ""
    print("    " + format_stats(stats))

    if not ocr_text.strip():
        print("\n⚠️ 没有拿到 OCR 文字，请先装一个本地引擎：")
        print("   pip install rapidocr-onnxruntime")
        return 1

    out_txt = os.path.join(OUT_DIR, "demo_OCR结果.txt")
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(ocr_text)

    ratio = difflib.SequenceMatcher(None, normalize(original), normalize(ocr_text)).ratio() * 100
    print("\n[4] OCR 结果与原文对比：")
    print("    字符级相似度：%.1f%%" % ratio)
    for kw in ("BMS", "PCS", "E10", "2.5V", "热失控"):
        print("    含 %-7s : %s" % (kw, "是" if kw in ocr_text else "否"))

    print("\n--- 原文（前 120 字）---")
    print(original.strip()[:120].replace("\n", " / "))
    print("--- OCR（前 120 字）---")
    print(ocr_text.strip()[:120].replace("\n", " / "))
    print("\n完整 OCR 结果已保存：" + out_txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
