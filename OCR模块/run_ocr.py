# 【模块 S-4】独立 OCR 工具 · 命令行入口
# -*- coding: utf-8 -*-
"""命令行入口：把 PDF（尤其扫描件）转成纯文本。

用法（在项目目录下执行）：
    python run_ocr.py                      # 处理 ./data 下所有 PDF -> ./ocr_output/*.ocr.txt
    python ocr_pdf.py --input a.pdf        # 指定文件（也可传目录）
    python ocr_pdf.py --engine vlm         # 强制引擎：rapidocr | tesseract | vlm
    python ocr_pdf.py --force              # 每页都 OCR（默认只 OCR 文字层不足的页）
    python ocr_pdf.py --dpi 300            # 提高渲染分辨率，扫描件更准
    python ocr_pdf.py --out ./ocr_out      # 自定义输出目录
"""

import argparse
import os
import sys
import time

try:  # 新版 PyMuPDF 用 pymupdf 这个包名，import fitz 会打弃用警告
    import pymupdf as fitz
except ImportError:  # 兼容旧版本
    import fitz

from ocr_settings import OCR_CONFIG
from pdf_ocr import (
    available_engines,
    extract_pages_text,
    format_engines,
    format_stats,
    meaningful_char_count,
    ocr_page,
    resolve_engine,
)

DEFAULT_OUT = "./ocr_output"


def collect_pdfs(target):
    if os.path.isfile(target):
        return [target]
    if not os.path.isdir(target):
        print("❌ 找不到路径：%s" % target)
        return []
    return [
        os.path.join(target, f)
        for f in sorted(os.listdir(target))
        if f.lower().endswith(".pdf")
    ]


def ocr_one(path, out_dir, engine, force, dpi):
    started = time.time()

    if force:
        doc = fitz.open(path)
        try:
            texts = [ocr_page(page, engine=engine, dpi=dpi) or "" for page in doc]
        finally:
            doc.close()
        stats = {
            "pages": len(texts),
            "need_ocr_pages": len(texts),
            "ocr_pages": len(texts),
            "engine": engine,
            "mode": "force",
            "empty_pages": sum(1 for t in texts if not meaningful_char_count(t)),
            "ocr_chars": sum(meaningful_char_count(t) for t in texts),
        }
    else:
        texts, stats = extract_pages_text(path, engine=engine)

    out_path = os.path.join(out_dir, "%s.ocr.txt" % os.path.splitext(os.path.basename(path))[0])
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# OCR 结果：%s\n" % os.path.basename(path))
        fh.write("# %s\n" % format_stats(stats))
        for i, text in enumerate(texts, 1):
            fh.write("\n===== 第 %d 页 =====\n" % i)
            fh.write((text or "").strip() + "\n")

    print("   ✅ 已写出 %s（耗时 %.1f 秒）" % (out_path, time.time() - started))
    print("      " + format_stats(stats))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="扫描件 OCR 工具：结果先导出成 txt，确认效果后再入库"
    )
    parser.add_argument("--input", default="./data", help="PDF 文件或目录，默认 ./data")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出目录，默认 ./ocr_output")
    parser.add_argument(
        "--engine",
        default=None,
        help="强制指定引擎：rapidocr | tesseract | vlm（默认按 ocr_settings.py）",
    )
    parser.add_argument("--force", action="store_true", help="每页都 OCR，而不是只 OCR 文字层不足的页")
    parser.add_argument("--dpi", type=int, default=None, help="渲染分辨率，默认取 OCR_CONFIG['dpi']")
    args = parser.parse_args()

    print("=" * 60)
    print(" OCR 引擎状态：" + format_engines(available_engines()))
    engine = args.engine or resolve_engine()
    if not engine:
        print(" ❌ 没有任何可用引擎，无法 OCR。")
        print("    最省事的办法：pip install rapidocr-onnxruntime")
        print("    或者或者设置 OCR_VLM_API_KEY 后把 OCR_ENGINE 设成 \"vlm\"（走云端，要花额度）。")
        return 1
    if args.engine:
        print(" 手动指定引擎：%s" % args.engine)
    print("=" * 60)

    if args.dpi:
        OCR_CONFIG["dpi"] = int(args.dpi)

    pdfs = collect_pdfs(args.input)
    if not pdfs:
        print("⚠️ %s 下没有找到 PDF。" % args.input)
        return 1

    os.makedirs(args.out, exist_ok=True)
    for i, path in enumerate(pdfs, 1):
        print("\n[%d/%d] %s" % (i, len(pdfs), os.path.basename(path)))
        ocr_one(path, args.out, engine, args.force, args.dpi)

    print("\n✨ 全部完成，结果在 %s" % os.path.abspath(args.out))
    print("   打开 txt 确认文字无误后，再拿它去做检索 / 抽取 / 入库。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
