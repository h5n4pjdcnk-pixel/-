# 【模块 S-4】独立 OCR 工具 · 批量转换
# -*- coding: utf-8 -*-
"""批量 OCR：把项目 data 目录下所有 PDF 转成 txt，输出到项目 OCR结果 目录。

文字层完好的页直接取文字，只有没有文字层的页才走 OCR。
输出目录镜像原来的分类结构，并写一份汇总报告。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from pdf_ocr import extract_pages_text, format_stats, resolve_engine, meaningful_char_count

ROOT = os.getenv("OCR_INPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.getenv("OCR_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OCR结果"))
WRAPPER = "基础资料 等5项"          # zip 解压出来的最外层目录，输出时去掉
PAGE_MARK = "===== 第 %d 页 ====="


def out_path_for(pdf_path):
    """把源 PDF 路径映射成输出 txt 路径：去掉解压包裹层、折叠重复的目录名。"""
    rel = os.path.relpath(pdf_path, ROOT)
    parts = rel.split(os.sep)
    if parts and parts[0] == WRAPPER:
        parts = parts[1:]

    dirs = []
    for d in parts[:-1]:
        if dirs and dirs[-1] == d:      # 基础资料/基础资料 -> 基础资料
            continue
        dirs.append(d)

    stem = os.path.splitext(parts[-1])[0]
    return os.path.join(OUT, *dirs, stem + ".txt")


def write_txt(pdf_path, pages, stats):
    dst = out_path_for(pdf_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rel = os.path.relpath(pdf_path, ROOT)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("# 源文件：%s\n" % rel)
        fh.write("# %s\n" % format_stats(stats))
        fh.write("# 说明：文字层完好的页直接取原文，没有文字层的页由 %s 引擎 OCR 得到。\n"
                 % (stats.get("engine") or "无"))
        fh.write("#" + "-" * 68 + "\n")
        for i, text in enumerate(pages, 1):
            fh.write("\n" + PAGE_MARK % i + "\n")
            fh.write((text or "").strip() + "\n")
    return dst


def main():
    if not os.path.isdir(ROOT):
        print("找不到目录：" + ROOT)
        return 1

    pdfs = []
    for dp, dn, fn in os.walk(ROOT):
        for f in fn:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dp, f))
    pdfs.sort()

    engine = resolve_engine()
    print("=" * 70, flush=True)
    print("批量 OCR：共 %d 个 PDF" % len(pdfs), flush=True)
    print("使用引擎：" + (engine or "无（不会做 OCR，只取文字层）"), flush=True)
    print("输出目录：" + OUT, flush=True)
    print("=" * 70, flush=True)

    os.makedirs(OUT, exist_ok=True)
    report = []
    t_all = time.time()
    total_pages = total_ocr = total_chars = 0

    for i, pdf in enumerate(pdfs, 1):
        name = os.path.relpath(pdf, ROOT)
        dst = out_path_for(pdf)
        t0 = time.time()

        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print("[%2d/%d] 跳过（已存在） %s" % (i, len(pdfs), name), flush=True)
            continue

        try:
            pages, stats = extract_pages_text(pdf)
            dst = write_txt(pdf, pages, stats)
            chars = sum(meaningful_char_count(t) for t in pages)
            total_pages += stats.get("pages", 0)
            total_ocr += stats.get("ocr_pages", 0)
            total_chars += chars
            cost = time.time() - t0
            report.append((name, stats.get("pages", 0), stats.get("ocr_pages", 0),
                           chars, cost, ""))
            print("[%2d/%d] %s" % (i, len(pdfs), name), flush=True)
            print("        %s   用时 %.1f 秒" % (format_stats(stats), cost), flush=True)
        except Exception as e:
            report.append((name, 0, 0, 0, time.time() - t0, str(e)))
            print("[%2d/%d] ❌ 失败 %s -> %s" % (i, len(pdfs), name, e), flush=True)

    # ---------------- 汇总报告 ----------------
    lines = []
    lines.append("原始资料 批量 OCR 结果汇总")
    lines.append("=" * 78)
    lines.append("生成时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("源目录  ：%s" % ROOT)
    lines.append("输出目录：%s" % OUT)
    lines.append("使用引擎：%s" % (engine or "无"))
    lines.append("文件总数：%d" % len(report))
    lines.append("总页数  ：%d" % total_pages)
    lines.append("其中 OCR：%d 页" % total_ocr)
    lines.append("总字符数：%d" % total_chars)
    lines.append("总耗时  ：%.1f 分钟" % ((time.time() - t_all) / 60.0))
    lines.append("=" * 78)
    lines.append("")
    lines.append("%-56s %6s %6s %9s %8s" % ("文件", "页数", "OCR页", "字符数", "秒"))
    lines.append("-" * 92)
    for name, pages, ocr, chars, cost, err in report:
        short = name if len(name) <= 54 else "..." + name[-51:]
        if err:
            lines.append("%-56s %6s %6s %9s %8.1f  失败: %s" % (short, "-", "-", "-", cost, err[:40]))
        else:
            lines.append("%-56s %6d %6d %9d %8.1f" % (short, pages, ocr, chars, cost))
    report_path = os.path.join(OUT, "_OCR汇总报告.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines[:11]), flush=True)
    print("\n汇总报告：" + report_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())