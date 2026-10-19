# 【模块 D-3】文本规范化：OCR 文本清洗与统一
from __future__ import annotations

import argparse
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


META_RE = re.compile(r"^#\s*(源文件：|共\s*\d+\s*页|说明：|-{5,})")
PAGE_MARKER_RE = re.compile(r"^=+\s*第\s*\d+\s*页\s*=+$")
PAGE_NUMBER_RE = re.compile(r"^(?:[-—–]\s*)?(?:\d{1,4}|[ivxlcdmIVXLCDM]{1,8}|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]{1,8})(?:\s*[-—–])?$")
LEADER_RE = re.compile(r"^[\.．。…·・\s]{6,}$")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WIDE_SPACE_RE = re.compile(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000\uFEFF]")
CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
BEFORE_PUNCT_RE = re.compile(r"\s+([,，.。;；:：!?？！、）】》%％])")
AFTER_OPEN_RE = re.compile(r"([（【《])\s+")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
STOP_RE = re.compile(r"[。！？；：:;.!?]$|[。！？；：:;.!?][”’）】》]$")
HEADING_RE = re.compile(
    r"^(目录|目次|前言|摘要|关键词|Abstract|Keywords|Key words|Foreword|Notice|Contents|参考文献)$"
    r"|^(第[一二三四五六七八九十百\d]+[章节篇部分]|附录[A-ZＡ-Ｚ]?|[一二三四五六七八九十]+[、.．]|[0-9]+(?:\.[0-9]+){0,4}[、.．]?\s*\S+)"
    r"|^(图|表)\s*\d+"
)
ASCII_HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9 /,&()\-]{1,60}$")
LIST_TABLE_RE = re.compile(
    r"^([•·●○◆◇■□\-–—]|[（(]?[0-9A-Za-z一二三四五六七八九十]+[）)])\s*"
    r"|^(图|表)\s*\d+"
    r"|^[0-9.+\-−×÷*/%％ ℃°A-Za-z]+$"
)

WEB_NAV_LINES = {
    "中国储能网欢迎您！",
    "请输入搜索内容",
    "首页",
    "政策标准",
    "产业资讯",
    "物理储能",
    "电化学储能",
    "电磁储能",
    "热储能",
    "氢储能",
    "社会责任",
}


@dataclass
class Stat:
    file: str
    before_chars: int
    after_chars: int
    before_lines: int
    after_lines: int

    @property
    def removed_chars(self) -> int:
        return self.before_chars - self.after_chars

    @property
    def removed_percent(self) -> float:
        if self.before_chars == 0:
            return 0.0
        return round(self.removed_chars / self.before_chars * 100, 2)


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    line = CONTROL_RE.sub("", line)
    line = WIDE_SPACE_RE.sub(" ", line)
    line = MULTI_SPACE_RE.sub(" ", line).strip()
    line = CJK_SPACE_RE.sub("", line)
    line = BEFORE_PUNCT_RE.sub(r"\1", line)
    line = AFTER_OPEN_RE.sub(r"\1", line)
    line = MULTI_SPACE_RE.sub(" ", line).strip()
    return line


def split_pages(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    for raw in lines:
        line = normalize_line(raw)
        if META_RE.match(line):
            continue
        if PAGE_MARKER_RE.match(line):
            pages.append(current)
            current = []
            continue
        current.append(line)
    pages.append(current)
    return pages


def is_web_noise(line: str, relative: str, nonblank_index: int) -> bool:
    if line == "万方数据":
        return True
    if "中国储能网" not in relative:
        return False
    if nonblank_index > 45:
        return False
    if line.startswith("当前位置："):
        return True
    return line in WEB_NAV_LINES


def clean_pages(pages: list[list[str]], relative: str) -> list[str]:
    output: list[str] = []
    nonblank_seen = 0
    for page in pages:
        nonblank_indexes = [i for i, line in enumerate(page) if line.strip()]
        boundary = set(nonblank_indexes[:2] + nonblank_indexes[-2:])
        for i, line in enumerate(page):
            if not line.strip():
                output.append("")
                continue
            nonblank_seen += 1
            if LEADER_RE.match(line):
                continue
            previous_line = page[i - 1] if i > 0 else ""
            next_line = page[i + 1] if i + 1 < len(page) else ""
            previous_nonblank = next((page[j] for j in range(i - 1, -1, -1) if page[j].strip()), "")
            next_nonblank = next((page[j] for j in range(i + 1, len(page)) if page[j].strip()), "")
            numeric_continuation = bool(
                re.search(r"[0-9]\.$", previous_nonblank)
                or re.search(r"[A-Za-z]+\.$", previous_nonblank)
                or re.match(r"^[0-9]+(?:[—–-]|[.)）])", next_nonblank)
            )
            if i in boundary and PAGE_NUMBER_RE.match(line) and not numeric_continuation:
                continue
            if PAGE_NUMBER_RE.match(line) and not previous_line.strip() and not next_line.strip() and not numeric_continuation:
                continue
            if is_web_noise(line, relative, nonblank_seen):
                continue
            output.append(line)
        output.append("")
    return output


def collapse_blanks(lines: list[str]) -> list[str]:
    result: list[str] = []
    previous_blank = True
    for line in lines:
        blank = not line.strip()
        if blank:
            if not previous_blank:
                result.append("")
                previous_blank = True
            continue
        result.append(line)
        previous_blank = False
    while result and not result[-1].strip():
        result.pop()
    return result


def is_heading_like(line: str) -> bool:
    if not line.strip() or len(line) > 80:
        return False
    return bool(HEADING_RE.match(line) or ASCII_HEADING_RE.match(line))


def is_list_or_table_like(line: str) -> bool:
    if not line.strip():
        return False
    return bool(LIST_TABLE_RE.match(line))


def should_join(a: str, b: str) -> bool:
    if not a.strip() or not b.strip():
        return False
    if a.endswith((":", "：")):
        return False
    if re.match(r"^(作者|来源|发布时间|浏览|Address|Website|E-mail|Tel|Fax|Description|Symbol)\b", b):
        return False
    if STOP_RE.search(a):
        return False
    if is_heading_like(a) or is_heading_like(b):
        return False
    if is_list_or_table_like(a) or is_list_or_table_like(b):
        return False
    if re.match(r"^(图|表)\s*\d+|^(Fig|Figure|Table)[. ]", b):
        return False
    if len(a) < 8 and len(b) < 40:
        return False
    return True


def join_lines(a: str, b: str) -> str:
    if re.search(r"[\u4e00-\u9fff]$", a) or re.match(r"^[\u4e00-\u9fff]", b):
        return a + b
    return a + " " + b


def merge_paragraph_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if not line.strip():
            if merged and merged[-1].strip():
                merged.append("")
            continue
        if merged and should_join(merged[-1], line):
            merged[-1] = join_lines(merged[-1], line)
        else:
            merged.append(line)
    return merged


def remove_consecutive_duplicates(lines: list[str]) -> list[str]:
    result: list[str] = []
    previous: str | None = None
    for line in lines:
        if line.strip() and line == previous:
            continue
        result.append(line)
        previous = line if line.strip() else None
    return result


def normalize_text(text: str, relative: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = text.split("\n")
    pages = split_pages(raw_lines)
    lines = clean_pages(pages, relative)
    lines = collapse_blanks(lines)
    lines = merge_paragraph_lines(lines)
    lines = remove_consecutive_duplicates(lines)
    lines = collapse_blanks(lines)
    if not lines:
        return ""
    return "\r\n".join(lines).strip() + "\r\n"


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def process(root: Path, apply: bool) -> list[Stat]:
    stats: list[Stat] = []
    for path in sorted(root.rglob("*.txt")):
        relative = str(path.relative_to(root))
        original = read_text(path)
        if path.name == "_OCR汇总报告.txt":
            stats.append(
                Stat(
                    file=relative,
                    before_chars=len(original),
                    after_chars=len(original),
                    before_lines=len(original.replace("\r\n", "\n").replace("\r", "\n").split("\n")),
                    after_lines=len(original.replace("\r\n", "\n").replace("\r", "\n").split("\n")),
                )
            )
            continue
        normalized = normalize_text(original, relative)
        if apply:
            path.write_text(normalized, encoding="utf-8", newline="")
        stats.append(
            Stat(
                file=relative,
                before_chars=len(original),
                after_chars=len(normalized),
                before_lines=len(original.replace("\r\n", "\n").replace("\r", "\n").split("\n")),
                after_lines=len(normalized.replace("\r\n", "\n").replace("\r", "\n").split("\n")) if normalized else 0,
            )
        )
    return stats


def write_previews(root: Path, preview_root: Path, sample_files: list[str]) -> None:
    preview_root.mkdir(parents=True, exist_ok=True)
    for sample in sample_files:
        source = root / sample
        if not source.exists():
            continue
        original = read_text(source)
        normalized = normalize_text(original, sample)
        destination = preview_root / sample
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(normalized, encoding="utf-8", newline="")


def copy_backup(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root.parent / f"{root.name}_备份_{timestamp}"
    shutil.copytree(root, backup)
    return backup


def print_stats(stats: list[Stat]) -> None:
    stats_sorted = sorted(stats, key=lambda item: item.removed_percent, reverse=True)
    print("file\tbefore\tafter\tremoved\tremoved_percent\tbefore_lines\tafter_lines")
    for item in stats_sorted:
        print(
            f"{item.file}\t{item.before_chars}\t{item.after_chars}\t"
            f"{item.removed_chars}\t{item.removed_percent:.2f}%\t{item.before_lines}\t{item.after_lines}"
        )
    total_before = sum(item.before_chars for item in stats)
    total_after = sum(item.after_chars for item in stats)
    total_removed = total_before - total_after
    total_ratio = (total_removed / total_before * 100) if total_before else 0.0
    print(f"TOTAL\t{total_before}\t{total_after}\t{total_removed}\t{total_ratio:.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--preview-root", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        print(f"Target root does not exist: {root}", file=sys.stderr)
        return 2

    if args.preview_root:
        write_previews(
            root,
            args.preview_root.resolve(),
            [
                r"标准规范\GBT+34120-2023.txt",
                r"故障案例\中国储能网 -用户侧 - 北京丰台区“4・16”储能电站较大火灾事故调查报告 }.txt",
                r"设备资料\User Manual (PCS 100kW)_BCS-B-HM X2 Series.txt",
                r"基础资料\锂离子电池热失控蔓延研究进展.txt",
            ],
        )

    backup: Path | None = None
    if args.apply:
        backup = copy_backup(root)
        print(f"BACKUP\t{backup}")

    stats = process(root, args.apply)
    print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
