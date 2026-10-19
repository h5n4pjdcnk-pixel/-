# 【模块 A-2】61 份 TXT 批量切分，生成 chunks.jsonl / manifest
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


MIN_CHARS = 500
MAX_CHARS = 1200
SOFT_TARGET = 900
ABS_MAX_CHARS = 1400

SENTENCE_END_RE = re.compile(r"(?<=[。！？!?；;])")
EN_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
HEADING_RE = re.compile(
    r"^(目录|目次|前言|摘要|关键词|Abstract|Keywords|Key words|Foreword|Notice|Contents|参考文献)$"
    r"|^(第[一二三四五六七八九十百\d]+[章节篇部分]|附录[A-ZＡ-Ｚ]?|[一二三四五六七八九十]+[、.．]|[0-9]+(?:\.[0-9]+){0,5}[、.．]?\s*\S+)"
    r"|^(图|表)\s*\d+"
)


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    chunk_index: int
    content: str


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heading(block: str) -> bool:
    line = block.strip()
    return bool(line and len(line) <= 90 and HEADING_RE.match(line))


def split_paragraphs(text: str) -> list[str]:
    blocks: list[str] = []
    for raw in re.split(r"\n\s*\n", normalize_text(text)):
        block = raw.strip()
        if not block:
            continue
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if len(lines) == 1:
            blocks.append(lines[0])
            continue

        buffer: list[str] = []
        for line in lines:
            if is_heading(line):
                if buffer:
                    blocks.append("\n".join(buffer).strip())
                    buffer = []
                blocks.append(line)
            else:
                buffer.append(line)
        if buffer:
            blocks.append("\n".join(buffer).strip())
    return blocks


def split_long_by_sentences(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= MAX_CHARS:
        return [text]

    pieces: list[str] = []
    for part in re.split(SENTENCE_END_RE, text):
        part = part.strip()
        if not part:
            continue
        if len(part) > MAX_CHARS and re.search(r"[A-Za-z]", part):
            en_parts = [p.strip() for p in EN_SENTENCE_END_RE.split(part) if p.strip()]
            pieces.extend(en_parts if len(en_parts) > 1 else [part])
        else:
            pieces.append(part)

    if len(pieces) <= 1:
        return split_overlong_fallback(text)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = join_piece(current, piece)
        if len(candidate) <= MAX_CHARS or (len(current) < MIN_CHARS and len(candidate) <= ABS_MAX_CHARS):
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(piece) > ABS_MAX_CHARS:
            chunks.extend(split_overlong_fallback(piece))
            current = ""
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def split_overlong_fallback(text: str) -> list[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        chunks: list[str] = []
        current = ""
        for line in lines:
            candidate = join_piece(current, line)
            if len(candidate) <= MAX_CHARS or len(current) < MIN_CHARS:
                current = candidate
            else:
                chunks.append(current)
                current = line
        if current:
            chunks.append(current)
        if all(len(item) <= ABS_MAX_CHARS for item in chunks):
            return chunks

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + SOFT_TARGET, len(text))
        chunks.append(text[start:end].strip())
        start = end
    return [chunk for chunk in chunks if chunk]


def join_piece(a: str, b: str) -> str:
    if not a:
        return b.strip()
    if not b:
        return a.strip()
    if "\n" in a or "\n" in b:
        return f"{a.rstrip()}\n{b.strip()}"
    return f"{a.rstrip()} {b.strip()}"


def chunk_blocks(blocks: list[str]) -> list[str]:
    chunks: list[str] = []
    current = ""

    for block in blocks:
        if not block.strip():
            continue

        for part in split_long_by_sentences(block):
            candidate = join_piece(current, part)
            if len(candidate) <= MAX_CHARS:
                current = candidate
                continue

            if len(current) >= MIN_CHARS:
                chunks.append(current)
                if len(part) > ABS_MAX_CHARS:
                    chunks.extend(split_overlong_fallback(part))
                    current = ""
                else:
                    current = part
                continue

            if current:
                if len(candidate) <= ABS_MAX_CHARS:
                    chunks.append(candidate)
                    current = ""
                else:
                    chunks.append(current)
                    current = part
            else:
                chunks.append(part)

    if current:
        chunks.append(current)

    return rebalance_short_chunks(chunks)


def rebalance_short_chunks(chunks: list[str]) -> list[str]:
    result: list[str] = []
    i = 0
    while i < len(chunks):
        current = chunks[i].strip()
        if not current:
            i += 1
            continue

        if len(current) < MIN_CHARS and i + 1 < len(chunks):
            candidate = join_piece(current, chunks[i + 1])
            if len(candidate) <= ABS_MAX_CHARS:
                result.append(candidate)
                i += 2
                continue

        if len(current) < MIN_CHARS and result:
            candidate = join_piece(result[-1], current)
            if len(candidate) <= ABS_MAX_CHARS:
                result[-1] = candidate
                i += 1
                continue

        result.append(current)
        i += 1

    return result


def quality_flags(chunk: str) -> list[str]:
    flags: list[str] = []
    length = len(chunk)
    if length == 0:
        flags.append("empty")
    if length < MIN_CHARS:
        flags.append("short")
    if length > MAX_CHARS:
        flags.append("long")
    if "\ufffd" in chunk:
        flags.append("replacement_char")
    if re.search(r"[。！？!?；;]$", chunk.strip()) is None and length >= MAX_CHARS:
        flags.append("possibly_incomplete_sentence")
    return flags


def build_chunks(source_root: Path) -> tuple[list[Chunk], dict[str, int]]:
    files = [
        path
        for path in sorted(source_root.rglob("*.txt"))
        if path.name != "_OCR汇总报告.txt"
    ]
    all_chunks: list[Chunk] = []
    per_file_counts: dict[str, int] = {}
    global_index = 1

    for path in files:
        relative = str(path.relative_to(source_root)).replace("\\", "/")
        text = read_text(path)
        chunks = chunk_blocks(split_paragraphs(text))
        per_file_counts[relative] = len(chunks)
        for local_index, content in enumerate(chunks, start=1):
            all_chunks.append(
                Chunk(
                    chunk_id=f"chunk_{global_index:04d}",
                    source_file=relative,
                    chunk_index=local_index,
                    content=content,
                )
            )
            global_index += 1

    return all_chunks, per_file_counts


def write_outputs(output_root: Path, chunks: list[Chunk], per_file_counts: dict[str, int], source_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    with (output_root / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    with (output_root / "chunks.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_id", "source_file", "chunk_index", "content"])
        writer.writeheader()
        for chunk in chunks:
            writer.writerow(asdict(chunk))

    lengths = [len(chunk.content) for chunk in chunks]
    flagged = [
        {
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "length": len(chunk.content),
            "flags": quality_flags(chunk.content),
        }
        for chunk in chunks
    ]
    flagged = [item for item in flagged if item["flags"]]

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "excluded_files": ["_OCR汇总报告.txt"],
        "source_file_count": len(per_file_counts),
        "chunk_count": len(chunks),
        "chunk_id_format": "chunk_0001",
        "fields": ["chunk_id", "source_file", "chunk_index", "content"],
        "length_policy": {
            "target_min_chars": MIN_CHARS,
            "target_max_chars": MAX_CHARS,
            "absolute_max_chars_for_semantic_integrity": ABS_MAX_CHARS,
        },
        "outputs": {
            "jsonl": "chunks.jsonl",
            "csv": "chunks.csv",
            "quality_report": "quality_report.md",
        },
        "per_file_counts": per_file_counts,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_root / "quality_report.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Chunk 切分质量报告\n\n")
        f.write(f"- 来源目录：`{source_root}`\n")
        f.write("- 排除文件：`_OCR汇总报告.txt`\n")
        f.write(f"- 原始 TXT 文件数：{len(per_file_counts)}\n")
        f.write(f"- Chunk 总数：{len(chunks)}\n")
        f.write(f"- 字符长度区间：{min(lengths)}–{max(lengths)}\n")
        f.write(f"- 平均长度：{statistics.mean(lengths):.1f}\n")
        f.write(f"- 中位数长度：{statistics.median(lengths):.1f}\n")
        in_range = sum(1 for length in lengths if MIN_CHARS <= length <= MAX_CHARS)
        f.write(f"- 500–1200 字符范围内：{in_range}/{len(chunks)}\n")
        f.write(f"- 需关注 Chunk 数：{len(flagged)}\n\n")
        f.write("## 说明\n\n")
        f.write("切分采用语义优先策略：优先按段落、标题、句末标点和换行边界切分；少数 chunk 低于或高于 500–1200 字符，是为了避免拆断目录、表格、专有名词或连贯语义段。\n\n")
        f.write("## 来源文件 Chunk 数\n\n")
        f.write("| source_file | chunks |\n| :--- | ---: |\n")
        for file, count in per_file_counts.items():
            f.write(f"| {file} | {count} |\n")
        f.write("\n## 需关注 Chunk\n\n")
        if not flagged:
            f.write("未发现空内容、乱码或长度异常 chunk。\n")
        else:
            f.write("| chunk_id | source_file | chunk_index | length | flags |\n| :--- | :--- | ---: | ---: | :--- |\n")
            for item in flagged:
                f.write(
                    f"| {item['chunk_id']} | {item['source_file']} | {item['chunk_index']} | "
                    f"{item['length']} | {', '.join(item['flags'])} |\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source root does not exist: {source_root}")

    chunks, per_file_counts = build_chunks(source_root)
    write_outputs(output_root, chunks, per_file_counts, source_root)
    print(f"source_files={len(per_file_counts)}")
    print(f"chunks={len(chunks)}")
    print(f"output_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
