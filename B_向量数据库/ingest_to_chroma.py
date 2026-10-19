# 【模块 B-2】将 Chunk 向量化并写入 Chroma
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
"""B「将 Chunk 向量化 + 建立 Chroma 向量数据库」的建库脚本。

默认吃 A 模块（文本切分）的产物：Chunk切分结果/chunks.jsonl，
每条 Chunk 向量化后连同「原文 + 编号 + 来源」一起写进 Chroma。
也可以用 --source pdf 走「PDF → OCR → 清洗 → 切块 → 段落修复」的老路径。

用法：
  python ingest_to_chroma.py                     # chunks.jsonl → 向量化 → Chroma
  python ingest_to_chroma.py --rebuild           # 清空集合重建（换向量模型后必须加）
  python ingest_to_chroma.py --backend local     # 临时改用本地离线向量模型
  python ingest_to_chroma.py --source pdf        # 从 ./data 的 PDF 现场切块入库
  python ingest_to_chroma.py --limit 200         # 只灌前 200 条（调试用）
"""
import argparse
import concurrent.futures
import json
import os
import re
import time

import chromadb
from tqdm import tqdm

from config import EMBEDDING_CONFIG, missing_keys
from embedding_model import EmbeddingError, local_model_path

DB_PATH = "./storage"
COLLECTION_NAME = "energy_storage_library"
CHUNKS_FILE = os.path.join("Chunk切分结果", "chunks.jsonl")
DATA_FOLDER = "./data"


# ======================= 读取待入库的 Chunk =======================
def load_chunks_from_jsonl(path, limit=None):
    """读取切分结果（A 模块产物）：一行一个 JSON，取 chunk_id/source_file/chunk_index/content。"""
    chunks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("content") or "").strip()
            if not text:
                continue
            source_file = obj.get("source_file") or ""
            chunks.append({
                "id": obj.get("chunk_id") or "chunk_%04d" % (len(chunks) + 1),
                "text": text,
                "source_file": source_file,
                "source": os.path.basename(source_file.replace(os.sep, "/")) or "未知来源",
                "page": 0,
                "chunk_index": int(obj.get("chunk_index") or 0),
            })
            if limit and len(chunks) >= limit:
                break
    return chunks


def is_garbage_rules(text):
    """粗过滤：太短、中文占比过低、连续乱码符号的片段直接丢掉。"""
    text = text.strip()
    if len(text) < 15:
        return True
    zh_chars = re.findall(r"[\u4e00-\u9fa5]", text)
    if len(text) > 0 and len(zh_chars) / len(text) < 0.5:
        return True
    if re.search(r"([^\w\s\u4e00-\u9fa5])\1{5,}", text):
        return True
    return False


def build_llm():
    from extractor import build_model
    return build_model("chat")


def ai_fix_paragraph(llm_model, chunk_text):
    """用大模型把 OCR 的异常换行/断裂段落拼回通顺段落。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = """你是一个严谨的文本排版与校对助手。以下文本是从 PDF OCR 提取的，存在异常换行、段落断裂（如两三个字就换行）等排版错乱。
【严格指令】：
1. 合并断裂文本：将不合逻辑的换行去掉，把断开的句子重新拼接成连贯、通顺的段落。
2. 保持原文语义：**严禁**增加原文没有的解释、**严禁**总结缩写、**严禁**改变原本的专业词汇。
3. 直接输出结果：不要添加说明性前缀或其他无关内容，直接输出修复后的文本。"""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content="待修复文本：\n%s" % chunk_text)]
    try:
        result = llm_model.invoke(messages).content.strip()
        return result if len(result) >= 10 else chunk_text
    except Exception:
        return chunk_text


def load_chunks_from_pdf(folder, max_workers=10):
    """旧路径：PDF → 逐页取文（扫描页自动 OCR）→ 清洗切块 → 大模型修段落。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from data_cleaner import clean_text
    from ocr_utils import extract_pages_text, format_stats

    pdf_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))
    if not pdf_files:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    llm_model = build_llm()
    chunks = []
    for filename in pdf_files:
        print("\n[预处理] %s" % filename)
        pages_text, ocr_stats = extract_pages_text(os.path.join(folder, filename))
        if ocr_stats.get("need_ocr_pages"):
            print("   OCR 兜底：" + format_stats(ocr_stats))
        for page_no, page_text in enumerate(pages_text, 1):
            raw = [c for c in splitter.split_text(clean_text(page_text)) if not is_garbage_rules(c)]
            if not raw:
                continue
            refined = [None] * len(raw)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(ai_fix_paragraph, llm_model, c): i for i, c in enumerate(raw)}
                for fut in concurrent.futures.as_completed(futures):
                    refined[futures[fut]] = fut.result()
            for seq, text in enumerate([r for r in refined if r], 1):
                chunks.append({
                    "id": "%s_p%d_c%d" % (filename, page_no, seq),
                    "text": text,
                    "source_file": filename,
                    "source": filename,
                    "page": page_no,
                    "chunk_index": seq,
                })
    return chunks


# ======================= 向量化 + 写库 =======================
def build_embedder(backend, need_fit_corpus=None, persist=True):
    """准备向量模型；local 后端在需要时用整份语料训练。

    persist=False（带 --limit 时）不落盘：用部分语料训出来的模型不该留给整库用，
    否则下次全量入库会静默复用一个只见过几百条数据的向量空间。
    """
    from embedding_model import LocalEmbedder, get_embedder, param_diffs

    if backend == "local":
        path = local_model_path(DB_PATH)
        if os.path.exists(path):
            model = LocalEmbedder.load(path)
            diffs = param_diffs(model)
            if not diffs:
                print("本地向量模型：复用已落盘的 %s" % path)
                return model
            print("本地模型参数与 config 不一致，自动重新训练（否则白改参数）：")
            for item in diffs:
                print("   · %s" % item)
        if need_fit_corpus is None:
            raise EmbeddingError("本地向量模型文件不存在，且没有语料可训练：%s" % path)
        print("本地向量模型：正在用 %d 条 Chunk 训练…" % len(need_fit_corpus))
        model = LocalEmbedder().fit(need_fit_corpus)
        if persist:
            model.save(path)
            print("   已落盘到 %s（下次检索直接复用）" % path)
        else:
            print("   本次带 --limit，跳过落盘：部分语料训出来的模型不给整库用。")
        return model
    return get_embedder(backend="cloud", db_path=DB_PATH)


def prepare_collection(client, rebuild, dimension, backend="", model_name=""):
    """建/取集合；换模型导致维度不一致时明确报错，别让库悄悄写坏。

    同时把「用哪个后端、哪个模型、多少维、什么时候建的」写进集合元数据——
    否则事后只能比维度，报不出「这个库是 local-256 建的、你现在配的是 cloud-4096」。
    """
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("已删除旧集合 %s" % COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_backend": str(backend),
            "embedding_model": str(model_name),
            "embedding_dim": int(dimension),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    if not rebuild and collection.count() > 0:
        try:
            got = collection.get(limit=1, include=["embeddings"])
            existing = got.get("embeddings")
            old_dim = len(existing[0]) if existing is not None and len(existing) else None
        except Exception:
            old_dim = None
        if old_dim and old_dim != dimension:
            raise SystemExit(
                "向量维度不一致：库里已有 %d 维，现在是 %d 维。\n"
                "换向量模型后必须重建，请加 --rebuild 重跑。" % (old_dim, dimension)
            )
    return collection


def write_chunks(collection, chunks, embedder, batch_size):
    """分批向量化并写入，返回「净增条数」。

    这里用 upsert 而不是 add：Chroma 的 add 撞到已存在的 id 会直接静默忽略
    （不报错、也不更新），重跑同一批 Chunk 时会让人误以为写进去了。
    upsert 则是有则覆盖、无则新增，换模型重建后重跑也能把旧向量刷成新的。
    """
    before = collection.count()
    for start in tqdm(range(0, len(chunks), batch_size), desc="向量化入库"):
        batch = chunks[start:start + batch_size]
        vectors = embedder.embed_documents([c["text"] for c in batch])
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vectors,
            metadatas=[{
                "chunk_id": c["id"],
                "source_file": c["source_file"],
                "source": c["source"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
            } for c in batch],
        )
    return collection.count() - before


def parse_args():
    parser = argparse.ArgumentParser(description="把 Chunk 向量化并写入 Chroma 向量库")
    parser.add_argument("--source", choices=["chunks", "pdf"], default="chunks",
                        help="chunks=A 模块切分结果（默认）；pdf=从 ./data 现场抽文切块")
    parser.add_argument("--chunks_file", default=CHUNKS_FILE, help="chunks.jsonl 路径")
    parser.add_argument("--data_dir", default=DATA_FOLDER, help="--source pdf 时的 PDF 目录")
    parser.add_argument("--backend", choices=["cloud", "local"], default=None,
                        help="向量后端，默认读 config.EMBEDDING_CONFIG['backend']")
    parser.add_argument("--rebuild", action="store_true", help="先删掉旧集合再重建")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")
    parser.add_argument("--retrain", action="store_true", help="local 后端强制重新训练向量模型")
    parser.add_argument("--db_path", default=DB_PATH, help="Chroma 落盘目录")
    return parser.parse_args()


def main():
    global DB_PATH
    args = parse_args()
    DB_PATH = args.db_path
    backend = args.backend or EMBEDDING_CONFIG.get("backend", "cloud")
    batch_size = int(EMBEDDING_CONFIG.get("batch_size", 32))

    if backend == "cloud":
        missing = missing_keys(["embedding"])
        if missing:
            print("还没填向量模型 API Key，无法向量化入库。请打开 config.py 填写，"
                  "或改用本地离线模型：python ingest_to_chroma.py --backend local")
            for item in missing:
                print("   · %s" % item)
            return 1

    if backend == "local" and args.retrain:
        stale = local_model_path(DB_PATH)
        if os.path.exists(stale):
            os.remove(stale)
            print("已删除旧的本地向量模型 %s，将重新训练。" % stale)

    # ---- 1. 取待入库的 Chunk ----
    if args.source == "chunks":
        if not os.path.exists(args.chunks_file):
            print("找不到切分结果 %s，先跑 A 模块的切分脚本，或改用 --source pdf。" % args.chunks_file)
            return 1
        chunks = load_chunks_from_jsonl(args.chunks_file, limit=args.limit)
        print("已读取 %d 条 Chunk（来源：%s）" % (len(chunks), args.chunks_file))
    else:
        if not os.path.isdir(args.data_dir):
            print("找不到文件夹 %s，请先放入 PDF。" % args.data_dir)
            return 1
        chunks = load_chunks_from_pdf(args.data_dir)
        if args.limit:
            chunks = chunks[:args.limit]
        print("已从 PDF 切出 %d 条 Chunk" % len(chunks))
    if not chunks:
        print("一条 Chunk 都没读到，向量库没有变化。")
        return 1

    # ---- 2. 选择向量模型（B-1）----
    print("向量后端：%s" % backend)
    embedder = build_embedder(backend, need_fit_corpus=[c["text"] for c in chunks],
                              persist=not args.limit)

    # ---- 3. 建 Chroma 向量库（B-3）----
    client = chromadb.PersistentClient(path=DB_PATH)
    dimension = embedder.dimension        # 云端在这里实测一次；本地是训练后的真实维度
    model_name = getattr(embedder, "model_name", None) or ("local-%dD" % dimension)
    collection = prepare_collection(client, args.rebuild, dimension, backend, model_name)
    print("向量库：%s / 集合 %s / %d 维 / 现有 %d 条"
          % (DB_PATH, COLLECTION_NAME, dimension, collection.count()))

    # ---- 4. 向量化并写入（B-2）----
    started = time.time()
    try:
        added = write_chunks(collection, chunks, embedder, batch_size)
    except EmbeddingError as exc:
        print("向量化失败：%s" % exc)
        return 1
    print("\n处理完成：处理 %d 条，净增 %d 条（同 id 的已覆盖），集合现有 %d 条，耗时 %.1f 秒"
          % (len(chunks), added, collection.count(), time.time() - started))
    print("下一步：python eval_semantic_search.py   # 评测语义检索效果")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
