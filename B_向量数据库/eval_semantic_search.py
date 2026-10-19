# 【模块 B-4】语义检索评测：命中率 / MRR / 延迟
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
"""B「实现并测试语义检索」的独立评测脚本。

干的事：拿一份「问题 → 该命中哪份资料」的评测集，把检索链路跑一遍，
算命中率 / MRR / 延迟，并对比「纯向量召回」与「向量召回 + 重排」两条路线，
最后落一份 markdown + json 报告。

用法（在项目根目录执行）：
  python eval_semantic_search.py                     # 内置评测集，两种模式对比
  python eval_semantic_search.py --k 10 --backend local
  python eval_semantic_search.py --auto 200          # 自动构造 200 条自检索样本（链路自检）
  python eval_semantic_search.py --mode vector       # 只测纯向量召回
  python eval_semantic_search.py --min-doc-hit 0.8   # 命中率低于 0.8 时以退出码 1 结束
"""
import argparse
import json
import os
import random
import re
import time
from datetime import datetime

from config import EMBEDDING_CONFIG, missing_keys
from embedding_model import local_model_path
from retriever import RetrieverError, SemanticRetriever

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DEFAULT = os.path.join(HERE, "语义检索评测集.jsonl")
REPORT_MD = os.path.join(HERE, "语义检索评测报告.md")
REPORT_JSON = os.path.join(HERE, "语义检索评测报告.json")


# ======================= 评测集 =======================
def load_benchmark(path, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if not obj.get("query"):
                continue
            items.append(obj)
            if limit and len(items) >= limit:
                break
    return items


def make_auto_benchmark(collection, n, seed=42):
    """自动样本：从库里随机抽 Chunk，取其中一句话当问题，看能否把它自己召回来。

    这是「自检索」冒烟测试，用来验证索引 / 维度 / 归一化没写错，
    不能代表真实问答质量——真实质量看内置评测集。
    """
    got = collection.get(include=["documents", "metadatas"])
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    if not ids:
        return []
    rng = random.Random(seed)
    picked = rng.sample(range(len(ids)), min(int(n), len(ids)))
    items = []
    for i in picked:
        text = docs[i] or ""
        pieces = [p.strip() for p in re.split(r"(?<=[。；！？])", text) if len(p.strip()) >= 25]
        if not pieces:
            continue
        meta = metas[i] or {}
        items.append({
            "query": rng.choice(pieces),
            "expected_chunk_id": ids[i],
            "expected_source": meta.get("source_file", ""),
            "expected_keywords": [],
            "note": "auto:self-retrieval",
        })
    return items


# ======================= 评测 =======================
def evaluate(retriever, items, k, use_rerank, mode_name):
    detail = []
    for item in items:
        query = item["query"]
        started = time.time()
        hits = retriever.search(query, top_k=k, use_rerank=use_rerank)
        latency = (time.time() - started) * 1000

        exp_source = item.get("expected_source") or ""
        exp_id = item.get("expected_chunk_id")
        keywords = item.get("expected_keywords") or []

        doc_ranks = [h["rank"] for h in hits
                     if exp_source and exp_source in (h.get("source_file") or "")]
        chunk_rank = next((h["rank"] for h in hits if h.get("chunk_id") == exp_id), None) if exp_id else None
        top1 = hits[0] if hits else None
        kw_top1 = any(kw in top1["text"] for kw in keywords) if (keywords and top1) else None

        detail.append({
            "query": query,
            "note": item.get("note", ""),
            "expected_source": exp_source,
            "doc_rank": min(doc_ranks) if doc_ranks else None,
            "chunk_rank": chunk_rank,
            "keyword_top1": kw_top1,
            "latency_ms": round(latency, 1),
            "top1_source": (top1.get("source_file") if top1 else ""),
            "top1_chunk": (top1.get("chunk_id") if top1 else ""),
            "top1_score": (round(top1.get("final_score", 0.0), 4) if top1 else None),
        })

    n = len(detail) or 1
    doc_ranks = [d["doc_rank"] for d in detail if d["doc_rank"]]
    chunk_ranks = [d["chunk_rank"] for d in detail if d["chunk_rank"]]
    kw_known = [d["keyword_top1"] for d in detail if d["keyword_top1"] is not None]
    summary = {
        "mode": mode_name,
        "use_rerank": use_rerank,
        "k": k,
        "n": len(detail),
        "doc_hit_rate": round(len(doc_ranks) / n, 4),
        "doc_mrr": round(sum(1.0 / r for r in doc_ranks) / n, 4),
        "chunk_hit_rate": round(len(chunk_ranks) / n, 4),
        "keyword_top1_rate": round(sum(1 for x in kw_known if x) / len(kw_known), 4) if kw_known else None,
        "avg_latency_ms": round(sum(d["latency_ms"] for d in detail) / n, 1),
        "detail": detail,
    }
    if not any(it.get("expected_chunk_id") for it in items):
        summary["chunk_hit_rate"] = None
    return summary


# ======================= 报告 =======================
def render_markdown(results, meta):
    lines = []
    lines.append("# 语义检索评测报告")
    lines.append("")
    lines.append("- 生成时间：%s" % meta["created_at"])
    lines.append("- 向量后端：`%s`（模型 %s）" % (meta["backend"], meta["model"]))
    lines.append("- 向量库：`%s` / 集合 `%s` / %d 条向量" % (meta["db_path"], meta["collection"], meta["vectors"]))
    lines.append("- 评测集：%s（%d 条 %s）" % (meta["benchmark"], meta["n"], meta["benchmark_kind"]))
    lines.append("- 评价口径：Top-%d 内是否命中「期望来源文档」；MRR 为首个命中的排名倒数" % meta["k"])
    if meta.get("rerank_status") == "degraded":
        lines.append("- 注意：重排接口本次调用失败，已自动降级为纯向量顺序，两种模式结果相同。")
    lines.append("")
    lines.append("## 一、总体对比")
    lines.append("")
    lines.append("| 模式 | 文档命中率@%d | MRR@%d | 关键词命中率(top1) | 平均延迟(ms) |" % (meta["k"], meta["k"]))
    lines.append("| --- | --- | --- | --- | --- |")
    for name, res in results.items():
        kw = res["keyword_top1_rate"]
        lines.append("| %s | %.2f%% | %.3f | %s | %.1f |" % (
            name, res["doc_hit_rate"] * 100, res["doc_mrr"],
            ("%.2f%%" % (kw * 100)) if kw is not None else "-",
            res["avg_latency_ms"]))
    lines.append("")
    lines.append("## 二、逐条结果")
    for name, res in results.items():
        lines.append("")
        lines.append("### %s" % name)
        lines.append("")
        lines.append("| 问题 | 期望来源 | 命中排名 | Top1 实际来源 | 备注 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for d in res["detail"]:
            rank = d["doc_rank"] if d["doc_rank"] else "未命中"
            lines.append("| %s | %s | %s | %s | %s |" % (
                d["query"], d["expected_source"] or "-", rank, d["top1_source"] or "-", d["note"] or "-"))
    lines.append("")
    lines.append("## 三、结论")
    lines.append("")
    if len(results) >= 2:
        names = list(results.keys())
        base, advanced = results[names[0]], results[names[1]]
        delta = (advanced["doc_hit_rate"] - base["doc_hit_rate"]) * 100
        lines.append("- 加了重排后文档命中率变化：%+.2f 个百分点。" % delta)
        if delta > 1:
            lines.append("- 重排对本次评测集有正向增益，线上默认开启。")
        elif delta < -1:
            lines.append("- 重排反而变差，检查重排模型或候选池大小（candidate_k / rerank_top_n）。")
        else:
            lines.append("- 两者差距很小，说明粗召回本身已经够准，重排主要起稳定作用。")
    lines.append("- 未命中条目请优先检查：该问题在语料里是否真的有对应内容、切分是否把关键段切碎。")
    lines.append("")
    return "\n".join(lines)


# ======================= 入口 =======================
def parse_args():
    parser = argparse.ArgumentParser(description="语义检索评测（命中率 / MRR / 延迟）")
    parser.add_argument("--benchmark", default=BENCH_DEFAULT, help="评测集 jsonl 路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--auto", type=int, default=0, help="自动构造 N 条自检索样本，代替人工评测集")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5, help="看前 K 条是否命中")
    parser.add_argument("--mode", choices=["both", "vector", "rerank"], default="both")
    parser.add_argument("--backend", choices=["cloud", "local"], default=None)
    parser.add_argument("--db_path", default="./storage")
    parser.add_argument("--collection", default="energy_storage_library")
    parser.add_argument("--min-doc-hit", type=float, default=0.0,
                        help="文档命中率下限，低于它则退出码为 1（默认 0 = 不卡）")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_embedder(args):
    from embedding_model import LocalEmbedder, get_embedder

    backend = args.backend or EMBEDDING_CONFIG.get("backend", "cloud")
    if backend == "cloud":
        if missing_keys(["embedding"]):
            raise SystemExit("向量模型 API Key 没填，无法在线评测；加 --backend local 用本地模型评测。")
        return backend, get_embedder(backend="cloud", db_path=args.db_path)
    path = local_model_path(args.db_path)
    if not os.path.exists(path):
        raise SystemExit("本地向量模型文件不存在：%s\n先跑 python ingest_to_chroma.py --backend local" % path)
    return backend, LocalEmbedder.load(path)


def main():
    args = parse_args()
    backend, embedder = build_embedder(args)
    model_name = getattr(embedder, "model_name", None) or "local-%dD" % embedder.dimension

    try:
        retriever = SemanticRetriever(db_path=args.db_path, collection_name=args.collection, embedder=embedder)
    except RetrieverError as exc:
        raise SystemExit(str(exc))

    vectors = retriever.count
    if vectors == 0:
        raise SystemExit("向量库是空的，先跑 python ingest_to_chroma.py 建库再评测。")

    if args.auto:
        items = make_auto_benchmark(retriever.collection, args.auto, args.seed)
        kind = "自动自检索样本"
        bench_name = "auto:%d" % args.auto
    else:
        if not os.path.exists(args.benchmark):
            raise SystemExit("找不到评测集：%s" % args.benchmark)
        items = load_benchmark(args.benchmark, args.limit)
        kind = "人工评测集"
        bench_name = os.path.basename(args.benchmark)
    if not items:
        raise SystemExit("评测集是空的。")

    modes = {"纯向量召回": False} if args.mode == "vector" else (
        {"向量召回+重排": True} if args.mode == "rerank" else {"纯向量召回": False, "向量召回+重排": True})

    results = {}
    for name, use_rerank in modes.items():
        if not args.quiet:
            print("正在评测：%s（%d 条，Top-%d）..." % (name, len(items), args.k))
        results[name] = evaluate(retriever, items, args.k, use_rerank, name)

    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend,
        "model": model_name,
        "db_path": args.db_path,
        "collection": args.collection,
        "vectors": vectors,
        "benchmark": bench_name,
        "benchmark_kind": kind,
        "n": len(items),
        "k": args.k,
    }

    meta["rerank_status"] = retriever.rerank_status
    if retriever.rerank_status == "degraded":
        print("提示：重排接口调用失败，已降级为纯向量顺序（检查 config.py 里 rerank 的 Key / 模型名）。")

    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(results, meta))
    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": results}, fh, ensure_ascii=False, indent=2)

    print("")
    print("向量后端 %s | 集合 %d 条 | 评测 %d 条 | 报告已写入：" % (backend, vectors, len(items)))
    for name, res in results.items():
        kw = res["keyword_top1_rate"]
        print("  %-12s 文档命中率@%d = %6.2f%% | MRR = %.3f | 平均延迟 %.1f ms%s" % (
            name, args.k, res["doc_hit_rate"] * 100, res["doc_mrr"], res["avg_latency_ms"],
            (" | 关键词命中 %.2f%%" % (kw * 100)) if kw is not None else ""))
    print("  " + REPORT_MD)

    best = max(res["doc_hit_rate"] for res in results.values())
    if args.min_doc_hit and best < args.min_doc_hit:
        print("未达到命中率下限 %.2f（实际 %.2f）" % (args.min_doc_hit, best))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())