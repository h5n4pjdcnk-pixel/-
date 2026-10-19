# 【模块 B-4】语义检索：向量召回 + 重排 + 来源加权
from __future__ import annotations

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
"""B「语义检索」核心：提问 → 向量召回 → 重排 → 加权 → Top-K。

单独抽成一个模块，是为了让「网页问答(app.py)」和「离线评测(eval_semantic_search.py)」
走完全相同的检索逻辑——否则评测出来的分数和线上效果对不上。
"""
import hashlib
import time

import chromadb
import requests

from config import LLM_CONFIG, RETRIEVAL_CONFIG
from embedding_model import get_embedder


class RetrieverError(RuntimeError):
    """检索相关的可预期错误（库不存在、维度不匹配等）。"""


def _source_weight(source_name, weights, source_file=""):
    """来源加权：命中 config.RETRIEVAL_CONFIG["source_weights"] 关键词就放大。

    匹配的是「完整路径」（如 标准规范/GBT+34120-2023.txt），加的是「哪一类资料」，
    不会被标题里恰好出现的字眼误伤（例如标题写成 “标准与规范” 的新闻稿）。
    """
    target = "%s %s" % (source_file or "", source_name or "")
    for keyword, factor in (weights or {}).items():
        if keyword in target:
            return float(factor)
    return 1.0


class SemanticRetriever:
    """语义检索器：向量召回 + 重排 + 来源加权。"""

    def __init__(self, db_path="./storage", collection_name="energy_storage_library",
                 embedder=None, config=None):
        self.db_path = db_path
        self.collection_name = collection_name
        self.cfg = dict(RETRIEVAL_CONFIG)
        if config:
            self.cfg.update(config)
        self.embedder = embedder or get_embedder(db_path=db_path)
        self._client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self._client.get_collection(name=collection_name)
        except Exception as exc:
            raise RetrieverError(
                "向量库 %s 里没有集合 %s，先跑 `python ingest_to_chroma.py` 建库。原始错误：%s"
                % (db_path, collection_name, exc)
            )
        # 集合的向量空间：默认 l2，重建时会显式设成 cosine
        meta = getattr(self.collection, "metadata", None) or {}
        self.space = str(meta.get("hnsw:space", "l2")).lower()
        # 重排状态：ok / off(没配) / degraded(调用失败已降级)
        self.rerank_status = "off"

    # ------------------------------------------------------------------ #
    @property
    def count(self):
        return self.collection.count()

    def _similarity(self, distance):
        """把 Chroma 的距离换算成 0~1 的相似度（越大越像）。"""
        distance = float(distance)
        if self.space == "cosine":
            return 1.0 - distance
        if self.space == "ip":
            return -distance
        # l2：向量都已 L2 归一化，欧氏距离 d 与余弦相似度的关系是 cos = 1 - d²/2
        return max(-1.0, 1.0 - (distance * distance) / 2.0)

    def _rerank(self, query, documents, top_n):
        """调用硅基流动重排接口。

        返回 (新顺序下标, 分数, 是否真的重排成功)。失败就原序降级——
        成功与否显式返回，不靠读 self.rerank_status（那种跨调用的实例状态容易被上一个
        请求污染，判断出错会导致整段排序逻辑被跳过）。
        """
        cfg = LLM_CONFIG.get("rerank") or {}
        baseline = (list(range(len(documents))), [None] * len(documents), False)
        if not cfg.get("model_name") or not documents:
            self.rerank_status = "off"
            return baseline
        payload = {
            "model": cfg["model_name"],
            "query": query,
            "top_n": max(1, min(top_n, len(documents))),
            "documents": documents,
        }
        # 必须带 Bearer 前缀：裸 Key 会被硅基流动判成 401 Token is invalid
        # （老版 app.py 的 api_rerank 就是裸 Key，所以这个项目的重排一直没真正生效过）
        headers = {"Authorization": "Bearer %s" % cfg["api_key"], "Content-Type": "application/json"}
        try:
            resp = requests.post(cfg["api_url"], json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            results = sorted(resp.json().get("results", []), key=lambda x: -x["relevance_score"])
            order = [int(item["index"]) for item in results]
            scores = [float(item["relevance_score"]) for item in results]
            if not order:
                self.rerank_status = "off"
                return baseline
            # 没进重排名单的尾巴按原顺序接在后面
            tail = [i for i in range(len(documents)) if i not in set(order)]
            self.rerank_status = "ok"
            return order + tail, scores + [None] * len(tail), True
        except Exception:
            self.rerank_status = "degraded"
            return baseline

    # ------------------------------------------------------------------ #
    def search(self, query, top_k=None, candidate_k=None, use_rerank=None,
               weights=None, with_trace=True):
        """检索主入口。

        query        用户问题
        top_k        最终返回条数，默认 config.RETRIEVAL_CONFIG["top_k"]
        candidate_k  向量粗召回条数，默认 config.RETRIEVAL_CONFIG["candidate_k"]
        use_rerank   是否重排，默认 config.RETRIEVAL_CONFIG["use_rerank"]
        weights      来源权重字典，默认 config.RETRIEVAL_CONFIG["source_weights"]
        返回：命中列表（按最终分数降序），每项是 dict
        """
        candidate_k = int(candidate_k or self.cfg.get("candidate_k", 60))
        top_k = int(top_k or self.cfg.get("top_k", 5))
        if use_rerank is None:
            use_rerank = bool(self.cfg.get("use_rerank", True))
        weights = self.cfg.get("source_weights", {}) if weights is None else weights

        total = self.count
        if total == 0:
            return []
        n_fetch = max(1, min(candidate_k, total))

        t0 = time.time()
        query_vec = self.embedder.embed_query(query)
        embed_ms = (time.time() - t0) * 1000

        raw = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_fetch,
            include=["documents", "metadatas", "distances"],
        )
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        if not docs:
            return []

        hits = []
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            meta = dict(meta or {})
            source_name = meta.get("source") or meta.get("source_file") or ""
            similarity = self._similarity(dist)
            weight = _source_weight(source_name, weights, meta.get("source_file", ""))
            hits.append({
                "chunk_id": meta.get("chunk_id", ""),
                "source_file": meta.get("source_file", source_name),
                "source": source_name,
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "text": doc,
                "vector_rank": rank,
                "vector_score": round(similarity, 4),
                "rerank_score": None,
                "weight": weight,
                "final_score": similarity * weight,
            })

        # 同一段文本可能重复命中（同一份资料在不同目录各存一份），先按内容去重：
        # 否则一个片段会占掉 Top-5 里好几个名额，还白烧重排额度
        seen_texts = set()
        unique_hits = []
        for item in hits:
            digest = hashlib.md5(item["text"].strip().encode("utf-8")).hexdigest()
            if digest in seen_texts:
                continue
            seen_texts.add(digest)
            unique_hits.append(item)
        hits = unique_hits

        rerank_ms = 0.0
        reranked = False   # 本次调用是否真的重排成功了（不能看 rerank_status，它是跨调用的实例状态）
        if use_rerank:
            head_n = int(self.cfg.get("rerank_top_n", len(hits)))
            pool = hits[:max(1, min(head_n, len(hits)))]
            tail = hits[len(pool):]
            t1 = time.time()
            order, scores, rerank_ok = self._rerank(query, [h["text"] for h in pool], len(pool))
            rerank_ms = (time.time() - t1) * 1000
            if rerank_ok:
                # 重排成功：重排分（0~1）和向量余弦（0~1）不是一个量纲，绝不能混在一个
                # sort 里比大小，否则没参与重排的尾巴会靠余弦分把重排结果整体压下去。
                # 这里走级联：重排看过的整体在前、没看过的整体在后，各自内部再排序。
                judged = []
                seen_idx = set()
                for pos, idx in enumerate(order):
                    item = pool[idx]
                    seen_idx.add(idx)
                    score = scores[pos] if pos < len(scores) else None
                    if score is not None:
                        item["rerank_score"] = round(float(score), 4)
                        item["final_score"] = float(score) * item["weight"]
                    judged.append(item)
                unjudged = [pool[i] for i in range(len(pool)) if i not in seen_idx]
                judged.sort(key=lambda x: x["final_score"], reverse=True)
                unjudged.sort(key=lambda x: x["final_score"], reverse=True)
                tail.sort(key=lambda x: x["final_score"], reverse=True)
                hits = judged + unjudged + tail
                reranked = True
            # 重排失败（降级）：所有条目都还带着原始向量分，落到下面统一全局排序

        if not reranked:
            # 没开重排 / 重排降级：final_score 在构造时已算成「相似度 × 来源权重」，全局排序即可
            hits.sort(key=lambda x: x["final_score"], reverse=True)

        hits = hits[:top_k]
        for out_rank, item in enumerate(hits, 1):
            item["rank"] = out_rank
            if not with_trace:
                for key in ("vector_rank", "vector_score", "rerank_score", "weight"):
                    item.pop(key, None)
        self.last_timing = {"embed_ms": round(embed_ms, 1), "rerank_ms": round(rerank_ms, 1)}
        return hits

    def search_texts(self, query, **kwargs):
        """只要文本，方便直接拼进 Prompt。"""
        return [h["text"] for h in self.search(query, **kwargs)]


_CACHE = {}


def get_retriever(db_path="./storage", collection_name="energy_storage_library",
                  embedder=None, config=None, use_cache=True):
    """进程内复用同一个检索器（Streamlit 反复渲染时不用重连数据库）。"""
    key = (
        os.path.abspath(db_path),
        collection_name,
        id(embedder) if embedder is not None else None,
        repr(sorted((config or {}).items())),
    )
    if use_cache and key in _CACHE:
        return _CACHE[key]
    retriever = SemanticRetriever(
        db_path=db_path, collection_name=collection_name, embedder=embedder, config=config
    )
    if use_cache:
        _CACHE[key] = retriever
    return retriever
