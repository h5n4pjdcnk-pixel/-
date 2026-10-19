# 【模块 B-1】选择向量模型：云端在线 / 本地离线统一封装
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
"""统一的向量模型封装 —— B「选择向量模型 / 将 Chunk 向量化」的实现落点。

两个后端共用一套接口，上层（ingest_to_chroma.py / retriever.py / app.py）不用关心是谁：

    embedder.embed_documents(list[str]) -> list[list[float]]   批量把文本变向量
    embedder.embed_query(str)           -> list[float]         把提问变向量
    embedder.dimension                  -> int                 向量维度
    embedder.backend                    -> str                 "cloud" / "local"

- cloud：走 config.EMBEDDING_CONFIG["model_name"] 指定的在线模型（硅基流动），
         中文语义最强，但要联网、要花额度。
- local：完全离线，字符 n-gram TF-IDF + TruncatedSVD（潜在语义分解 LSA）。
         零成本、断网可用，语义能力弱于云端大模型，只用于没 Key / 没网时
         把整条链路和评测跑通。
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config import EMBEDDING_CONFIG, LLM_CONFIG

LOCAL_MODEL_FILENAME = "local_embedder.joblib"


class EmbeddingError(RuntimeError):
    """向量化相关的可预期错误（Key 没填、模型文件缺失、维度不一致等）。"""


def local_model_path(db_path):
    return os.path.join(db_path, LOCAL_MODEL_FILENAME)


class CloudEmbedder:
    """硅基流动在线向量模型（默认后端）。"""

    backend = "cloud"

    def __init__(self, batch_size=None, max_retries=None, max_workers=None):
        from langchain_openai import OpenAIEmbeddings

        cfg = LLM_CONFIG["embedding"]
        self.model_name = cfg["model_name"]
        self.batch_size = int(batch_size or EMBEDDING_CONFIG.get("batch_size", 32))
        self.max_retries = int(max_retries or EMBEDDING_CONFIG.get("max_retries", 3))
        self.max_workers = int(max_workers or EMBEDDING_CONFIG.get("max_workers", 1)) or 1
        self._config_dimension = EMBEDDING_CONFIG.get("dimension") or None
        self._probed_dimension = None
        self._warned_dimension_mismatch = False
        self._client = OpenAIEmbeddings(
            openai_api_base=cfg["api_url"],
            openai_api_key=cfg["api_key"],
            model=self.model_name,
            # 关键：langchain 默认会先用 tiktoken 把文本切成 token 数组再发出去，
            # 那是 OpenAI 的私有扩展，硅基流动不认——实测同一句话拿到的向量和直连 API
            # 的余弦只有 0.31（等于把 token id 当文本喂给了模型），检索质量直接崩掉。
            check_embedding_ctx_length=False,
        )

    @property
    def dimension(self):
        """向量维度：以实测为准（换过模型后 config 里那个数字很容易忘改）。

        实测要花一次 API 调用，所以进程内只探一次；探不通才退回 config 里写的值。
        """
        if self._probed_dimension is None:
            try:
                self._probed_dimension = len(self._embed_batch(["维度探测"])[0])
            except Exception as exc:
                self._probed_dimension = self._config_dimension
                if self._probed_dimension:
                    print("⚠️ 向量维度探测失败（%s），先按 config 里的 %d 维处理。"
                          % (exc, self._probed_dimension))
                else:
                    raise EmbeddingError("探测向量维度失败，且 config 里没写 dimension：%s" % exc)
        if (self._config_dimension and self._config_dimension != self._probed_dimension
                and not self._warned_dimension_mismatch):
            self._warned_dimension_mismatch = True
            print("⚠️ EMBEDDING_CONFIG['dimension']=%d 与实测的 %d 维不一致，以实测为准（建议把 config 改过来）。"
                  % (self._config_dimension, self._probed_dimension))
        return int(self._probed_dimension)

    def _embed_batch(self, texts):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                return self._client.embed_documents(list(texts))
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise EmbeddingError("云端向量模型 %s 调用失败：%s" % (self.model_name, last_err))

    def embed_documents(self, texts):
        """分批向量化；批次之间按 max_workers 并发（纯网络 I/O，并发收益接近线性）。"""
        texts = list(texts)
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        if self.max_workers <= 1 or len(batches) <= 1:
            results = [self._embed_batch(batch) for batch in batches]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(self._embed_batch, batches))   # map 保序，且首个异常会抛出
        vectors = []
        for chunk in results:
            vectors.extend(chunk)
        return vectors

    def embed_query(self, text):
        return self._embed_batch([text])[0]


class LocalEmbedder:
    """本地离线向量模型：字符 n-gram TF-IDF + TruncatedSVD（潜在语义分解）。

    需要先在语料上 fit() 一次（ingest 阶段自动完成并落盘），之后可反复 embed。
    设计成「可持久化」是为了建库和查询用的是同一套空间——否则向量没有可比性。
    """

    backend = "local"

    def __init__(self, dimension=None, ngram_range=None, min_df=None, drop_top=None, verbose=True):
        self.requested_dimension = int(dimension or EMBEDDING_CONFIG.get("local_dimension", 256))
        self.dimension = self.requested_dimension      # fit 之后会变成实际输出维度
        self.ngram_range = tuple(ngram_range or EMBEDDING_CONFIG.get("local_ngram_range", [1, 3]))
        self.min_df = int(min_df or EMBEDDING_CONFIG.get("local_min_df", 2))
        self.drop_top = int(EMBEDDING_CONFIG.get("local_drop_top", 2) if drop_top is None else drop_top)
        self.params = None        # 训练时的参数指纹，用来判断 config 改了没有
        self.verbose = verbose
        self.vectorizer = None
        self.reducer = None
        self.normalizer = None

    def fit(self, texts):
        """在整份语料上学习词表与语义子空间（只做一次）。"""
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import Normalizer
        from sklearn.random_projection import SparseRandomProjection

        corpus = [t if t and t.strip() else "空" for t in texts]
        min_df = self.min_df if len(corpus) >= 20 else 1
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            dtype=np.float32,
        )
        matrix = self.vectorizer.fit_transform(corpus)
        n_samples, n_features = matrix.shape

        # TruncatedSVD 要求 n_components < min(样本数, 特征数)；语料太小时退化为随机投影
        target = min(self.dimension, max(2, min(n_samples, n_features) - 1))
        if target >= 8:
            self.reducer = TruncatedSVD(n_components=target, random_state=42)
            self.fit_method = "TruncatedSVD(LSA)"
        else:
            self.reducer = SparseRandomProjection(
                n_components=max(2, min(self.dimension, n_features)),
                random_state=42,
                dense_output=True,
            )
            self.fit_method = "SparseRandomProjection"
        self.reducer.fit(matrix)
        self.normalizer = Normalizer(copy=False)
        self.dimension = int(self.reducer.components_.shape[0])
        self.params = self.snapshot_params()
        if self.verbose:
            print("   🧠 本地向量模型已训练：%s，词表 %d 维 → 输出 %d 维（语料 %d 条）"
                  % (self.fit_method, n_features, self.dimension, n_samples))
        return self

    def snapshot_params(self):
        """这套向量空间是用什么参数训练出来的——存进模型文件，供别人比对。"""
        return {
            "local_dimension": self.requested_dimension,
            "local_ngram_range": list(self.ngram_range),
            "local_min_df": self.min_df,
            "local_drop_top": self.drop_top,
        }

    def _require_fitted(self):
        if self.vectorizer is None or self.reducer is None:
            raise EmbeddingError(
                "本地向量模型还没训练。先运行 `python ingest_to_chroma.py` 建库（会自动 fit 并落盘），"
                "或显式调用 fit(语料)。"
            )

    def _transform(self, texts):
        self._require_fitted()
        clean = [t if t and t.strip() else "空" for t in texts]
        sparse = self.vectorizer.transform(clean)
        dense = self.reducer.transform(sparse).astype(np.float32)
        # 丢掉最靠前的若干主成分（all-but-the-top）：这些维度主要承载「整份语料都差不多」
        # 的共性主题，留着会让所有 Chunk 彼此都很像，反而压掉真正的区分度
        if self.drop_top > 0:
            dense[:, :min(self.drop_top, dense.shape[1])] = 0.0
        # 统一做 L2 归一化：这样内积就等于余弦相似度，distances 可直接当相似度看
        return self.normalizer.transform(dense).tolist()

    def embed_documents(self, texts):
        return self._transform(list(texts))

    def embed_query(self, text):
        return self._transform([text])[0]

    def save(self, path):
        import joblib

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "reducer": self.reducer,
                "dimension": self.dimension,
                "ngram_range": list(self.ngram_range),
                "min_df": self.min_df,
                "drop_top": self.drop_top,
                "params": self.params,
                "fit_method": getattr(self, "fit_method", ""),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path, verbose=False):
        import joblib
        from sklearn.preprocessing import Normalizer

        blob = joblib.load(path)
        inst = cls(
            dimension=blob["dimension"],
            ngram_range=blob["ngram_range"],
            min_df=blob["min_df"],
            drop_top=blob.get("drop_top", 0),
            verbose=verbose,
        )
        inst.vectorizer = blob["vectorizer"]
        inst.reducer = blob["reducer"]
        inst.fit_method = blob.get("fit_method", "")
        inst.params = blob.get("params")
        if inst.params:
            inst.requested_dimension = int(inst.params.get("local_dimension", inst.requested_dimension))
        inst.normalizer = Normalizer(copy=False)
        return inst


def param_diffs(embedder):
    """把本地模型的训练参数和当前 config 比一比，返回不一致项的文字说明。

    为什么要比：改了 config 里的 local_* 参数但没重建向量库时，模型文件和库都还是旧
    空间的，代码却以为变了——轻则参数白改，重则查询向量和库里的向量不在一个空间。
    """
    params = getattr(embedder, "params", None)
    if not params:
        return ["模型文件里没有参数指纹（旧版本生成的），建议 python ingest_to_chroma.py --retrain"]
    want = {
        "local_dimension": int(EMBEDDING_CONFIG.get("local_dimension", 256)),
        "local_ngram_range": list(EMBEDDING_CONFIG.get("local_ngram_range", [1, 3])),
        "local_min_df": int(EMBEDDING_CONFIG.get("local_min_df", 2)),
        "local_drop_top": int(EMBEDDING_CONFIG.get("local_drop_top", 2)),
    }
    return [
        "%s: 建库时 %s → 现在 %s" % (key, params.get(key), value)
        for key, value in want.items() if params.get(key) != value
    ]


def get_embedder(backend=None, db_path="./storage", need_fit=False):
    """按配置返回向量模型实例。

    backend   None 用 config.EMBEDDING_CONFIG["backend"]；也可显式传 "cloud"/"local"
    db_path   local 后端读取已落盘模型文件的目录
    need_fit  True 时（建库流程）允许返回未训练的 LocalEmbedder 供后续 fit
    """
    backend = str(backend or EMBEDDING_CONFIG.get("backend", "cloud")).lower()
    if backend == "cloud":
        return CloudEmbedder()
    if backend == "local":
        path = local_model_path(db_path)
        if os.path.exists(path):
            model = LocalEmbedder.load(path)
            diffs = param_diffs(model)
            if diffs:
                # 查询侧必须继续用「和库里向量同一个空间」的旧模型，不能按新 config 重建，
                # 所以这里只警告不报错，改完参数记得重建库。
                print("⚠️ 本地向量模型是用旧参数训练的，与当前 config 不一致：")
                for item in diffs:
                    print("   · %s" % item)
                print("   向量库仍是旧空间，参数改动要生效请重建：python ingest_to_chroma.py --retrain --rebuild")
            return model
        if need_fit:
            return LocalEmbedder()
        raise EmbeddingError(
            "本地向量模型文件不存在：%s\n"
            "请先用 local 后端建一次库：python ingest_to_chroma.py --backend local" % path
        )
    raise EmbeddingError("未知的向量后端：%s（可选 cloud / local）" % backend)
