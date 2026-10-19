# 【模块 S-7】启动前环境自检
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
# -*- coding: utf-8 -*-
"""启动前自检：一条命令看清到底还缺什么。

用法（必须在本项目目录下执行）：
    python check_env.py
"""
import io
import os
import sys

# Windows 控制台默认 GBK，中文/符号可能编码失败，这里兜底
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

OK, BAD, WARN = "[OK] ", "[!!] ", "[??] "

DEPENDENCIES = [
    ("streamlit", "streamlit"),
    ("chromadb", "chromadb"),
    ("neo4j", "neo4j"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_community", "langchain-community"),
    ("langchain_text_splitters", "langchain-text-splitters"),
    ("fitz", "pymupdf"),
    ("pypdf", "pypdf"),
    ("opencc", "opencc"),
    ("pyvis", "pyvis"),
    ("networkx", "networkx"),
    ("numpy", "numpy"),
    ("tqdm", "tqdm"),
    ("requests", "requests"),
    ("PIL", "pillow"),
    ("tiktoken", "tiktoken"),
    ("matplotlib", "matplotlib"),
]


def main():
    blockers = []

    print("=" * 64)
    print(" 储能运维知识图谱 + RAG 系统 —— 启动前自检")
    print("=" * 64)

    # ---------- 1. 运行位置 ----------
    print("\n[1] 运行位置")
    if not os.path.exists("config.py"):
        print(BAD + "当前目录不是项目目录：%s" % os.getcwd())
        print(BAD + "请先 cd 到项目根目录再运行，否则 ./data ./storage 会找错地方。")
        return
    print(OK + "项目目录：%s" % os.getcwd())
    print(OK + "Python %s" % sys.version.split()[0])

    # ---------- 2. 依赖包 ----------
    print("\n[2] 依赖包")
    missing_mods = []
    for module, name in DEPENDENCIES:
        try:
            __import__(module)
            print(OK + name)
        except Exception:
            print(BAD + "%s 缺失" % name)
            missing_mods.append(name)
    if missing_mods:
        blockers.append("依赖包缺失：%s（执行 pip install -r requirements.txt）" % "、".join(missing_mods))

    # ---------- 3. 大模型 API Key ----------
    print("\n[3] 大模型 API Key")
    try:
        from config import missing_keys
        miss = missing_keys()
    except Exception as exc:
        print(BAD + "无法读取 config.py：%s" % exc)
        miss = []
        blockers.append("config.py 读取失败")
    if miss:
        for item in miss:
            print(BAD + "%s 还是占位符，没填" % item)
        print("     四组配置（chat/vision/embedding/rerank）都走硅基流动，")
        print("     只需在 .env 里填好 SILICONFLOW_KEY 一个 Key（照着 .env.example 抄）。")
        blockers.append("大模型 API Key 未填写")
    else:
        print(OK + "四组 Key 都已填写")

    # ---------- 4. Neo4j ----------
    print("\n[4] Neo4j 图数据库")
    if not os.path.exists(r"C:\jdk-17.0.2"):
        print(WARN + r"C:\jdk-17.0.2 不存在，启动Neo4j.bat 需要 JAVA_HOME 指向它")
    if not os.path.exists(r"C:\neo4j-community-5.26.30"):
        print(WARN + r"C:\neo4j-community-5.26.30 不存在")
    try:
        from neo4j_storage import Neo4jClient
        db = Neo4jClient()
        db.verify()
        db.close()
        print(OK + "连接成功（Bolt 端口通、密码正确）")
    except Exception as exc:
        print(BAD + "连不上：%s" % exc)
        print("     请先双击运行 启动Neo4j.bat，并保持那个黑窗口开着。")
        print("     若提示认证失败，检查 config.py 里 NEO4J_CONF 的密码。")
        blockers.append("Neo4j 未启动或密码不对")

    # ---------- 5. Chroma 向量库 ----------
    print("\n[5] Chroma 向量库")
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./storage")
        names = [c.name for c in client.list_collections()]
        print(OK + "storage/ 可读，现有集合：%s" % (names or "无"))
        target = None
        for c in client.list_collections():
            if c.name == "energy_storage_library":
                target = c
        if target is None:
            print(WARN + "还没有 energy_storage_library，跑一次 ingest_to_chroma.py 会自动建。")
        elif target.count() == 0:
            print(WARN + "energy_storage_library 是空的，需要跑 ingest_to_chroma.py 灌数据。")
        else:
            print(OK + "energy_storage_library 已有 %d 条向量" % target.count())
            _report_embedding(target, blockers)
    except Exception as exc:
        print(BAD + "读取 ./storage 失败：%s" % exc)

    # ---------- 6. PDF 素材 ----------
    print("\n[6] PDF 素材（./data）")
    chunks_file = os.path.join("Chunk切分结果", "chunks.jsonl")
    if os.path.exists(chunks_file):
        with open(chunks_file, encoding="utf-8") as fh:
            chunk_n = sum(1 for line in fh if line.strip())
        print(OK + "Chunk切分结果/chunks.jsonl 存在，共 %d 条 Chunk，可直接给 ingest_to_chroma.py 用。" % chunk_n)
    else:
        print(WARN + "找不到 Chunk切分结果/chunks.jsonl，向量入库只能走 --source pdf。")

    pdfs = [f for f in os.listdir("./data") if f.lower().endswith(".pdf")] if os.path.isdir("./data") else []
    if not pdfs:
        print(WARN + "./data 里没有 PDF，main.py 和 ingest_to_chroma.py 都会空跑。")
        print("     注意：document_loader.py 会按储能关键词过滤，非储能领域文档会被整体丢掉。")
    else:
        print(OK + "发现 %d 个 PDF：%s" % (len(pdfs), "、".join(pdfs[:5])))

    # ---------- 7. OCR（扫描件兜底） ----------
    print("\n[7] 扫描件 OCR 兜底")
    try:
        from ocr_utils import TIPS, available_engines, resolve_engine
        avail = available_engines()
        for name in ("rapidocr", "tesseract", "vlm"):
            if avail.get(name):
                print(OK + "%s：可用" % name)
            else:
                print(WARN + "%s：不可用 —— %s" % (name, TIPS[name]))
        chosen = resolve_engine()
        if chosen:
            print(OK + "auto 模式下会使用：%s" % chosen)
        else:
            print(WARN + "没有任何 OCR 引擎可用：扫描件仍会抽出 0 条三元组。")
            print("     最省事的办法：pip install rapidocr-onnxruntime")
            print("     或者把 config.py 里 OCR_CONFIG[\"engine\"] 设成 \"vlm\"（走云端，要花额度）。")
    except Exception as exc:
        print(BAD + "OCR 模块加载失败：%s" % exc)

    # ---------- 结论 ----------
    print("\n" + "=" * 64)
    if blockers:
        print("结论：还不能跑，先解决下面 %d 项：" % len(blockers))
        for i, item in enumerate(blockers, 1):
            print("  %d) %s" % (i, item))
    else:
        print("结论：环境就绪，按顺序执行：")
        print("  python main.py --input_dir ./data")
        print("  python ingest_to_chroma.py")
        print("  python -m streamlit run app.py")
    print("=" * 64)


def _report_embedding(collection, blockers):
    """报告当前用哪个向量模型，并检查它和库里已有向量的维度是否对得上。"""
    try:
        from config import EMBEDDING_CONFIG
        from embedding_model import get_embedder

        backend = EMBEDDING_CONFIG.get("backend", "cloud")
        if backend == "cloud":
            print("     向量后端：cloud（%s）" % EMBEDDING_CONFIG.get("model_name", "?"))
        else:
            print("     向量后端：local（本地离线字符 n-gram + LSA）")
        meta = getattr(collection, "metadata", None) or {}
        if meta.get("embedding_model"):
            print("     建库信息：backend=%s  model=%s  dim=%s  built_at=%s"
                  % (meta.get("embedding_backend"), meta.get("embedding_model"),
                     meta.get("embedding_dim"), meta.get("built_at")))
        got = collection.get(limit=1, include=["embeddings"])
        emb = got.get("embeddings")
        lib_dim = len(emb[0]) if emb is not None and len(emb) else None
        want_dim = get_embedder(db_path="./storage").dimension
        if lib_dim and want_dim and lib_dim != want_dim:
            print(BAD + "维度不一致：库里是 %d 维、当前向量模型是 %d 维。" % (lib_dim, want_dim))
            print("     请重建向量库：python ingest_to_chroma.py --rebuild")
            blockers.append("向量库维度与向量模型不一致")
        elif lib_dim:
            print(OK + "向量维度一致（%d 维）" % lib_dim)
    except Exception as exc:
        print(WARN + "向量模型/维度检查跳过：%s" % exc)


if __name__ == "__main__":
    main()
