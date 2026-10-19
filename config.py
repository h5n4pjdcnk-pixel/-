# 【模块 S-1】全局配置：模型 Key / Neo4j / 切分参数 / OCR 参数
import os
import sys

# Windows 控制台默认编码是 GBK，脚本里大量 emoji 会抛 UnicodeEncodeError。
# 这里做一次兜底：无法编码的字符降级为 ?，保证中文正常、程序不崩。
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# 强制绕过代理
os.environ['NO_PROXY'] = '127.0.0.1,localhost'

def _load_env_file(path=None):
    """载入同目录下的 .env（十行极简实现，不额外引入 python-dotenv 依赖）。

    优先级：已有的系统环境变量 > .env 文件里的值。
    .env 已被 .gitignore 忽略，所以 Key 不会跟着代码提交到 GitHub。
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                name = name.strip()
                value = value.strip().strip('"').strip("'")
                if name and name not in os.environ:
                    os.environ[name] = value
    except Exception:
        pass


_load_env_file()

# ==========================================================
#  【API Key 从环境变量读 —— 不写死在代码里】
#
#  只需一家平台：硅基流动 —— 对话/三元组抽取、向量化、图片理解、重排全部走它，
#      注册：https://cloud.siliconflow.cn  →  API 密钥
#      一个 Key 搞定所有环节，不再需要 DeepSeek 官方直连。
#
#  本地怎么配：把 .env.example 复制成 .env，填上自己的 Key，
#  config.py 启动时会自动读它 —— 不用改代码，也不会误传上 GitHub。
# ==========================================================

SILICONFLOW_BASE = os.getenv("SILICONFLOW_BASE", "https://api.siliconflow.cn/v1")
SILICONFLOW_KEY  = os.getenv("SILICONFLOW_KEY", "")

# 大模型配置（上层代码用 LLM_CONFIG["chat"] 这种键名来取）
LLM_CONFIG = {
    # 对话 + 三元组抽取 + 段落修复
    # 可选模型：deepseek-ai/DeepSeek-V3.2（非推理模型，快且便宜，推荐）
    #           deepseek-ai/DeepSeek-V4-Pro（能力更强，价格更高）
    #           deepseek-ai/DeepSeek-R1（推理强，但会先输出思考过程，慢且贵）
    "chat": {
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "api_key": SILICONFLOW_KEY,
        "api_url": SILICONFLOW_BASE,
    },
    # 图片理解（vision_extractor.py，可选步骤）
    "vision": {
        "model_name": "Qwen/Qwen3-VL-32B-Instruct",
        "api_key": SILICONFLOW_KEY,
        "api_url": SILICONFLOW_BASE,
    },
    # 向量化（ingest_to_chroma.py / app.py / knowledge_fusion.py）：必需
    # 实际选用的模型以 EMBEDDING_CONFIG["model_name"] 为准（见下方【B-1 选择向量模型】）
    "embedding": {
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "api_key": SILICONFLOW_KEY,
        "api_url": SILICONFLOW_BASE,
    },
    # 检索重排（app.py 的 api_rerank，可选，缺了会自动降级取前 3 条）
    "rerank": {
        "model_name": "BAAI/bge-reranker-v2-m3",
        "api_key": SILICONFLOW_KEY,
        "api_url": SILICONFLOW_BASE + "/rerank",
    },
}

# 上面四组现在都指向硅基流动，日常只需维护 SILICONFLOW_KEY 一个 Key。
# 想换回 DeepSeek 官方直连的话，把 "chat" 那组改成：
#   "model_name": "deepseek-chat",
#   "api_key": "<官方 Key>",
#   "api_url": "https://api.deepseek.com/v1",

# ==========================================================
#  【B-1 选择向量模型】—— 把文本变成向量的那个模型
#
#  两种后端，改 "backend" 切换：
#    cloud  硅基流动在线向量模型（默认）。中文语义强，但要联网、要花额度。
#    local  本地离线兜底模型（字符 n-gram TF-IDF + 潜在语义分解 LSA），
#           零成本、断网可用；语义能力弱于云端大模型，只用于没 Key / 没网时
#           把「向量化 → 建库 → 检索 → 评测」整条链路跑通。
#
#  云端候选模型（以硅基流动控制台实际列表为准，改 model_name 即切换）：
#    Qwen/Qwen3-Embedding-8B               默认，中文/多语言检索第一梯队，4096 维，最贵
#    Qwen/Qwen3-Embedding-4B               效果接近 8B、价格更低，性价比首选
#    Qwen/Qwen3-Embedding-0.6B             最便宜最快，适合先跑通再升级
#    BAAI/bge-m3                           经典多语言检索模型，1024 维，生态成熟
#    netease-youdao/bce-embedding-base_v1  中文检索专精，768 维
#  注意：换模型 = 换向量维度 = 旧库作废，必须 python ingest_to_chroma.py --rebuild 重建。
# ==========================================================
EMBEDDING_CONFIG = {
    # 默认走云端；断网 / 想省额度时改成 "local" 用本地离线模型（效果弱一些）。
    # 换后端或换模型名后，向量空间会变，必须重建：python ingest_to_chroma.py --rebuild
    "backend": "cloud",             # cloud | local
    "model_name": "Qwen/Qwen3-Embedding-8B",
    "dimension": 4096,              # Qwen3-Embedding-8B 的输出维度，建库前用它做一致性校验
    "batch_size": 32,               # 云端每批提交多少条文本
    "max_workers": 10,              # 云端并发线程数
    "max_retries": 3,               # 单批失败重试次数
    # ---- local 后端参数 ----
    "local_dimension": 256,         # 本地模型最终向量维度
    "local_ngram_range": [1, 3],    # 字符 n-gram 范围
    "local_min_df": 2,              # 词表最小文档频率，过滤偶发片段
    "local_drop_top": 2,            # 丢弃最靠前的若干主成分（"all-but-the-top"），提纯区分度
}

# ==========================================================
#  【B-4 语义检索】—— 检索阶段参数
# ==========================================================
RETRIEVAL_CONFIG = {
    "candidate_k": 60,              # 向量粗召回条数
    "top_k": 5,                     # 重排后最终交给大模型的条数
    "use_rerank": True,             # 是否调用 BAAI/bge-reranker-v2-m3 重排
    "rerank_top_n": 20,             # 只对前 N 条做重排，省额度
    "source_weights": {             # 命中所属资料类别（路径前缀）时的加权，权威资料优先
        "标准规范": 1.5,
        "运维规程": 1.3,
        "白皮书": 1.5,
    },
}

# LLM_CONFIG["embedding"]["model_name"] 统一从 EMBEDDING_CONFIG 取值，避免两处写不一致
LLM_CONFIG["embedding"]["model_name"] = EMBEDDING_CONFIG["model_name"]

# Neo4j 图数据库（本机 Community 版，双击 启动Neo4j.bat 启动）
NEO4J_CONF = {
    "uri": os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", ""),
}

# PDF处理配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ==========================================================
#  扫描件 OCR 配置（ocr_utils.py 读取）
#
#  默认只对「文字层不足」的页触发 OCR，正常电子版 PDF 完全不受影响。
#  三种引擎任选，engine="auto" 时按 rapidocr -> tesseract -> vlm 挑第一个可用的：
#    rapidocr   本地离线，推荐。装法：pip install rapidocr-onnxruntime
#    tesseract  本地离线，需另装 Tesseract 可执行文件（含 chi_sim 语言包）
#    vlm        云端，复用 LLM_CONFIG["vision"] 的 Qwen3-VL，零安装但按页计费
# ==========================================================
OCR_CONFIG = {
    "engine": "auto",            # auto | rapidocr | tesseract | vlm | off
    "min_text_chars": 80,        # 单页有效字符低于此值就判定为扫描页，去走 OCR
    "dpi": 200,                  # 渲染分辨率，越高越准也越慢（200~300 够用）
    "tesseract_lang": "chi_sim+eng",
    "tessdata_dir": None,        # Tesseract 语言包目录，None 表示用系统默认
    "vlm_max_side": 1600,        # 走 vlm 时图片长边上限，控制 token 消耗
    "vlm_max_pages": 50,         # 走 vlm 时单次运行最多 OCR 多少页，防止误烧额度
}


def missing_keys(modes=None):
    """检查 API Key 是否还是占位符（没填就跑不通）。

    modes 传 None 时检查全部四组配置；也可传 ["chat"] 只检查某几组。
    返回未填写项的说明列表，空列表代表都填好了。
    """
    targets = list(LLM_CONFIG.keys()) if modes is None else modes
    missing = []
    for name in targets:
        cfg = LLM_CONFIG.get(name) or {}
        key = str(cfg.get("api_key", "")).strip()
        # 占位符形如 "sk-在此填你的Key"，光看 sk- 前缀区分不出来
        if not key.startswith("sk-") or "在这里填" in key or len(key) < 20:
            missing.append('LLM_CONFIG["%s"]["api_key"]' % name)
    return missing
