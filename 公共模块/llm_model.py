# 【模块 S-2】公共模块 · 大模型统一工厂
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
"""统一的大模型工厂 —— 全项目只在这里 new ChatOpenAI。

`extractor.build_model` 就是本函数的再导出，所以老的 `from extractor import build_model`
不用改。统一在一处的好处：
  - 超时/重试参数一致：以前没有设 timeout，抽取几百个 Chunk 时一旦某次请求挂住，
    整个 main.py 会一直卡在那里；现在单次请求 120 秒、失败自动重试 3 次。
  - 温度一致：抽取和问答都用 0.1，避免有的地方发散、有的地方死板。
  - 换模型/换厂商只改 config.py 一处。
"""
from langchain_openai import ChatOpenAI

from config import LLM_CONFIG

DEFAULT_TIMEOUT = 120       # 单次请求超时（秒）
DEFAULT_MAX_RETRIES = 3     # 网络抖动 / 限流时的自动重试次数


class LLMConfigError(RuntimeError):
    """大模型配置有问题（分组名写错、Key 没填等）。"""


def build_model(mode="chat", temperature=0.1,
                timeout=DEFAULT_TIMEOUT, max_retries=DEFAULT_MAX_RETRIES):
    """按 config.LLM_CONFIG[mode] 建一个大模型客户端。

    mode: chat / vision /（embedding、rerank 不走这里，那两个是向量与重排接口）
    """
    cfg = LLM_CONFIG.get(mode)
    if not cfg:
        raise LLMConfigError(
            "config.py 的 LLM_CONFIG 里没有 %r 这组配置，可选：%s"
            % (mode, "、".join(LLM_CONFIG.keys()))
        )
    return ChatOpenAI(
        model=cfg["model_name"],
        openai_api_key=cfg["api_key"],
        openai_api_base=cfg["api_url"],
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
