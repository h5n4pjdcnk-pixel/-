# 【模块 C-1】确定实体与关系类型 + 从 Chunk 抽取三元组
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
import json, re
from langchain_core.messages import SystemMessage, HumanMessage

# 大模型统一从 公共模块/llm_model.py 来，这里只做再导出，老的 `from extractor import build_model` 照旧可用
from llm_model import LLMConfigError, build_model  # noqa: F401

def extract_triples(llm, text):
    system_prompt = """
    你是一名储能运维、电化学与安全合规专家。请从输入的文本中提取结构化的知识图谱三元组。

    【提取规则与分类】
    请提取文本中的逻辑链路，并将其转化为独立的“主体-关系-客体”三元组。对于每个三元组，必须准确判断其所属类别，并提取对应的 properties（边缘属性），若无对应信息严格设为 null：

    1. 类别："运维与故障" (以故障现象或报警为中心)
    - properties.condition：具体的触发条件、阈值或环境因素（如: 温度>75℃）。
    - properties.severity：报警严重度或故障影响（如: 致命/降额）。
    - properties.solution：具体的排查步骤、维修动作或解决策略。
    - properties.requirement：null。

    2. 类别："电化学机理" (以宏观现象或微观反应为中心)
    - properties.condition：微观电化学反应机制、催化条件或前置诱发因素。
    - properties.severity：机理衰减或危害程度（如: 电池容量不可逆损失）。
    - properties.solution：延缓衰减、抑制反应或热失控的预防措施。
    - properties.requirement：null。

    3. 类别："安全与合规" (以安全标准、消防预案、系统架构为中心)
    - properties.condition：适用场景、前提要求或特定部署环境。
    - properties.severity：违规风险级别。
    - properties.solution：null。
    - properties.requirement：具体的合规指标、国家标准参数或强制执行动作。

    【关系词约束】
    请尽量使用标准化、简练的关系词，候选库：[导致、属于、包含、触发、预防、参考标准、解决方式、依赖于、诱发]。

    【严格输出约束】(非常重要)
    1. 必须且只能输出合法的 JSON 数组，不要包含任何解释性文字。
    2. 不要输出 Markdown 标记（如 ```json 和 ```），直接输出中括号 [ 开始的纯文本。
    3. 实体(subject/object)必须是极简的专业名词（如“BMS”、“七氟丙烷”、“枝晶生长”）。
    4. 如果文本中未能明确提及某个 property 的值，请将其设为 null，切勿自行捏造。

    【JSON 数据结构模板】
    [
    {
        "subject": "头实体 (如: PCS过温告警)",
        "relation": "关系词",
        "object": "尾实体 (如: 防尘网堵塞)",
        "category": "运维与故障 | 电化学机理 | 安全与合规",
        "properties": {
        "condition": "...",
        "severity": "...",
        "solution": "...",
        "requirement": "..."
        }
    }
    ]
    """
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]
    
    try:
        response_text = llm.invoke(messages).content
        
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # 提取第一个 [ 到最后一个 ] 之间的内容
        match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
        if not match: 
            return []
        
        raw_json = match.group()
        # 预清洗：处理一些常见的 JSON 语法错误（如末尾多出的逗号）
        raw_json = re.sub(r',\s*\]', ']', raw_json)
        
        # 解析为 Python 字典列表
        parsed_data = json.loads(raw_json)
        
        validated_triplets = []
        if isinstance(parsed_data, list):
            for item in parsed_data:
                if isinstance(item, dict) and all(k in item for k in ("subject", "relation", "object")):
                    
                    # 防御：如果大模型漏掉了 category 或 properties，手动补齐，防止后续落库时报错
                    if "category" not in item:
                        item["category"] = "未分类"
                    if "properties" not in item:
                        item["properties"] = {"condition": None, "severity": None}
                    
                    # 只有格式完美的数据才会被采纳
                    validated_triplets.append(item)
                    
        return validated_triplets

    # 细化报错信息，方便你调试
    except json.JSONDecodeError as e:
        print(f"      ❌ JSON解析崩溃: {e}")
        # 如果解析失败，打印出模型原本回复的前50个字符看看是什么鬼样子
        print(f"      📝 模型原文片段: {response_text[:50]}...") 
        return []
    except Exception as e:
        print(f"      ❌ 抽取过程发生未知错误: {e}")
        return []