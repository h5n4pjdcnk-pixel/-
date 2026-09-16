from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import json
import re
import os

# 强制绕过代理，解决之前遇到的 502 问题
os.environ['NO_PROXY'] = '127.0.0.1,localhost'

# 1. 大模型配置字典
llm_config = {

}

def build_model(mode=""):
    config = llm_config[mode]
    LLM = ChatOpenAI(
        model=config["model_name"],
        openai_api_key=config["api_key"],
        openai_api_base=config["api_url"],
        max_retries=2,
        temperature=0.1
    )
    return LLM

def stream_invoke(llm_model, messages):
    full_response = ""
    results = llm_model.stream(messages)
    for chunk in results:
        print(chunk.content, end="", flush=True)
        full_response += chunk.content
    return full_response

# ==========================================
# 2. 核心：三元组提取逻辑 
# ==========================================
def extract_triples(llm_model, text):
    system_prompt = """你是专业储能知识三元组提取器。
    1. 提取(主体, 关系, 客体)三元组。
    2. 为每个三元组增加属性字段"properties"，包含"condition"(触发条件)和"severity"(严重程度)。
    3. 必须用JSON数组格式返回。
    输出格式示例：[{"subject": "BMS", "relation": "触发", "object": "报警", "properties": {"condition": "温度>60", "severity": "高"}}]
    不要输出 ```json 代码块，仅输出纯内容。
    """
    
    user_input = f"提取三元组及详细属性：\n{text}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
    
    print("\n[AI 正在流式生成响应...]")
    full_response = stream_invoke(llm_model, messages).strip()
    print("\n\n[接收完成，开始解析...]")
    
    try:
        clean_response = re.sub(r'```json\s*|\s*```', '', full_response)
        return json.loads(clean_response)
    except Exception as e:
        print(f"解析失败: {e}")
        json_match = re.search(r'\[.*\]', full_response, re.DOTALL)
        return json.loads(json_match.group()) if json_match else []

# ==========================================
# 3. 知识图谱构建与可视化 
# ==========================================
def build_knowledge_graph(triples):
    if not triples: return None
    entities = set()
    for t in triples:
        entities.add(t["subject"])
        entities.add(t["object"])
    relations = [{"source": t["subject"], "target": t["object"], "type": t["relation"], "props": t.get("properties", {})} for t in triples]
    return {"entities": [{"id": e, "name": e} for e in entities], "relations": relations}

def visualize_knowledge_graph(graph, output_file="energy_storage_kg.html"):
    if not graph: return None
    abs_path = os.path.abspath(output_file)
    
    net = Network(directed=True, height="700px", width="100%", bgcolor="#f5f5f5", font_color="black", notebook=False)
    for entity in graph["entities"]:
        net.add_node(entity["id"], label=entity["name"], color="#4CAF50")
    for relation in graph["relations"]:
        p = relation["props"]
        hover_text = f"关系: {relation['type']}\n条件: {p.get('condition')}\n严重度: {p.get('severity')}"
        net.add_edge(relation["source"], relation["target"], label=relation["type"], title=hover_text, color="#FF9800")
    
    net.set_options('{"nodes": {"size": 30, "font": {"size": 14}}, "edges": {"font": {"size": 12}, "length": 200}}')
    
    try:
        net.write_html(output_file, open_browser=False)
        print(f"✅ 可视化 HTML 已保存至: {abs_path}")
        print(f"💡 提示：你可以直接去文件夹双击它，或用浏览器打开。")
    except Exception as e:
        print(f"⚠️ 保存 HTML 失败: {e}")

# ==========================================
# 4. Neo4j 图数据库入库逻辑 
# ==========================================
class Neo4jStorage:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def insert_triplets(self, triples):
        if not triples: return
        print("\n[正在将带属性的数据写入 Neo4j...]")
        with self.driver.session() as session:
            for triple in triples:
                s, r, o = triple.get("subject"), triple.get("relation"), triple.get("object")
                p = triple.get("properties", {})
                if not all([s, r, o]): continue
                
                cypher = f"""
                MERGE (s:Entity {{name: $s}}) 
                MERGE (o:Entity {{name: $o}}) 
                MERGE (s)-[rel:`{r}`]->(o)
                SET rel.condition = $cond, rel.severity = $sev
                """
                session.run(cypher, s=s, o=o, cond=p.get("condition"), sev=p.get("severity"))
                print(f"  -> 已入库并关联属性: ({s})-[{r}]->({o})")

# ==========================================
# 5. 主控流程整合 
# ==========================================
def process_text_to_graph(text, mode="local_llama3", neo4j_uri="bolt://localhost:7687", neo4j_pwd="your_password"):
    print("="*50)
    print(f"🔋 启动知识抽取引擎 (当前模型: {mode})")
    print("="*50)
    
    llm = build_model(mode=mode)
    triples = extract_triples(llm, text)
    
    if not triples:
        print("❌ 提取失败。")
        return
        
    graph_data = build_knowledge_graph(triples)
    visualize_knowledge_graph(graph_data)
    
    try:
        neo_db = Neo4jStorage(neo4j_uri, "neo4j", neo4j_pwd)
        neo_db.insert_triplets(triples)
        neo_db.close()
    except Exception as e:
        print(f"\n⚠️ Neo4j 入库失败: {e}")

if __name__ == "__main__":
    sample_text = """
    当电池单体电压低于2.5V时，BMS会触发“深度放电”报警，并上报E10故障。该报警严重程度极高。
    """
    process_text_to_graph(
        text=sample_text,
        mode="",
        neo4j_pwd=""
    )