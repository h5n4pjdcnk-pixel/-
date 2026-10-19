# 【模块 C-3】建立 Neo4j 图谱：写入 / 查询 / 可视化
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
import os
from neo4j import GraphDatabase
from pyvis.network import Network
from config import NEO4J_CONF
import json  

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_CONF["uri"], auth=(NEO4J_CONF["user"], NEO4J_CONF["password"]))
    def close(self): self.driver.close()
    def verify(self):
        """主动探活：连不上立刻抛异常，避免后面所有操作静默失败。"""
        self.driver.verify_connectivity()

    def insert_triples(self, triples):
        """写入三元组，返回真正写进图库的条数。

        单条失败（例如关系名里带反引号）不会中断整批，但结束时汇总打印，
        避免出现「界面显示成功、其实一条都没入库」的情况。
        """
        inserted, failed = 0, 0
        first_errors = []
        with self.driver.session() as s:
            for t in triples:
                sub = str(t.get('subject', '')).strip()
                obj = str(t.get('object', '')).strip()
                rel = str(t.get('relation', '')).strip() # 提取关系

                # 校验关系和实体名称
                if not rel or rel == "": rel = "相关"
                if not sub or not obj: continue

                props = t.get('properties', {}) or {}
                # 模型没给该字段时是 null，不能 str(None) 写成字符串 "None"
                cond = props.get('condition') or '常规'
                seve = props.get('severity') or '普通'

                cypher = (
                    f"MERGE (a:Entity {{name: $s}}) "
                    f"MERGE (b:Entity {{name: $o}}) "
                    f"MERGE (a)-[r:`{rel}`]->(b) "
                    f"SET r.condition=$c, r.severity=$v"
                )
                try:
                    s.run(cypher, s=sub, o=obj, c=cond, v=seve)
                    inserted += 1
                except Exception as e:
                    failed += 1
                    if len(first_errors) < 3:
                        first_errors.append(f"({sub})-[{rel}]->({obj}): {e}")

        if failed:
            print(f"    ⚠️ 本批有 {failed} 条写入失败（成功 {inserted} 条）：")
            for msg in first_errors:
                print(f"       · {msg}")
        return inserted
    def search_knowledge(self, query_text):
        """根据用户问题，反向匹配图谱中的实体和关系"""
        with self.driver.session() as s:
            # Cypher 语法：如果用户的 query 包含了节点 n 或 m 的名字，就把它们的关系提取出来
            cypher = """
            MATCH (n:Entity)-[r]->(m:Entity)
            WHERE $query CONTAINS n.name OR $query CONTAINS m.name
            RETURN n.name AS s, type(r) AS rel, m.name AS o, r.condition AS cond, r.severity AS sev
            LIMIT 5
            """
            try:
                result = s.run(cypher, parameters={"query": query_text})
                return [{"s": rec["s"], "rel": rec["rel"], "o": rec["o"], "cond": rec["cond"], "sev": rec["sev"]} for rec in result]
            except Exception as e:
                print(f"图谱检索出错: {e}")
                return []

def visualize_kg(triples, filename="output.html"):
    if not triples: return
    net = Network(directed=True, height="700px", width="100%", bgcolor="#f5f5f5")
    
    # 限制可视化数量：只展示前 300 条，否则浏览器会卡死
    display_triples = triples[:300] 
    
    for t in display_triples:
        # 转换为字符串以保证节点 ID 类型一致
        s_id = str(t.get('subject', 'Unknown'))
        o_id = str(t.get('object', 'Unknown'))
        
        net.add_node(s_id, color="#4CAF50")
        net.add_node(o_id, color="#4CAF50")
        
        rel_label = str(t.get('relation', '相关'))
        props = t.get('properties', {})
        hover = f"条件:{props.get('condition')}\n严重度:{props.get('severity')}"
        
        net.add_edge(s_id, o_id, label=rel_label, title=hover)
    
    net.write_html(filename)