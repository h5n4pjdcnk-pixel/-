# 【模块 C-5】知识融合：实体向量相似度合并去重
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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
from collections import defaultdict
from neo4j import GraphDatabase
from config import EMBEDDING_CONFIG, NEO4J_CONF

# 和 B 向量库共用同一个 storage 目录，便于复用已经训练好的本地向量模型
DB_PATH = "./storage"

class UnionFind:
    """并查集实现，用于实体合并"""
    def __init__(self, elements):
        self.parent = {element: element for element in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return
        # 选择字典序较小的作为主实体
        if px < py:
            self.parent[py] = px
        else:
            self.parent[px] = py


def load_entities(uri, user, password):
    """从 Neo4j 加载所有 Entity 节点及其所属文档名"""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    entities = []
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.name AS name, e.doc_name AS doc_name")
        for record in result:
            name = record["name"]
            doc_names = record["doc_name"]
            if isinstance(doc_names, str):
                doc_names = [doc_names]
            elif not isinstance(doc_names, list):
                doc_names = []
            entities.append((name, doc_names))
    driver.close()
    return entities


def encode_entities_api(entity_names, mode="embedding"):
    """取实体的向量表示（和 B 向量库共用同一套向量模型封装）。

    - cloud：硅基流动在线模型，全局一致的向量空间；
    - local：优先复用 ./storage/local_embedder.joblib（和 RAG 检索是同一个空间，
      实体向量存进 Neo4j 后跨次运行仍然可比）；该文件不存在时，退而在
      「本次实体名列表」上现场训练一个，只用于本次消歧，不落盘。

    mode 参数是历史遗留，实际用哪个模型由 config.EMBEDDING_CONFIG 决定。
    """
    from embedding_model import LocalEmbedder, get_embedder, local_model_path

    names = list(entity_names)
    if not names:
        return np.zeros((0, 0))

    backend = EMBEDDING_CONFIG.get("backend", "cloud")
    if backend == "cloud":
        embedder = get_embedder(backend="cloud", db_path=DB_PATH)
    else:
        path = local_model_path(DB_PATH)
        if os.path.exists(path):
            embedder = LocalEmbedder.load(path)
            print("   使用本地向量模型 %s（和向量库同一空间）" % path)
        else:
            print("   本地还没有落盘的向量模型，改为在 %d 个实体名上现场训练一份（不回写 storage）。" % len(names))
            embedder = LocalEmbedder(verbose=False).fit(names)
    return np.array(embedder.embed_documents(names))


def calculate_similarity_matrix(embeddings):
    """使用 Numpy 矩阵运算加速余弦相似度计算"""
    # 向量归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # 避免除以零
    norms[norms == 0] = 1e-10
    normalized_embeddings = embeddings / norms
    
    # 矩阵点乘得到 N x N 的相似度矩阵
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    return similarity_matrix


def merge_entities(entity_names, similarity_matrix, threshold):
    """根据相似度矩阵和阈值，使用并查集进行实体分组"""
    uf = UnionFind(entity_names)
    n = len(entity_names)
    
    # 提取上三角矩阵中大于阈值的索引
    indices = np.where(similarity_matrix > threshold)
    for i, j in zip(*indices):
        if i < j:  # 只处理上三角部分，避免重复
            uf.union(entity_names[i], entity_names[j])
    
    groups = defaultdict(list)
    for entity in entity_names:
        parent = uf.find(entity)
        groups[parent].append(entity)
        
    return groups


def perform_merge(uri, user, password, groups):
    """在图数据库中执行合并操作，重定向关系并删除冗余节点"""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for primary, group in groups.items():
            if len(group) <= 1:
                continue
                
            # 1. 汇总所属文档名
            all_doc_names = set()
            for entity in group:
                result = session.run("MATCH (e:Entity {name: $name}) RETURN e.doc_name AS doc_name", name=entity)
                for record in result:
                    doc_names = record["doc_name"]
                    if isinstance(doc_names, str):
                        all_doc_names.add(doc_names)
                    elif isinstance(doc_names, list):
                        all_doc_names.update(doc_names)
            
            merged_doc_names = sorted(all_doc_names)
            # 存为 Neo4j 原生列表格式
            session.run(
                "MATCH (e:Entity {name: $primary}) SET e.doc_name = $doc_names",
                primary=primary, doc_names=merged_doc_names
            )
            
            # 2. 关系重定向与冗余节点删除
            for entity in group:
                if entity == primary:
                    continue
                
                # 转移旧节点的出度关系
                out_rels = session.run(
                    "MATCH (old:Entity {name: $old_name})-[r]->(b) RETURN type(r) AS rel_type, b.name AS target_name, properties(r) AS props",
                    old_name=entity
                )
                for record in out_rels:
                    rel_type = record["rel_type"]
                    target = record["target_name"]
                    props = record["props"]
                    # 使用 Python f-string 动态写入关系类型
                    session.run(
                        f"""
                        MATCH (p:Entity {{name: $primary}})
                        MATCH (b:Entity {{name: $target}})
                        MERGE (p)-[r:`{rel_type}`]->(b)
                        SET r = $props
                        """,
                        primary=primary, target=target, props=props
                    )
                
                # 转移旧节点的入度关系
                in_rels = session.run(
                    "MATCH (a)-[r]->(old:Entity {name: $old_name}) RETURN a.name AS source_name, type(r) AS rel_type, properties(r) AS props",
                    old_name=entity
                )
                for record in in_rels:
                    rel_type = record["rel_type"]
                    source = record["source_name"]
                    props = record["props"]
                    session.run(
                        f"""
                        MATCH (a:Entity {{name: $source}})
                        MATCH (p:Entity {{name: $primary}})
                        MERGE (a)-[r:`{rel_type}`]->(p)
                        SET r = $props
                        """,
                        source=source, primary=primary, props=props
                    )
                
                # 安全删除冗余旧节点及其关联的旧关系
                session.run("MATCH (e:Entity {name: $name}) DETACH DELETE e", name=entity)
    driver.close()


def save_embeddings_as_properties(uri, user, password, entity_names, embeddings):
    """将向量作为属性直接写入对应的实体节点"""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for name, embedding in zip(entity_names, embeddings):
            session.run(
                """
                MATCH (e:Entity {name: $name})
                SET e.embedding = $vector
                """,
                name=name, vector=embedding.tolist()
            )
    driver.close()


def main(uri, user, password, merge_threshold):
    print("1. 加载实体数据...")
    entities = load_entities(uri, user, password)
    if not entities:
        print("无可用实体数据。")
        return
        
    entity_names = [e[0] for e in entities]
    print(f"共加载 {len(entity_names)} 个实体。")
    
    print("2. 获取实体向量...")
    if EMBEDDING_CONFIG.get("backend", "cloud") != "cloud":
        print("   提示：当前是本地离线向量模型，相似度数值区间和云端模型不同，")
        print("   合并阈值 MERGE_THRESHOLD=%.2f 是按云端模型调的，本地跑可能一个都不合并（这是安全的默认行为）。" % merge_threshold)
    embeddings = encode_entities_api(entity_names)
    
    print("3. 计算相似度矩阵...")
    similarity_matrix = calculate_similarity_matrix(embeddings)
    
    print(f"4. 按照阈值 {merge_threshold} 进行实体分组...")
    groups = merge_entities(entity_names, similarity_matrix, merge_threshold)
    
    print("5. 执行图数据库节点合并与关系重定向...")
    perform_merge(uri, user, password, groups)
    
    print("6. 将向量存入主节点属性...")
    # 合并后只需为剩余的主节点保存向量即可
    primary_entities = list(groups.keys())
    # 获取主节点对应的向量索引
    primary_indices = [entity_names.index(p) for p in primary_entities]
    primary_embeddings = embeddings[primary_indices]
    
    save_embeddings_as_properties(uri, user, password, primary_entities, primary_embeddings)
    print("流程执行完毕。")

if __name__ == "__main__":
    # 直接使用 config.py 中的配置
    NEO4J_URI = NEO4J_CONF["uri"]
    NEO4J_USER = NEO4J_CONF["user"]
    NEO4J_PASSWORD = NEO4J_CONF["password"]
    
    MERGE_THRESHOLD = 0.92  # 合并阈值
    
    main(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MERGE_THRESHOLD)