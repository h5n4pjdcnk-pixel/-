import os
import csv
import sys
from neo4j import GraphDatabase 
from config import NEO4J_CONF  # 导入 Neo4j 配置

class Neo4jImageImporter:
    def __init__(self, uri, user, pwd):
        """初始化 Neo4j 数据库连接"""
        print(f"📡 正在连接 Neo4j 数据库 ({uri})...")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            # 简单验证连接
            self.driver.verify_connectivity()
            print("✅ 数据库连接成功！")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            sys.exit(1)

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            print("🔌 数据库连接已关闭。。")

    def run_import(self, csv_file_path):
        """读取 CSV 文件并执行 Cypher 语句导入数据"""
        if not os.path.exists(csv_file_path):
            print(f"❌ 找不到 CSV 文件: {csv_file_path}")
            return

        print(f"🚀 开始读取 CSV 并导入多模态知识: {os.path.basename(csv_file_path)}")
        
        count = 0
        
        # 定义 Cypher 导入语句 
        # 使用 MERGE 保证文档节点唯一，使用 CREATE 创建新的图片节点和关系
        cypher_query = """
        // 1. 确保文档节点存在
        MERGE (d:Document {name: $document_name})
        ON CREATE SET d.created_at = timestamp()
        
        // 2. 创建图片知识节点
        CREATE (i:Image {
            id: $image_id, 
            title: $title, 
            relative_path: $relative_path
        })
        
        // 3. 建立关系线
        CREATE (d)-[:CONTAINS]->(i)
        """

        try:
            with self.driver.session() as session:
                encoding_to_use = 'utf-8-sig'
                try:
                    with open(csv_file_path, 'r', encoding='utf-8-sig') as test_f:
                        test_f.read()
                except UnicodeDecodeError:
                    encoding_to_use = 'gbk'  # UTF-8 解码失败时尝试使用 GBK
                
                # 用正确的编码打开文件
                with open(csv_file_path, 'r', encoding=encoding_to_use) as csvfile:
                    reader = csv.DictReader(csvfile, fieldnames=['图片ID', '关联文档', '图谱描述', '本地路径'])
                    next(reader) # 跳过标题行

                    for row in reader:
                        # 参数清洗
                        params = {
                            "image_id": row['图片ID'].strip(),
                            "document_name": row['关联文档'].strip(),
                            "title": row['图谱描述'].strip(),
                            # 存入图谱的是相对于 vision_output 的相对路径，方便异地迁移
                            "relative_path": row['本地路径'].strip() 
                        }

                        if not params["title"] or "fail" in params["title"]:
                            print(f"  ⚠️ 跳过识别失败的图片: {params['image_id']}")
                            continue

                        # 执行 Cypher 语句
                        session.run(cypher_query, params)
                        count += 1
                        print(f"  ✅ 已将图片 [{params['image_id']}] 作为多模态知识挂载到文档 [{params['document_name']}]")

            print(f"\n✨ 多模态图谱构建完成！总共建立了 {count} 张图片的视觉知识节点。")
            print("💡 你可以在 Neo4j Browser 中运行语句 `MATCH p=()-[:CONTAINS]->() RETURN p` 查看结果。")

        except Exception as e:
            print(f"❌ 导入过程中发生严重错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    TARGET_CSV = os.getenv("VISION_CSV_PATH","vision_output/image_knowledge.csv")
    
    print(f"======== 开始强制导入: {TARGET_CSV} ========")
    
    # 直接使用 config.py 中的配置
    uri = NEO4J_CONF["uri"]
    user = NEO4J_CONF["user"]
    pwd = NEO4J_CONF["password"]
    
    # 初始化导入器
    importer = Neo4jImageImporter(uri, user, pwd)
    
    # 执行导入
    importer.run_import(TARGET_CSV)
    
    # 释放资源
    importer.close()


