# 【模块 S-5】离线建库入口：切块 → 抽三元组 → 写 Neo4j → 可视化（A + C）
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
import time
import argparse
import datetime
from document_loader import load_pdf_and_chunk
from data_cleaner import clean_text
from extractor import build_model, extract_triples
from neo4j_storage import Neo4jClient, visualize_kg
from config import missing_keys

def main_pipeline(input_dir, model_name, log_path):
    # 前置检查：别跑到一半才发现 Key 没填、数据库没开
    missing = missing_keys([model_name])
    if missing:
        print("❌ 还没填大模型 API Key，无法抽取三元组。请打开 config.py 填写：")
        for item in missing:
            print(f"   · {item}")
        return

    print("🔋 正在初始化 Neo4j 图数据库...")
    db = Neo4jClient()
    try:
        db.verify()
    except Exception as e:
        print(f"❌ 连不上 Neo4j：{e}")
        print("   请先双击运行 启动Neo4j.bat，并保持那个黑窗口开着。")
        return
    
    print(f"🤖 正在加载知识抽取大模型: {model_name}...")
    llm = build_model(model_name)
    
    all_extracted_triples = []
    processed_files = [] # 用于记录日志

    # 获取所有 PDF 文件
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"⚠️ 在 {input_dir} 目录下未找到任何 PDF 文件。")
        return

    print(f"📂 发现 {len(pdf_files)} 个 PDF 文件，开始流水线处理...")

    for filename in pdf_files:
        print(f"\n🚀 开始处理: {filename}")
        file_path = os.path.join(input_dir, filename)
        
        try:
            chunks = load_pdf_and_chunk(file_path)
            print(f"  📑 文档已切分为 {len(chunks)} 个高价值文本块。")
            
            file_triples_count = 0
            
            for i, chunk in enumerate(chunks):
                print(f"  -> 处理文本块 {i+1}/{len(chunks)}...")
                cleaned_chunk = clean_text(chunk) 
                
                if not cleaned_chunk or len(cleaned_chunk.strip()) < 5:
                    continue

                try:
                    triples = extract_triples(llm, cleaned_chunk)
                    if triples:
                        inserted = db.insert_triples(triples)
                        all_extracted_triples.extend(triples)
                        file_triples_count += inserted
                        if inserted < len(triples):
                            print(f"    ⚠️ 提取 {len(triples)} 条，成功入库 {inserted} 条。")
                        else:
                            print(f"    ✅ 成功提取并存入 {inserted} 条逻辑！")
                except Exception as e:
                    print(f"    ❌ 抽取发生错误: {e}")
                    time.sleep(2) 
            
            processed_files.append({"file": filename, "status": "success", "count": file_triples_count})
            
        except Exception as file_e:
            print(f"❌ 处理文件 {filename} 失败: {file_e}")
            processed_files.append({"file": filename, "status": "failed", "error": str(file_e)})

    # ================= 写入处理日志 =================
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("图谱抽取流水线日志\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"输入目录: {input_dir}\n")
            log_file.write(f"使用模型: {model_name}\n")
            log_file.write("=" * 60 + "\n\n")
            
            for item in processed_files:
                if item["status"] == "success":
                    log_file.write(f"✅ [成功] {item['file']} -> 共提取 {item['count']} 条三元组\n")
                else:
                    log_file.write(f"❌ [失败] {item['file']} -> 报错: {item.get('error', '未知错误')}\n")
                    
        print(f"\n✨ 全部任务完成！详细运行日志已保存至: {log_path}")
    except Exception as e:
        print(f"\n⚠️ 日志写入失败: {e}")

    # 释放资源与可视化
    if hasattr(db, 'close'):
        db.close() 
        
    if all_extracted_triples:
        print("🕸️ 正在生成图谱可视化 HTML...")
        visualize_kg(all_extracted_triples)
    else:
        print("⚠️ 没有提取到任何三元组，图谱没有更新。常见原因：")
        print("   1) PDF 不属于储能领域 —— document_loader.py 按储能关键词过滤，非本领域文档会被整体丢掉；")
        print("   2) PDF 是扫描件，没有文字层，需要先做 OCR。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="储能领域知识图谱抽取流水线")
    parser.add_argument("--input_dir", default="./data", help="输入的 PDF 文件夹路径")
    parser.add_argument("--model", default="chat", help="config.py 中配置的模型名称")
    parser.add_argument("--log_path", default="./kg_processing_log.txt", help="运行日志保存路径")
    
    args = parser.parse_args()
    
    main_pipeline(args.input_dir, args.model, args.log_path)