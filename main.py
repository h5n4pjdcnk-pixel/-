import os
import time
import argparse
import datetime
from document_loader import load_pdf_and_chunk
from data_cleaner import clean_text
from extractor import build_model, extract_triples
from neo4j_storage import Neo4jClient, visualize_kg

def main_pipeline(input_dir, model_name, log_path):
    print("🔋 正在初始化 Neo4j 图数据库...")
    db = Neo4jClient()
    
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
                        db.insert_triples(triples)
                        all_extracted_triples.extend(triples)
                        file_triples_count += len(triples)
                        print(f"    ✅ 成功提取并存入 {len(triples)} 条逻辑！")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="储能领域知识图谱抽取流水线")
    parser.add_argument("--input_dir", default="./data", help="输入的 PDF 文件夹路径")
    parser.add_argument("--model", default="DeepSeek-V3.2-Fast", help="config.py 中配置的模型名称")
    parser.add_argument("--log_path", default="./kg_processing_log.txt", help="运行日志保存路径")
    
    args = parser.parse_args()
    
    main_pipeline(args.input_dir, args.model, args.log_path)