import os
import re
import chromadb
import time
import concurrent.futures 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm
from data_cleaner import clean_text
from extractor import build_model #  引入你的大模型组件

# --- 1. 基础配置 ---
DB_PATH = "./storage"      
DATA_FOLDER = "./data" 
COLLECTION_NAME = "energy_storage_library"
MAX_WORKERS = 10 #  并发线程数（根据 API 限流情况可调整，10 通常比较安全）

if not os.path.exists(DATA_FOLDER):
    print(f"❌ 找不到文件夹 {DATA_FOLDER}")
    exit()

# --- 2. 初始化组件 ---
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

print("☁️ 正在连接云端向量模型...")
embedding_model = OpenAIEmbeddings(
    openai_api_base="",
    openai_api_key="",
    model="Qwen/Qwen3-Embedding-8B"  
)

print("🤖 正在连接云端 LLM (用于段落修复)...")
llm_model = build_model("")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def is_garbage_rules(text):
    text = text.strip()
    if len(text) < 15: return True
    zh_chars = re.findall(r'[\u4e00-\u9fa5]', text)
    if len(text) > 0 and len(zh_chars) / len(text) < 0.5: return True
    if re.search(r'([^\w\s\u4e00-\u9fa5])\1{5,}', text): return True
    return False

def ai_fix_paragraph(chunk_text):
    system_prompt = """你是一个严谨的文本排版与校对助手。以下文本是从 PDF OCR 提取的，存在存在异常换行、段落断裂（如两三个字就换行）等排版错乱。
【严格指令】：
1. 合并断裂文本：将不合逻辑的换行去掉，把断开的句子重新拼接成连贯、通顺的段落。
2. 保持原文语义：**严禁**增加原文没有的解释、**严禁**总结缩写、**严禁**改变原本的专业词汇。
3. 直接输出结果：不要添加说明性前缀或其他无关内容，直接输出修复后的文本。。"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"待修复文本：\n{chunk_text}")
    ]
    
    try:
        response = llm_model.invoke(messages)
        result = response.content.strip()
        # 若模型返回内容异常过短，则回退使用原文本
        if len(result) < 10:
            return chunk_text 
        return result
    except Exception as e:
        # 网络波动等报错时，用原文本兜底，不让程序崩溃
        return chunk_text 

# --- 执行极速入库 ---
pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]

for filename in pdf_files:
    file_path = os.path.join(DATA_FOLDER, filename)
    print(f"\n[🚀 预处理与入库中]: {filename}")
    
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        for i, page in tqdm(enumerate(pages), total=len(pages), desc="进度"):
            cleaned_content = clean_text(page.page_content)
            raw_chunks = text_splitter.split_text(cleaned_content)
            
            if not raw_chunks: continue
                
            # 1. 过滤明显异常文本
            suspect_chunks = [c for c in raw_chunks if not is_garbage_rules(c)]
            if not suspect_chunks: continue
            
            # 2. 并发执行段落修复
            refined_chunks = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # 并发提交当前页的文本片段
                future_to_chunk = {executor.submit(ai_fix_paragraph, chunk): chunk for chunk in suspect_chunks}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    clean_chunk = future.result()
                    if clean_chunk:
                        refined_chunks.append(clean_chunk)
            
            if not refined_chunks: continue
            
            # --- 3. 将清洗后的文本向量化并写入 Chroma ---
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectors = embedding_model.embed_documents(refined_chunks)
                    metadatas = [{"source": filename, "page": i + 1} for _ in refined_chunks]
                    ids = [f"{filename}_p{i}_c{j}" for j in range(len(refined_chunks))]
                    
                    collection.add(documents=refined_chunks, embeddings=vectors, metadatas=metadatas, ids=ids)
                    break 
                    
                except Exception as api_e:
                    if attempt < max_retries - 1:
                        time.sleep(3)
                    else:
                        pass 

    except Exception as file_e:
        print(f"处理文件 {filename} 彻底出错: {file_e}")
            
print("\n✅ 所有文档处理完成，已写入 ChromaDB 向量库！")