# 【模块 C-6】多模态扩展：PDF 图片抽取 + 视觉描述
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
import io
import sys
import base64
import json
import csv
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF库，用于提取PDF图片

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import LLM_CONFIG 

class VisionAPIExtractor:
    def __init__(self, mode="vision"):
        """
        初始化基于云端 API 的图片提取与标题生成器
        """
        print(f"正在初始化云端视觉大模型 API (模式: {mode})...")
        try:
            cfg = LLM_CONFIG[mode]
            self.llm = ChatOpenAI(
                model=cfg["model_name"], 
                openai_api_key=cfg["api_key"], 
                openai_api_base=cfg["api_url"], 
                temperature=0.1
            )
            print("视觉大模型 API 加载成功。")
        except Exception as e:
            print(f"API 加载失败，请检查 config.py: {e}")
            sys.exit(1)

    def extract_images_from_pdf(self, pdf_path):
        """从 PDF 文档中提取有效图片"""
        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_index in range(len(doc)):
                page = doc[page_index]
                # 获取该页所有的图片对象
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    # 提取图片字节流
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        # 过滤机制：剔除宽度或高度小于 100 像素的图片（通常是Logo或装饰线条）
                        if image.width >= 100 and image.height >= 100:
                            images.append(image.convert("RGB"))
                    except Exception as img_e:
                        print(f"  警告: 无法处理第 {page_index+1} 页的图像: {img_e}")
        except Exception as doc_e:
            print(f"处理文档 {os.path.basename(pdf_path)} 时出错: {doc_e}")

        print(f"  从文档中提取了 {len(images)} 张有效图片 (已过滤微小图标)。")
        return images

    def encode_image_to_base64(self, image):
        """将 PIL 图片转换为 Base64 编码"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_title_with_api(self, image):
        """调用云端大模型为图片生成描述"""
        try:
            base64_image = self.encode_image_to_base64(image)

            # 储能领域专属提示词
            system_prompt = "你是一名储能系统与电气设备图像分析助手。"

            user_prompt = (
                "请用简短的一句话描述图片内容。"
                "若图片涉及储能电池舱、BMS 界面、电路图或运维故障，"
                "请准确指出设备名称或故障类型。描述控制在15个字以内，不使用标点符号。"
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=[
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                )
            ]

            response = self.llm.invoke(messages)
            title = response.content.strip()
            
            if title.lower().startswith("这是一张"):
                title = title.replace("这是一张", "")
                
            return title

        except Exception as e:
            print(f"生成标题时 API 出错: {e}")
            return "生成标题失败_API错误"

    def process_pdf(self, pdf_path):
        """处理单个 PDF 文档"""
        result = {}
        doc_name = os.path.basename(pdf_path).replace('.pdf', '').replace('.PDF', '')
        print(f"\n开始处理文档: {doc_name}")
        
        images = self.extract_images_from_pdf(pdf_path)

        for i, img in enumerate(images):
            img_key = f"{doc_name}_img_{i+1}"
            print(f"  -> 正在识别图片 {i+1}/{len(images)} ...")
            title = self.generate_title_with_api(img)
            
            result[img_key] = {
                "image": img,
                "title": title,
                "document": pdf_path
            }
            print(f"    识别结果: [{title}]")

        return result

    def process_folder(self, folder_path):
        """遍历文件夹处理所有 PDF 文档"""
        all_results = {}
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    result = self.process_pdf(pdf_path)
                    all_results.update(result)
        return all_results

def save_results(results, output_base_dir="vision_output"):
    """保存提取的图片和生成的元数据结构"""
    if not results:
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "image_knowledge.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['图片ID', '关联文档', '图谱描述', '本地路径'])
        writer.writeheader()
        
        for img_key, data in results.items():
            doc_name = os.path.basename(data['document']).split('.')[0]
            doc_dir = os.path.join(output_dir, doc_name)
            os.makedirs(doc_dir, exist_ok=True)
            
            img_path = os.path.join(doc_dir, f"{img_key}.jpg")
            data['image'].save(img_path, "JPEG")
            
            writer.writerow({
                '图片ID': img_key,
                '关联文档': doc_name,
                '图谱描述': data['title'],
                '本地路径': os.path.relpath(img_path, output_dir)
            })
            
    print(f"\n多模态知识已保存至目录: {output_dir}")
    print(f"数据索引文件: {csv_path}")

if __name__ == "__main__":
    TARGET_DIR = "./data" 
    
    extractor = VisionAPIExtractor(mode="vision") 
    extracted_knowledge = extractor.process_folder(TARGET_DIR)
    
    save_results(extracted_knowledge)