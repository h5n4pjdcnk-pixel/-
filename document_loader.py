import fitz  # PyMuPDF
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf_and_chunk(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc])
    doc.close()
    
    #领域关键词过滤
    keywords = [
        # 运维与故障类
        "电池", "BMS", "电压", "故障", "报警", "PCS", "维护", 
        # 电化学机理类
        "电化学", "SEI", "内阻", "热失控", "锂离子", "析锂", "电解液",
        # 规范、安全与操作规程类 
        "规范", "标准", "安全", "消防", "应急", "规程", "操作", 
        "白皮书", "双链", "预警", "防护", "隔离", "绝缘", "接地"
    ]
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i:i + CHUNK_SIZE]
        if any(k in chunk for k in keywords):
            chunks.append(chunk)
    return chunks