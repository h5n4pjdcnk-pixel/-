import re
from opencc import OpenCC 

# 初始化繁体转简体 (统一图谱节点的知识口径，避免出现 "电极" 和 "電極" 两个不同节点)
cc = OpenCC('t2s')  

# 扩展的下标字符映射 (解决 LiFePO₄ -> LiFePO4 等化学式问题)
SUBSCRIPT_MAP = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    'ₐ': 'A', 'ₑ': 'E', 'ₒ': 'O', 'ₓ': 'X',
    'ₔ': 'Ǝ', 'ₕ': 'H', 'ₖ': 'K', 'ₗ': 'L',
    'ₘ': 'M', 'ₙ': 'N', 'ₚ': 'P', 'ₛ': 'S', 'ₜ': 'T',
}

# 扩展的上标字符映射 (解决 Li⁺ -> Li+, m² -> m2 等单位和离子问题)
SUPERSCRIPT_MAP = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')',
    'ⁱ': 'I', 'ⁿ': 'N',
    # 圆圈数字映射转为普通数字
    '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
    '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10'
}

def replace_subscripts_and_superscripts(text):
    """统一处理上下标字符"""
    charmap = {**SUBSCRIPT_MAP, **SUPERSCRIPT_MAP}
    return ''.join([charmap.get(c, c) for c in text])

# ================= 主力清洗函数 =================
def clean_text(text):
    """
    通用文本清洗函数：融合了 PDF 排版降噪与专业领域字符规范。
    """
    if not text:
        return ""

    # --- 第一阶段：微观字符规范化  ---
    # 1. 繁体转简体
    text = cc.convert(text)
    # 2. 替换上下标字符，保证化学式和单位的检索一致性
    text = replace_subscripts_and_superscripts(text)


    # --- 第二阶段：宏观排版降噪  ---
    # 3. 过滤独立页码 (匹配如 "12", "- 12 -", "第12页" 等单独占据一行的页码)
    text = re.sub(r'^\s*-?\s*第?\s*\d+\s*页?\s*-?\s*$', '', text, flags=re.MULTILINE)
    
    # 4. 清理不可见字符与零宽字符 (防止大模型解析异常)
    text = text.replace('\u200b', '').replace('\ufeff', '').replace('\xa0', ' ')

    # 5. 过滤纯符号构成的无意义行 (例如 "-------" 或 "...")
    lines = text.split('\n')
    valid_lines = []
    for line in lines:
        stripped_line = line.strip()
        # 保留空行（用于后续分段），或者包含实际内容（中英数）的行
        if stripped_line == "" or re.search(r'[a-zA-Z0-9\u4e00-\u9fa5]', stripped_line):
            valid_lines.append(stripped_line)
    
    # 重新组合文本
    text = '\n'.join(valid_lines)

    # 6. 规范化空白与换行 (将3个以上的连续换行符统一替换为2个，保留合理的段落间距)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 保留连字符(-)、百分号(%)、小数点(.)等，保证专业名词不被破坏
    return text.strip()