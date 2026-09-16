import streamlit as st
import chromadb
import os
import requests
import time
import uuid 
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings

# --- 导入模块化组件 ---
from config import LLM_CONFIG  
from extractor import build_model
from neo4j_storage import Neo4jClient

st.set_page_config(page_title="储能运维大模型助手", page_icon="🔋", layout="wide")

with st.sidebar:
    st.markdown("### 🔬 消融实验控制台")
    st.markdown("通过拨动下方开关，实时测试各模块对系统的增益效果。")
    st.session_state.use_vector = st.checkbox("✅ 启用向量说明书检索 (Vector RAG)", value=False) 
    st.session_state.use_graph = st.checkbox("✅ 启用图谱规则推理 (Graph RAG)", value=True)
    st.session_state.use_reflect = st.checkbox("✅ 启用历史反思经验库 (Reflection)", value=True)

st.markdown("""
<style>
    header {visibility: hidden;}
    blockquote {
        border-left: 4px solid #f63366;
        background-color: #2b2b2b;
        padding: 10px 15px;
        color: #e0e0e0;
        border-radius: 5px;
        margin-top: 10px;
    }
    div.stButton > button {
        border-radius: 20px;
        border: 1px solid #4a4a4a;
        background-color: #1e1e1e;
        color: #d1d1d1;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        border-color: #1E90FF;
        color: #1E90FF;
        background-color: #2b2b2b;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_resources():
    try:
        client = chromadb.PersistentClient(path="./storage")
        collection = client.get_collection(name="energy_storage_library")
        collection_reflect = client.get_or_create_collection(name="reflection_library")
    except Exception as e:
        st.warning(f"向量库连接失败，请检查 ./storage 目录。")
        collection, collection_reflect = None, None
        
    embed_model = OpenAIEmbeddings(
        openai_api_base="",        #填写你的api
        openai_api_key="",
        model="Qwen/Qwen3-Embedding-8B"  
    )
    llm = build_model("")
    neo_db = Neo4jClient()
    return collection, collection_reflect, embed_model, llm, neo_db

collection, collection_reflect, embedding_model, llm_model, neo4j_client = init_resources()

# --- 检索重排 API ---
def api_rerank(query, docs):
    cfg = LLM_CONFIG.get("bge-reranker-v2-m3-free")
    if not cfg: return docs[:3], [1.0]*min(3, len(docs)), list(range(min(3, len(docs))))
    url = ""  #填写你的url
    headers = {"Authorization": f"{cfg['api_key']}", "Content-Type": "application/json"}
    payload = {"model": cfg["model_name"], "query": query, "top_n": 10, "documents": docs}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
        sorted_results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        ranked_docs = [docs[item["index"]] for item in sorted_results]
        scores = [item["relevance_score"] for item in sorted_results]
        original_indices = [item["index"] for item in sorted_results]
        return ranked_docs, scores, original_indices 
    except:
        return docs[:3], [1.0]*min(3, len(docs)), list(range(min(3, len(docs))))

# --- 核心推理与生成流 ---
def get_graphrag_answer_stream(query, status):
    vector_context, reflect_context, final_sources = "未启用该模块", "未启用该模块", []
    status.write("🔍 **[意图识别]** 正在解析提问...")

    if collection_reflect and st.session_state.get('use_reflect', True):
        try:
            status.write("🧠 **[经验回溯]** 检索历史高分反思库...")
            query_vec = embedding_model.embed_query(query)
            ref_results = collection_reflect.query(query_embeddings=query_vec, n_results=1)
            if ref_results['documents'] and ref_results['documents'][0] and len(ref_results['documents'][0]) > 0:
                reflect_context = ref_results['documents'][0][0]
                status.write("💡 **[反思命中]** 提取到相关历史经验。")
        except: pass
    elif not st.session_state.get('use_reflect', True):
        status.write("⚠️ **[消融状态]** 反思库检索已切断。")

    if collection and st.session_state.get('use_vector', True):
        try:
            status.write("🗄️ 正在搜索规程与白皮书...")
            query_vec = embedding_model.embed_query(query)
            results = collection.query(query_embeddings=query_vec, n_results=60)
            if results['documents'] and results['documents'][0]:
                raw_docs, raw_metas = results['documents'][0], results['metadatas'][0]
                status.write("⚖️ 正在执行语义重排序...")
                ranked_docs, raw_scores, original_indices = api_rerank(query, raw_docs)
                
                weighted_results = []
                for doc, score, idx in zip(ranked_docs, raw_scores, original_indices):
                    src_name = raw_metas[idx].get('source', '')
                    weight = 1.5 if "规范" in src_name or "白皮书" in src_name else 1.0
                    weighted_results.append((score * weight, doc, idx))
                
                weighted_results.sort(key=lambda x: x[0], reverse=True)
                top_k_indices = [item[2] for item in weighted_results[:5]]
                final_sources = [raw_metas[idx] for idx in top_k_indices]
                vector_context = "\n---\n".join([item[1] for item in weighted_results[:5]])
                status.write("🎯 重排序完成，选取 Top-5 文本片段。。")
        except Exception as e: status.write(f"❌ 检索异常: {e}")
    elif not st.session_state.get('use_vector', True):
        status.write("⚠️ **[消融状态]** 向量说明书检索已切断。")

    graph_context = "【图谱逻辑关系】: 未命中相关图谱规则\n"
    if st.session_state.get('use_graph', True):
        status.write("🕸️ **[图谱查询]** 提取 Neo4j 逻辑规则...")
        graph_data = neo4j_client.search_knowledge(query)
        if graph_data:
            graph_context = "【图谱逻辑关系】:\n" + "\n".join([f"- ({r['s']}) --[{r['rel']}]--> ({r['o']}) | 条件: {r['cond']}" for r in graph_data])
    else:
        status.write("⚠️ **[消融状态]** 图谱逻辑链推理已关闭。")
        graph_context = "【图谱逻辑关系】: 未启用该模块\n"

    system_prompt = """你是一名储能系统运维领域的专业问答助手。请结合提供的检索上下文和图谱信息，对用户问题进行分析并给出专业回答。

【回答要求】：
1. 回答时直接组织专业结论，不主动描述内部检索流程或上下文组织方式。
2. 内化知识：直接、自信、专业地陈述事实、分析故障和给出建议。

【🎨 公式与符号清洗及渲染指令】：
1. 强制 LaTeX 渲染：输出图谱条件、阈值参数、物理量等表达式时，必须使用 LaTeX 语法（行内公式用单美元符号包裹，复杂公式用双美元符号包裹）。
2. 碎片清洗与格式化：遇到 PDF 提取产生的重复或断裂文本（例如原文中孤立的 "1C \n 1C: dT/dt..."），应先进行格式清理后再组织输出！你必须自动清理重复或无意义的文本片段，必要时整理为结构清晰的 Markdown 列表
👉 示例格式：
- **0.5C 充电倍率**：$\frac{dT}{dt} > 0.8^\circ\text{C}/\text{min}$
- **1C 充电倍率**：$\frac{dT}{dt} > 1.2^\circ\text{C}/\text{min}$
- **2C 充电倍率**：$\frac{dT}{dt} > 1.8^\circ\text{C}/\text{min}$

【⚠️ 格式强制指令】：
无论回答多长，在回答的最末尾，必须严格按以下格式，单独起一行输出 3 个启发式的延伸探索问题（切勿漏掉分割线和💡图标）：

---
💡 **您可以继续探索以下问题：**
1. [延伸问题1]
2. [延伸问题2]
3. [延伸问题3]"""

    full_prompt = f"【反思库经验】:\n{reflect_context}\n\n【说明书原文】:\n{vector_context}\n\n{graph_context}\n\n问题：{query}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=full_prompt)]
    return llm_model.stream(messages), final_sources, vector_context, graph_context, reflect_context

# ==========================================
# 状态机与页面渲染
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "need_rating" not in st.session_state:
    st.session_state.need_rating = False

def handle_submit(query_text):
    st.session_state.messages.append({"role": "user", "content": query_text})
    st.session_state.need_rating = False

# --- 场景 A：首页搜索 ---
if len(st.session_state.messages) == 0:
    st.markdown("<div style='height: 18vh;'></div>", unsafe_allow_html=True)
    _, col_img, _ = st.columns([4, 1, 4])
    with col_img:
        try: st.image("image_bd18e4.png", use_column_width=True) 
        except: st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🔋</h1>", unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: center;'>
            <h2 style='font-size: 2.2rem; margin-bottom: 30px;'>储能系统智能问答助手</h2>
        </div>
    """, unsafe_allow_html=True)
    
    _, col_center, _ = st.columns([1.5, 4, 1.5])
    with col_center:
        user_input = st.text_input("搜索框", placeholder="在此输入您的储能运维故障或疑问...", label_visibility="collapsed")
        if user_input:
            handle_submit(user_input)
            st.rerun()
            
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.button("电池单体电压过低怎么办？", on_click=handle_submit, args=("电池单体电压过低怎么办？",), use_container_width=True)
        c2.button("电池热失控怎么预防？", on_click=handle_submit, args=("电池热失控怎么预防？",), use_container_width=True)
        c3.button("逆变器通讯中断如何排查？", on_click=handle_submit, args=("逆变器通讯中断如何排查？",), use_container_width=True)

# --- 场景 B：聊天工作台 ---
else:
    st.caption("🔋 储能运维专家系统 | RAG & Reflection Engine")
    st.markdown("---")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.popover("📚 核心参考出处"):
                    unique_docs = list(set([src.get('source', '未知文档') for src in msg["sources"]]))
                    for doc in unique_docs[:5]: st.caption(f"📄 {doc}")

    if st.session_state.messages[-1]["role"] == "user":
        latest_query = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.status("🧠 **系统正在推理...**", expanded=True) as status:
                stream_response, sources, vec_ctx, graph_ctx, reflect_ctx = get_graphrag_answer_stream(latest_query, status)
                status.update(label="✅ **逻辑构建完成**", state="complete", expanded=False)
            
            with st.expander("🛠️ 查看底层召回片段与逻辑链", expanded=False):
                st.markdown("#### 1. 📖 参考文本片段")
                clean_vec_ctx = re.sub(r'---第\s*\d+\s*页.*?页', '', vec_ctx) 
                display_text = clean_vec_ctx[:1000] + "\n...(仅展示前1000字符)" if len(clean_vec_ctx) > 1000 else clean_vec_ctx
                st.markdown(f"```text\n{display_text}\n```")
                
                st.markdown("#### 2. 🕸️ 图谱推理链")
                if "发现规则" in graph_ctx: st.success(graph_ctx)
                else: st.warning(graph_ctx)
                
                if reflect_ctx:
                    st.markdown("#### 🧠 历史反思增强")
                    st.info(reflect_ctx)

                st.markdown("#### 📑 物理溯源档案")
                if sources:
                    unique_docs = list(set([src.get('source', '未知文档') for src in sources]))
                    for doc in unique_docs[:5]: st.markdown(f"- 📄 `{doc}`")

            # 从流式响应中分离正文与推荐问题
            placeholder = st.empty() 
            full_answer = ""
            display_answer = ""
            
            for chunk in stream_response:
                full_answer += chunk.content
                # 检测推荐问题标记，仅显示正文部分
                marker_idx = full_answer.find("💡")
                if marker_idx != -1 and "您可以继续探索以下问题" in full_answer[marker_idx:]:
                    dash_idx = full_answer.rfind("---", 0, marker_idx)
                    display_answer = full_answer[:dash_idx].strip() if dash_idx != -1 else full_answer[:marker_idx].strip()
                else:
                    display_answer = full_answer
                    
                placeholder.markdown(display_answer + " ▌") 
                
            placeholder.markdown(display_answer) 
            
            # 提取推荐问题并保存至 follow_ups 字段
            follow_ups = []
            if "您可以继续探索以下问题" in full_answer:
                q_part = full_answer.split("您可以继续探索以下问题")[-1]
                follow_ups = [q.strip() for q in re.findall(r'\d+\.\s*(.+)', q_part) if q.strip()]
            
            # 历史记录仅保存回答正文
            st.session_state.messages.append({
                "role": "assistant", 
                "content": display_answer, 
                "sources": sources,
                "follow_ups": follow_ups # 按钮数据单独存
            })
            st.session_state.need_rating = True
            st.rerun()

    if prompt := st.chat_input("继续提问..."):
        handle_submit(prompt)
        st.rerun()

    # --- 🌟 快捷追问与评分模块 ---
    if st.session_state.need_rating and len(st.session_state.messages) >= 2:
        # 使用 Streamlit 原生分隔线
        st.markdown("---")
        
        # 从最后一条历史记录里取出提取好的问题
        last_msg = st.session_state.messages[-1]
        follow_ups = last_msg.get("follow_ups", [])
        
        if follow_ups:
            st.markdown("#### ✨ 您可以继续探索以下问题：")
            for idx, q in enumerate(follow_ups[:3]): 
                clean_q = q.replace("**", "") 
                st.button(f"👉 {clean_q}", on_click=handle_submit, args=(clean_q,), use_container_width=True, key=f"fq_btn_{idx}")
            st.markdown("---")
        
        st.markdown("#### 📝 请评估此回答以帮助系统进化：")
        cols = st.columns([1,1,1,1,2])
        
        def handle_rate(score):
            st.session_state.need_rating = False
            if score >= 8:
                try:
                    qa_text = f"Q: {st.session_state.messages[-2]['content']}\nA: {st.session_state.messages[-1]['content']}"
                    vec = embedding_model.embed_documents([qa_text])
                    collection_reflect.add(documents=[qa_text], embeddings=vec, ids=[f"ref_{uuid.uuid4().hex[:6]}"], metadatas=[{"score": score}])
                    st.toast("🎉 高分验证通过！经验已萃取入库。", icon="✅")
                except: pass
            else:
                st.toast("收到反馈！我们会持续精进算法。", icon="🙏")

        def cancel_rate():
            st.session_state.need_rating = False

        cols[0].button("1分", on_click=handle_rate, args=(1,), use_container_width=True)
        cols[1].button("5分", on_click=handle_rate, args=(5,), use_container_width=True)
        cols[2].button("8分", on_click=handle_rate, args=(8,), use_container_width=True)
        cols[3].button("10分", on_click=handle_rate, args=(10,), use_container_width=True)
        cols[4].button("✖️ 取消打分", on_click=cancel_rate, use_container_width=True)