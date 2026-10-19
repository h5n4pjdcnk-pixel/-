# 【模块 S-6】在线问答入口：Streamlit 界面，向量检索 + 图谱推理 + 反思库（B + C）
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
import streamlit as st
import chromadb
import os
import time
import uuid 
import re
from langchain_core.messages import HumanMessage, SystemMessage

# --- 导入模块化组件 ---
from config import LLM_CONFIG, RETRIEVAL_CONFIG, missing_keys  
from extractor import build_model
from neo4j_storage import Neo4jClient

st.set_page_config(page_title="储能运维大模型助手", page_icon="🔋", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.markdown("### 🔬 消融实验控制台")
    st.markdown("通过拨动下方开关，实时测试各模块对系统的增益效果。")
    st.session_state.use_vector = st.checkbox("✅ 启用向量说明书检索 (Vector RAG)", value=False) 
    st.session_state.use_graph = st.checkbox("✅ 启用图谱规则推理 (Graph RAG)", value=True)
    st.session_state.use_reflect = st.checkbox("✅ 启用历史反思经验库 (Reflection)", value=True)

    # 缺 Key 时直接说清楚，省得后面报 401 摸不着头脑
    _missing = missing_keys()
    if _missing:
        st.error("以下 API Key 还没填，检索与回答会失败：\n" + "\n".join(f"- `{m}`" for m in _missing))

st.markdown("""
<style>
    header {visibility: hidden;}
    /* 展开/收起侧边栏的按钮渲染在 header 内部，必须放行，
       否则窗口一窄侧边栏自动收起后就再也打不开了 */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
    }
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
        
    # 向量模型走统一封装：config.EMBEDDING_CONFIG["backend"] 决定用云端模型还是本地离线模型
    from embedding_model import get_embedder
    try:
        embed_model = get_embedder(db_path="./storage")
    except Exception as exc:
        st.warning(f"向量模型初始化失败：{exc}")
        embed_model = None

    # 检索器：向量召回 + 重排 + 来源加权；和离线评测脚本共用同一套逻辑
    retriever = None
    if collection is not None and embed_model is not None:
        try:
            from retriever import SemanticRetriever
            retriever = SemanticRetriever(db_path="./storage", embedder=embed_model)
        except Exception as exc:
            st.warning(f"检索器初始化失败：{exc}")

    # 向量库 / 反思库必须和当前向量模型同维度，否则检索会直接报错，这里提前讲清楚
    def _collection_dim(col):
        try:
            if col is None or col.count() == 0:
                return None
            got = col.get(limit=1, include=["embeddings"])
            emb = got.get("embeddings")
            return len(emb[0]) if emb is not None and len(emb) else None
        except Exception:
            return None

    if embed_model is not None:
        try:
            want_dim = embed_model.dimension
        except Exception:
            want_dim = None
        if want_dim:
            lib_meta = getattr(collection, "metadata", None) or {}
            lib_dim = _collection_dim(collection)
            if lib_dim and lib_dim != want_dim:
                built = "%s / %s" % (lib_meta.get("embedding_backend", "?"),
                                     lib_meta.get("embedding_model", "?"))
                st.error(f"向量库是 {lib_dim} 维（建库时用的是 {built}），"
                         f"当前向量模型是 {want_dim} 维，对不上。"
                         "请先重建向量库：python ingest_to_chroma.py --rebuild")
                retriever = None
            reflect_dim = _collection_dim(collection_reflect)
            if reflect_dim and reflect_dim != want_dim:
                st.warning("旧反思库的向量维度和当前模型不一致，已停用（不影响向量检索）。")
                collection_reflect = None

    llm = build_model("chat")
    neo_db = Neo4jClient()
    try:
        neo_db.verify()
    except Exception:
        st.warning("连不上 Neo4j，图谱推理会失效。请先双击运行 启动Neo4j.bat 并保持窗口开着。")
    return collection, collection_reflect, embed_model, retriever, llm, neo_db

collection, collection_reflect, embedding_model, retriever, llm_model, neo4j_client = init_resources()

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

    if retriever is not None and st.session_state.get('use_vector', True):
        try:
            status.write("🗄️ 正在搜索规程与白皮书...")
            hits = retriever.search(query, top_k=RETRIEVAL_CONFIG["top_k"],
                                    candidate_k=RETRIEVAL_CONFIG["candidate_k"])
            if hits:
                status.write(f"⚖️ 语义重排完成，选取 Top-{len(hits)} 文本片段。")
                vector_context = "\n---\n".join(h["text"] for h in hits)
                final_sources = [{"source": h["source"], "page": h["page"]} for h in hits]
        except Exception as e:
            status.write(f"❌ 检索异常: {e}")
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
        try: st.image("image_bd18e4.png", width="stretch") 
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
        c1.button("电池单体电压过低怎么办？", on_click=handle_submit, args=("电池单体电压过低怎么办？",), width="stretch")
        c2.button("电池热失控怎么预防？", on_click=handle_submit, args=("电池热失控怎么预防？",), width="stretch")
        c3.button("逆变器通讯中断如何排查？", on_click=handle_submit, args=("逆变器通讯中断如何排查？",), width="stretch")

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
                st.button(f"👉 {clean_q}", on_click=handle_submit, args=(clean_q,), width="stretch", key=f"fq_btn_{idx}")
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

        cols[0].button("1分", on_click=handle_rate, args=(1,), width="stretch")
        cols[1].button("5分", on_click=handle_rate, args=(5,), width="stretch")
        cols[2].button("8分", on_click=handle_rate, args=(8,), width="stretch")
        cols[3].button("10分", on_click=handle_rate, args=(10,), width="stretch")
        cols[4].button("✖️ 取消打分", on_click=cancel_rate, width="stretch")