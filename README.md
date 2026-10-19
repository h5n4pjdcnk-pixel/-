# 储能知识助手（脱敏代码版）

这是一个面向公开仓库的脱敏版本，展示储能运维领域的 **RAG + GraphRAG** 项目结构：

- OCR：PDF 文字层检测与扫描页兜底识别
- 文本规范化：清洗 OCR 噪声、页眉页脚和多余空白
- 文本切分：生成 500–1200 字左右的语义 Chunk
- 向量检索：Chroma + 云端/本地向量模型 + 重排
- 知识图谱：三元组抽取、Neo4j 存储、实体融合
- 问答界面：Streamlit，支持向量检索 / 图谱推理 / 反思库消融开关

> 本仓库不包含完整业务语料、原始 PDF 全集、向量库和运行日志。完整 OCR 文本、设备手册、标准规范等可能涉及第三方隐私或版权，未随仓库发布。

## 目录结构

```text
.
├── app.py                         Streamlit 问答入口
├── main.py                        PDF → 切块 → 三元组 → Neo4j
├── check_env.py                   环境自检
├── config.py                      全局配置
├── .env.example                   环境变量模板
├── A_文本切分                     文本切分模块说明
├── B_向量数据库                   向量化、入库、检索、评测
├── C_知识图谱                     三元组抽取、Neo4j、知识融合
├── Chunk切分结果                  切分脚本与输出格式说明
├── 公共模块                       OCR、清洗、文档读取、大模型工厂
├── 多模态                         PDF 图片抽取与视觉描述
├── OCR模块                        独立 OCR 工具
├── 文本规范化                     OCR 文本规范化脚本
├── data                           测试样例 PDF
└── lib                            前端静态依赖
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

复制配置模板并填写自己的 Key：

```bash
copy .env.example .env
python check_env.py
```

`.env` 已被 `.gitignore` 忽略，不会进入仓库。API Key 只从环境变量读取，`config.py` 中没有硬编码密钥。

## 样例数据链路

仓库只保留 `data\测试样例_储能运维.pdf`，可按以下顺序生成中间产物：

```bash
python "OCR模块\batch_ocr.py"
python "文本规范化\OCR文本规范化处理记录\normalize_ocr_texts_safe.py" --root ".\OCR结果" --apply
python "Chunk切分结果\chunk_texts.py" --source-root ".\OCR结果" --output-root ".\Chunk切分结果"
python "B_向量数据库\ingest_to_chroma.py" --rebuild
python "B_向量数据库\eval_semantic_search.py"
```

启动网页：

```bash
启动网页.bat
```

知识图谱流程：

```bash
python main.py --input_dir .\data
```

运行 `main.py` 前需要启动 Neo4j，并配置 `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`。

## 配置说明

| 变量 | 用途 |
| --- | --- |
| `SILICONFLOW_KEY` | 对话、向量化、视觉理解、重排共用的 API Key |
| `NEO4J_URI` | Neo4j Bolt 地址，默认 `bolt://127.0.0.1:7687` |
| `NEO4J_USER` | Neo4j 用户名 |
| `NEO4J_PASSWORD` | Neo4j 密码 |
| `OCR_VLM_API_KEY` | 可选，云端 OCR 引擎使用 |

向量后端可在 `config.py` 的 `EMBEDDING_CONFIG` 中切换：

- `cloud`：硅基流动在线向量模型，默认使用 `Qwen/Qwen3-Embedding-8B`
- `local`：字符 n-gram TF-IDF + LSA，本地离线运行，适合无 Key 时跑通链路

切换后端或更换向量模型后，需要重建向量库：

```bash
python "B_向量数据库\ingest_to_chroma.py" --rebuild
```

## 数据与隐私

- 仓库不包含 `.env`、真实 API Key、Neo4j 密码或 Git 凭证。
- 仓库不包含完整 OCR 语料、规范化全文、Chunk 全文和 Chroma 向量库。
- 代码中的本机绝对路径已改为项目相对路径或环境变量。
- 上传前请确认没有把本地 PDF、日志、备份目录加入 Git。

更详细的上传检查见 `GitHub上传说明.md`。
