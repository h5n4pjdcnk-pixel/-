# PDF 扫描件 OCR 模块

把**没有文字层**的 PDF（扫描件、整页图片）识别成文字。既能当库调用，也能直接用命令行。

## 一、解决什么问题

PDF 其实分两种：

| 类型 | `page.get_text()` 能读到什么 |
|---|---|
| 电子版（Word 导出等，有文字层） | 正常读到全部文字 |
| **扫描件**（整页就是一张图） | **一个字符都读不到** |

扫描件用常规方式取文，拿到的是**空字符串** —— 下游不管是做检索、信息抽取还是建知识库，
都会「什么结果都没有，却看不出错在哪」。

本模块的处理策略是**按页兜底**：文字层够用就直接用，不够才渲染成图片走 OCR。
所以正常电子版 PDF **零额外开销**，不会被拖慢。

## 二、安装

```bash
pip install -r requirements.txt
pip install rapidocr-onnxruntime      # 推荐的本地离线引擎
```

## 三、快速开始

**一键演示**（不需要自备素材，脚本会自己造一份扫描件出来）：

```bash
python demo.py
```

它会：生成一份电子版 PDF → 栅格化成扫描件 → 跑 OCR → 输出相似度和关键词命中情况。

**命令行**：

```bash
python run_ocr.py                              # 处理 ./data 下所有 PDF
python run_ocr.py --input 扫描件.pdf            # 指定文件（也可传目录）
python run_ocr.py --input ./pdfs --out ./out    # 指定输入 / 输出目录
python run_ocr.py --engine rapidocr             # 强制指定引擎
python run_ocr.py --force --dpi 300             # 每页都 OCR，并提高渲染清晰度
```

结果导出为 `./ocr_output/<文件名>.ocr.txt`，按页分段，方便先确认效果再决定要不要入库。

**当库调用**：

```python
from pdf_ocr import extract_pages_text

pages, stats = extract_pages_text("扫描件.pdf")
print(pages[0])   # 第 1 页的文字
print(stats)      # {'pages': 3, 'need_ocr_pages': 2, 'ocr_pages': 2, 'engine': 'rapidocr', ...}
```

也可以只对某一页、指定引擎处理：

```python
import fitz
from pdf_ocr import ocr_page

doc = fitz.open("扫描件.pdf")
text = ocr_page(doc[0], engine="rapidocr", dpi=300)
```

## 四、工作原理

1. **判断**：逐页统计有效字符数（中文字、英文字母、数字），少于 `MIN_TEXT_CHARS`
   （默认 80）就判定这页没有文字层。
2. **渲染**：把该页按 `DPI`（默认 200）渲染成图片。
3. **识别**：交给 OCR 引擎取文字。
4. **后处理**：`normalize_ocr_text()` 修掉 OCR 的两个典型副作用 ——
   - **合并被拆散的英文缩写**：`B M S` → `BMS`、`E 1 0` → `E10`
   - **还原全角数字**：`2．5` → `2.5`

   第 4 步不是锦上添花。多模态模型经常把缩写识别成单个字母用空格隔开，
   一旦 `BMS` 保持成 `B M S`，下游所有按关键词 `BMS` 做的过滤和检索就全部失配，
   等于 OCR 白做。实测加这一步之后，与原文的相似度从 99.1% 提升到 100%。

## 五、三种引擎

| 引擎 | 位置 | 费用 | 安装方式 |
|---|---|---|---|
| `rapidocr` | 本地 | 免费 | `pip install rapidocr-onnxruntime` |
| `tesseract` | 本地 | 免费 | 装 Tesseract 程序 + `chi_sim` 语言包，并把安装目录加入 PATH |
| `vlm` | 云端 | 按量计费 | 设置环境变量 `OCR_VLM_API_KEY` |

由 `OCR_ENGINE` 决定用哪个（默认 `auto`）：

- `auto`：按 `rapidocr → tesseract → vlm` 顺序挑第一个可用的
- 指定引擎名：强制使用；若该引擎不可用，会打印提示并回退到 `auto`
- `off`：完全关闭 OCR

## 六、配置

改 `ocr_settings.py`，或直接用环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `OCR_ENGINE` | `auto` | 引擎选择 |
| `OCR_MIN_TEXT_CHARS` | `80` | 低于此有效字符数判为无文字层 |
| `OCR_DPI` | `200` | 渲染分辨率 |
| `OCR_TESSERACT_LANG` | `chi_sim+eng` | Tesseract 语言 |
| `OCR_VLM_API_KEY` | 空 | 云端 VLM 密钥（仅 `vlm` 引擎需要） |
| `OCR_VLM_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | 云端模型名 |
| `OCR_VLM_API_URL` | 硅基流动地址 | 云端接口地址 |
| `OCR_VLM_MAX_PAGES` | `50` | 云端单次最多 OCR 页数（防误烧额度） |

> 为便于分享和提交，代码里**不写死任何密钥**，云端引擎一律通过环境变量提供。

## 七、实测效果

用 `demo.py` 造的扫描件（中文 + 英文缩写 + 数字混排）测试：

| 引擎 | 字符级相似度 | 说明 |
|---|---|---|
| `rapidocr` | 99.5% | 本地、免费，约 6~13 秒/页（含首次加载模型） |
| `vlm` | 99.1% → **100%**（后处理后） | 云端、按页计费 |

`BMS`、`E10`、`PCS`、`2.5V` 等关键缩写和数字经后处理后都能正确还原。

> 相似度会随页面内容、`OCR_DPI`、引擎版本小幅浮动，上表是同一份演示页面的实测值。
> 复现方法：跑 `python demo.py`，它会打印本次的相似度和关键词命中情况。

## 八、注意事项

- OCR 免不了有错字，**数字、单位和单个字母最容易错**。实测 `2.5V` 会被认成 `2.5√`，
  提高 `OCR_DPI` 有时能改善、有时没区别，**关键数字请务必人工抽查**。
- 首次运行 `rapidocr` 会加载模型，会慢几秒；同一进程内只加载一次。
- 走 `vlm` 时每页都是一次 API 调用，处理整本书之前先把 `OCR_VLM_MAX_PAGES` 设小一点试水。
- 本模块只处理 PDF。单张图片（jpg/png）不在范围内，但可以先用 PyMuPDF 包成 PDF 再处理。

## 九、文件说明

| 文件 | 作用 |
|---|---|
| `pdf_ocr.py` | 核心模块：引擎抽象、逐页兜底、结果后处理 |
| `ocr_settings.py` | 全部可调参数（支持环境变量覆盖） |
| `run_ocr.py` | 命令行入口，结果导出为 txt |
| `demo.py` | 一键演示：自动造扫描件并验证效果 |
| `requirements.txt` | 依赖清单 |
