# 文本 Chunk 切分结果

本目录保留 Chunk 切分脚本和输出格式说明。完整 `chunks.jsonl`、`chunks.csv`、`manifest.json` 与质量报告不随公开仓库提供，需要在本地对自有语料运行脚本生成。

## 生成方式

```bash
python chunk_texts.py --source-root "..\OCR结果" --output-root "."
```

## 输出文件

- `chunks.jsonl`：一行一个 JSON 对象，字段为 `chunk_id`、`source_file`、`chunk_index`、`content`
- `chunks.csv`：与 JSONL 内容一致，便于表格工具查看
- `manifest.json`：切分任务元数据和每个源文件的 Chunk 数
- `quality_report.md`：长度分布、来源映射和需关注条目

## 字段含义

- `chunk_id`：全局唯一连续编号，如 `chunk_0001`
- `source_file`：Chunk 所属 TXT 文件的相对路径
- `chunk_index`：Chunk 在原文件内的片段序号，从 1 开始
- `content`：Chunk 文本内容

切分采用语义优先策略，目标长度为 500–1200 字符；短文档、表格或连续语义段允许适当超出范围，以保持内容完整。