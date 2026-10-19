# A_文本切分

本目录是 A / B / C 三段式结构中的占位目录。实际切分脚本位于 `Chunk切分结果\chunk_texts.py`。

完整 Chunk 数据不随公开仓库发布。使用自有文本运行：

```bash
python "Chunk切分结果\chunk_texts.py" --source-root ".\OCR结果" --output-root ".\Chunk切分结果"
```

生成的 `chunks.jsonl`、`chunks.csv`、`manifest.json` 和 `quality_report.md` 均已被 `.gitignore` 排除。