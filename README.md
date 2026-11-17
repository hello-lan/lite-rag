# 财报 RAG（BM25 + 向量 + RRF）

本地查询财报和投资者问答。向量放在 Chroma，chunk 原文放在 sqlite。BM25 负责年份、科目、指标名这类字面匹配，BGE-M3 负责语义，两路结果用 RRF 合并。

换 embedding 模型时只重算向量，不用重新解析 PDF，也不用重建 BM25。

## 目录

```
docs/            财报 PDF、Markdown，采集脚本也写到这里
models/bge-m3/   本地 BGE-M3 模型
data/            入库生成的索引：meta.db、bm25_index.pkl、chroma/
config.yaml      切分、模型、召回条数、大模型
main.py          命令行入口：ingest、query、reembed
lite_rag/        解析、索引、检索、回答
scripts/         上证 e 互动、深交所互动易采集
```

`docs/` 里有两份虚构的星澜科技示例，用来先跑通流程。真实财报直接放进 `docs/` 即可。

## 环境

需要 Python 3.10 及以上。终端里如果 `python` 不可用，用 [uv](https://docs.astral.sh/uv/) 即可：

```powershell
uv python install 3.12
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

已经装好 Python 时：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

默认从项目里的 `models/bge-m3` 加载模型。如果改回 Hub 名 `BAAI/bge-m3`，第一次入库会下载模型（大约 2GB）。国内网络可以先指定镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

只想确认 PDF / Markdown 能不能切开、暂时不下载模型：

```powershell
python main.py ingest --check
```

内存紧张时，把 `config.yaml` 里的 `embedding.model_name` 临时改成 `BAAI/bge-small-zh-v1.5`，确认流程后再换回 `BAAI/bge-m3`，并执行 `python main.py reembed`。

## 采集

问答先写成 Markdown，再和财报一起入库。两个脚本共用代码列表、文本清洗和写出，站点解析各自保留。

```powershell
python scripts/crawl_sse.py --code 600519
python scripts/crawl_irm.py --code 000001
```

## 入库

```powershell
python main.py ingest
```

支持 `.pdf`、`.md`、`.markdown`。文字版 PDF 用 pypdf 抽文本，Markdown 用 marko 抽文本（含表格）。扫出来的图片版财报抽不到字，需要另外做 OCR，本项目不包含这一步。

再次运行时，原文和切分配置都没变的文件会跳过。文件从 `docs/` 删除后，对应 chunk 会从 sqlite、Chroma 和 BM25 里清掉。

```powershell
python main.py ingest --force       # 全部重新切分并写入
python main.py ingest --bm25-only   # 只按 sqlite 重建 BM25
```

## 查询

```powershell
python main.py query 2025年毛利率是多少
python main.py query 公司主要风险有哪些 --mode vector
python main.py query 分红方案 --mode bm25
python main.py query 2025年资本开支是多少 --mode hybrid --top-k 4
```

默认打印命中的原文。`--mode` 不写时使用 `config.yaml` 的 `retrieval.mode`。

混合检索时，BM25 和向量各取 `candidate_k` 条（默认 10），再按 RRF 截到 `top_k` 条（默认 4）。RRF 只看名次：`1 / (60 + rank + 1)`，不会把两种分数直接相加。

要生成回答，配置一个 OpenAI 兼容接口。默认是 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="你的密钥"
python main.py query 2025年毛利率是多少 --answer
```

也可以把 `config.yaml` 里的 `llm.enabled` 改成 `true`，查询时就会自动调用。`--no-answer` 可以临时关掉。

## 更换 embedding

1. 修改 `config.yaml` 的 `embedding.model_name`（以及需要区分的 `embedding.version`）。
2. 执行 `python main.py reembed`。

脚本会读出 sqlite 里的全部 `chunk_text`，清空 Chroma 后写入新向量，并更新 `embed_model_name`。BM25 不重建。

改 `chunk_size`、`chunk_overlap` 或 `split.version` 才需要重新执行 `python main.py ingest`。这时原文变了，BM25 会一起重建。

## 边界

几十到上百份年报可以在单机上跑。这个结构不面向百万级文档，也不面向高并发在线服务。Chroma 只存 `chunk_id`、文件名和文档 id；回答和重算向量都以 sqlite 里的原文为准。
