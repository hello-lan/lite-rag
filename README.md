# 财报 RAG（BM25 + 向量 + RRF）

本地查询公司材料和投资通识。向量放在 Chroma，chunk 原文和元信息放在同一份 sqlite（`financial_report_chunks` / `knowledge_chunks` 分表），BM25 也按 scope 分开。BM25 负责年份、科目、指标名这类字面匹配，BGE-M3 负责语义，两路结果用 RRF 合并。查某一家公司或某一类材料时，两路使用同一套筛选条件。

换 embedding 模型时只重算向量，不用重新解析 PDF，也不用重建 BM25。

## 目录

```
corpus/02_processed/financial_report/filing/     年报、半年报、季报
corpus/02_processed/financial_report/ir_qa/      互动易、上证 e 互动
corpus/02_processed/financial_report/survey/     投资者调研
corpus/02_processed/financial_report/research/   研报
corpus/02_processed/financial_report/news/       新闻
corpus/02_processed/knowledge/investment_theory/   投资理论
corpus/02_processed/knowledge/beginner/ 投资入门答疑
corpus/raw/sse、corpus/raw/irm       采集脚本的默认输出，入库时用 --dir 或改 --output
corpus/03_splited/                  已经切好的 JSON，入库时只做嵌入
corpus/tangchao/               唐朝书房原文
corpus/BRK-Annual-Meeting/     伯克希尔股东大会记录
models/bge-m3/               本地 BGE-M3 模型
data/                        入库生成的索引：meta.db（financial_report_chunks / knowledge_chunks）、financial_report_bm25.pkl、knowledge_bm25.pkl、chroma/
config.yaml                  切分、模型、召回条数、大模型
main.py                      命令行入口：ingest-split、query-knowledge、query-financial-report、reembed、serve
api.py                       查询接口：/knowledge、/financial_report 各自的 retrieve 与 query
lite_rag/                    索引、检索、回答；只导入已切分 JSON
scripts/                     切分、抽取、采集（年报/互动易等）
```

`corpus/02_processed/` 放加工中的原文或中间结果；入库只认 `corpus/03_splited/` 里的预切分 JSON。JSON 的 `meta` 决定公司、年份、章节等字段。

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

只想确认预切分 JSON 能不能读入、暂时不下载模型：

```powershell
python main.py ingest-split --check
```

内存紧张时，把 `config.yaml` 里的 `embedding.model_name` 临时改成 `BAAI/bge-small-zh-v1.5`，确认流程后再换回 `BAAI/bge-m3`，并执行 `python main.py reembed`。

## 采集

问答先写成 Markdown，再由业务脚本切成 JSON 后 `ingest-split`。采集脚本共用代码列表、文本清洗和写出，站点解析各自保留。默认分别写到 `corpus/raw/sse` 和 `corpus/raw/irm`；也可用 `--output` 指到 `corpus/02_processed/financial_report/ir_qa`。

```powershell
python scripts/crawl_sse.py --code 600519
python scripts/crawl_irm.py --code 000001
python scripts/crawl_sse.py --code 600519 --output corpus/02_processed/financial_report/ir_qa
```

## 入库（只导入已切分 JSON）

`lite_rag` 不再切分文档。PDF / Markdown 的切分由业务脚本完成（如 `scripts/split_annual_reports.py`、`scripts/ingest_tangchao.py`），写出到 `corpus/03_splited/`，再由 `ingest-split` 嵌入入库。

一个 JSON 文件对应一份逻辑文档，可以放在 `corpus/03_splited/` 的任意子目录里。类型以文件里的 `meta` 为准，不看子目录名字。

`meta` 里 `scope` 和 `doc_type` 必填，其余可以不写。单条 chunk 可以改自己的年份、日期或其他字段，没写的继承文档级 `meta`。`split` 用来标记这套切分；不写时记为 `external`。改 `config.yaml` 里的 `chunk_size` 不会自动重算这些 chunk——需要先重跑对应切分脚本，再 `ingest-split`。

`scope` 只能是 `financial_report` 或 `knowledge`；`doc_type` 在公司侧是 `filing`、`ir_qa`、`survey`、`research`、`news`，在通识侧是 `investment_theory`、`beginner`。`investment_theory` 还要写 `figure` 和 `occasion`：`figure` 是 `唐朝`、`buffett`（巴菲特）、`munger`（芒格）；`occasion` 是 `letter`（致投资者的信）、`meeting`（股东大会）、`speech`（演讲）、`article`（文章）。

财报向量只写非空的 `stock_name`、`stock_code`、`period`、`section`、`entity`，不写 `year`。`section` 可以是 `profile`、`segment`、`mda`、`risk`、`material`、`governance`、`financials`、`notes`。`entity` 只在 `financials` 或 `notes` 上使用，取 `consolidated` 或 `parent`。查财报用 `period`，不要用 `year`；`year` 只筛投资通识。

```json
{
  "split": "qa-pair-v1",
  "meta": {
    "scope": "financial_report",
    "doc_type": "ir_qa",
    "title": "贵州茅台互动问答",
    "stock_name": "贵州茅台",
    "stock_code": "600519"
  },
  "chunks": [
    {"text": "问：……\n答：……", "published_on": "2025-03-12", "year": 2025}
  ]
}
```

`chunks` 不能为空，`text` 必须是非空字符串。字段不认识、年份和报告期对不上时，会报出文件路径并停下。

```powershell
python main.py ingest-split --check      # 只校验 JSON
python main.py ingest-split              # 嵌入并写入
python main.py ingest-split --force      # 即使没变也重新嵌入
python main.py ingest-split --bm25-only  # 只按 sqlite 重建 BM25
```

不传 `--dir` 时读取 `paths.split_dir`，默认是 `corpus/03_splited`。相对路径相对项目根目录。`ingest-split` 只删除本次扫描范围内已经不见的 JSON 对应 chunk。`python main.py reembed` 会按 sqlite 里的全部原文重算向量。

## 查询

```powershell
python main.py query-financial-report 2025年毛利率是多少
python main.py query-financial-report 公司主要风险有哪些 --mode vector
python main.py query-financial-report 分红方案 --mode bm25 --code 600519 --doc-type filing
python main.py query-financial-report 2025年资本开支是多少 --mode hybrid --top-k 4 --stock-name 贵州茅台 --period 2025 --section financials --entity consolidated
python main.py query-knowledge 什么是安全边际 --doc-type investment_theory --figure 唐朝 --occasion article --year 2018
```

`query-financial-report` 查公司材料，`query-knowledge` 查投资通识，筛选参数不共用。两边都支持 `--mode`、`--top-k`、`--output`；`--mode` 不写时使用 `config.yaml` 的 `retrieval.mode`。

`query-financial-report` 可选 `--doc-type`（`filing`、`ir_qa`、`survey`、`research`、`news`）、`--stock-name`、`--code`、`--period`、`--section`、`--entity`。传入公司或代码后，向量和 BM25 都只保留这家公司。报告期用 `--period`，不要用 `--year`。

`query-knowledge` 可选 `--doc-type`（`investment_theory`、`beginner`）、`--figure`、`--occasion`、`--year`。`--figure` 和 `--occasion` 只用于 `investment_theory`。

不传上述筛选时，在该命令范围内搜索全部材料。程序不会从问题文本里猜测公司和类型。

混合检索时，BM25 和向量各取 `candidate_k` 条（默认 10），再按 RRF 截到 `top_k` 条（默认 4）。RRF 只看名次：`1 / (60 + rank + 1)`，不会把两种分数直接相加。

要生成回答，配置一个 OpenAI 兼容接口。默认是 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="你的密钥"
python main.py query-financial-report 2025年毛利率是多少 --output both
```

`--output` 可以是 `hits`（只打印命中）、`answer`（只打印回答）或 `both`。不传时看 `config.yaml` 里的 `llm.enabled`：`true` 相当于 `both`，`false` 相当于 `hits`。

## 查询接口

```powershell
python api.py
python api.py --host 127.0.0.1 --port 8100
python main.py serve
python main.py serve --host 127.0.0.1 --port 8100
```

`python api.py` 和 `python main.py serve` 启动的是同一个查询接口。

`embedding.keep_in_memory` 控制向量模型是否常驻。设为 `true` 时，服务启动就加载模型，后续请求直接编码；设为 `false` 时，每次向量检索临时加载，请求结束后释放。启动参数可以覆盖配置：`python api.py --keep-model` 或 `python main.py serve --no-keep-model`。`GET /health` 会返回 `keep_in_memory` 和 `model_loaded`。

默认只监听本机的 `8100` 端口。knowledge 和 financial_report 是两套接口，路径已经分开，请求体里不再传 `collection`。把另一套的筛选字段写进来会返回 422。

`POST /knowledge/retrieve` 和 `POST /knowledge/query` 只查投资通识。可选筛选是 `doc_type`、`figure`、`occasion` 和 `year`。目前 `doc_type` 只有 `investment_theory`。`figure` 是 `唐朝`、`buffett`、`munger`；`occasion` 是 `letter`、`meeting`、`speech`、`article`。不传表示该层不限制。

`POST /financial_report/retrieve` 和 `POST /financial_report/query` 只查定期报告。可选筛选是 `stock_name`、`stock_code`、`period`、`section` 和 `entity`。`section` 是 `profile`、`segment`、`mda`、`risk`、`material`、`governance`、`financials`、`notes`；`entity` 是 `consolidated` 或 `parent`，只用于财务报表和附注。`period` 写成 `2024`、`2024Q1` 或 `2024H1`，格式不对返回 400。

`question` 必填且不能为空白。`mode` 只能是 `hybrid`、`bm25` 或 `vector`，`top_k` 必须是大于 0 的整数。不传时分别使用 `retrieval.mode` 和 `retrieval.top_k`。字段说明见 `http://127.0.0.1:8100/docs`。`GET /health` 返回服务是否在运行，以及索引文件是否存在。

`/knowledge/retrieve` 和 `/financial_report/retrieve` 只返回命中，不调用大模型。对应的 `/query` 用 `output` 选择返回内容：`hits` 只返回命中，`answer` 只返回回答，`both` 两者都返回。不传 `output` 时看 `llm.enabled`：`true` 相当于 `both`，`false` 相当于 `hits`。

```powershell
$body = @{ question = "2025年毛利率是多少"; stock_code = "600519"; period = "2025" } | ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:8100/financial_report/retrieve -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

四个接口都返回 `question`、`collection` 和 `mode`。`collection` 由路径决定，是 `knowledge` 或 `financial_report`。`retrieve` 带 `hits`。`query` 再带实际采用的 `output`：`hits` 和 `both` 带 `hits`，`answer` 和 `both` 带 `answer`。大模型失败时返回 502，并带上已经检索到的 `hits`。每条命中仍有 `scope`、`doc_type`、`stock_name`、`stock_code`、`year`、`period`、`published_on`、`section`、`entity`、`figure`、`occasion`；这份材料用不到的字段是空字符串，没有年份时 `year` 为 0。投资理论的展示名来自 `figure`。要让其他机器访问，把 `--host` 改成 `0.0.0.0`。

## 更换 embedding

1. 修改 `config.yaml` 的 `embedding.model_name`（以及需要区分的 `embedding.version`）。
2. 执行 `python main.py reembed`。

脚本会读出 sqlite 里的全部 `chunk_text`，清空 Chroma 后写入新向量，并更新 `embed_model_name`。BM25 不重建。

改切分参数后，先重跑对应业务切分脚本更新 `corpus/03_splited/` 的 JSON，再执行 `python main.py ingest-split`。BM25 会随原文变化一起重建。

## 边界

几十到上百份文档可以在单机上跑。这个结构不面向百万级文档，也不面向高并发在线服务。Chroma 存向量和同一套元信息；回答和重算向量都以 sqlite 里的原文为准。
