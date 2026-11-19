# 财报 RAG（BM25 + 向量 + RRF）

本地查询公司材料和投资通识。向量放在 Chroma，chunk 原文和元信息放在 sqlite。BM25 负责年份、科目、指标名这类字面匹配，BGE-M3 负责语义，两路结果用 RRF 合并。查某一家公司或某一类材料时，两路使用同一套筛选条件。

换 embedding 模型时只重算向量，不用重新解析 PDF，也不用重建 BM25。

## 目录

```
corpus/02_processed/company/filing/     年报、半年报、季报
corpus/02_processed/company/ir_qa/      互动易、上证 e 互动
corpus/02_processed/company/survey/     投资者调研
corpus/02_processed/company/research/   研报
corpus/02_processed/company/news/       新闻
corpus/02_processed/knowledge/investment_theory/   投资理论
corpus/02_processed/knowledge/beginner/ 投资入门答疑
corpus/raw/sse、corpus/raw/irm       采集脚本的默认输出，入库时用 --dir 或改 --output
corpus/03_splited/                  已经切好的 JSON，入库时只做嵌入
corpus/tangchao/               唐朝书房原文
corpus/BRK-Annual-Meeting/     伯克希尔股东大会记录
models/bge-m3/               本地 BGE-M3 模型
data/                        入库生成的索引：meta.db、bm25_index.pkl、chroma/
config.yaml                  切分、模型、召回条数、大模型
main.py                      命令行入口：ingest、ingest-split、query、reembed、serve
api.py                       查询接口：/knowledge、/financial_report 各自的 retrieve 与 query
lite_rag/                    解析、索引、检索、回答
scripts/                     上证 e 互动、深交所互动易采集
```

直接放在 `corpus/02_processed/` 根目录、且不在上述子目录里的文件，记成公司定期报告。文件名用 `600519_贵州茅台_2024年报.pdf` 这种形式时，会解析出代码、简称、年份和报告期。一份文件的所有 chunk 共用这一套元信息。

材料如果还在旧的 `docs/` 或 `corpus/raw/`，原文挪到 `corpus/02_processed/` 下的同样位置，已经切好的 JSON 挪到 `corpus/03_splited/`。相对这两个目录的路径不变时，已经入库的文档 id 也不变。

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

问答先写成 Markdown，再和财报一起入库。两个脚本共用代码列表、文本清洗和写出，站点解析各自保留。默认分别写到 `corpus/raw/sse` 和 `corpus/raw/irm`。要交给默认入库目录，可以把 `--output` 指到 `corpus/02_processed/company/ir_qa`，或入库时加上 `--dir corpus/raw`。`sse`、`irm` 这两个目录名入库时记成互动问答。

```powershell
python scripts/crawl_sse.py --code 600519
python scripts/crawl_irm.py --code 000001
python scripts/crawl_sse.py --code 600519 --output corpus/02_processed/company/ir_qa
```

## 入库

```powershell
python main.py ingest
python main.py ingest --dir corpus/02_processed/company/filing
```

不传 `--dir` 时读取 `paths.docs_dir`，默认是 `corpus/02_processed`。相对路径相对项目根目录。指定目录落在 `corpus/02_processed` 里面时，文档 id 仍相对这个配置目录，并且只清理该子目录里已经不见的文件。指定目录在配置目录外面时，id 相对本次目录，也只清理这次扫到的范围。

支持 `.pdf`、`.md`、`.markdown`。文字版 PDF 用 pypdf 抽文本，Markdown 用 marko 抽文本（含表格）。扫出来的图片版财报抽不到字，需要另外做 OCR，本项目不包含这一步。

Markdown 可以在文首用 YAML 覆盖目录和文件名解析出的元信息。这段内容不会进入 chunk。和目录不一致时采用文首，并在日志里说明。`scope` 只能是 `company` 或 `knowledge`；`doc_type` 在公司侧是 `filing`、`ir_qa`、`survey`、`research`、`news`，在通识侧是 `investment_theory`、`beginner`。`investment_theory` 还要写 `figure` 和 `occasion`：`figure` 是 `唐朝`、`buffett`（巴菲特）、`munger`（芒格）；`occasion` 是 `letter`（致投资者的信）、`meeting`（股东大会）、`speech`（演讲）、`article`（文章）。一份材料只标一个主要人物，不写 `author`。

```yaml
---
doc_type: research
company: 贵州茅台
stock_code: "600519"
year: 2025
period: "2025Q1"
published_on: "2025-04-02"
author: 中信证券
title: 贵州茅台2025年一季报点评
---
```

`stock_code` 建议加引号，避免 `000001` 被 YAML 当成数字。`period` 写成 `2024`、`2024Q1` 或 `2024H1`，表示累计口径。没有的文本字段留空，没有的年份记为 0。

财报向量只写非空的 `company`、`stock_code`、`period`、`section`、`entity`，不写 `year`。`section` 可以是 `profile`（公司与业务）、`segment`（分产品分地区）、`mda`（经营讨论）、`risk`（风险因素）、`material`（重要事项）、`governance`（股东与治理）、`financials`（财务报表）、`notes`（附注）。`entity` 只在 `financials` 或 `notes` 上使用，取 `consolidated`（合并）或 `parent`（母公司）。这两项不会自动识别，写在文首或预切分 JSON 里才会入库。查财报用 `period`，不要用 `year`；`year` 只筛投资通识。

再次运行时，原文和切分配置都没变的文件会跳过；只有元信息变了时，只更新 sqlite 和 Chroma 的标签，不重新算向量。文件从 `corpus/02_processed/` 删除后，对应 chunk 会从 sqlite、Chroma 和 BM25 里清掉。`corpus/03_splited/` 里的 JSON 不在这次扫描里，不会被删掉。用 `--dir` 只扫一个子目录时，这个子目录以外的 PDF 和 Markdown 也不会被删掉。

已有库如果只缺 `section`、`entity`、`figure` 或 `occasion`，下次入库会补上这些列，不清空原文。更早的表结构仍会清空 sqlite、Chroma 和 BM25，再按这次扫描的目录重新入库。`--bm25-only` 和 `reembed` 遇到那种旧结构会提示先完整入库。

```powershell
python main.py ingest --force       # 全部重新切分并写入
python main.py ingest --bm25-only   # 只按 sqlite 重建 BM25
```

## 预切分入库

问答这类材料不适合按 `chunk_size` 切开时，用自己的脚本先切好，存成 JSON，再只做嵌入和入库。一个 JSON 文件对应一份逻辑文档，可以放在 `corpus/03_splited/` 的任意子目录里。类型以文件里的 `meta` 为准，不看子目录名字。

`meta` 的字段和 Markdown 文首相同。`scope` 和 `doc_type` 必填，其余可以不写。单条 chunk 可以改自己的年份、日期或其他字段，没写的继承文档级 `meta`。`split` 用来标记这套切分；不写时记为 `external`。改 `config.yaml` 里的 `chunk_size` 不会重算这些 chunk，文本或 `split` 变了才会重写。

```json
{
  "split": "qa-pair-v1",
  "meta": {
    "scope": "company",
    "doc_type": "ir_qa",
    "title": "贵州茅台互动问答",
    "company": "贵州茅台",
    "stock_code": "600519"
  },
  "chunks": [
    {"text": "问：……\n答：……", "published_on": "2025-03-12", "year": 2025}
  ]
}
```

`chunks` 不能为空，`text` 必须是非空字符串。字段不认识、年份和报告期对不上时，会报出文件路径并停下。

```powershell
python main.py ingest-split --check   # 只校验 JSON
python main.py ingest-split           # 嵌入并写入
python main.py ingest-split --force   # 即使没变也重新嵌入
```

`ingest-split` 只删除 `corpus/03_splited/` 里已经不见的 JSON。`corpus/02_processed/` 里的 PDF 和 Markdown 留在索引中。两边都写入之后，`python main.py reembed` 会按 sqlite 里的全部原文重算向量。

## 查询

```powershell
python main.py query 2025年毛利率是多少 --scope company
python main.py query 公司主要风险有哪些 --scope company --mode vector
python main.py query 分红方案 --scope company --mode bm25 --code 600519 --doc-type filing
python main.py query 2025年资本开支是多少 --scope company --mode hybrid --top-k 4 --company 贵州茅台 --period 2025 --section financials --entity consolidated
python main.py query 什么是安全边际 --scope knowledge --doc-type investment_theory --figure 唐朝 --occasion article --year 2018
```

默认打印命中的原文。`--scope` 必填，只能是 `company` 或 `knowledge`。`--mode` 不写时使用 `config.yaml` 的 `retrieval.mode`。不传 `--doc-type`、`--figure`、`--occasion`、`--company`、`--code`、`--year`、`--period`、`--section`、`--entity` 时，在该范围内搜索全部材料。`--figure` 和 `--occasion` 只用于 `investment_theory`。传入公司或代码后，向量和 BM25 都只保留这家公司。程序不会从问题文本里猜测公司和类型。查询范围包含财报时不能传 `--year`，报告期用 `--period`。`--year` 只用于投资通识。

混合检索时，BM25 和向量各取 `candidate_k` 条（默认 10），再按 RRF 截到 `top_k` 条（默认 4）。RRF 只看名次：`1 / (60 + rank + 1)`，不会把两种分数直接相加。

要生成回答，配置一个 OpenAI 兼容接口。默认是 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="你的密钥"
python main.py query 2025年毛利率是多少 --scope company --output both
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

`POST /financial_report/retrieve` 和 `POST /financial_report/query` 只查定期报告。可选筛选是 `company`、`stock_code`、`period`、`section` 和 `entity`。`section` 是 `profile`、`segment`、`mda`、`risk`、`material`、`governance`、`financials`、`notes`；`entity` 是 `consolidated` 或 `parent`，只用于财务报表和附注。`period` 写成 `2024`、`2024Q1` 或 `2024H1`，格式不对返回 400。

`question` 必填且不能为空白。`mode` 只能是 `hybrid`、`bm25` 或 `vector`，`top_k` 必须是大于 0 的整数。不传时分别使用 `retrieval.mode` 和 `retrieval.top_k`。字段说明见 `http://127.0.0.1:8100/docs`。`GET /health` 返回服务是否在运行，以及索引文件是否存在。

`/knowledge/retrieve` 和 `/financial_report/retrieve` 只返回命中，不调用大模型。对应的 `/query` 用 `output` 选择返回内容：`hits` 只返回命中，`answer` 只返回回答，`both` 两者都返回。不传 `output` 时看 `llm.enabled`：`true` 相当于 `both`，`false` 相当于 `hits`。

```powershell
$body = @{ question = "2025年毛利率是多少"; stock_code = "600519"; period = "2025" } | ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:8100/financial_report/retrieve -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

四个接口都返回 `question`、`collection` 和 `mode`。`collection` 由路径决定，是 `knowledge` 或 `financial_report`。`retrieve` 带 `hits`。`query` 再带实际采用的 `output`：`hits` 和 `both` 带 `hits`，`answer` 和 `both` 带 `answer`。大模型失败时返回 502，并带上已经检索到的 `hits`。每条命中仍有 `scope`、`doc_type`、`company`、`stock_code`、`year`、`period`、`published_on`、`author`、`section`、`entity`、`figure`、`occasion`；这份材料用不到的字段是空字符串，没有年份时 `year` 为 0。投资理论的展示名来自 `figure`，`author` 留空。要让其他机器访问，把 `--host` 改成 `0.0.0.0`。

## 更换 embedding

1. 修改 `config.yaml` 的 `embedding.model_name`（以及需要区分的 `embedding.version`）。
2. 执行 `python main.py reembed`。

脚本会读出 sqlite 里的全部 `chunk_text`，清空 Chroma 后写入新向量，并更新 `embed_model_name`。BM25 不重建。

改 `chunk_size`、`chunk_overlap` 或 `split.version` 才需要重新执行 `python main.py ingest`。这时原文变了，BM25 会一起重建。`corpus/03_splited/` 里的 JSON 不走这套切分；要更新它们，改 JSON 后再运行 `python main.py ingest-split`。

## 边界

几十到上百份文档可以在单机上跑。这个结构不面向百万级文档，也不面向高并发在线服务。Chroma 存向量和同一套元信息；回答和重算向量都以 sqlite 里的原文为准。
