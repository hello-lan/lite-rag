"""财报 RAG。

documents 负责把 PDF / Markdown 切成 chunk，store 负责 sqlite、Chroma 和 BM25，
search 负责召回，answer 负责组织回答。入库在 ingest。
项目根目录的 main.py 是命令行入口。
"""
