"""调用 POST /query 的示例。先在项目根目录启动服务：python api.py"""

import json

import requests

URL = "http://127.0.0.1:8008/query"


def query(question: str, **options) -> dict:
    """options 可传 mode、top_k、answer、scope、doc_type、company、stock_code、year。不传则使用 config.yaml，筛选默认不限制。"""
    resp = requests.post(
        URL,
        json={"question": question, **options},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    data = query(
        "2023年分红是多少？",
        mode="hybrid",  # hybrid | bm25 | vector
        top_k=5,  # >= 1
        answer=True,  # 是否调用大模型
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))

    # 只查贵州茅台 2023 年定期报告：
    # data = query("2023年营业收入是多少？", stock_code="600519", doc_type="filing", year=2023)
    # 只查投资通识：
    # data = query("什么是安全边际？", scope="knowledge", doc_type="theory")


if __name__ == "__main__":
    main()
