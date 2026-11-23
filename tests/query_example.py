"""调用查询接口的示例。先在项目根目录启动服务：python api.py"""

import json

import requests

KNOWLEDGE_URL = "http://127.0.0.1:8008/knowledge/query"
FINANCIAL_REPORT_URL = "http://127.0.0.1:8008/financial_report/query"


def query(url: str, question: str, **options) -> dict:
    """knowledge 可传 doc_type、figure、occasion、year。financial_report 可传 stock_name、stock_code、period、section、entity。还可传 mode、top_k、output。不传 mode、top_k、output 时使用 config.yaml。"""
    resp = requests.post(
        url,
        json={"question": question, **options},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    data = query(
        FINANCIAL_REPORT_URL,
        "2023年分红是多少？",
        stock_code="600519",
        period="2023",
        mode="hybrid",  # hybrid | bm25 | vector
        top_k=5,  # >= 1
        output="both",  # hits | answer | both；不传则看 llm.enabled
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))

    # 只查投资通识：
    # data = query(
    #     KNOWLEDGE_URL,
    #     "什么是安全边际？",
    #     doc_type="investment_theory",
    #     figure="唐朝",
    #     occasion="article",
    #     year=2018,
    #     output="hits",
    # )


if __name__ == "__main__":
    main()
