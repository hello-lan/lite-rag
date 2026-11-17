"""把命中的 chunk 交给 OpenAI 兼容接口，生成回答。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from lite_rag.meta import source_label


def build_context(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            f"[{hit['rank']}] 来源：{source_label(hit)}（chunk #{hit['chunk_idx']}）\n{hit['chunk_text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(cfg: dict, question: str, hits: list[dict]) -> str:
    if not hits:
        return "没有检索到可用材料，无法生成回答。"
    llm = cfg["llm"]
    env_name = llm["api_key_env"]
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        raise SystemExit(f"未设置环境变量 {env_name}，无法调用大模型。")

    payload = {
        "model": llm["model"],
        "temperature": float(llm.get("temperature", 0.1)),
        "messages": [
            {"role": "system", "content": llm.get("system_prompt") or "只根据材料回答。"},
            {"role": "user", "content": f"材料：\n{build_context(hits)}\n\n问题：{question}"},
        ],
    }
    request = urllib.request.Request(
        llm["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    timeout = float(llm.get("timeout", 120))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"大模型接口返回 {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"无法连接大模型接口: {exc.reason}") from exc
    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"大模型返回了无法识别的内容: {body}") from exc
