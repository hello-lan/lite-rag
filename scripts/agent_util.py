"""供切分脚本导入的轻量 LLM agent。

调用方传入 system_prompt（决策规则、切分策略、输出结构等）和待处理原文 text，
本模块负责按 config.yaml 的 llm 配置调用 OpenAI 兼容接口，返回模型原文回复。

示例：

    from scripts.agent_util import SimpleAgent, run_agent

    agent = SimpleAgent(system_prompt="把正文按问答切开，只输出 JSON …")
    reply = agent.run(raw_text)

    # 或函数式
    reply = run_agent(system_prompt, raw_text)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lite_rag.config import load_config, load_dotenv

log = logging.getLogger("agent_util")

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_JSON_NUDGE = "请只输出合法 JSON，不要 Markdown 代码块以外的解释文字。"
_RETRY_SLEEP = 1.0


class AgentError(RuntimeError):
    """调用大模型失败或返回无法解析。"""


def _resolve_llm(cfg: dict | None = None) -> dict:
    load_dotenv()
    cfg = cfg if cfg is not None else load_config()
    llm = cfg.get("llm")
    if not isinstance(llm, dict):
        raise AgentError("config.yaml 缺少 llm 配置")
    for key in ("base_url", "api_key_env", "model"):
        if not llm.get(key):
            raise AgentError(f"config.yaml 缺少 llm.{key}")
    return llm


def _is_retryable_http(code: int) -> bool:
    return code == 429 or code >= 500


def _chat_once(
    *,
    system_prompt: str,
    text: str,
    llm: dict,
    api_key: str,
    temperature: float,
    timeout: float,
    model: str,
) -> str:
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
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
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AgentError(f"大模型接口返回 {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AgentError(f"无法连接大模型接口: {exc.reason}") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AgentError(f"大模型返回了无法识别的内容: {body}") from exc
    if not isinstance(content, str):
        raise AgentError(f"大模型返回 content 不是字符串: {body}")
    return content.strip()


def _chat(
    *,
    system_prompt: str,
    text: str,
    cfg: dict | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    model: str | None = None,
    retries: int = 1,
) -> str:
    if not system_prompt or not system_prompt.strip():
        raise ValueError("system_prompt 不能为空")
    if text is None:
        raise ValueError("text 不能为 None")

    llm = _resolve_llm(cfg)
    env_name = llm["api_key_env"]
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        raise AgentError(f"未设置环境变量 {env_name}，无法调用大模型")

    resolved_model = model or llm["model"]
    resolved_temp = float(
        llm.get("temperature", 0.1) if temperature is None else temperature
    )
    wait = float(llm.get("timeout", 120) if timeout is None else timeout)
    attempts = max(1, int(retries) + 1)

    log.info("LLM 请求 model=%s chars=%s attempts=%s", resolved_model, len(text), attempts)

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _chat_once(
                system_prompt=system_prompt.strip(),
                text=text,
                llm=llm,
                api_key=api_key,
                temperature=resolved_temp,
                timeout=wait,
                model=resolved_model,
            )
        except AgentError as exc:
            last_error = exc
            msg = str(exc)
            retryable = "无法连接" in msg
            if not retryable and "大模型接口返回" in msg:
                try:
                    code = int(msg.split("返回", 1)[1].split(":", 1)[0].strip())
                except (ValueError, IndexError):
                    code = 0
                retryable = _is_retryable_http(code)
            if not retryable or attempt + 1 >= attempts:
                raise
            log.warning("LLM 瞬时失败，重试 %s/%s: %s", attempt + 1, attempts - 1, msg[:200])
            time.sleep(_RETRY_SLEEP)

    raise AgentError(f"大模型调用失败: {last_error}") from last_error


def parse_json_reply(reply: str) -> Any:
    """从模型回复中取出 JSON（支持裸 JSON 或 ```json 代码块）。"""
    text = reply.strip()
    if not text:
        raise AgentError("模型回复为空，无法解析 JSON")

    candidates = [text]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    # 兜底：截取第一个 { 或 [ 到最后一个 } 或 ]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    errors: list[Exception] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(exc)

    raise AgentError(f"无法从模型回复解析 JSON: {reply[:500]}") from (
        errors[-1] if errors else None
    )


class SimpleAgent:
    """用固定 system_prompt 处理多段原文的可复用 agent。"""

    def __init__(
        self,
        system_prompt: str,
        *,
        cfg: dict | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        model: str | None = None,
        retries: int = 1,
    ) -> None:
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt 不能为空")
        self.system_prompt = system_prompt.strip()
        self.cfg = cfg
        self.temperature = temperature
        self.timeout = timeout
        self.model = model
        self.retries = max(0, int(retries))

    def run(self, text: str) -> str:
        """把 text 交给模型，返回原始回复文本。"""
        return _chat(
            system_prompt=self.system_prompt,
            text=text,
            cfg=self.cfg,
            temperature=self.temperature,
            timeout=self.timeout,
            model=self.model,
            retries=self.retries,
        )

    def run_json(self, text: str) -> Any:
        """同 run，但把回复解析成 JSON；解析失败时再请求一次并附带 JSON 提醒。"""
        reply = self.run(text)
        try:
            return parse_json_reply(reply)
        except AgentError as first_exc:
            log.warning("JSON 解析失败，再请求一次: %s", str(first_exc)[:200])
            nudged = (
                f"{text.rstrip()}\n\n---\n{_JSON_NUDGE}"
                if text
                else _JSON_NUDGE
            )
            retry_prompt = f"{self.system_prompt}\n\n{_JSON_NUDGE}"
            reply2 = _chat(
                system_prompt=retry_prompt,
                text=nudged,
                cfg=self.cfg,
                temperature=self.temperature,
                timeout=self.timeout,
                model=self.model,
                retries=self.retries,
            )
            try:
                return parse_json_reply(reply2)
            except AgentError as second_exc:
                log.warning("二次 JSON 仍失败，回复前缀: %s", reply2[:300])
                raise second_exc from first_exc


def run_agent(
    system_prompt: str,
    text: str,
    *,
    cfg: dict | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    model: str | None = None,
    retries: int = 1,
    as_json: bool = False,
) -> str | Any:
    """一次性调用：system_prompt 定义规则，text 为待切原文。

    as_json=True 时解析并返回 JSON 对象，否则返回模型原文。
    """
    agent = SimpleAgent(
        system_prompt,
        cfg=cfg,
        temperature=temperature,
        timeout=timeout,
        model=model,
        retries=retries,
    )
    return agent.run_json(text) if as_json else agent.run(text)
