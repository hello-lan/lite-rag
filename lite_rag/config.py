"""读取 config.yaml，并把相对路径解析到项目根目录。"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml

log = logging.getLogger("rag")
ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """读取项目根目录的 .env。已经存在的环境变量不覆盖。"""
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def configure_stdio() -> None:
    load_dotenv()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.ERROR)


def load_config(path: Path | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"找不到配置文件: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"配置文件不是合法的 YAML 映射: {cfg_path}")
    cfg["_root"] = cfg_path.parent.resolve()
    cfg["_config_path"] = cfg_path.resolve()
    overlap = int(cfg["split"]["chunk_overlap"])
    size = int(cfg["split"]["chunk_size"])
    if overlap >= size:
        raise SystemExit("split.chunk_overlap 必须小于 split.chunk_size")
    mode = cfg["retrieval"]["mode"]
    if mode not in {"hybrid", "bm25", "vector"}:
        raise SystemExit("retrieval.mode 只能是 hybrid、bm25 或 vector")
    embedding = cfg.get("embedding")
    if isinstance(embedding, dict) and "keep_in_memory" in embedding:
        if not isinstance(embedding["keep_in_memory"], bool):
            raise SystemExit("embedding.keep_in_memory 只能是 true 或 false")
    return cfg


def resolve_path(cfg: dict, relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return cfg["_root"] / path


def split_config_of(cfg: dict) -> str:
    split = cfg["split"]
    return (
        f"{split['version']}:chunk_size={int(split['chunk_size'])},"
        f"overlap={int(split['chunk_overlap'])}"
    )
