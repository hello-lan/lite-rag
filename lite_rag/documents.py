"""读入 corpus/03_splited 里已切分的 JSON。

lite_rag 只导入预切分结果；PDF / Markdown 抽取与切分由 scripts/ 完成。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lite_rag.meta import DocMeta, doc_meta_from_fields

_SPLIT_KEYS = frozenset({"split", "meta", "chunks"})


def iter_split_files(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        return []
    files = [
        path
        for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".json"
    ]
    return sorted(files, key=lambda path: path.as_posix().lower())


def load_split_file(path: Path) -> tuple[list[str], list[DocMeta], str]:
    """读入一份已切分 JSON。返回正文、逐条元信息和 split 标记。"""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 无法解析: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("JSON 必须是对象")
    unknown = sorted(set(loaded) - _SPLIT_KEYS)
    if unknown:
        raise ValueError("不认识的字段: " + "、".join(unknown))
    meta = loaded.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("meta 必须是对象")
    chunks = loaded.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks 不能为空")
    split_name = loaded.get("split", "external")
    if not isinstance(split_name, str) or not split_name.strip():
        raise ValueError("split 必须是非空字符串")

    try:
        base = doc_meta_from_fields(meta, title_default=path.stem)
    except ValueError as exc:
        raise ValueError(f"meta: {exc}") from exc
    texts: list[str] = []
    docs: list[DocMeta] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"chunks[{index}] 必须是对象")
        if "text" not in chunk:
            raise ValueError(f"chunks[{index}] 缺少 text")
        text = chunk["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"chunks[{index}].text 必须是非空字符串")
        override = {key: value for key, value in chunk.items() if key != "text"}
        try:
            doc = doc_meta_from_fields(override, base=base) if override else base
        except ValueError as exc:
            raise ValueError(f"chunks[{index}]: {exc}") from exc
        texts.append(text.strip())
        docs.append(doc)
    return texts, docs, split_name.strip()


def source_doc_id(docs_dir: Path, path: Path) -> str:
    relative = path.resolve().relative_to(docs_dir.resolve()).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()


def relative_to_root(cfg: dict, path: Path) -> str:
    root = cfg["_root"]
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()
