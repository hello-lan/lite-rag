"""从 docs/raw 读出 PDF 和 Markdown，或读入 docs/split 里已切分的 JSON。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from lite_rag.meta import DocMeta, build_doc_meta, doc_meta_from_fields

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}
_SPLIT_KEYS = frozenset({"split", "meta", "chunks"})
_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)


def iter_source_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        return []
    files = [
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda path: path.as_posix().lower())


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


def md_to_text(src: str) -> str:
    import marko
    from marko.ext.gfm import GFM

    document = marko.Markdown(extensions=[GFM]).parse(src)
    return _render_markdown(document).strip()


def _render_markdown(node) -> str:
    if isinstance(node, str):
        return node
    name = type(node).__name__
    children = getattr(node, "children", None)
    if name == "TableRow" and children and not isinstance(children, str):
        cells = [_render_markdown(child).strip() for child in children]
        return " | ".join(cell for cell in cells if cell) + "\n"
    if children is None:
        return ""
    inner = children if isinstance(children, str) else "".join(_render_markdown(child) for child in children)
    if name == "Link":
        dest = getattr(node, "dest", "") or ""
        if dest and dest not in inner:
            return f"{inner} ({dest})"
        return inner
    if name == "ListItem":
        return "- " + inner.strip() + "\n"
    if name == "TableCell":
        return inner.strip()
    if name in {"Paragraph", "Heading", "Quote", "FencedCode", "CodeBlock", "BlankLine", "ThematicBreak"}:
        text = inner.strip()
        return f"{text}\n\n" if text else "\n"
    if name in {"List", "Table", "TableHead", "TableBody"}:
        return inner + ("\n" if inner and not inner.endswith("\n") else "")
    return inner


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page).strip()


def load_source(docs_dir: Path, path: Path) -> tuple[str, DocMeta, str | None]:
    """读出正文和文档元信息。文首 YAML 不进入正文。"""
    try:
        relative = path.resolve().relative_to(docs_dir.resolve()).as_posix()
    except ValueError as exc:
        raise SystemExit(f"{path} 不在文档目录 {docs_dir} 里") from exc
    suffix = path.suffix.lower()
    front: dict | None = None
    try:
        if suffix == ".pdf":
            text = pdf_to_text(path)
        elif suffix in {".md", ".markdown"}:
            front, body = _split_front_matter(path.read_text(encoding="utf-8-sig"))
            text = md_to_text(body)
        else:
            raise ValueError(f"不支持的文件类型: {path.name}")
        doc, warning = build_doc_meta(relative, path.stem, front)
    except ValueError as exc:
        raise SystemExit(f"{relative}: {exc}") from exc
    return text, doc, warning


def _split_front_matter(raw: str) -> tuple[dict | None, str]:
    matched = _FRONT_MATTER.match(raw)
    if not matched:
        return None, raw
    try:
        loaded = yaml.safe_load(matched.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(f"文首 YAML 无法解析: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("文首信息必须是 YAML 映射")
    return loaded, raw[matched.end() :]


def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_text(path)
    if suffix in {".md", ".markdown"}:
        return md_to_text(path.read_text(encoding="utf-8-sig"))
    raise ValueError(f"不支持的文件类型: {path}")


def split_document(text: str, cfg: dict) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    split = cfg["split"]
    separators = list(split.get("separators") or ["\n\n", "\n", "。", "；", "，", " ", ""])
    kwargs = dict(
        chunk_size=int(split["chunk_size"]),
        chunk_overlap=int(split["chunk_overlap"]),
        separators=separators,
        length_function=len,
    )
    try:
        splitter = RecursiveCharacterTextSplitter(keep_separator=True, **kwargs)
    except TypeError:
        splitter = RecursiveCharacterTextSplitter(**kwargs)
    return [part.strip() for part in splitter.split_text(text) if part and part.strip()]
