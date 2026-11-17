"""从 docs/ 读出 PDF 和 Markdown，并切成 chunk。"""

from __future__ import annotations

import hashlib
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}


def iter_source_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        return []
    files = [
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda path: path.as_posix().lower())


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
