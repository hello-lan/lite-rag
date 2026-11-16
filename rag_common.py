"""财报 RAG 的共用能力：解析、切分、sqlite、Chroma、BM25、RRF。"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import sqlite3
import sys
from pathlib import Path

import yaml

log = logging.getLogger("rag")

ROOT = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}
_PUNCT = re.compile(r"^[\s\W_]+$", re.UNICODE)
_jieba_ready = False
_model = None


def configure_stdio() -> None:
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


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = "DEFERRED"
    return conn


def init_meta_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_db(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_doc_id TEXT,
                source_name TEXT,
                source_path TEXT,
                chunk_text TEXT,
                chunk_idx INTEGER,
                split_config TEXT,
                embed_model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_doc_id)"
        )
        conn.commit()
    finally:
        conn.close()


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


def tokenize(text: str) -> list[str]:
    global _jieba_ready
    import jieba

    if not _jieba_ready:
        jieba.setLogLevel(logging.WARNING)
        jieba.initialize()
        _jieba_ready = True
    raw = [token.strip() for token in jieba.lcut(text) if token.strip()]
    tokens = [token for token in raw if not _PUNCT.match(token)]
    return tokens or raw or ["空"]


def open_chroma(cfg: dict):
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise SystemExit("缺少 chromadb，请先执行 pip install -r requirements.txt") from exc

    path = resolve_path(cfg, cfg["paths"]["chroma_dir"])
    path.mkdir(parents=True, exist_ok=True)
    try:
        return chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
    except TypeError:
        return chromadb.PersistentClient(path=str(path))


def open_collection(client, cfg: dict, reset: bool = False):
    name = cfg["paths"]["collection_name"]
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def chroma_meta(chunk_id: int, source_name: str, source_doc_id: str) -> dict:
    return {
        "chunk_id": int(chunk_id),
        "source_name": source_name,
        "source_doc_id": source_doc_id,
    }


def chroma_upsert(collection, ids: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
    batch_size = 128
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


def resolve_embedding_source(cfg: dict) -> str:
    name = cfg["embedding"]["model_name"]
    path = Path(name)
    if not path.is_absolute():
        path = (cfg.get("_root") or ROOT) / name
    if path.is_dir():
        return str(path)
    return name


def get_model(cfg: dict):
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("缺少 sentence-transformers，请先执行 pip install -r requirements.txt") from exc

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = cfg["embedding"].get("device") or "auto"
    kwargs = {}
    if device != "auto":
        kwargs["device"] = device
    model_name = resolve_embedding_source(cfg)
    log.info("加载 embedding 模型 %s", model_name)
    model = SentenceTransformer(model_name, **kwargs)
    model.max_seq_length = int(cfg["embedding"].get("max_seq_length", 1024))
    _model = model
    return model


def embed_texts(model, texts: list[str], cfg: dict, is_query: bool = False) -> list[list[float]]:
    prefix = ""
    if is_query:
        prefix = cfg["embedding"].get("query_instruction") or ""
    payload = [prefix + text for text in texts] if prefix else list(texts)
    batch_size = int(cfg["embedding"].get("batch_size", 8))
    normalize = bool(cfg["embedding"].get("normalize", True))
    vectors = model.encode(
        payload,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=len(payload) > batch_size,
    )
    return [row.tolist() for row in vectors]


def load_doc_chunks(conn: sqlite3.Connection, source_doc_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, chunk_idx, chunk_text, split_config, source_name, source_doc_id
        FROM chunks
        WHERE source_doc_id = ?
        ORDER BY chunk_idx
        """,
        (source_doc_id,),
    ).fetchall()


def same_chunks(rows: list[sqlite3.Row], texts: list[str], split_cfg: str) -> bool:
    if len(rows) != len(texts):
        return False
    for index, (row, text) in enumerate(zip(rows, texts)):
        if row["split_config"] != split_cfg or row["chunk_idx"] != index or row["chunk_text"] != text:
            return False
    return True


def chroma_has_ids(collection, chunk_ids: list[int]) -> bool:
    if not chunk_ids:
        return True
    found: set[str] = set()
    ids = [str(chunk_id) for chunk_id in chunk_ids]
    for start in range(0, len(ids), 100):
        got = collection.get(ids=ids[start : start + 100])
        found.update(got.get("ids") or [])
    return len(found) == len(ids)


def delete_doc(conn: sqlite3.Connection, collection, source_doc_id: str) -> int:
    rows = conn.execute("SELECT id FROM chunks WHERE source_doc_id = ?", (source_doc_id,)).fetchall()
    ids = [str(row["id"]) for row in rows]
    if ids:
        try:
            collection.delete(ids=ids)
        except Exception:
            log.debug("按 chunk id 删除向量失败", exc_info=True)
    try:
        collection.delete(where={"source_doc_id": source_doc_id})
    except Exception:
        log.debug("按文档 id 删除向量失败", exc_info=True)
    conn.execute("DELETE FROM chunks WHERE source_doc_id = ?", (source_doc_id,))
    return len(ids)


def insert_chunks(
    conn: sqlite3.Connection,
    *,
    source_doc_id: str,
    source_name: str,
    source_path: str,
    texts: list[str],
    split_cfg: str,
    embed_model_name: str,
) -> list[int]:
    ids: list[int] = []
    for index, text in enumerate(texts):
        cursor = conn.execute(
            """
            INSERT INTO chunks (
                source_doc_id, source_name, source_path, chunk_text,
                chunk_idx, split_config, embed_model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source_doc_id, source_name, source_path, text, index, split_cfg, embed_model_name),
        )
        ids.append(int(cursor.lastrowid))
    return ids


def rebuild_bm25(conn: sqlite3.Connection, index_path: Path) -> int:
    from rank_bm25 import BM25Okapi

    rows = conn.execute("SELECT id, chunk_text FROM chunks ORDER BY id").fetchall()
    chunk_ids = [int(row["id"]) for row in rows]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not chunk_ids:
        payload = {"chunk_ids": [], "bm25": None}
    else:
        tokenized = [tokenize(row["chunk_text"]) for row in rows]
        payload = {"chunk_ids": chunk_ids, "bm25": BM25Okapi(tokenized)}
    with index_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return len(chunk_ids)


def load_bm25(index_path: Path) -> dict:
    if not index_path.exists():
        raise SystemExit("没有找到 bm25_index.pkl，请先运行 python rag.py ingest")
    with index_path.open("rb") as f:
        payload = pickle.load(f)
    if not payload.get("chunk_ids") or payload.get("bm25") is None:
        raise SystemExit("BM25 索引是空的，请先运行 python rag.py ingest")
    return payload


def bm25_search(cfg: dict, question: str, top_k: int) -> list[tuple[int, float]]:
    import numpy as np

    payload = load_bm25(resolve_path(cfg, cfg["paths"]["bm25_index"]))
    scores = payload["bm25"].get_scores(tokenize(question))
    order = np.argsort(scores)[::-1]
    hits: list[tuple[int, float]] = []
    for position in order:
        score = float(scores[position])
        if score <= 0:
            break
        hits.append((int(payload["chunk_ids"][int(position)]), score))
        if len(hits) >= top_k:
            break
    return hits


def vector_search(cfg: dict, question: str, top_k: int) -> list[tuple[int, float]]:
    client = open_chroma(cfg)
    collection = open_collection(client, cfg)
    count = collection.count()
    if count <= 0:
        raise SystemExit("向量库是空的，请先运行 python rag.py ingest")
    model = get_model(cfg)
    vector = embed_texts(model, [question], cfg, is_query=True)[0]
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, count),
        include=["distances"],
    )
    ids = result.get("ids") or [[]]
    distances = result.get("distances") or [[]]
    hits: list[tuple[int, float]] = []
    for chunk_id, distance in zip(ids[0], distances[0] if distances else []):
        hits.append((int(chunk_id), float(distance)))
    return hits


def reciprocal_rank_fusion(results_list: list[list[int]], k: int = 60) -> tuple[list[int], dict[int, float]]:
    """RRF：score = 1 / (k + rank + 1)，只比较名次，不把 BM25 分和余弦距离直接相加。"""
    fused_scores: dict[int, float] = {}
    for ranked_list in results_list:
        for rank, chunk_id in enumerate(ranked_list):
            chunk_id = int(chunk_id)
            score = 1.0 / (k + rank + 1)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score
    sorted_ids = sorted(fused_scores, key=lambda item: fused_scores[item], reverse=True)
    return sorted_ids, fused_scores


def fetch_chunks(conn: sqlite3.Connection, chunk_ids: list[int]) -> list[sqlite3.Row]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT id, source_doc_id, source_name, source_path, chunk_text, chunk_idx,
               split_config, embed_model_name
        FROM chunks
        WHERE id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    by_id = {int(row["id"]): row for row in rows}
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def retrieve(cfg: dict, question: str) -> list[dict]:
    mode = cfg["retrieval"]["mode"]
    top_k = int(cfg["retrieval"]["top_k"])
    candidate_k = int(cfg["retrieval"].get("candidate_k", max(top_k, 10)))
    rrf_k = int(cfg["retrieval"].get("rrf_k", 60))
    db_path = resolve_path(cfg, cfg["paths"]["meta_db"])
    if not db_path.exists():
        raise SystemExit("没有找到 meta.db，请先运行 python rag.py ingest")

    if mode == "bm25":
        pairs = bm25_search(cfg, question, top_k)
        ordered_ids = [chunk_id for chunk_id, _score in pairs]
        score_map = {chunk_id: score for chunk_id, score in pairs}
        score_name = "bm25"
    elif mode == "vector":
        pairs = vector_search(cfg, question, top_k)
        ordered_ids = [chunk_id for chunk_id, _score in pairs]
        score_map = {chunk_id: score for chunk_id, score in pairs}
        score_name = "distance"
    else:
        bm25_ids = [chunk_id for chunk_id, _score in bm25_search(cfg, question, candidate_k)]
        vector_ids = [chunk_id for chunk_id, _score in vector_search(cfg, question, candidate_k)]
        fused_ids, score_map = reciprocal_rank_fusion([bm25_ids, vector_ids], k=rrf_k)
        ordered_ids = fused_ids[:top_k]
        score_name = "rrf"

    conn = connect_db(db_path)
    try:
        rows = fetch_chunks(conn, ordered_ids)
    finally:
        conn.close()

    hits = []
    for rank, row in enumerate(rows, start=1):
        hits.append(
            {
                "rank": rank,
                "id": int(row["id"]),
                "source_name": row["source_name"],
                "source_path": row["source_path"],
                "chunk_idx": int(row["chunk_idx"]),
                "chunk_text": row["chunk_text"],
                "score": score_map.get(int(row["id"])),
                "score_name": score_name,
            }
        )
    return hits


def build_context(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            f"[{hit['rank']}] 来源：{hit['source_name']}（chunk #{hit['chunk_idx']}）\n{hit['chunk_text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(cfg: dict, question: str, hits: list[dict]) -> str:
    import json
    import urllib.error
    import urllib.request

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
