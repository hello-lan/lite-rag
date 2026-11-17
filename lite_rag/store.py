"""索引：chunk 原文在 sqlite，向量在 Chroma，关键词在 BM25。"""

from __future__ import annotations

import logging
import os
import pickle
import re
import sqlite3
from pathlib import Path

from lite_rag.config import ROOT, log, resolve_path

_PUNCT = re.compile(r"^[\s\W_]+$", re.UNICODE)
_jieba_ready = False
_model = None


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


def chroma_has_ids(collection, chunk_ids: list[int]) -> bool:
    if not chunk_ids:
        return True
    found: set[str] = set()
    ids = [str(chunk_id) for chunk_id in chunk_ids]
    for start in range(0, len(ids), 100):
        got = collection.get(ids=ids[start : start + 100])
        found.update(got.get("ids") or [])
    return len(found) == len(ids)


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
        raise SystemExit(f"没有找到 {index_path}，请先运行 python main.py ingest")
    with index_path.open("rb") as f:
        payload = pickle.load(f)
    if not payload.get("chunk_ids") or payload.get("bm25") is None:
        raise SystemExit("BM25 索引是空的，请先运行 python main.py ingest")
    return payload
