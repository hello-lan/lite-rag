"""索引：chunk 原文在 sqlite，向量在 Chroma，关键词在 BM25。"""

from __future__ import annotations

import gc
import logging
import os
import pickle
import re
import sqlite3
from pathlib import Path

from lite_rag.config import ROOT, log, resolve_path
from lite_rag.meta import TYPES_BY_SCOPE, DocMeta, SearchFilter, doc_from_row

_PUNCT = re.compile(r"^[\s\W_]+$", re.UNICODE)
_jieba_ready = False
_model = None


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = "DEFERRED"
    return conn


_REQUIRED_COLUMNS = frozenset({
    "id",
    "source_doc_id",
    "source_name",
    "source_path",
    "chunk_text",
    "chunk_idx",
    "split_config",
    "embed_model_name",
    "created_at",
    "scope",
    "doc_type",
    "title",
    "company",
    "stock_code",
    "year",
    "period",
    "published_on",
    "author",
})

_CHUNK_COLUMNS = """
    id, source_doc_id, source_name, source_path, chunk_text, chunk_idx,
    split_config, embed_model_name, scope, doc_type, title, company,
    stock_code, year, period, published_on, author
"""


def _table_columns(conn: sqlite3.Connection) -> set[str] | None:
    found = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'chunks'"
    ).fetchone()
    if not found:
        return None
    return {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}


def _as_current_columns(columns: set[str] | None) -> set[str] | None:
    """把尚未改名的 publisher 列看成 author，避免因此判定成旧库。"""
    if columns is None:
        return None
    if "author" not in columns and "publisher" in columns:
        return (columns - {"publisher"}) | {"author"}
    return columns


def _rename_author_column(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn)
    if columns is not None and "publisher" in columns and "author" not in columns:
        conn.execute("ALTER TABLE chunks RENAME COLUMN publisher TO author")


def schema_is_current(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    conn = connect_db(db_path)
    try:
        columns = _as_current_columns(_table_columns(conn))
    finally:
        conn.close()
    return columns is not None and _REQUIRED_COLUMNS.issubset(columns)


def init_meta_db(db_path: Path) -> bool:
    """确保 chunks 是当前结构。旧表直接删除，返回是否清掉了旧库。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_db(db_path)
    reset = False
    try:
        _rename_author_column(conn)
        columns = _table_columns(conn)
        if columns is not None and not _REQUIRED_COLUMNS.issubset(columns):
            conn.execute("DROP TABLE chunks")
            reset = True
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_doc_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_path TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                split_config TEXT NOT NULL,
                embed_model_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                scope TEXT NOT NULL DEFAULT '',
                doc_type TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                company TEXT NOT NULL DEFAULT '',
                stock_code TEXT NOT NULL DEFAULT '',
                year INTEGER NOT NULL DEFAULT 0,
                period TEXT NOT NULL DEFAULT '',
                published_on TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_doc_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_scope ON chunks(scope, doc_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_stock ON chunks(stock_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_company ON chunks(company)")
        conn.commit()
    finally:
        conn.close()
    return reset


def load_doc_chunks(conn: sqlite3.Connection, source_doc_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        f"""
        SELECT {_CHUNK_COLUMNS}
        FROM chunks
        WHERE source_doc_id = ?
        ORDER BY chunk_idx
        """,
        (source_doc_id,),
    ).fetchall()


def delete_doc(conn: sqlite3.Connection, client, cfg: dict, source_doc_id: str) -> int:
    rows = conn.execute("SELECT id FROM chunks WHERE source_doc_id = ?", (source_doc_id,)).fetchall()
    ids = [str(row["id"]) for row in rows]
    delete_vector_ids(client, cfg, ids)
    _delete_where_source(client, cfg, source_doc_id)
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
    docs: list[DocMeta],
) -> list[int]:
    if len(docs) != len(texts):
        raise ValueError("元信息数量与 chunk 数量不一致")
    ids: list[int] = []
    for index, (text, doc) in enumerate(zip(texts, docs)):
        cursor = conn.execute(
            """
            INSERT INTO chunks (
                source_doc_id, source_name, source_path, chunk_text,
                chunk_idx, split_config, embed_model_name,
                scope, doc_type, title, company, stock_code,
                year, period, published_on, author
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_doc_id,
                source_name,
                source_path,
                text,
                index,
                split_cfg,
                embed_model_name,
                doc.scope,
                doc.doc_type,
                doc.title,
                doc.company,
                doc.stock_code,
                doc.year,
                doc.period,
                doc.published_on,
                doc.author,
            ),
        )
        ids.append(int(cursor.lastrowid))
    return ids


def update_chunk_metas(conn: sqlite3.Connection, chunk_ids: list[int], docs: list[DocMeta]) -> None:
    if len(chunk_ids) != len(docs):
        raise ValueError("元信息数量与 chunk 数量不一致")
    for chunk_id, doc in zip(chunk_ids, docs):
        conn.execute(
            """
            UPDATE chunks
            SET scope = ?, doc_type = ?, title = ?, company = ?, stock_code = ?,
                year = ?, period = ?, published_on = ?, author = ?
            WHERE id = ?
            """,
            (
                doc.scope,
                doc.doc_type,
                doc.title,
                doc.company,
                doc.stock_code,
                doc.year,
                doc.period,
                doc.published_on,
                doc.author,
                chunk_id,
            ),
        )


def filter_chunk_ids(conn: sqlite3.Connection, filt: SearchFilter) -> set[int] | None:
    if not filt.active():
        return None
    clause, params = filt.sql()
    rows = conn.execute(f"SELECT id FROM chunks WHERE {clause}", params).fetchall()
    return {int(row["id"]) for row in rows}


def fetch_chunks(conn: sqlite3.Connection, chunk_ids: list[int]) -> list[sqlite3.Row]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT {_CHUNK_COLUMNS}
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


LEGACY_COLLECTION = "finance_chunks"
_COLLECTION_META = {"hnsw:space": "cosine"}


def collection_map(cfg: dict) -> dict[str, str]:
    raw = (cfg.get("paths") or {}).get("collections")
    if not isinstance(raw, dict) or not raw:
        raise SystemExit("配置缺少 paths.collections")
    mapping: dict[str, str] = {}
    for doc_type, name in raw.items():
        kind = str(doc_type).strip()
        collection = str(name).strip()
        if not kind or not collection:
            raise SystemExit("paths.collections 里的类型和 collection 名都不能为空")
        mapping[kind] = collection
    return mapping


def collection_name(cfg: dict, doc_type: str) -> str:
    name = collection_map(cfg).get(doc_type)
    if not name:
        raise SystemExit(f"doc_type={doc_type} 还没有对应的 Chroma collection")
    return name


def configured_collection_names(cfg: dict) -> list[str]:
    names: list[str] = []
    for name in collection_map(cfg).values():
        if name not in names:
            names.append(name)
    return names


def doc_types_for_filter(cfg: dict, filt: SearchFilter) -> list[str]:
    """按筛选决定要查哪些 doc_type。未指定类型和范围时返回配置里的全部类型。"""
    mapping = collection_map(cfg)
    if filt.doc_type:
        collection_name(cfg, filt.doc_type)
        return [filt.doc_type]
    if filt.scope:
        types = sorted(doc_type for doc_type in TYPES_BY_SCOPE[filt.scope] if doc_type in mapping)
        if not types:
            raise SystemExit(f"scope={filt.scope} 下还没有可查询的 collection")
        return types
    if filt.company or filt.stock_code:
        types = sorted(doc_type for doc_type in TYPES_BY_SCOPE["company"] if doc_type in mapping)
        if not types:
            raise SystemExit("公司材料还没有对应的 Chroma collection")
        return types
    return list(mapping)


def _listed_collection_names(client) -> set[str]:
    names: set[str] = set()
    for item in client.list_collections():
        if isinstance(item, str):
            names.add(item)
        else:
            name = getattr(item, "name", None)
            if name:
                names.add(name)
    return names


def open_named_collection(client, name: str, reset: bool = False):
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name=name, metadata=_COLLECTION_META)


def _delete_collection(client, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        log.debug("删除 collection %s 失败", name, exc_info=True)


def reset_collections(client, cfg: dict) -> None:
    """索引结构重置时清掉旧 collection 和当前配置的 collection。"""
    for name in set(configured_collection_names(cfg)) | {LEGACY_COLLECTION}:
        _delete_collection(client, name)
    for name in configured_collection_names(cfg):
        _mark_collection_slim(open_named_collection(client, name))


def _as_vectors(embeddings) -> list[list[float]]:
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    vectors: list[list[float]] = []
    for row in embeddings:
        if hasattr(row, "tolist"):
            row = row.tolist()
        vectors.append(list(row))
    return vectors


def _copy_collection(source, target) -> None:
    total = source.count()
    offset = 0
    batch_size = 128
    while offset < total:
        got = source.get(include=["embeddings", "metadatas"], limit=batch_size, offset=offset)
        ids = list(got.get("ids") or [])
        if not ids:
            break
        embeddings = got.get("embeddings")
        metadatas = list(got.get("metadatas") or [])
        if embeddings is None or len(metadatas) != len(ids):
            raise RuntimeError(f"无法从 {getattr(source, 'name', LEGACY_COLLECTION)} 读出完整向量，迁移中止")
        target.upsert(ids=ids, embeddings=_as_vectors(embeddings), metadatas=metadatas)
        offset += len(ids)


def ensure_collections(client, cfg: dict) -> None:
    """把旧的 finance_chunks 原样迁到投资理论 collection，并建好配置里的 collection。"""
    mapping = collection_map(cfg)
    existing = _listed_collection_names(client)
    theory_name = mapping.get("theory")
    if theory_name and LEGACY_COLLECTION in existing:
        legacy = client.get_collection(LEGACY_COLLECTION)
        legacy_count = legacy.count()
        if theory_name in existing:
            theory = client.get_collection(theory_name)
            theory_count = theory.count()
        else:
            theory = None
            theory_count = 0
        if legacy_count and theory_count:
            log.warning(
                "同时存在 %s（%s 条）和 %s（%s 条），未迁移，避免重复写入。",
                LEGACY_COLLECTION,
                legacy_count,
                theory_name,
                theory_count,
            )
        elif legacy_count:
            created = theory is None
            if theory is None:
                theory = open_named_collection(client, theory_name)
            try:
                log.info("把 %s 的 %s 条向量迁到 %s", LEGACY_COLLECTION, legacy_count, theory_name)
                _copy_collection(legacy, theory)
                copied = theory.count()
            except Exception as exc:
                if created or theory_count == 0:
                    _delete_collection(client, theory_name)
                raise SystemExit(f"迁移 {LEGACY_COLLECTION} 失败：{exc}") from exc
            if copied != legacy_count:
                if created or theory_count == 0:
                    _delete_collection(client, theory_name)
                raise SystemExit(
                    f"迁移后 {theory_name} 有 {copied} 条，与 {LEGACY_COLLECTION} 的 {legacy_count} 条不一致"
                )
            theory = _replace_with_slim(client, theory, "theory")
            _mark_collection_slim(theory)
            _delete_collection(client, LEGACY_COLLECTION)
            if LEGACY_COLLECTION in _listed_collection_names(client):
                raise SystemExit(f"向量已复制到 {theory_name}，但删除 {LEGACY_COLLECTION} 失败")
            existing.discard(LEGACY_COLLECTION)
        else:
            _delete_collection(client, LEGACY_COLLECTION)
    for name in configured_collection_names(cfg):
        open_named_collection(client, name)
    _slim_configured_collections(client, cfg)


def delete_vector_ids(client, cfg: dict, ids: list[str]) -> None:
    if not ids:
        return
    for name in configured_collection_names(cfg):
        try:
            collection = client.get_collection(name)
        except Exception:
            continue
        for start in range(0, len(ids), 128):
            try:
                collection.delete(ids=ids[start : start + 128])
            except Exception:
                log.debug("按 chunk id 删除向量失败", exc_info=True)


def _delete_where_source(client, cfg: dict, source_doc_id: str) -> None:
    for name in configured_collection_names(cfg):
        try:
            collection = client.get_collection(name)
            collection.delete(where={"source_doc_id": source_doc_id})
        except Exception:
            log.debug("按文档 id 删除向量失败", exc_info=True)


def clear_source_vectors(client, cfg: dict, source_doc_ids: set[str], ids: list[str]) -> None:
    """从所有 collection 删掉这些 chunk，并按文档 id 再清一次。"""
    delete_vector_ids(client, cfg, ids)
    for source_doc_id in source_doc_ids:
        _delete_where_source(client, cfg, source_doc_id)


_CHROMA_QUERY_FIELDS = {
    "theory": frozenset({"year"}),
    "filing": frozenset({"company", "stock_code", "year"}),
}
META_VERSION = 2


def _collection_version(collection) -> int:
    raw = (collection.metadata or {}).get("meta_version")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _mark_collection_slim(collection) -> None:
    collection.modify(metadata={"meta_version": META_VERSION})


def _slim_stored_meta(doc_type: str, old: dict | None) -> dict:
    old = old or {}
    meta: dict = {}
    source_doc_id = str(old.get("source_doc_id") or "")
    if source_doc_id:
        meta["source_doc_id"] = source_doc_id
    author = str(old.get("author") or old.get("publisher") or "")
    company = str(old.get("company") or "")
    stock_code = str(old.get("stock_code") or "")
    try:
        year = int(old.get("year") or 0)
    except (TypeError, ValueError):
        year = 0
    if doc_type == "theory":
        if author:
            meta["author"] = author
        if year:
            meta["year"] = year
    elif doc_type == "filing":
        if company:
            meta["company"] = company
        if stock_code:
            meta["stock_code"] = stock_code
        if year:
            meta["year"] = year
    return meta


def _replace_with_slim(client, collection, doc_type: str):
    """新建 collection 写入瘦身后的元信息，再换回原名。upsert 会保留旧键。"""
    name = collection.name
    original_count = collection.count()
    temp_name = f"{name}__slim"
    _delete_collection(client, temp_name)
    temp = client.create_collection(name=temp_name, metadata=dict(_COLLECTION_META))
    try:
        listed = collection.get(include=[])
        ids = list(listed.get("ids") or [])
        for start in range(0, len(ids), 128):
            batch = ids[start : start + 128]
            got = collection.get(ids=batch, include=["embeddings", "metadatas"])
            found = list(got.get("ids") or [])
            embeddings = _as_vectors(got.get("embeddings"))
            metadatas = list(got.get("metadatas") or [])
            by_id = {chunk_id: index for index, chunk_id in enumerate(found)}
            aligned_ids = []
            aligned_vectors = []
            aligned_metas = []
            for chunk_id in batch:
                index = by_id.get(chunk_id)
                if index is None:
                    continue
                aligned_ids.append(chunk_id)
                aligned_vectors.append(embeddings[index])
                aligned_metas.append(_slim_stored_meta(doc_type, metadatas[index]))
            if aligned_ids:
                temp.upsert(ids=aligned_ids, embeddings=aligned_vectors, metadatas=aligned_metas)
        if temp.count() != original_count:
            raise SystemExit(f"收紧 {name} 后有 {temp.count()} 条，原来是 {original_count} 条")
    except Exception:
        _delete_collection(client, temp_name)
        raise
    _delete_collection(client, name)
    temp.modify(name=name)
    return client.get_collection(name)


def _slim_configured_collections(client, cfg: dict) -> None:
    seen: set[str] = set()
    for doc_type, name in collection_map(cfg).items():
        if name in seen:
            continue
        seen.add(name)
        collection = client.get_collection(name)
        if _collection_version(collection) >= META_VERSION:
            continue
        if collection.count():
            log.info("收紧 %s 的向量元信息", name)
            collection = _replace_with_slim(client, collection, doc_type)
        _mark_collection_slim(collection)


def chroma_meta(source_doc_id: str, doc: DocMeta) -> dict:
    """写入 Chroma 的字段。空值和 year=0 不写。"""
    meta: dict = {"source_doc_id": source_doc_id}
    if doc.doc_type == "theory":
        if doc.author:
            meta["author"] = doc.author
        if doc.year:
            meta["year"] = int(doc.year)
        return meta
    if doc.doc_type == "filing":
        if doc.company:
            meta["company"] = doc.company
        if doc.stock_code:
            meta["stock_code"] = doc.stock_code
        if doc.year:
            meta["year"] = int(doc.year)
        return meta
    raise SystemExit(f"doc_type={doc.doc_type} 还没有对应的 Chroma collection")


def chroma_where_fits(doc_types: list[str], filt: SearchFilter) -> bool:
    """筛选字段必须是每个目标 collection 都有的，否则这次向量检索没有命中。"""
    needed: set[str] = set()
    if filt.company:
        needed.add("company")
    if filt.stock_code:
        needed.add("stock_code")
    if filt.year:
        needed.add("year")
    if not needed:
        return True
    for doc_type in doc_types:
        if not needed <= _CHROMA_QUERY_FIELDS.get(doc_type, frozenset()):
            return False
    return True


def chroma_meta_from_row(row) -> dict:
    return chroma_meta(row["source_doc_id"], doc_from_row(row))


def chroma_upsert(collection, ids: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
    batch_size = 128
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


def chroma_update_meta(collection, ids: list[str], metadatas: list[dict]) -> None:
    batch_size = 128
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.update(ids=ids[start:end], metadatas=metadatas[start:end])


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


def keep_model_loaded(cfg: dict) -> bool:
    """查询路径是否在本次编码后继续持有 embedding 模型。缺省为不常驻。"""
    embedding = cfg.get("embedding") or {}
    return bool(embedding.get("keep_in_memory", False))


def model_is_loaded() -> bool:
    return _model is not None


def release_model() -> None:
    """丢掉进程内的 embedding 模型，便于下一次检索重新加载。"""
    global _model
    if _model is None:
        return
    _model = None
    log.info("释放 embedding 模型")
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        raise SystemExit(f"没有找到 {index_path}，请先运行 python main.py ingest 或 python main.py ingest-split")
    with index_path.open("rb") as f:
        payload = pickle.load(f)
    if not payload.get("chunk_ids") or payload.get("bm25") is None:
        raise SystemExit("BM25 索引是空的，请先运行 python main.py ingest 或 python main.py ingest-split")
    return payload
