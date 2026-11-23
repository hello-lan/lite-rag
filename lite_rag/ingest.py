"""把 corpus/03_splited 的预切分 JSON 嵌入入库，或按 sqlite 原文重算向量。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from lite_rag.config import log, resolve_path
from lite_rag.documents import (
    iter_split_files,
    load_split_file,
    relative_to_root,
    source_doc_id,
)
from lite_rag.meta import DocMeta, doc_from_row
from lite_rag.store import (
    CHUNK_TABLES,
    bm25_paths,
    chroma_has_ids,
    chroma_meta,
    chroma_meta_from_row,
    chroma_update_meta,
    chroma_upsert,
    chunk_table,
    clear_source_vectors,
    collection_map,
    collection_name,
    configured_collection_names,
    connect_db,
    count_chunks,
    delete_doc,
    embed_texts,
    ensure_collections,
    get_model,
    init_meta_db,
    insert_chunks,
    iter_scope_rows,
    list_source_docs,
    load_doc_chunks,
    open_chroma,
    open_named_collection,
    rebuild_all_bm25,
    reset_collections,
    schema_is_current,
    update_chunk_metas,
)


def same_chunks(rows: list[sqlite3.Row], texts: list[str], split_cfg: str) -> bool:
    if len(rows) != len(texts):
        return False
    for index, (row, text) in enumerate(zip(rows, texts)):
        if row["split_config"] != split_cfg or row["chunk_idx"] != index or row["chunk_text"] != text:
            return False
    return True


def split_dir_of(cfg: dict) -> Path:
    raw = (cfg.get("paths") or {}).get("split_dir")
    if not raw:
        raise SystemExit("配置缺少 paths.split_dir")
    return resolve_path(cfg, raw)


def _scan_dirs(configured: Path, directory: Path | str | None, cfg: dict) -> tuple[Path, Path]:
    """返回本次扫描目录，以及计算文档 id 用的根目录。

    指定目录落在配置目录里面时，id 仍相对配置目录。
    只扫一个子目录时，已经入库的文档不会被当成新文档。
    """
    if directory is None:
        return configured, configured
    chosen = resolve_path(cfg, directory)
    try:
        chosen.resolve().relative_to(configured.resolve())
    except ValueError:
        return chosen, chosen
    return chosen, configured


def split_scan_dirs(cfg: dict, directory: Path | str | None) -> tuple[Path, Path]:
    return _scan_dirs(split_dir_of(cfg), directory, cfg)


def _path_under(source_path: str, directory: Path, root: Path) -> bool:
    if not source_path:
        return False
    path = Path(source_path)
    if not path.is_absolute():
        path = root / path
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _chroma_client(cfg: dict, *, reset: bool):
    client = open_chroma(cfg)
    if reset:
        reset_collections(client, cfg)
    else:
        ensure_collections(client, cfg)
    return client


def _require_mapped(cfg: dict, docs: list[DocMeta], source_name: str) -> None:
    mapping = collection_map(cfg)
    unknown = sorted({doc.doc_type for doc in docs if doc.doc_type not in mapping})
    if unknown:
        kinds = "、".join(unknown)
        raise SystemExit(f"{source_name} 的 doc_type（{kinds}）还没有对应的 Chroma collection")


def _vectors_cover(client, cfg: dict, chunk_ids: list[int], docs: list[DocMeta]) -> bool:
    grouped: dict[str, list[int]] = {}
    for chunk_id, doc in zip(chunk_ids, docs):
        grouped.setdefault(doc.doc_type, []).append(chunk_id)
    for doc_type, ids in grouped.items():
        collection = open_named_collection(client, collection_name(cfg, doc_type))
        if not chroma_has_ids(collection, ids):
            return False
    return True


def _upsert_groups(client, cfg: dict, ids: list[str], vectors, metas: list[dict], docs: list[DocMeta]) -> None:
    grouped: dict[str, dict] = {}
    for chunk_id, vector, meta, doc in zip(ids, vectors, metas, docs):
        bucket = grouped.setdefault(doc.doc_type, {"ids": [], "vectors": [], "metas": []})
        bucket["ids"].append(chunk_id)
        bucket["vectors"].append(vector)
        bucket["metas"].append(meta)
    for doc_type, bucket in grouped.items():
        collection = open_named_collection(client, collection_name(cfg, doc_type))
        chroma_upsert(collection, bucket["ids"], bucket["vectors"], bucket["metas"])


def _drop_foreign_ids(client, cfg: dict, ids: list[str], docs: list[DocMeta]) -> None:
    for name in configured_collection_names(cfg):
        foreign = [
            chunk_id
            for chunk_id, doc in zip(ids, docs)
            if collection_name(cfg, doc.doc_type) != name
        ]
        if not foreign:
            continue
        try:
            collection = client.get_collection(name)
            collection.delete(ids=foreign)
        except Exception:
            log.debug("清理其它 collection 中的向量失败", exc_info=True)


def prune_missing(
    conn,
    client,
    cfg: dict,
    *,
    live_ids: set[str],
    root: Path,
    skip_under: Path | None = None,
    only_under: Path | None = None,
) -> int:
    rows = list_source_docs(conn)
    removed = 0
    for row in rows:
        source_path = row["source_path"] or ""
        if skip_under is not None and _path_under(source_path, skip_under, root):
            continue
        if only_under is not None and not _path_under(source_path, only_under, root):
            continue
        if row["source_doc_id"] in live_ids:
            continue
        delete_doc(conn, client, cfg, row["source_doc_id"])
        removed += 1
        log.info("删除已不存在的文档: %s", row["source_name"])
    return removed


def upsert_existing(conn, client, rows, cfg: dict, docs: list[DocMeta] | None = None) -> None:
    if not rows:
        return
    meta_docs = [docs[index] if docs is not None else doc_from_row(row) for index, row in enumerate(rows)]
    _require_mapped(cfg, meta_docs, rows[0]["source_name"])
    model = get_model(cfg)
    texts = [row["chunk_text"] for row in rows]
    vectors = embed_texts(model, texts, cfg)
    model_name = cfg["embedding"]["model_name"]
    ids = [str(row["id"]) for row in rows]
    metas = [
        chroma_meta(row["source_doc_id"], doc)
        for row, doc in zip(rows, meta_docs)
    ]
    clear_source_vectors(client, cfg, {row["source_doc_id"] for row in rows}, ids)
    _upsert_groups(client, cfg, ids, vectors, metas, meta_docs)
    scopes = {doc.scope for doc in meta_docs}
    if len(scopes) != 1:
        raise SystemExit("同一批补写的 chunk 必须属于同一个 scope")
    table = chunk_table(next(iter(scopes)))
    conn.execute(
        f"UPDATE {table} SET embed_model_name = ? WHERE id IN ({','.join('?' * len(ids))})",
        [model_name, *[int(chunk_id) for chunk_id in ids]],
    )


def _retag(conn, client, cfg: dict, rows, docs: list[DocMeta]) -> None:
    source_id = rows[0]["source_doc_id"]
    source_name = rows[0]["source_name"]
    chunk_ids = [int(row["id"]) for row in rows]
    update_chunk_metas(conn, chunk_ids, docs)
    grouped: dict[str, list[tuple[int, DocMeta]]] = {}
    for chunk_id, doc in zip(chunk_ids, docs):
        grouped.setdefault(doc.doc_type, []).append((chunk_id, doc))
    for doc_type, pairs in grouped.items():
        collection = open_named_collection(client, collection_name(cfg, doc_type))
        chroma_update_meta(
            collection,
            [str(chunk_id) for chunk_id, _doc in pairs],
            [chroma_meta(source_id, doc) for _chunk_id, doc in pairs],
        )
    _drop_foreign_ids(client, cfg, [str(chunk_id) for chunk_id in chunk_ids], docs)


def _same_chroma_meta(stored: dict, expected: dict) -> bool:
    if set(stored) != set(expected):
        return False
    return all(stored.get(key) == value for key, value in expected.items())


def _chroma_meta_matches(client, cfg: dict, rows, docs: list[DocMeta]) -> bool:
    """向量上的元信息是否已经和当前文档一致。收紧字段后，原文没变也要补写。"""
    grouped: dict[str, list[tuple[str, dict]]] = {}
    for row, doc in zip(rows, docs):
        grouped.setdefault(doc.doc_type, []).append(
            (str(row["id"]), chroma_meta(row["source_doc_id"], doc))
        )
    for doc_type, pairs in grouped.items():
        collection = open_named_collection(client, collection_name(cfg, doc_type))
        expected = {chunk_id: meta for chunk_id, meta in pairs}
        got = collection.get(ids=list(expected), include=["metadatas"])
        found = list(got.get("ids") or [])
        metadatas = list(got.get("metadatas") or [])
        stored_by_id = {chunk_id: meta or {} for chunk_id, meta in zip(found, metadatas)}
        if set(stored_by_id) != set(expected):
            return False
        for chunk_id, meta in expected.items():
            if not _same_chroma_meta(stored_by_id[chunk_id], meta):
                return False
    return True


def _expand_docs(doc: DocMeta | list[DocMeta], count: int) -> list[DocMeta]:
    if isinstance(doc, DocMeta):
        return [doc] * count
    if len(doc) != count:
        raise SystemExit("chunk 元信息数量与文本数量不一致")
    return list(doc)


def store_chunks(
    conn,
    client,
    cfg: dict,
    *,
    doc_id: str,
    source_name: str,
    source_path: str,
    texts: list[str],
    doc: DocMeta | list[DocMeta],
    split_cfg: str,
    force: bool,
) -> str:
    docs = _expand_docs(doc, len(texts))
    _require_mapped(cfg, docs, source_name)
    existing = load_doc_chunks(conn, doc_id)
    if not force and same_chunks(existing, texts, split_cfg):
        ids = [int(row["id"]) for row in existing]
        meta_same = all(item.same_as(row) for item, row in zip(docs, existing))
        if not meta_same and _vectors_cover(client, cfg, ids, docs):
            _retag(conn, client, cfg, existing, docs)
            log.info("元信息已更新: %s", source_name)
            return "retagged"
        if meta_same and _vectors_cover(client, cfg, ids, docs):
            if not _chroma_meta_matches(client, cfg, existing, docs):
                _retag(conn, client, cfg, existing, docs)
                log.info("元信息已更新: %s", source_name)
                return "retagged"
            log.info("未改动，跳过: %s（%s 个 chunk）", source_name, len(texts))
            return "skipped"
        if not meta_same:
            update_chunk_metas(conn, ids, docs)
        log.info("原文还在，向量缺失，正在补写: %s", source_name)
        upsert_existing(conn, client, existing, cfg, docs)
        return "repaired"

    vectors = embed_texts(get_model(cfg), texts, cfg)
    if existing:
        delete_doc(conn, client, cfg, doc_id)
    chunk_ids = insert_chunks(
        conn,
        source_doc_id=doc_id,
        source_name=source_name,
        source_path=source_path,
        texts=texts,
        split_cfg=split_cfg,
        embed_model_name=cfg["embedding"]["model_name"],
        docs=docs,
    )
    ids = [str(chunk_id) for chunk_id in chunk_ids]
    metas = [chroma_meta(doc_id, item) for item in docs]
    _upsert_groups(client, cfg, ids, vectors, metas, docs)
    head = docs[0]
    who = head.stock_code or head.stock_name or "-"
    log.info("写入 %s：%s 个 chunk [%s/%s %s]", source_name, len(chunk_ids), head.scope, head.doc_type, who)
    return "written"


def check_split_documents(split_dir: Path) -> None:
    files = iter_split_files(split_dir)
    if not files:
        raise SystemExit(f"{split_dir} 里没有 JSON。请把切分结果放进该目录。")
    for path in files:
        try:
            texts, docs, split_cfg = load_split_file(path)
        except ValueError as exc:
            raise SystemExit(f"{_relative(split_dir, path)}: {exc}") from exc
        doc = docs[0]
        who = doc.stock_name or doc.title
        log.info(
            "%s  %s/%s  %s  split %s  chunk %s",
            path.name,
            doc.scope,
            doc.doc_type,
            who,
            split_cfg,
            len(texts),
        )


def ingest_split_file(conn, client, cfg, split_dir, path, force: bool, *, id_root: Path) -> str:
    try:
        texts, docs, split_cfg = load_split_file(path)
    except ValueError as exc:
        raise SystemExit(f"{_relative(split_dir, path)}: {exc}") from exc
    return store_chunks(
        conn,
        client,
        cfg,
        doc_id=source_doc_id(id_root, path),
        source_name=path.name,
        source_path=relative_to_root(cfg, path),
        texts=texts,
        doc=docs,
        split_cfg=split_cfg,
        force=force,
    )


def ingest_split_documents(
    cfg: dict,
    *,
    force: bool,
    check: bool,
    bm25_only: bool = False,
    split_dir: Path | str | None = None,
) -> None:
    directory, id_root = split_scan_dirs(cfg, split_dir)
    db_path = resolve_path(cfg, cfg["paths"]["meta_db"])
    bm25_files = list(bm25_paths(cfg).values())

    if check:
        if not directory.exists():
            raise SystemExit(f"预切分目录不存在: {directory}")
        check_split_documents(directory)
        return

    if bm25_only:
        if db_path.exists() and not schema_is_current(db_path):
            raise SystemExit(
                "索引结构已更换。请运行 python main.py ingest-split，旧库会在入库时清空并重建。"
            )
        init_meta_db(db_path)
        conn = connect_db(db_path)
        try:
            count = rebuild_all_bm25(conn, cfg)
            log.info("BM25 已按 sqlite 重建，共 %s 个 chunk", count)
        finally:
            conn.close()
        return

    reset = init_meta_db(db_path)
    if reset:
        log.info("旧索引已删除，这次只写入预切分 JSON：%s", directory)
        for path in bm25_files:
            if path.exists():
                path.unlink()
    client = _chroma_client(cfg, reset=reset)
    conn = connect_db(db_path)
    try:
        if not directory.exists():
            raise SystemExit(f"预切分目录不存在: {directory}")
        files = iter_split_files(directory)
        if not files:
            raise SystemExit(f"{directory} 里没有 JSON。")
        removed = prune_missing(
            conn,
            client,
            cfg,
            live_ids={source_doc_id(id_root, path) for path in files},
            root=cfg["_root"],
            only_under=directory,
        )
        conn.commit()
        counts = {"written": 0, "repaired": 0, "retagged": 0, "skipped": 0}
        for path in files:
            try:
                status = ingest_split_file(conn, client, cfg, directory, path, force, id_root=id_root)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            counts[status] += 1

        bm25_missing = any(not path.exists() for path in bm25_files)
        dirty = any(counts[key] for key in ("written", "repaired")) or removed or bm25_missing
        if dirty:
            count = rebuild_all_bm25(conn, cfg)
            log.info("BM25 已更新，共 %s 个 chunk", count)
        else:
            log.info("文档和向量都没有变化，BM25 保持原样")
        total = count_chunks(conn)
        parts = "，".join(f"{scope}={count_chunks(conn, scope)}" for scope in CHUNK_TABLES)
        log.info(
            "完成。写入 %s，补向量 %s，更新元信息 %s，跳过 %s，删除缺失 %s，sqlite 现有 %s 个 chunk（%s）",
            counts["written"],
            counts["repaired"],
            counts["retagged"],
            counts["skipped"],
            removed,
            total,
            parts,
        )
    finally:
        conn.close()


def reembed_vectors(cfg: dict, *, force: bool) -> None:
    db_path = resolve_path(cfg, cfg["paths"]["meta_db"])
    if not db_path.exists():
        raise SystemExit(f"没有找到 {db_path}，请先运行 python main.py ingest-split")
    if not schema_is_current(db_path):
        raise SystemExit(
            "索引结构已更换。请运行 python main.py ingest-split，旧库会在入库时清空并重建。"
        )
    init_meta_db(db_path)
    conn = connect_db(db_path)
    try:
        rows: list = []
        for scope in CHUNK_TABLES:
            rows.extend(iter_scope_rows(conn, scope))
        if not rows:
            raise SystemExit("sqlite 里没有 chunk，请先运行 python main.py ingest-split")

        target = cfg["embedding"]["model_name"]
        mapping = collection_map(cfg)
        unknown = sorted({row["doc_type"] for row in rows} - set(mapping))
        if unknown:
            kinds = "、".join(unknown)
            raise SystemExit(f"这些 doc_type 还没有对应的 Chroma collection：{kinds}")
        client = open_chroma(cfg)
        ensure_collections(client, cfg)
        groups: dict[str, list] = {}
        for row in rows:
            groups.setdefault(row["doc_type"], []).append(row)
        same_model = all(row["embed_model_name"] == target for row in rows)
        aligned = all(
            open_named_collection(client, name).count() == len(groups.get(doc_type, []))
            for doc_type, name in mapping.items()
        )
        if same_model and aligned and not force:
            log.info("Chroma 已与当前模型 %s 对齐。更换模型，或加上 --force，才会重写向量。", target)
            log.info("BM25 只依赖原文，这次不需要重建。")
            return

        log.info("开始重算 %s 个 chunk，模型 %s。BM25 索引保持不变。", len(rows), target)
        vectors = embed_texts(get_model(cfg), [row["chunk_text"] for row in rows], cfg)
        # 按 (scope, id) 区分，避免两表自增 id 撞车。
        by_key = {(row["scope"], int(row["id"])): vector for row, vector in zip(rows, vectors)}
        for doc_type, name in mapping.items():
            group = groups.get(doc_type, [])
            collection = open_named_collection(client, name, reset=True)
            if not group:
                continue
            chroma_upsert(
                collection,
                [str(row["id"]) for row in group],
                [by_key[(row["scope"], int(row["id"]))] for row in group],
                [chroma_meta_from_row(row) for row in group],
            )
        for scope, table in CHUNK_TABLES.items():
            conn.execute(f"UPDATE {table} SET embed_model_name = ?", (target,))
        conn.commit()
        log.info("Chroma 已重写，sqlite 中的 embed_model_name 已更新为 %s", target)
    finally:
        conn.close()
