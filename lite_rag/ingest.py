"""把 docs/ 写入索引，或只按 sqlite 里的原文重算向量。"""

from __future__ import annotations

import sqlite3

from lite_rag.config import log, resolve_path, split_config_of
from lite_rag.documents import (
    iter_source_files,
    parse_file,
    relative_to_root,
    source_doc_id,
    split_document,
)
from lite_rag.store import (
    chroma_has_ids,
    chroma_meta,
    chroma_upsert,
    connect_db,
    delete_doc,
    embed_texts,
    get_model,
    init_meta_db,
    insert_chunks,
    load_doc_chunks,
    open_chroma,
    open_collection,
    rebuild_bm25,
)


def same_chunks(rows: list[sqlite3.Row], texts: list[str], split_cfg: str) -> bool:
    if len(rows) != len(texts):
        return False
    for index, (row, text) in enumerate(zip(rows, texts)):
        if row["split_config"] != split_cfg or row["chunk_idx"] != index or row["chunk_text"] != text:
            return False
    return True


def prune_missing(conn, collection, docs_dir) -> int:
    live_ids = {source_doc_id(docs_dir, path) for path in iter_source_files(docs_dir)}
    rows = conn.execute("SELECT DISTINCT source_doc_id, source_name FROM chunks").fetchall()
    removed = 0
    for row in rows:
        if row["source_doc_id"] in live_ids:
            continue
        delete_doc(conn, collection, row["source_doc_id"])
        removed += 1
        log.info("删除已不存在的文档: %s", row["source_name"])
    return removed


def check_documents(docs_dir, cfg: dict) -> None:
    files = iter_source_files(docs_dir)
    if not files:
        raise SystemExit(f"{docs_dir} 里没有 PDF 或 Markdown。请把财报放进该目录。")
    for path in files:
        text = parse_file(path)
        if not text:
            log.info("未抽出文本（扫描件需要另外做 OCR）: %s", path.name)
            continue
        chunks = split_document(text, cfg)
        log.info("%s  字符 %s  chunk %s", path.name, len(text), len(chunks))


def upsert_existing(conn, collection, rows, cfg: dict) -> None:
    if not rows:
        return
    model = get_model(cfg)
    texts = [row["chunk_text"] for row in rows]
    vectors = embed_texts(model, texts, cfg)
    model_name = cfg["embedding"]["model_name"]
    ids = [str(row["id"]) for row in rows]
    metas = [chroma_meta(row["id"], row["source_name"], row["source_doc_id"]) for row in rows]
    for source_id in {row["source_doc_id"] for row in rows}:
        try:
            collection.delete(where={"source_doc_id": source_id})
        except Exception:
            log.debug("补写前清理旧向量失败", exc_info=True)
    chroma_upsert(collection, ids, vectors, metas)
    conn.execute(
        f"UPDATE chunks SET embed_model_name = ? WHERE id IN ({','.join('?' * len(ids))})",
        [model_name, *[int(chunk_id) for chunk_id in ids]],
    )


def ingest_file(conn, collection, cfg, docs_dir, path, force: bool) -> str:
    text = parse_file(path)
    doc_id = source_doc_id(docs_dir, path)
    texts = split_document(text, cfg) if text else []
    if not texts:
        log.info("跳过（未抽出文本，扫描件需要 OCR）: %s", path.name)
        if load_doc_chunks(conn, doc_id):
            delete_doc(conn, collection, doc_id)
            return "dropped"
        return "empty"

    split_cfg = split_config_of(cfg)
    existing = load_doc_chunks(conn, doc_id)
    if not force and same_chunks(existing, texts, split_cfg):
        ids = [int(row["id"]) for row in existing]
        if chroma_has_ids(collection, ids):
            log.info("未改动，跳过: %s（%s 个 chunk）", path.name, len(texts))
            return "skipped"
        log.info("原文还在，向量缺失，正在补写: %s", path.name)
        upsert_existing(conn, collection, existing, cfg)
        return "repaired"

    vectors = embed_texts(get_model(cfg), texts, cfg)
    if existing:
        delete_doc(conn, collection, doc_id)
    chunk_ids = insert_chunks(
        conn,
        source_doc_id=doc_id,
        source_name=path.name,
        source_path=relative_to_root(cfg, path),
        texts=texts,
        split_cfg=split_cfg,
        embed_model_name=cfg["embedding"]["model_name"],
    )
    chroma_upsert(
        collection,
        [str(chunk_id) for chunk_id in chunk_ids],
        vectors,
        [chroma_meta(chunk_id, path.name, doc_id) for chunk_id in chunk_ids],
    )
    log.info("写入 %s：%s 个 chunk", path.name, len(chunk_ids))
    return "written"


def ingest_documents(cfg: dict, *, force: bool, check: bool, bm25_only: bool) -> None:
    docs_dir = resolve_path(cfg, cfg["paths"]["docs_dir"])
    db_path = resolve_path(cfg, cfg["paths"]["meta_db"])
    bm25_path = resolve_path(cfg, cfg["paths"]["bm25_index"])

    if check:
        check_documents(docs_dir, cfg)
        return

    init_meta_db(db_path)
    conn = connect_db(db_path)
    try:
        if bm25_only:
            count = rebuild_bm25(conn, bm25_path)
            log.info("BM25 已按 sqlite 重建，共 %s 个 chunk", count)
            return

        if not docs_dir.exists():
            raise SystemExit(f"文档目录不存在: {docs_dir}")
        files = iter_source_files(docs_dir)
        if not files:
            raise SystemExit(f"{docs_dir} 里没有 PDF 或 Markdown。")

        client = open_chroma(cfg)
        collection = open_collection(client, cfg)
        removed = prune_missing(conn, collection, docs_dir)
        conn.commit()
        counts = {"written": 0, "repaired": 0, "skipped": 0, "empty": 0, "dropped": 0}
        for path in files:
            try:
                status = ingest_file(conn, collection, cfg, docs_dir, path, force)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            counts[status] += 1

        dirty = any(counts[key] for key in ("written", "repaired", "dropped")) or removed or not bm25_path.exists()
        if dirty:
            count = rebuild_bm25(conn, bm25_path)
            log.info("BM25 已更新，共 %s 个 chunk → %s", count, bm25_path.name)
        else:
            log.info("文档和向量都没有变化，BM25 保持原样")
        total = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        log.info(
            "完成。写入 %s，补向量 %s，跳过 %s，无文本 %s，清除空文档 %s，删除缺失 %s，sqlite 现有 %s 个 chunk",
            counts["written"],
            counts["repaired"],
            counts["skipped"],
            counts["empty"],
            counts["dropped"],
            removed,
            total,
        )
    finally:
        conn.close()


def reembed_vectors(cfg: dict, *, force: bool) -> None:
    db_path = resolve_path(cfg, cfg["paths"]["meta_db"])
    if not db_path.exists():
        raise SystemExit(f"没有找到 {db_path}，请先运行 python main.py ingest")
    init_meta_db(db_path)
    conn = connect_db(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, chunk_text, source_name, source_doc_id, embed_model_name
            FROM chunks
            ORDER BY id
            """
        ).fetchall()
        if not rows:
            raise SystemExit("sqlite 里没有 chunk，请先运行 python main.py ingest")

        target = cfg["embedding"]["model_name"]
        client = open_chroma(cfg)
        collection = open_collection(client, cfg)
        same_model = all(row["embed_model_name"] == target for row in rows)
        if same_model and collection.count() == len(rows) and not force:
            log.info("Chroma 已与当前模型 %s 对齐。更换模型，或加上 --force，才会重写向量。", target)
            log.info("BM25 只依赖原文，这次不需要重建。")
            return

        log.info("开始重算 %s 个 chunk，模型 %s。BM25 索引保持不变。", len(rows), target)
        vectors = embed_texts(get_model(cfg), [row["chunk_text"] for row in rows], cfg)
        collection = open_collection(client, cfg, reset=True)
        chroma_upsert(
            collection,
            [str(row["id"]) for row in rows],
            vectors,
            [chroma_meta(row["id"], row["source_name"], row["source_doc_id"]) for row in rows],
        )
        conn.execute("UPDATE chunks SET embed_model_name = ?", (target,))
        conn.commit()
        log.info("Chroma 已重写，sqlite 中的 embed_model_name 已更新为 %s", target)
    finally:
        conn.close()
