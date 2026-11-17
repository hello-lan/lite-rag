"""把 docs/raw 写入索引，把 docs/split 的预切分 JSON 嵌入入库，或只按 sqlite 原文重算向量。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from lite_rag.config import log, resolve_path, split_config_of
from lite_rag.documents import (
    iter_source_files,
    iter_split_files,
    load_source,
    load_split_file,
    relative_to_root,
    source_doc_id,
    split_document,
)
from lite_rag.meta import DocMeta
from lite_rag.store import (
    chroma_has_ids,
    chroma_meta,
    chroma_meta_from_row,
    chroma_update_meta,
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


def prune_missing(
    conn,
    collection,
    *,
    live_ids: set[str],
    root: Path,
    skip_under: Path | None = None,
    only_under: Path | None = None,
) -> int:
    rows = conn.execute(
        "SELECT DISTINCT source_doc_id, source_name, source_path FROM chunks"
    ).fetchall()
    removed = 0
    for row in rows:
        source_path = row["source_path"] or ""
        if skip_under is not None and _path_under(source_path, skip_under, root):
            continue
        if only_under is not None and not _path_under(source_path, only_under, root):
            continue
        if row["source_doc_id"] in live_ids:
            continue
        delete_doc(conn, collection, row["source_doc_id"])
        removed += 1
        log.info("删除已不存在的文档: %s", row["source_name"])
    return removed


def check_documents(docs_dir, cfg: dict) -> None:
    files = iter_source_files(docs_dir)
    if not files:
        raise SystemExit(f"{docs_dir} 里没有 PDF 或 Markdown。请把材料放进该目录。")
    for path in files:
        text, doc, warning = load_source(docs_dir, path)
        if warning:
            log.warning(warning)
        if not text:
            log.info("未抽出文本（扫描件需要另外做 OCR）: %s", path.name)
            continue
        chunks = split_document(text, cfg)
        who = doc.company or doc.title
        log.info(
            "%s  %s/%s  %s  字符 %s  chunk %s",
            path.name,
            doc.scope,
            doc.doc_type,
            who,
            len(text),
            len(chunks),
        )


def upsert_existing(conn, collection, rows, cfg: dict, docs: list[DocMeta] | None = None) -> None:
    if not rows:
        return
    model = get_model(cfg)
    texts = [row["chunk_text"] for row in rows]
    vectors = embed_texts(model, texts, cfg)
    model_name = cfg["embedding"]["model_name"]
    ids = [str(row["id"]) for row in rows]
    metas = []
    for index, row in enumerate(rows):
        if docs is None:
            metas.append(chroma_meta_from_row(row))
        else:
            metas.append(chroma_meta(int(row["id"]), row["source_name"], row["source_doc_id"], docs[index]))
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


def _retag(conn, collection, rows, docs: list[DocMeta]) -> None:
    source_id = rows[0]["source_doc_id"]
    source_name = rows[0]["source_name"]
    chunk_ids = [int(row["id"]) for row in rows]
    update_chunk_metas(conn, chunk_ids, docs)
    chroma_update_meta(
        collection,
        [str(chunk_id) for chunk_id in chunk_ids],
        [chroma_meta(chunk_id, source_name, source_id, doc) for chunk_id, doc in zip(chunk_ids, docs)],
    )


def _expand_docs(doc: DocMeta | list[DocMeta], count: int) -> list[DocMeta]:
    if isinstance(doc, DocMeta):
        return [doc] * count
    if len(doc) != count:
        raise SystemExit("chunk 元信息数量与文本数量不一致")
    return list(doc)


def store_chunks(
    conn,
    collection,
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
    existing = load_doc_chunks(conn, doc_id)
    if not force and same_chunks(existing, texts, split_cfg):
        ids = [int(row["id"]) for row in existing]
        meta_same = all(item.same_as(row) for item, row in zip(docs, existing))
        if not meta_same and chroma_has_ids(collection, ids):
            _retag(conn, collection, existing, docs)
            log.info("元信息已更新: %s", source_name)
            return "retagged"
        if meta_same and chroma_has_ids(collection, ids):
            log.info("未改动，跳过: %s（%s 个 chunk）", source_name, len(texts))
            return "skipped"
        if not meta_same:
            update_chunk_metas(conn, ids, docs)
        log.info("原文还在，向量缺失，正在补写: %s", source_name)
        upsert_existing(conn, collection, existing, cfg, docs)
        return "repaired"

    vectors = embed_texts(get_model(cfg), texts, cfg)
    if existing:
        delete_doc(conn, collection, doc_id)
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
    chroma_upsert(
        collection,
        [str(chunk_id) for chunk_id in chunk_ids],
        vectors,
        [chroma_meta(chunk_id, source_name, doc_id, item) for chunk_id, item in zip(chunk_ids, docs)],
    )
    head = docs[0]
    who = head.stock_code or head.company or "-"
    log.info("写入 %s：%s 个 chunk [%s/%s %s]", source_name, len(chunk_ids), head.scope, head.doc_type, who)
    return "written"


def ingest_file(conn, collection, cfg, docs_dir, path, force: bool) -> str:
    text, doc, warning = load_source(docs_dir, path)
    if warning:
        log.warning(warning)
    doc_id = source_doc_id(docs_dir, path)
    texts = split_document(text, cfg) if text else []
    if not texts:
        log.info("跳过（未抽出文本，扫描件需要 OCR）: %s", path.name)
        if load_doc_chunks(conn, doc_id):
            delete_doc(conn, collection, doc_id)
            return "dropped"
        return "empty"
    return store_chunks(
        conn,
        collection,
        cfg,
        doc_id=doc_id,
        source_name=path.name,
        source_path=relative_to_root(cfg, path),
        texts=texts,
        doc=doc,
        split_cfg=split_config_of(cfg),
        force=force,
    )


def ingest_documents(cfg: dict, *, force: bool, check: bool, bm25_only: bool) -> None:
    docs_dir = resolve_path(cfg, cfg["paths"]["docs_dir"])
    db_path = resolve_path(cfg, cfg["paths"]["meta_db"])
    bm25_path = resolve_path(cfg, cfg["paths"]["bm25_index"])

    if check:
        check_documents(docs_dir, cfg)
        return

    if bm25_only:
        if db_path.exists() and not schema_is_current(db_path):
            raise SystemExit(
                "索引结构已更换。请运行 python main.py ingest 或 python main.py ingest-split，"
                "旧库会在入库时清空并重建。"
            )
        init_meta_db(db_path)
        conn = connect_db(db_path)
        try:
            count = rebuild_bm25(conn, bm25_path)
            log.info("BM25 已按 sqlite 重建，共 %s 个 chunk", count)
        finally:
            conn.close()
        return

    reset = init_meta_db(db_path)
    if reset:
        log.info("旧索引已删除，将按 docs/raw 重新入库。预切分 JSON 需要再运行 python main.py ingest-split")
        if bm25_path.exists():
            bm25_path.unlink()
        open_collection(open_chroma(cfg), cfg, reset=True)
    conn = connect_db(db_path)
    try:
        if not docs_dir.exists():
            raise SystemExit(f"文档目录不存在: {docs_dir}")
        files = iter_source_files(docs_dir)
        if not files:
            raise SystemExit(f"{docs_dir} 里没有 PDF 或 Markdown。")

        client = open_chroma(cfg)
        collection = open_collection(client, cfg)
        removed = prune_missing(
            conn,
            collection,
            live_ids={source_doc_id(docs_dir, path) for path in files},
            root=cfg["_root"],
            skip_under=split_dir_of(cfg),
        )
        conn.commit()
        counts = {"written": 0, "repaired": 0, "retagged": 0, "skipped": 0, "empty": 0, "dropped": 0}
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
            "完成。写入 %s，补向量 %s，更新元信息 %s，跳过 %s，无文本 %s，清除空文档 %s，删除缺失 %s，sqlite 现有 %s 个 chunk",
            counts["written"],
            counts["repaired"],
            counts["retagged"],
            counts["skipped"],
            counts["empty"],
            counts["dropped"],
            removed,
            total,
        )
    finally:
        conn.close()


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
        who = doc.company or doc.title
        log.info(
            "%s  %s/%s  %s  split %s  chunk %s",
            path.name,
            doc.scope,
            doc.doc_type,
            who,
            split_cfg,
            len(texts),
        )


def ingest_split_file(conn, collection, cfg, split_dir, path, force: bool) -> str:
    try:
        texts, docs, split_cfg = load_split_file(path)
    except ValueError as exc:
        raise SystemExit(f"{_relative(split_dir, path)}: {exc}") from exc
    return store_chunks(
        conn,
        collection,
        cfg,
        doc_id=source_doc_id(split_dir, path),
        source_name=path.name,
        source_path=relative_to_root(cfg, path),
        texts=texts,
        doc=docs,
        split_cfg=split_cfg,
        force=force,
    )


def ingest_split_documents(cfg: dict, *, force: bool, check: bool) -> None:
    split_dir = split_dir_of(cfg)
    db_path = resolve_path(cfg, cfg["paths"]["meta_db"])
    bm25_path = resolve_path(cfg, cfg["paths"]["bm25_index"])

    if check:
        if not split_dir.exists():
            raise SystemExit(f"预切分目录不存在: {split_dir}")
        check_split_documents(split_dir)
        return

    reset = init_meta_db(db_path)
    if reset:
        log.info("旧索引已删除，这次只写入 docs/split。PDF 和 Markdown 需要再运行 python main.py ingest")
        if bm25_path.exists():
            bm25_path.unlink()
        open_collection(open_chroma(cfg), cfg, reset=True)
    conn = connect_db(db_path)
    try:
        if not split_dir.exists():
            raise SystemExit(f"预切分目录不存在: {split_dir}")
        files = iter_split_files(split_dir)
        if not files:
            raise SystemExit(f"{split_dir} 里没有 JSON。")

        client = open_chroma(cfg)
        collection = open_collection(client, cfg)
        removed = prune_missing(
            conn,
            collection,
            live_ids={source_doc_id(split_dir, path) for path in files},
            root=cfg["_root"],
            only_under=split_dir,
        )
        conn.commit()
        counts = {"written": 0, "repaired": 0, "retagged": 0, "skipped": 0}
        for path in files:
            try:
                status = ingest_split_file(conn, collection, cfg, split_dir, path, force)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            counts[status] += 1

        dirty = any(counts[key] for key in ("written", "repaired")) or removed or not bm25_path.exists()
        if dirty:
            count = rebuild_bm25(conn, bm25_path)
            log.info("BM25 已更新，共 %s 个 chunk → %s", count, bm25_path.name)
        else:
            log.info("文档和向量都没有变化，BM25 保持原样")
        total = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        log.info(
            "完成。写入 %s，补向量 %s，更新元信息 %s，跳过 %s，删除缺失 %s，sqlite 现有 %s 个 chunk",
            counts["written"],
            counts["repaired"],
            counts["retagged"],
            counts["skipped"],
            removed,
            total,
        )
    finally:
        conn.close()


def reembed_vectors(cfg: dict, *, force: bool) -> None:
    db_path = resolve_path(cfg, cfg["paths"]["meta_db"])
    if not db_path.exists():
        raise SystemExit(f"没有找到 {db_path}，请先运行 python main.py ingest 或 python main.py ingest-split")
    if not schema_is_current(db_path):
        raise SystemExit(
            "索引结构已更换。请运行 python main.py ingest 或 python main.py ingest-split，"
            "旧库会在入库时清空并重建。"
        )
    init_meta_db(db_path)
    conn = connect_db(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, chunk_text, source_name, source_doc_id, embed_model_name,
                   scope, doc_type, title, company, stock_code, year, period,
                   published_on, publisher
            FROM chunks
            ORDER BY id
            """
        ).fetchall()
        if not rows:
            raise SystemExit("sqlite 里没有 chunk，请先运行 python main.py ingest 或 python main.py ingest-split")

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
            [chroma_meta_from_row(row) for row in rows],
        )
        conn.execute("UPDATE chunks SET embed_model_name = ?", (target,))
        conn.commit()
        log.info("Chroma 已重写，sqlite 中的 embed_model_name 已更新为 %s", target)
    finally:
        conn.close()
