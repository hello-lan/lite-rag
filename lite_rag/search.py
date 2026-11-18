"""召回：BM25、向量，以及按名次融合的 RRF。"""

from __future__ import annotations

from lite_rag.config import resolve_path
from lite_rag.meta import SearchFilter, public_meta
from lite_rag.store import (
    chroma_where_fits,
    collection_name,
    configured_collection_names,
    connect_db,
    doc_types_for_filter,
    embed_texts,
    ensure_collections,
    fetch_chunks,
    filter_chunk_ids,
    get_model,
    keep_model_loaded,
    load_bm25,
    open_chroma,
    open_named_collection,
    release_model,
    tokenize,
)


def bm25_search(
    cfg: dict,
    question: str,
    top_k: int,
    allowed: set[int] | None = None,
) -> list[tuple[int, float]]:
    import numpy as np

    payload = load_bm25(resolve_path(cfg, cfg["paths"]["bm25_index"]))
    scores = payload["bm25"].get_scores(tokenize(question))
    order = np.argsort(scores)[::-1]
    hits: list[tuple[int, float]] = []
    for position in order:
        score = float(scores[position])
        if score <= 0:
            break
        chunk_id = int(payload["chunk_ids"][int(position)])
        if allowed is not None and chunk_id not in allowed:
            continue
        hits.append((chunk_id, score))
        if len(hits) >= top_k:
            break
    return hits


def vector_search(
    cfg: dict,
    question: str,
    top_k: int,
    where: dict | None = None,
    filt: SearchFilter | None = None,
) -> list[tuple[int, float]]:
    client = open_chroma(cfg)
    ensure_collections(client, cfg)
    chosen = filt if filt is not None else SearchFilter()
    doc_types = doc_types_for_filter(cfg, chosen)
    if not chroma_where_fits(doc_types, chosen):
        return []
    counts: list[tuple[object, int]] = []
    for doc_type in doc_types:
        collection = open_named_collection(client, collection_name(cfg, doc_type))
        counts.append((collection, collection.count()))
    if all(total <= 0 for _collection, total in counts):
        selected = {collection_name(cfg, doc_type) for doc_type in doc_types}
        others_have_data = any(
            open_named_collection(client, name).count() > 0
            for name in configured_collection_names(cfg)
            if name not in selected
        )
        if not others_have_data:
            raise SystemExit("向量库是空的，请先运行 python main.py ingest")
        return []

    searchable: list[tuple[object, int]] = []
    for collection, total in counts:
        if total <= 0:
            continue
        if where:
            matched = collection.get(where=where, include=[])
            count = len(matched.get("ids") or [])
            if count <= 0:
                continue
        else:
            count = total
        searchable.append((collection, count))
    if not searchable:
        return []

    model = get_model(cfg)
    try:
        vector = embed_texts(model, [question], cfg, is_query=True)[0]
    finally:
        if not keep_model_loaded(cfg):
            model = None
            release_model()
    best: dict[int, float] = {}
    for collection, count in searchable:
        query = {
            "query_embeddings": [vector],
            "n_results": min(top_k, count),
            "include": ["distances"],
        }
        if where:
            query["where"] = where
        result = collection.query(**query)
        ids = result.get("ids") or [[]]
        distances = result.get("distances") or [[]]
        for chunk_id, distance in zip(ids[0], distances[0] if distances else []):
            chunk_id = int(chunk_id)
            distance = float(distance)
            if chunk_id not in best or distance < best[chunk_id]:
                best[chunk_id] = distance
    ordered = sorted(best, key=lambda item: best[item])[:top_k]
    return [(chunk_id, best[chunk_id]) for chunk_id in ordered]


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


def retrieve(
    cfg: dict,
    question: str,
    *,
    scope: str | None = None,
    doc_type: str | None = None,
    company: str | None = None,
    stock_code: str | None = None,
    year: int | None = None,
) -> list[dict]:
    filt = SearchFilter.parse(
        scope=scope,
        doc_type=doc_type,
        company=company,
        stock_code=stock_code,
        year=year,
    )
    mode = cfg["retrieval"]["mode"]
    top_k = int(cfg["retrieval"]["top_k"])
    candidate_k = int(cfg["retrieval"].get("candidate_k", max(top_k, 10)))
    rrf_k = int(cfg["retrieval"].get("rrf_k", 60))
    db_path = resolve_path(cfg, cfg["paths"]["meta_db"])
    if not db_path.exists():
        raise SystemExit(f"没有找到 {db_path}，请先运行 python main.py ingest")

    where = filt.chroma_where()
    conn = connect_db(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        if total == 0:
            raise SystemExit("索引是空的，请先运行 python main.py ingest")
        allowed = filter_chunk_ids(conn, filt)
        if allowed is not None and not allowed:
            return []

        if mode == "bm25":
            pairs = bm25_search(cfg, question, top_k, allowed)
            ordered_ids = [chunk_id for chunk_id, _score in pairs]
            score_map = {chunk_id: score for chunk_id, score in pairs}
            score_name = "bm25"
        elif mode == "vector":
            pairs = vector_search(cfg, question, top_k, where, filt)
            ordered_ids = [chunk_id for chunk_id, _score in pairs]
            score_map = {chunk_id: score for chunk_id, score in pairs}
            score_name = "distance"
        else:
            bm25_ids = [chunk_id for chunk_id, _score in bm25_search(cfg, question, candidate_k, allowed)]
            vector_ids = [chunk_id for chunk_id, _score in vector_search(cfg, question, candidate_k, where, filt)]
            fused_ids, score_map = reciprocal_rank_fusion([bm25_ids, vector_ids], k=rrf_k)
            ordered_ids = fused_ids[:top_k]
            score_name = "rrf"

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
                **public_meta(row),
            }
        )
    return hits
