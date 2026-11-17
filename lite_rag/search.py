"""召回：BM25、向量，以及按名次融合的 RRF。"""

from __future__ import annotations

from lite_rag.config import resolve_path
from lite_rag.store import (
    connect_db,
    embed_texts,
    fetch_chunks,
    get_model,
    load_bm25,
    open_chroma,
    open_collection,
    tokenize,
)


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
        raise SystemExit("向量库是空的，请先运行 python main.py ingest")
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


def retrieve(cfg: dict, question: str) -> list[dict]:
    mode = cfg["retrieval"]["mode"]
    top_k = int(cfg["retrieval"]["top_k"])
    candidate_k = int(cfg["retrieval"].get("candidate_k", max(top_k, 10)))
    rrf_k = int(cfg["retrieval"].get("rrf_k", 60))
    db_path = resolve_path(cfg, cfg["paths"]["meta_db"])
    if not db_path.exists():
        raise SystemExit(f"没有找到 {db_path}，请先运行 python main.py ingest")

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
