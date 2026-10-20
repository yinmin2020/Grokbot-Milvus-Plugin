"""Native dense, BM25 and hybrid search shared by the script entry points."""

from mvbot_common import ensure_loaded, validate_vectors


def search(
    c,
    collection,
    *,
    mode,
    text=None,
    vector=None,
    filter="",
    top_k=5,
    candidates=20,
    ranker="rrf",
    rrf_k=60,
    weights=(0.7, 0.3),
    output_fields=None,
    dense_field="dense",
    sparse_field="sparse",
    metric="COSINE",
):
    from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

    if not 1 <= top_k <= candidates <= 16384:
        raise ValueError("Require 1 <= top_k <= candidates <= 16384")
    if mode != "bm25":
        if not vector:
            raise ValueError("Dense/hybrid search requires a query vector")
        validate_vectors([vector], len(vector))
    if mode != "dense" and not (text and text.strip()):
        raise ValueError("BM25/hybrid search requires query text")
    ensure_loaded(c, collection)
    common = dict(
        collection_name=collection,
        limit=top_k,
        output_fields=output_fields or ["*"],
        consistency_level="Strong",
    )
    if mode == "dense":
        return c.search(
            data=[vector],
            anns_field=dense_field,
            filter=filter,
            search_params={"metric_type": metric},
            **common,
        )[0]
    if mode == "bm25":
        return c.search(
            data=[text],
            anns_field=sparse_field,
            filter=filter,
            search_params={"metric_type": "BM25"},
            **common,
        )[0]
    if rrf_k <= 0 or len(weights) != 2 or any(not 0 <= w <= 1 for w in weights) or not any(weights):
        raise ValueError("Require positive RRF k and two weights in [0, 1] with positive sum")
    requests = [
        AnnSearchRequest(
            data=[vector],
            anns_field=dense_field,
            param={"metric_type": metric},
            limit=candidates,
            expr=filter,
        ),
        AnnSearchRequest(
            data=[text], anns_field=sparse_field, param={"metric_type": "BM25"}, limit=candidates, expr=filter
        ),
    ]
    reranker = RRFRanker(rrf_k) if ranker == "rrf" else WeightedRanker(*weights)
    return c.hybrid_search(reqs=requests, ranker=reranker, **common)[0]
