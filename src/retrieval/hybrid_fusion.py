from src.config import get_config

def rrf_fusion(dense_result, sparse_result, alpha: float = 0.5, top_k: int = 10):
    """
    Reciprocal Rank Fusion: combine two ranked result lists using only each
    node's RANK in each list, not its raw similarity score. This avoids
    dense scores (which can cluster tightly and look falsely confident)
    drowning out sparse/BM25 matches the way relative score fusion can.
    """
    rrf_k = get_config()["retrieval"]["rrf_k"]
    from llama_index.core.vector_stores import VectorStoreQueryResult

    scores: dict = {}
    node_map: dict = {}

    for rank, (node, node_id) in enumerate(zip(dense_result.nodes, dense_result.ids)):
        scores[node_id] = scores.get(node_id, 0) + 1 / (rrf_k + rank + 1)
        node_map[node_id] = node

    for rank, (node, node_id) in enumerate(zip(sparse_result.nodes, sparse_result.ids)):
        scores[node_id] = scores.get(node_id, 0) + 1 / (rrf_k + rank + 1)
        node_map[node_id] = node

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return VectorStoreQueryResult(
        nodes=[node_map[i] for i in ranked_ids],
        similarities=[scores[i] for i in ranked_ids],
        ids=ranked_ids,
    )