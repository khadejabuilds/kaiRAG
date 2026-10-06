from llama_index.core.postprocessor import SentenceTransformerRerank
from src.config import get_config

_reranker = None


def get_reranker() -> SentenceTransformerRerank:
    reranker_model = get_config()["retrieval"]["reranker_model"]
    global _reranker
    if _reranker is None:
        _reranker = SentenceTransformerRerank(model=reranker_model, top_n=5)
    return _reranker