import gc
import torch
from src.config import get_config
from parsing.docling_parser import split_pdf, build_converter
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from retrieval.hybrid_fusion import rrf_fusion

embed_model = get_config()["indexing"]["embed_model"]
pages_per_chunk = get_config()["parsing"]["pages_per_chunk"]

def get_qdrant_client() -> QdrantClient:
    """
    Local-mode Qdrant holds an OS file lock on QDRANT_PATH for as long as
    the QdrantClient object is alive. Every function that used to call
    `QdrantClient(path=QDRANT_PATH)` independently (get_qdrant_vector_store,
    quick_test's collection-reset block, etc.) was opening a SEPARATE
    client on the same path — in a long-lived Jupyter/Colab kernel, an
    earlier cell's client object often outlives the cell it was created in,
    so a later call opening a second client on the same path hits
    "Storage folder ... already accessed by another instance" even within
    the same process. One shared client, reused everywhere, avoids that.

    If you still hit the lock error after this fix, it means something
    OUTSIDE this module — another notebook, another kernel, a previous run
    that was interrupted without cleanup — still holds the lock; restarting
    the runtime clears it.
    """
    qdrant_path = get_config()["paths"]["qdrant_path"]
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=qdrant_path)
    return _qdrant_client


def get_qdrant_vector_store() -> QdrantVectorStore:
    """
    Local-mode Qdrant, persisted to Drive so it survives Colab disconnects.
    enable_hybrid=True turns on sparse (BM25-style) vectors alongside dense —
    fastembed runs the sparse encoder on CPU, no VRAM cost, so it doesn't
    compete with Qwen2-VL for GPU memory.

    fusion method: RRF if USE_RRF else LlamaIndex's default relative score
    fusion. Set USE_RRF=False above to compare the two directly.
    """
    collection_name = get_config()["indexing"]["collection"]
    sparse_model = get_config()["indexing"]["sparse_model"]
    use_rrf = get_config()["retrieval"]["use_rrf"]
    client = get_qdrant_client()
    kwargs = dict(
        client=client,
        collection_name=collection_name,
        enable_hybrid=True,
        fastembed_sparse_model=sparse_model,
    )
    if use_rrf:
        kwargs["hybrid_fusion_fn"] = rrf_fusion
    return QdrantVectorStore(**kwargs)


def build_index(pdf_path: str) -> VectorStoreIndex:
    print("[1/3] Splitting PDF into memory-safe chunks...")
    chunk_paths = split_pdf(pdf_path, pages_per_chunk)

    print("[2/3] Processing each chunk and building structured nodes...")
    # Build the converter ONCE, then reuse it for every chunk. The shared
    # VLM (get_vlm()) is lazy-loaded on first use and also persists across
    # chunks — it is not tied to the converter's lifecycle.
    converter = build_converter()

    all_nodes: list[TextNode] = []
    active_hierarchy: dict[int, str] = {}   # threaded across chunks, not reset each time
    for chunk_path in chunk_paths:
        print(f"  Processing {chunk_path.name}...")
        nodes, active_hierarchy = process_chunk(chunk_path, converter, DOC_SOURCE, active_hierarchy)
        print(f"    -> {len(nodes)} nodes")
        all_nodes.extend(nodes)

    del converter
    gc.collect()
    torch.cuda.empty_cache()

    table_count = sum(1 for n in all_nodes if n.metadata.get("type") == "table")
    workflow_count = sum(1 for n in all_nodes if n.metadata.get("type") == "workflow_diagram")
    print(f"\n  Total: {len(all_nodes)} nodes | {table_count} tables | {workflow_count} workflow diagrams")

    print("[3/3] Embedding + indexing into Qdrant (hybrid dense + sparse)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)

    vector_store = get_qdrant_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Upserts each node's dense + sparse vectors into Qdrant. Sparse vectors
    # are computed per-node independently (fastembed BM25), so adding a new
    # document later (see add_document()) is a cheap append, not a rebuild.
    index = VectorStoreIndex(
        all_nodes,
        storage_context=storage_context,
    )

    return index


def load_index() -> VectorStoreIndex:
    """
    Reopen an existing Qdrant collection without re-running Docling/VLM.
    Use this in a new Colab session once the collection already exists on Drive.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)
    vector_store = get_qdrant_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)


def add_document(pdf_path: str, doc_source: str):
    """
    Incremental ingestion: process a new PDF and upsert its nodes into the
    SAME Qdrant collection, tagging doc_source so metadata filtering can
    distinguish it later (e.g. "GEMCL" vs "FFS-1"). No full-corpus rebuild.
    """
    
    print(f"[1/3] Splitting {pdf_path} into memory-safe chunks...")
    chunk_paths = split_pdf(pdf_path, pages_per_chunk)

    print("[2/3] Processing chunks...")
    converter = build_converter()
    all_nodes: list[TextNode] = []
    active_hierarchy: dict[int, str] = {}   # threaded across chunks, not reset each time
    for chunk_path in chunk_paths:
        nodes, active_hierarchy = process_chunk(chunk_path, converter, doc_source, active_hierarchy)
        all_nodes.extend(nodes)
        print(f"  {chunk_path.name} -> {len(nodes)} nodes")

    del converter
    gc.collect()
    torch.cuda.empty_cache()

    print("[3/3] Upserting into existing Qdrant collection...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)
    vector_store = get_qdrant_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(all_nodes, storage_context=storage_context)
    print(f"  Added {len(all_nodes)} nodes tagged doc_source={doc_source}")
    return index
