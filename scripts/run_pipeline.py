"""
CLI entrypoint for kaiRAG. Wires parsing -> vision -> indexing -> retrieval ->
generation together. This is the ONLY file that should import from every
layer — everything else stays within the dependency direction:

    parsing, vision  ->  indexing  ->  retrieval  ->  generation  ->  (here)

IMPORTANT: PYTORCH_CUDA_ALLOC_CONF must be set before torch is imported
anywhere, including indirectly via `src....` imports below. That's why this
line comes before any other import in this file, not just before `import
torch` — a later `from src.vision.vlm_analyzer import ...` would pull torch
in regardless of import order within this file otherwise.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import sys
from pathlib import Path

import torch

from src.config import get_config
from src.parsing.docling_parser import build_converter, split_pdf
from src.vision.vlm_analyzer import unload_vlm
from src.indexing.qdrant_store import (
    build_index,
    load_index,
    add_document,
    get_qdrant_client,
)
from src.indexing.nodes import build_nodes
from src.generation.query_engine import query_with_generation


def cmd_build(args: argparse.Namespace) -> None:
    """Full ingestion of a new document into the production collection."""
    cfg = get_config()
    doc_source = args.doc_source or cfg["indexing"].get("default_doc_source", "DEFAULT")
    print(f"[build] Ingesting {args.pdf} as doc_source={doc_source}")
    build_index(args.pdf, doc_source=doc_source)
    print("[build] Done.")


def cmd_add(args: argparse.Namespace) -> None:
    """Incremental ingestion into an existing production collection."""
    if not args.doc_source:
        print("error: --doc-source is required for add", file=sys.stderr)
        sys.exit(1)
    print(f"[add] Adding {args.pdf} as doc_source={args.doc_source}")
    add_document(args.pdf, args.doc_source)
    print("[add] Done.")


def cmd_test(args: argparse.Namespace) -> None:
    """
    Validate node-construction on a page subset, against the TEST collection.
    Mirrors the old quick_test(): crosses real chunk boundaries so hierarchy
    threading is actually exercised, and optionally resets the test
    collection first so repeated runs don't accumulate stale duplicates.
    """
    cfg = get_config()
    doc_source = args.doc_source or cfg["indexing"].get("default_doc_source", "DEFAULT")

    all_chunk_paths = split_pdf(args.pdf, cfg["parsing"]["pages_per_chunk"])

    if args.start or args.end:
        start = args.start or 1
        end = args.end or (len(all_chunk_paths) * cfg["parsing"]["pages_per_chunk"])
        chunk_paths = []
        for cp in all_chunk_paths:
            n, m = cp.stem.replace("chunk_", "").split("_")
            n, m = int(n), int(m)
            if m >= start and n <= end:
                chunk_paths.append(cp)
        print(f"[test] Using pages {start}-{end} -> {len(chunk_paths)} chunks")
    else:
        n_chunks = max(1, args.max_pages // cfg["parsing"]["pages_per_chunk"])
        chunk_paths = all_chunk_paths[:n_chunks]
        print(f"[test] Using {len(chunk_paths)} chunks ({len(chunk_paths) * cfg['parsing']['pages_per_chunk']} pages)")

    converter = build_converter()
    active_hierarchy: dict[int, str] = {}
    all_nodes = []
    for cp in chunk_paths:
        print(f"  Processing {cp.name}...")
        doc = converter.convert(str(cp)).document if not args.use_cache else None
        # build_nodes threads active_hierarchy through and owns caching
        # internally when use_cache=True (see docling_parser.process_chunk_cached
        # equivalent, wired inside build_nodes/iter_items_with_headings).
        nodes, active_hierarchy = build_nodes(
            chunk_path=cp,
            converter=converter,
            doc_source=doc_source,
            active_hierarchy=active_hierarchy,
            use_cache=args.use_cache,
        )
        print(f"    -> {len(nodes)} nodes | hierarchy now: {active_hierarchy}")
        all_nodes.extend(nodes)

    del converter
    gc.collect()
    torch.cuda.empty_cache()

    tables = [n for n in all_nodes if n.metadata["type"] == "table"]
    workflows = [n for n in all_nodes if n.metadata["type"] == "workflow_diagram"]
    texts = [n for n in all_nodes if n.metadata["type"] == "text"]
    print(f"\n[test] {len(all_nodes)} nodes total | {len(tables)} tables | {len(workflows)} figures | {len(texts)} text")

    if args.no_index:
        print("[test] --no-index set, skipping Qdrant indexing.")
        return

    test_collection = cfg["indexing"]["test_collection"]
    if args.reset:
        client = get_qdrant_client()
        if client.collection_exists(test_collection):
            client.delete_collection(test_collection)
            print(f"[test] Cleared existing test collection '{test_collection}'.")

    from llama_index.core import VectorStoreIndex, Settings, StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from src.retrieval.hybrid_fusion import rrf_fusion

    Settings.embed_model = HuggingFaceEmbedding(model_name=cfg["indexing"]["embed_model"])
    client = get_qdrant_client()
    kwargs = dict(
        client=client,
        collection_name=test_collection,
        enable_hybrid=True,
        fastembed_sparse_model=cfg["indexing"]["sparse_model"],
    )
    if cfg["retrieval"]["use_rrf"]:
        kwargs["hybrid_fusion_fn"] = rrf_fusion
    test_store = QdrantVectorStore(**kwargs)
    storage_context = StorageContext.from_defaults(vector_store=test_store)
    VectorStoreIndex(all_nodes, storage_context=storage_context)
    print(f"[test] Indexed into '{test_collection}'. Use --mode query --collection test to try retrieval.")


def cmd_query(args: argparse.Namespace) -> None:
    """Run a single query against production (or test) and print the answer."""
    cfg = get_config()
    unload_vlm()  # free the 7B vision model before loading the 7B generation model

    collection = cfg["indexing"]["test_collection"] if args.collection == "test" else None
    index = load_index(collection_override=collection)

    answer = query_with_generation(index, args.question, doc_source=args.doc_source)
    print("\n" + "=" * 60)
    print(answer)
    print("=" * 60)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_pipeline", description="kaiRAG CLI")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_build = sub.add_parser("build", help="Full ingestion into the production collection")
    p_build.add_argument("pdf", help="Path to the source PDF")
    p_build.add_argument("--doc-source", help="Tag for metadata filtering, e.g. FFS-1")
    p_build.set_defaults(func=cmd_build)

    p_add = sub.add_parser("add", help="Incremental ingestion into an existing collection")
    p_add.add_argument("pdf", help="Path to the source PDF")
    p_add.add_argument("--doc-source", required=True, help="Tag for metadata filtering")
    p_add.set_defaults(func=cmd_add)

    p_test = sub.add_parser("test", help="Validate node construction on a page subset")
    p_test.add_argument("pdf", help="Path to the source PDF")
    p_test.add_argument("--doc-source", help="Tag for metadata filtering")
    p_test.add_argument("--max-pages", type=int, default=100)
    p_test.add_argument("--start", type=int, default=None, help="1-indexed start page")
    p_test.add_argument("--end", type=int, default=None, help="1-indexed end page")
    p_test.add_argument("--use-cache", action="store_true", help="Reuse cached Docling parse if present")
    p_test.add_argument("--no-index", action="store_true", help="Build nodes only, skip Qdrant")
    p_test.add_argument("--reset", action="store_true", default=True, help="Reset test collection first (default True)")
    p_test.set_defaults(func=cmd_test)

    p_query = sub.add_parser("query", help="Run a single query end-to-end")
    p_query.add_argument("question", help="The question to ask")
    p_query.add_argument("--doc-source", default=None, help="Restrict to one doc_source")
    p_query.add_argument("--collection", choices=["prod", "test"], default="prod")
    p_query.set_defaults(func=cmd_query)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Make the configured path the one get_config() reads, before any module
    # that calls get_config() at import time would see a stale default.
    get_config.cache_clear()
    get_config(args.config)

    args.func(args)


if __name__ == "__main__":
    main()