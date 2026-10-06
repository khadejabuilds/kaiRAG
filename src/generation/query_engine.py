import torch
from llama_index.core import VectorStoreIndex, Settings
from src.retrieval.reranker import get_reranker
from expander import TableWorkflowExpander
from src.vision.vlm_analyzer import unload_vlm

def get_llm():
    """
    Lazy-loaded local generation LLM — Qwen2.5-7B-Instruct, 4-bit, text-only
    (no vision tower, unlike get_vlm()'s Qwen2-VL). Used only at query time
    for response synthesis. Call unload_vlm() before this if ingestion ran
    earlier in the same session.
    """
    global _llm
    if _llm is None:
        from llama_index.llms.huggingface import HuggingFaceLLM
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        _llm = HuggingFaceLLM(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
            device_map="cuda",
            model_kwargs={"quantization_config": bnb_config},
            generate_kwargs={"do_sample": False},
            context_window= 4096,
            max_new_tokens = 512,
            system_prompt=(
                "You are answering questions about engineering standards "
                "(API 579-1/ASME FFS-1) and internal design documents. "
                "Answer only from the provided context. If the context "
                "doesn't contain the answer, say so explicitly. Preserve "
                "any formulas, variable names, and section references "
                "exactly as given."
            ),
        )
    return _llm

def query(index: VectorStoreIndex, question: str, doc_source: str | None = None):
    """
    Hybrid retrieval: dense (bge-small) + sparse (BM25 via fastembed), fused
    by Qdrant server-side, then reranked with bge-reranker-large.
    Pass doc_source="FFS-1" or "GEMCL" to hard-filter before search.
    """
    filters = None
    if doc_source:
        from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
        filters = MetadataFilters(filters=[ExactMatchFilter(key="doc_source", value=doc_source)])

    retriever = index.as_retriever(
        vector_store_query_mode="hybrid",
        similarity_top_k=20,   # dense candidates
        sparse_top_k=20,       # sparse candidates
        filters=filters,
    )
    initial_results = retriever.retrieve(question)

    reranker = get_reranker()

    reranked_results = reranker.postprocess_nodes(
        nodes=initial_results,
        query_str=question
    )

    print(f"\nQuestion: {question}")
    print("─" * 60)
    for i, node in enumerate(reranked_results, 1):
        print(f"\n[Result {i}] Score: {node.score:.4f} | Type: {node.metadata.get('type')} | Source: {node.metadata.get('doc_source')}")
        print("Embedded text:", node.text)
        if "structured_table" in node.metadata:
            print("Structured table:", node.metadata["structured_table"])
        if "structured_workflow" in node.metadata:
            print("Structured workflow:", node.metadata["structured_workflow"])
    print("─" * 60)

#index = build_index(PDF_PATH)

def query_with_generation(index: VectorStoreIndex, question: str, doc_source: str | None = None):
    filters = None
    if doc_source:
        from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
        filters = MetadataFilters(filters=[ExactMatchFilter(key="doc_source", value=doc_source)])

    unload_vlm()          # free Qwen2-VL-7B if it's still resident from ingestion this session
    Settings.llm = get_llm()

    query_engine = index.as_query_engine(
        vector_store_query_mode="hybrid",
        similarity_top_k=20,
        sparse_top_k=20,
        filters=filters,
        node_postprocessors=[get_reranker(), TableWorkflowExpander()],
        response_mode="compact",
    )

    response = query_engine.query(question)

    print(f"\nQuestion: {question}")
    print("─" * 60)
    print(response)
    print("─" * 60)
    for node in response.source_nodes:
        print(f"  [{node.metadata.get('type')}] {node.metadata.get('section')} (score {node.score:.4f})")

    return response

