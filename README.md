
# kaiRAG: Local Multimodal RAG for Engineering & Technical Documents

kaiRAG is a production-grade, fully local Retrieval-Augmented Generation (RAG) pipeline designed to ingest, index, and query complex engineering and functional design documents (including multi-level tables, flowcharts, diagrams, and formulas).

## Architecture Overview
1. **Document Parsing**: Powered by **Docling** for structured text, table, and figure extraction.
2. **Vision Analysis**: **Qwen2-VL-7B-Instruct (4-bit)** for visual description of complex diagrams and flowcharts.
3. **Indexing**: **Qdrant** (local mode) storing **bge-small-en-v1.5** dense embeddings and **BM25** sparse vectors.
4. **Hybrid Retrieval**: Bi-encoder + BM25 search combined via **Reciprocal Rank Fusion (RRF)**, reranked with **bge-reranker-large**.
5. **Generation**: **LlamaIndex Query Engine** paired with **Qwen2.5-7B-Instruct** and `TableWorkflowExpander` for full row-level tabular detail.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/kaiRAG.git](https://github.com/your-username/kaiRAG.git)
   cd kaiRAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Memory:**
   Ensure CUDA segment expansion is set before execution to minimize OOM risk:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

## Usage

* **Ingest a Document:**
  ```bash
  python scripts/run_pipeline.py --ingest --pdf_path doc_source/nasa_systems_engineering_handbook.pdf
  ```
* **Query the System:**
  ```bash
  python scripts/run_pipeline.py --query "What is the criteria for tailoring and SE NPR requirement?"
  ```