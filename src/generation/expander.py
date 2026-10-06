from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

class TableWorkflowExpander(BaseNodePostprocessor):
    """
    Runs after reranking, before synthesis. Retrieval/reranking already
    happened on the short embed_text — this only changes what the LLM
    reads at generation time, injecting the full structured_table JSON or
    workflow description instead of the compressed summary.
    """
    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None):
        for n in nodes:
            md = n.node.metadata
            if md.get("type") == "table" and "structured_table" in md:
                n.node.text = f"{n.node.text}\nFull table data: {md['structured_table']}"
            elif md.get("type") == "workflow_diagram" and "structured_workflow" in md:
                n.node.text = f"{n.node.text}\nFull diagram data: {md['structured_workflow']}"
        return nodes
