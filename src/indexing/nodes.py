import json
import re
from docling_core.types.doc.document import TableItem, PictureItem
from llama_index.core.schema import TextNode
from vision.vlm_analyzer import describe_picture_with_vlm

def _to_str(item) -> str:
    """
    Coerce a single list item into a display string. This prompt doesn't
    request JSON output, so the VLM's response is expected to be free text —
    but if it ever does emit JSON-shaped content (e.g. nested dicts inside
    'steps'/'decisions'), this keeps str.join() from crashing on it.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("step", "action", "text", "name", "label", "description", "value"):
            if key in item and item[key]:
                return str(item[key])
        return json.dumps(item)
    return str(item)


def _to_str_list(items) -> list[str]:
    if not isinstance(items, list):
        return [] if items in (None, "") else [_to_str(items)]
    return [_to_str(i) for i in items]


def _pd_to_numeric_safe(series):
    import pandas as pd
    coerced = pd.to_numeric(series, errors="coerce")
    # Only treat the column as numeric if most values actually converted —
    # otherwise a mostly-text column with one stray number would get
    # mis-summarized as a numeric range.
    if coerced.notna().mean() >= 0.8:
        return coerced.dropna()
    return None


def _summarize_column(df, col_idx: int, col_name: str, max_categories: int = 6) -> str:
    """
    Describe one column without dumping every row: numeric columns get a
    min/max range, categorical/text columns get their distinct values (or
    a count if there are too many to list). This keeps embed_text short and
    semantically distinct per table, instead of a long "col is val" chain
    that (a) risks truncation past bge-small's 512-token limit and (b)
    makes structurally different tables look near-identical once truncated.

    Selects by POSITION (iloc), not by label (df[col_name]) — Docling
    sometimes produces duplicate/unnamed column labels (e.g. merged-header
    tables come back as [0, 1, 2]), and selecting a duplicated label with
    df[col_name] returns a DataFrame instead of a Series, which breaks
    .str accessor calls. Position-based selection is always unambiguous.
    """
    series = df.iloc[:, col_idx]
    series = series[series.notna()].astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        return f"{col_name}: no data"

    numeric = _pd_to_numeric_safe(series)
    if numeric is not None:
        return f"{col_name} ranges {numeric.min():g} to {numeric.max():g}"

    uniques = series.unique().tolist()
    if len(uniques) <= max_categories:
        return f"{col_name}: {', '.join(uniques)}"
    return f"{col_name}: {len(uniques)} distinct values (e.g. {', '.join(uniques[:3])}, ...)"


def table_to_node(table_item: TableItem, doc, headings: list[str], doc_source: str) -> TextNode:
    df = table_item.export_to_dataframe(doc=doc)
    table_json = df.to_dict(orient="records")
    columns = list(df.columns)

    # Summarize each column (range for numeric, distinct values for
    # categorical) instead of dumping every row as "col is val" text.
    # Full row-level data still lives in metadata["structured_table"] for
    # retrieval-time display — this only changes what gets EMBEDDED.
    if any(not isinstance(c, str) for c in columns):
        print(f"  [table_to_node] non-string column names found: {columns}")
    column_names = [str(c) for c in columns]  # columns can be non-string (e.g. unnamed/int headers)

    # Disambiguate duplicate labels (e.g. two columns both named "Value")
    # so the summary below doesn't blend two different columns together
    # under one printed name.
    seen: dict[str, int] = {}
    display_names = []
    for name in column_names:
        seen[name] = seen.get(name, 0) + 1
        display_names.append(name if seen[name] == 1 else f"{name} ({seen[name]})")

    col_summaries = [
        _summarize_column(df, idx, display_names[idx]) for idx in range(len(columns))
    ]
    embed_text = f"Table with {len(df)} rows, columns: {', '.join(display_names)}. " + "; ".join(col_summaries)

    # FIX: no more "System Architecture Section" GEMCL leftover as a silent
    # default — that string was making every heading-less table (e.g. one
    # landing right at a chunk boundary before the hierarchy tracker caught
    # a heading) look identical to the embedding model. Use a source-neutral
    # fallback that doesn't inject fake shared context.
    section_context = " > ".join(headings) if headings else "Unlabeled section"
    embed_text = f"[Table under: {section_context}] {embed_text}"

    return TextNode(
        text=embed_text,
        metadata={
            "type": "table",
            "section": section_context,
            "structured_table": json.dumps(table_json),
            "columns": columns,
            "doc_source": doc_source,
        },
    )

def picture_to_node(picture_item: PictureItem, headings: list[str], doc_source: str, raw_text: str) -> TextNode | None:
    """
    raw_text is now generated by describe_picture_with_vlm() and passed in
    explicitly, rather than read from picture_item.annotations — that
    field was populated by Docling's own picture-description pipeline,
    which is no longer running (do_picture_description=False).
    """
    if not raw_text:
        return None

    raw_text = raw_text.strip()
    section_context = " > ".join(headings) if headings else "Operational Process Flow"

    # This prompt asks for free-text technical description, not JSON, so most
    # pictures will land in the raw-text fallback below. The JSON branch is
    # kept as a defensive fallback in case the VLM emits JSON anyway.
    parsed = {"steps": [], "decisions": [], "owner": "unspecified department"}
    is_json_parsed = False

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            is_json_parsed = True
        except json.JSONDecodeError:
            pass

    if is_json_parsed and parsed.get("steps"):
        steps = _to_str_list(parsed.get("steps", []))
        decisions = _to_str_list(parsed.get("decisions", []))
        embed_text = (
            f"[Workflow diagram under: {section_context}] "
            f"Steps: {' -> '.join(steps) if steps else 'none'}. "
            f"Decision points: {'; '.join(decisions) if decisions else 'none'}."
        )
    else:
        # FALLBACK: expected path for this prompt — index the free-text VLM
        # description directly. This preserves LaTeX formulas, labels, etc.
        if len(raw_text) < 15:  # Skip low-signal descriptions or empty spaces
            return None
        embed_text = f"[Visual Workflow diagram under: {section_context}] {raw_text}"
        parsed["raw_vlm_description"] = raw_text

    return TextNode(
        text=embed_text,
        metadata={
            "type": "workflow_diagram",
            "section": section_context,
            "structured_workflow": json.dumps(parsed),
            "doc_source": doc_source,
        },
    )

def build_nodes(doc, doc_source, active_hierarchy):
    nodes = []
    for item, headings, section in iter_items_with_headings(doc, active_hierarchy):
        if isinstance(item, TableItem):
            patch_table_cells(item, doc)
            nodes.append(table_to_node(item, doc, headings, doc_source))
        elif isinstance(item, PictureItem):
            n = picture_to_node(item, headings, doc_source, describe_picture_with_vlm(item, doc))
            if n: nodes.append(n)
        elif getattr(item, "text", "").strip() and len(item.text.strip()) > 3:
            nodes.append(TextNode(text=f"[Context: {section}] {item.text.strip()}",
                                  metadata={"type": "text", "section": section,
                                            "parent_headings": headings,
                                            "doc_source": doc_source}))
    return nodes