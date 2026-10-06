from pypdf import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.document import SectionHeaderItem
from config import get_config


chunk_dir = get_config()["paths"]["chunk_dir"]

def split_pdf(pdf_path: str, chunk_size: int) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    chunk_paths = []
    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        writer = PdfWriter()
        for page in reader.pages[start:end]:
            writer.add_page(page)

        chunk_path = chunk_dir / f"chunk_{start + 1}_{end}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)
        chunk_paths.append(chunk_path)
        print(f"  Wrote {chunk_path.name} (pages {start + 1}-{end})")

    print(f"Split {total_pages} pages into {len(chunk_paths)} chunks of ~{chunk_size} pages.\n")
    return chunk_paths


def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    # Picture description is now handled manually via the shared VLM
    # (describe_picture_with_vlm / picture_to_node below), NOT by Docling's
    # own internal pipeline. Docling's built-in picture-description model
    # loaded a second Qwen2-VL instance onto the GPU independently of
    # get_vlm() — with the 4-bit 7B model also resident, that pushed VRAM
    # past the ~7.4GB ceiling this pipeline targets. Keeping this off means
    # only one VLM is ever loaded on the GPU at a time.
    pipeline_options.do_picture_description = False
    pipeline_options.do_formula_enrichment = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.images_scale = 2.0  # Keep safe for 7.4GB memory ceiling
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

def _is_noise_heading(text: str) -> bool:
    normalized = text.strip().lower().rstrip(":")
    return normalized in {p.rstrip(":") for p in NOISE_HEADING_PATTERNS}

def iter_items_with_headings(doc, active_hierarchy):
    """Yields (item, headings, section_context). Owns the hierarchy tracking."""
    for item, level in doc.iterate_items():
        if isinstance(item, SectionHeaderItem) and item.text:
            text = item.text.strip()
            if not _is_noise_heading(text):
                lvl = level or 0
                active_hierarchy[lvl] = text
                for stale in [l for l in active_hierarchy if l > lvl]:
                    del active_hierarchy[stale]
                continue
        headings = [active_hierarchy[l] for l in sorted(active_hierarchy)]
        yield item, headings, " > ".join(headings) or "General Project Structure"