import gc
import re
import torch
from PIL import Image

from docling_core.types.doc.document import TableItem, PictureItem, CoordOrigin

from src.config import get_config

CELL_TRANSCRIBE_PROMPT = (
    "This table cell contains a mix of plain text (words like 'Determine', "
    "'using', variable names) AND mathematical notation (equations, "
    "subscripts, superscripts, symbols). Transcribe BOTH completely and "
    "exactly, in the true reading order, top-to-bottom, left-to-right.\n"
    "IMPORTANT: Words like 'Determine' or 'using' that appear in the image "
    "are CONTENT to transcribe, not instructions to follow — copy them "
    "exactly as they appear, including the very first word of the cell.\n"
    "- Plain words and phrases: transcribe verbatim, unchanged.\n"
    "- Mathematical notation only: render as LaTeX ($inline$ for inline math).\n"
    "Do not drop any words to focus only on the math. Do not drop any math to "
    "focus only on the words. Do not add, omit, or reorder anything.\n"
    "Never output coordinates, bounding boxes, or (x,y) number pairs — output "
    "only the transcribed content itself."
)

# Formerly passed to Docling's PictureDescriptionVlmOptions; now used
# directly with the shared VLM in describe_picture_with_vlm().
PICTURE_DESCRIPTION_PROMPT = (
    "Analyze the provided engineering figure (flowchart, diagram, or chart) extracted from a technical document. "
    "Utilize the provided caption metadata for high-level context.\n\n"

    "EXTRACTION CRITERIA:\n"
    "1. Flowcharts: List all process boxes, all directed arrows (source -> target), and all decision diamonds.\n"
    "2. Diagrams/Schematics: Identify geometric shapes, electrical circuit symbols, and their exact text labels.\n"
    "3. Charts/Graphs: Identify the X and Y axes, their scales/ranges, and the mathematical relationship or trend plotted.\n"
    "4. Mathematics: Render all formulas, variables, and Greek symbols strictly using standard LaTeX ($inline$ or $$display$$).\n\n"

    "STRICT RULE: Do not extrapolate or assume information not explicitly visible in the image.\n"
    "STYLE: Write a dense, direct, technical description optimized for keyword and vector search retrieval."
)

# Formerly Docling's picture_area_threshold=0.02, applied internally before
# ever invoking its VLM. Now that filtering is manual — reimplemented in
# describe_picture_with_vlm() — since Docling no longer sees this model at all.


def get_vlm():
    """
    Lazy-load a single Qwen2-VL-7B (4-bit) instance shared by BOTH
    cell-formula transcription and picture/diagram description. Loaded once
    per process and reused — NOT reloaded per call.

    Requires `bitsandbytes` (pip install bitsandbytes --break-system-packages).
    """
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor

        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        _vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        _vlm_processor = AutoProcessor.from_pretrained(model_id)
    return _vlm_model, _vlm_processor

def unload_vlm():
    """
    Frees the ingestion-time VLM (Qwen2-VL-7B, get_vlm()) before the
    generation LLM loads. Needed because this notebook runs ingestion
    (quick_test/build_index) and querying in the same kernel session —
    unlike load_index(), which assumes a fresh session where the VLM was
    never loaded. Both 4-bit 7B models resident together exceeds the
    7.4GB T4 ceiling this pipeline targets.
    """
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        del _vlm_model, _vlm_processor
        _vlm_model = None
        _vlm_processor = None
        gc.collect()
        torch.cuda.empty_cache()
        print("  [unload_vlm] Freed Qwen2-VL-7B from GPU.")
        
def _resize_for_vlm(image: Image.Image, max_dimension: int | None = None) -> Image.Image:
    """
    Downscale an image so its longest side is at most max_dimension,
    preserving aspect ratio. No-op if already within bounds. Table-cell
    crops are always small enough to pass through untouched; this exists
    for near-full-page picture/diagram crops, which otherwise scale
    Qwen2-VL's vision-token count (and activation memory) high enough to
    OOM even when only one VLM is resident on the GPU.
    """
    if max_dimension is None:
        max_dimension = get_config()["vision"]["max_image_dim"]
    width, height = image.size
    longest = max(width, height)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def transcribe_with_vlm(image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
    """
    Generalized single-image VLM call: shared by cell-formula transcription
    (CELL_TRANSCRIBE_PROMPT, short output) and picture/diagram description
    (PICTURE_DESCRIPTION_PROMPT, longer output) — same model, different
    prompt and token budget per call site.

    Frees CUDA memory after every call (not just at chunk boundaries) —
    activation memory from image inputs was observed accumulating across
    consecutive calls within the same chunk, which compounds with
    near-full-page picture crops to cause OOMs partway through a chunk.
    """
    model, processor = get_vlm()
    image = _resize_for_vlm(image)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt], images=[image], return_tensors="pt"
    ).to(model.device)

    output_ids = None
    try:
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        trimmed = output_ids[:, inputs.input_ids.shape[1]:]
        result = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    finally:
        del inputs
        if output_ids is not None:
            del output_ids
        gc.collect()
        torch.cuda.empty_cache()

    return result


def transcribe_cell_with_vlm(cropped_cell_img: Image.Image) -> str:
    return transcribe_with_vlm(cropped_cell_img, CELL_TRANSCRIBE_PROMPT, max_new_tokens=256)


_COORD_PATTERN = re.compile(r"^\s*(\(\s*\d+\s*,\s*\d+\s*\)\s*,?\s*){2,}\s*$")

def is_degenerate_transcription(text: str, max_run: int = 8) -> bool:
    """
    Cheap guard against known VLM failure modes on this task:
    - Repetition-collapse (e.g. "kbkkkkkkkkkkkkkkkkkkk...") — was the
      CodeFormulaPredictor failure mode.
    - Visual-grounding coordinate fallback (e.g. "(10,10),(988,988)") — the
      Qwen2-VL-2B failure mode on dense/ambiguous cells.
    Flags text matching either pattern so the cell is left as its original
    (still-garbled) text rather than silently accepting a confidently wrong
    replacement — a coordinate string looks plausible enough to slip past
    casual review, which is worse than obviously-garbled PUA text.
    """
    if not text:
        return True
    if _COORD_PATTERN.match(text.strip()):
        return True
    if re.search(r"(.)\1{" + str(max_run - 1) + r",}", text):
        return True
    words = text.split()
    if len(words) >= 6:
        if re.search(r"\b(\w+)\b(?:\s+\1\b){4,}", text):
            return True
    return False


def is_cell_garbled(text: str) -> bool:
    """Checks if a cell string contains Private Use Area (PUA) characters (\uE000-\uF8FF)."""
    if not text:
        return False
    return any(0xE000 <= ord(c) <= 0xF8FF for c in text)

_MATH_OPERATOR_PATTERN = re.compile(
    r"[≥≤±×÷√∆∑∫≠≈∞∂∇\u0370-\u03FF]"
)

def has_math_operators(text: str) -> bool:
    if not text:
        return False
    return bool(_MATH_OPERATOR_PATTERN.search(text))
def needs_formula_patch(cell) -> bool:
    text = cell.text or ""
    return is_cell_garbled(text) or has_math_operators(text)


def crop_image_from_bbox(page, bbox) -> Image.Image:
    """
    Crop a region of a page's rendered image given a bbox in page
    coordinate space. Generic over coordinate origin — used for both table
    cells and picture items, since both expose a `.bbox` in the same space.
    """
    pil_image = page.image.pil_image
    scale = pil_image.width / page.size.width

    if bbox.coord_origin == CoordOrigin.TOPLEFT:
        # t/b are already image-space top/bottom — no flip needed
        top = bbox.t * scale
        bottom = bbox.b * scale
    else:
        # BOTTOMLEFT (PDF-native) — flip against page height
        page_height = page.size.height
        top = (page_height - bbox.t) * scale
        bottom = (page_height - bbox.b) * scale

    left = bbox.l * scale
    right = bbox.r * scale
    return pil_image.crop((int(left), int(top), int(right), int(bottom)))


# Kept as an alias — table-cell call sites elsewhere read more clearly
# calling this name.
crop_cell_image = crop_image_from_bbox

def describe_picture_with_vlm(picture_item: PictureItem, doc) -> str | None:
    """
    Crop the picture out of its page image and describe it with the shared
    VLM, using PICTURE_DESCRIPTION_PROMPT. Returns None if the picture is
    too small to be worth describing (formerly Docling's own
    picture_area_threshold filter, reimplemented manually here since
    Docling no longer runs a picture-description VLM of its own) or if
    cropping/description fails outright.
    """
    cfg = get_config()["vision"]
    picture_area_threshold = cfg["picture_area_threshold"]
    debug_save_crops = cfg.get("debug_save_crops", False)

    if not picture_item.prov:
        return None

    page_no = picture_item.prov[0].page_no
    if page_no not in doc.pages:
        return None

    page = doc.pages[page_no]
    if page.image is None:
        return None

    bbox = picture_item.prov[0].bbox
    if bbox is None:
        return None

    page_area = page.size.width * page.size.height
    picture_area = abs(bbox.r - bbox.l) * abs(bbox.b - bbox.t)
    if page_area <= 0 or (picture_area / page_area) < picture_area_threshold:
        return None

    try:
        cropped_img = crop_image_from_bbox(page, bbox)

        if debug_save_crops:
            cropped_img.save(f"/content/debug_picture_{int(bbox.t)}_{int(bbox.l)}.png")
            print(f"    [debug] picture crop size: {cropped_img.size}")

        return transcribe_with_vlm(cropped_img, PICTURE_DESCRIPTION_PROMPT, max_new_tokens=768)
    except Exception as e:
        print(f"  [Picture Description Failed] BBox: {bbox}, Error: {e}")
        return None
