"""
docling_spec_parser.py
======================
Plug-and-play pipeline: Specification file(s) → JSON + Markdown
using Docling with OCR enabled.

Supported input formats: PDF, DOCX, XLSX, PPTX, images, HTML, and more.
Output:
  - <filename>.json   → Structured DoclingDocument (for downstream LLM tasks)
  - <filename>.md     → Human-readable Markdown

Usage:
  1. Set INPUT_PATH to a single file or a directory of spec files.
  2. Set OUTPUT_DIR to where you want results saved.
  3. Choose your OCR_ENGINE below (see comments).
  4. Run: python docling_spec_parser.py

Install dependencies:
  pip install docling
  # For EasyOCR (default, no extra setup):
  pip install easyocr
  # For Tesseract:
  sudo apt install tesseract-ocr  (Linux) or brew install tesseract (macOS)
  pip install pytesseract
  # For RapidOCR:
  pip install rapidocr-onnxruntime
"""

import json
import logging
from pathlib import Path
from typing import Union

# ─────────────────────────────────────────────
# ✅ USER CONFIGURATION — Edit these values
# ─────────────────────────────────────────────

INPUT_PATH: Union[str, Path] = "./specs"   # Single file or directory
OUTPUT_DIR: Union[str, Path] = "./output"  # Where JSON + MD files are saved

# OCR Engine options: "easyocr" | "tesseract" | "rapidocr" | "granite_vision"
# See Section 2 in the README comments below for details on each.
OCR_ENGINE: str = "easyocr"

# Languages for OCR (ISO 639-1 codes). EasyOCR & RapidOCR use these.
# Tesseract uses codes like "eng", "deu" etc.
OCR_LANGUAGES: list[str] = ["en"]

# If True, forces OCR even on text-layer PDFs (useful for scanned/image PDFs)
FORCE_FULL_OCR: bool = False

# Docling table structure model (True = use TableFormer ML model for better tables)
ENABLE_TABLE_STRUCTURE: bool = True

# Set to True to also export images embedded in documents
EXPORT_IMAGES: bool = False

# ─────────────────────────────────────────────
# Pipeline code — no need to edit below
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def build_pipeline_options():
    """Build PipelineOptions with the configured OCR engine."""
    from docling.datamodel.pipeline_options import (
        PipelineOptions,
        EasyOcrOptions,
        TesseractOcrOptions,
        RapidOcrOptions,
    )

    pipeline_options = PipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = ENABLE_TABLE_STRUCTURE

    if OCR_ENGINE == "easyocr":
        ocr_options = EasyOcrOptions(
            lang=OCR_LANGUAGES,
            force_full_page_ocr=FORCE_FULL_OCR,
            use_gpu=False,  # Set True if CUDA GPU available
        )
        pipeline_options.ocr_options = ocr_options
        log.info("OCR engine: EasyOCR | Languages: %s", OCR_LANGUAGES)

    elif OCR_ENGINE == "tesseract":
        # Tesseract uses different lang codes: "eng", "deu", "fra", etc.
        tesseract_langs = ["eng"] if OCR_LANGUAGES == ["en"] else OCR_LANGUAGES
        ocr_options = TesseractOcrOptions(
            lang=tesseract_langs,
            force_full_page_ocr=FORCE_FULL_OCR,
        )
        pipeline_options.ocr_options = ocr_options
        log.info("OCR engine: Tesseract | Languages: %s", tesseract_langs)

    elif OCR_ENGINE == "rapidocr":
        ocr_options = RapidOcrOptions(
            force_full_page_ocr=FORCE_FULL_OCR,
        )
        pipeline_options.ocr_options = ocr_options
        log.info("OCR engine: RapidOCR")

    elif OCR_ENGINE == "granite_vision":
        # Granite Vision OCR — IBM's multimodal model used as an OCR backend
        # Requires: pip install docling[granite]
        # Note: Uses a local or remote Granite Vision model.
        # See Section 3 in comments for when to use this.
        try:
            from docling.datamodel.pipeline_options import GraniteVisionOcrOptions
            ocr_options = GraniteVisionOcrOptions(
                force_full_page_ocr=FORCE_FULL_OCR,
            )
            pipeline_options.ocr_options = ocr_options
            log.info("OCR engine: Granite Vision (IBM)")
        except ImportError:
            log.error("GraniteVision not available. Install with: pip install 'docling[granite]'")
            raise

    else:
        raise ValueError(
            f"Unknown OCR_ENGINE '{OCR_ENGINE}'. "
            "Choose from: easyocr, tesseract, rapidocr, granite_vision"
        )

    return pipeline_options


def convert_file(input_file: Path, output_dir: Path, converter) -> dict:
    """
    Convert a single file using Docling and save JSON + Markdown outputs.

    Returns a summary dict with paths to output files.
    """
    log.info("Processing: %s", input_file.name)
    stem = input_file.stem
    json_path = output_dir / f"{stem}.json"
    md_path   = output_dir / f"{stem}.md"

    try:
        result = converter.convert(str(input_file))
        doc = result.document

        # ── Export to Markdown ──────────────────────────────────────────────
        markdown_text = doc.export_to_markdown()
        md_path.write_text(markdown_text, encoding="utf-8")
        log.info("  ✅ Markdown saved: %s", md_path)

        # ── Export to JSON ──────────────────────────────────────────────────
        # doc.export_to_dict() produces a rich structured dict:
        # {
        #   "schema_name": "DoclingDocument",
        #   "version": "...",
        #   "name": "...",
        #   "body": { ... },          ← full document tree
        #   "texts": [...],           ← all text elements with bounding boxes
        #   "tables": [...],          ← structured table data
        #   "pictures": [...],        ← embedded images (if any)
        #   "pages": {...},           ← per-page metadata & layout
        # }
        doc_dict = doc.export_to_dict()

        # Attach pipeline metadata for traceability
        doc_dict["_pipeline_meta"] = {
            "source_file": str(input_file),
            "ocr_engine": OCR_ENGINE,
            "ocr_languages": OCR_LANGUAGES,
            "force_full_ocr": FORCE_FULL_OCR,
            "table_structure_enabled": ENABLE_TABLE_STRUCTURE,
        }

        json_path.write_text(
            json.dumps(doc_dict, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("  ✅ JSON saved:     %s", json_path)

        return {
            "source": str(input_file),
            "status": "success",
            "json_output": str(json_path),
            "md_output": str(md_path),
            "page_count": len(doc.pages) if hasattr(doc, "pages") else None,
            "text_elements": len(doc.texts) if hasattr(doc, "texts") else None,
            "table_count": len(doc.tables) if hasattr(doc, "tables") else None,
        }

    except Exception as e:
        log.error("  ❌ Failed to process %s: %s", input_file.name, e)
        return {
            "source": str(input_file),
            "status": "error",
            "error": str(e),
        }


def run_pipeline(input_path: Path, output_dir: Path):
    """
    Main pipeline entry point.
    Accepts a single file or a directory of files.
    Returns a list of result summaries.
    """
    from docling.document_converter import DocumentConverter, FormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_options = build_pipeline_options()

    # Register supported formats with their backends
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: FormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,       # Fast, reliable PDF backend
            ),
            InputFormat.DOCX: FormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,
            ),
            InputFormat.XLSX: FormatOption(
                pipeline_options=pipeline_options,
            ),
            InputFormat.PPTX: FormatOption(
                pipeline_options=pipeline_options,
            ),
            InputFormat.IMAGE: FormatOption(
                pipeline_options=pipeline_options,     # OCR directly on images
            ),
            InputFormat.HTML: FormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )

    # Collect files
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx",
                             ".ppt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
                             ".html", ".htm"}
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = [
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        log.info("Found %d supported file(s) in %s", len(files), input_path)
    else:
        raise FileNotFoundError(f"INPUT_PATH not found: {input_path}")

    if not files:
        log.warning("No supported files found.")
        return []

    # Process all files
    results = [convert_file(f, output_dir, converter) for f in files]

    # Write pipeline summary JSON
    summary_path = output_dir / "_pipeline_summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    log.info("\n📋 Summary saved: %s", summary_path)

    # Print summary table
    ok  = [r for r in results if r["status"] == "success"]
    err = [r for r in results if r["status"] == "error"]
    log.info("=" * 50)
    log.info("✅ Succeeded: %d | ❌ Failed: %d", len(ok), len(err))
    for r in err:
        log.warning("  FAILED: %s → %s", r["source"], r.get("error"))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO USE THE OUTPUT JSON IN YOUR LLM PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
#
# The JSON output (DoclingDocument) is self-contained and rich. Example usage:
#
#   import json
#   doc = json.load(open("output/my_spec.json"))
#
#   # All text blocks (paragraphs, headings, captions)
#   texts = [t["text"] for t in doc.get("texts", [])]
#
#   # All tables as row/column data
#   for table in doc.get("tables", []):
#       for row in table.get("data", {}).get("grid", []):
#           cells = [cell.get("text", "") for cell in row]
#
#   # Page-level layout info
#   for page_num, page_data in doc.get("pages", {}).items():
#       print(f"Page {page_num}: {page_data}")
#
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    results = run_pipeline(
        input_path=Path(INPUT_PATH),
        output_dir=Path(OUTPUT_DIR),
    )
