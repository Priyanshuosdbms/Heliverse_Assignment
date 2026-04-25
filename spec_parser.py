"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        spec_parser.py                                        ║
║  Specification Document → Enriched JSON                                      ║
║                                                                              ║
║  Stage 1 — Docling (layout + OCR)                                            ║
║    Converts PDF / DOCX / XLSX / PPTX / images → DoclingDocument              ║
║    OCR engines: easyocr | tesseract | rapidocr | granite_vision              ║
║                                                                              ║
║  Stage 2 — VLM Rescue (optional, per-page)                                   ║
║    Low-confidence or image-heavy pages → Qwen2.5-VL (vLLM)                  ║
║    Falls back gracefully if vLLM is not reachable                            ║
║                                                                              ║
║  Stage 3 — LLM Enrichment (optional)                                         ║
║    Structured extraction of key spec fields via Ollama                       ║
║    Models: gemma3:12b (vision-aware summary) | qwen3 (reasoning/Q&A)         ║
║                                                                              ║
║  Output (per file):                                                          ║
║    <stem>.json  → Final enriched JSON  ← THIS IS THE PIPELINE OUTPUT        ║
║    <stem>.md    → Markdown (human-readable, for debugging)                   ║
║    _pipeline_summary.json → Run manifest                                     ║
║                                                                              ║
║  Install:                                                                    ║
║    pip install docling easyocr httpx pypdfium2                               ║
║    # Optional for Tesseract: sudo apt install tesseract-ocr                  ║
║    # Optional for RapidOCR:  pip install rapidocr-onnxruntime                ║
║    # Optional for Granite:   pip install 'docling[granite]'                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

pip install docling easyocr httpx pypdfium2

# Put files in ./specs/, then:
python spec_parser.py


"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

# ══════════════════════════════════════════════════════════════════════════════
#  ✅  USER CONFIGURATION — only section you need to edit
# ══════════════════════════════════════════════════════════════════════════════

# ── Input / Output ────────────────────────────────────────────────────────────
INPUT_PATH: Union[str, Path] = "./specs"    # single file  OR  directory
OUTPUT_DIR: Union[str, Path] = "./output"   # all outputs land here

# ── Stage 1: Docling OCR engine ───────────────────────────────────────────────
# Options: "easyocr" | "tesseract" | "rapidocr" | "granite_vision"
OCR_ENGINE: str = "easyocr"

# ISO-639-1 language codes for EasyOCR / RapidOCR  (e.g. ["en", "de"])
# For Tesseract use its own codes:                  (e.g. ["eng", "deu"])
OCR_LANGUAGES: list[str] = ["en"]

# Force OCR even when a text layer exists (set True for scanned specs)
FORCE_FULL_OCR: bool = False

# TableFormer ML model for table structure recognition (recommended: True)
ENABLE_TABLE_STRUCTURE: bool = True

# ── Stage 2: VLM Rescue via vLLM (Qwen2.5-VL) ────────────────────────────────
# Pages whose OCR confidence falls below this threshold are re-extracted
# by Qwen2.5-VL.  Set ENABLE_VLM_RESCUE = False to skip this stage entirely.
ENABLE_VLM_RESCUE: bool = True
VLM_CONFIDENCE_THRESHOLD: float = 0.75     # 0.0 – 1.0
VLLM_BASE_URL: str = "http://localhost:8000/v1"
VLLM_MODEL: str = "Qwen/Qwen2.5-VL-7B-Instruct"
VLLM_MAX_TOKENS: int = 4096
VLLM_TIMEOUT_SEC: int = 120
# Render resolution multiplier for page→image conversion (2 ≈ 144 DPI)
PAGE_RENDER_SCALE: float = 2.0

# ── Stage 3: LLM Enrichment via Ollama ───────────────────────────────────────
# Runs a structured extraction pass over the assembled Markdown to produce
# a clean "spec_summary" section inside the final JSON.
# Set ENABLE_LLM_ENRICHMENT = False to skip.
ENABLE_LLM_ENRICHMENT: bool = True
OLLAMA_BASE_URL: str = "http://localhost:11434"
# "gemma3:12b"  → good at structured extraction from long text
# "qwen3:latest" → text-only, stronger reasoning; use for Q&A downstream
OLLAMA_MODEL: str = "gemma3:12b"
OLLAMA_TIMEOUT_SEC: int = 180

# Fields to extract in Stage 3. Adjust to match your spec domain.
SPEC_FIELDS_TO_EXTRACT: list[str] = [
    "document_title",
    "document_number",
    "revision",
    "issue_date",
    "author_or_owner",
    "scope_or_purpose",
    "applicable_standards",
    "key_requirements",          # list
    "materials_or_components",   # list
    "test_or_acceptance_criteria",
    "safety_notes",
]

# ══════════════════════════════════════════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spec_parser")

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — Docling OCR
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf", ".docx", ".doc",
    ".xlsx", ".xls",
    ".pptx", ".ppt",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
    ".html", ".htm",
}


def _build_pipeline_options():
    """Construct Docling PipelineOptions for the chosen OCR engine."""
    from docling.datamodel.pipeline_options import (  # type: ignore
        PipelineOptions,
        EasyOcrOptions,
        TesseractOcrOptions,
        RapidOcrOptions,
    )

    opts = PipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = ENABLE_TABLE_STRUCTURE

    if OCR_ENGINE == "easyocr":
        opts.ocr_options = EasyOcrOptions(
            lang=OCR_LANGUAGES,
            force_full_page_ocr=FORCE_FULL_OCR,
            use_gpu=False,          # flip to True if CUDA available
        )
        log.info("OCR engine : EasyOCR  | langs=%s", OCR_LANGUAGES)

    elif OCR_ENGINE == "tesseract":
        opts.ocr_options = TesseractOcrOptions(
            lang=OCR_LANGUAGES,
            force_full_page_ocr=FORCE_FULL_OCR,
        )
        log.info("OCR engine : Tesseract | langs=%s", OCR_LANGUAGES)

    elif OCR_ENGINE == "rapidocr":
        opts.ocr_options = RapidOcrOptions(
            force_full_page_ocr=FORCE_FULL_OCR,
        )
        log.info("OCR engine : RapidOCR")

    elif OCR_ENGINE == "granite_vision":
        try:
            from docling.datamodel.pipeline_options import GraniteVisionOcrOptions  # type: ignore
            opts.ocr_options = GraniteVisionOcrOptions(
                force_full_page_ocr=FORCE_FULL_OCR,
            )
            log.info("OCR engine : Granite Vision (IBM)")
        except ImportError:
            raise ImportError(
                "Granite Vision not installed. Run: pip install 'docling[granite]'"
            )
    else:
        raise ValueError(
            f"Unknown OCR_ENGINE '{OCR_ENGINE}'. "
            "Choose: easyocr | tesseract | rapidocr | granite_vision"
        )

    return opts


def _build_converter(pipeline_options):
    """Build a Docling DocumentConverter with format-specific backends."""
    from docling.document_converter import DocumentConverter, FormatOption  # type: ignore
    from docling.datamodel.base_models import InputFormat                   # type: ignore
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend   # type: ignore
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend  # type: ignore

    return DocumentConverter(
        format_options={
            InputFormat.PDF: FormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            ),
            InputFormat.DOCX: FormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,
            ),
            InputFormat.XLSX:  FormatOption(pipeline_options=pipeline_options),
            InputFormat.PPTX:  FormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: FormatOption(pipeline_options=pipeline_options),
            InputFormat.HTML:  FormatOption(pipeline_options=pipeline_options),
        }
    )


def run_docling(file_path: Path, converter) -> tuple[dict, str]:
    """
    Run Docling on a single file.

    Returns
    -------
    doc_dict : dict
        Raw DoclingDocument exported as dict.
        Schema overview:
          {
            "schema_name": "DoclingDocument",
            "name":    str,               # filename stem
            "body":    { ... },           # hierarchical document tree
            "texts":   [ {text, label,    # paragraphs / headings / captions
                           prov:[{page_no, bbox}]} ],
            "tables":  [ {data:{grid:[[{text,col_span,row_span}]]},
                           prov:[{page_no, bbox}]} ],
            "pictures":[ {prov:[{page_no, bbox}]} ],
            "pages":   { "1": {size:{w,h}, ocr_confidence?}, ... },
          }
    markdown : str
        Full document as Markdown (for human review + Stage 3 input).
    """
    result   = converter.convert(str(file_path))
    doc      = result.document
    doc_dict = doc.export_to_dict()
    markdown = doc.export_to_markdown()
    return doc_dict, markdown


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — VLM Rescue (Qwen2.5-VL via vLLM)
# ══════════════════════════════════════════════════════════════════════════════

def _render_pdf_page_to_b64(pdf_path: Path, page_index: int) -> str:
    """
    Render a single PDF page to a base64-encoded PNG string.
    page_index is 0-based.
    """
    import pypdfium2 as pdfium  # type: ignore
    pdf    = pdfium.PdfDocument(str(pdf_path))
    page   = pdf[page_index]
    bitmap = page.render(scale=PAGE_RENDER_SCALE)
    pil    = bitmap.to_pil()
    buf    = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _qwen_vl_extract_page(pdf_path: Path, page_index: int) -> str | None:
    """
    Send a rendered PDF page to Qwen2.5-VL (vLLM OpenAI-compatible endpoint).
    Returns extracted Markdown text, or None on failure.
    """
    import httpx  # type: ignore

    try:
        b64 = _render_pdf_page_to_b64(pdf_path, page_index)
    except Exception as e:
        log.warning("  Page render failed (page %d): %s", page_index + 1, e)
        return None

    payload = {
        "model": VLLM_MODEL,
        "temperature": 0.0,         # deterministic — important for extraction
        "max_tokens": VLLM_MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are a technical document extraction assistant.\n"
                            "Extract ALL text from this specification page exactly as it appears.\n"
                            "Rules:\n"
                            "- Preserve every table using Markdown table syntax.\n"
                            "- Preserve section headings as Markdown headings (#, ##, ###).\n"
                            "- Do NOT summarise, skip, or paraphrase any content.\n"
                            "- Do NOT add commentary or preamble.\n"
                            "- Output ONLY the extracted content."
                        ),
                    },
                ],
            }
        ],
    }

    try:
        resp = httpx.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload,
            timeout=VLLM_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning("  vLLM call failed (page %d): %s", page_index + 1, e)
        return None


def _check_vllm_reachable() -> bool:
    """Quick health-check — returns True if vLLM endpoint responds."""
    import httpx  # type: ignore
    try:
        r = httpx.get(f"{VLLM_BASE_URL}/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def apply_vlm_rescue(
    file_path: Path,
    doc_dict: dict,
) -> dict:
    """
    For each page in doc_dict whose OCR confidence is below threshold,
    re-extract text using Qwen2.5-VL and attach it under:
      doc_dict["pages"][page_num]["vlm_extraction"]

    Only operates on PDF files (requires page rendering).
    Returns the (possibly mutated) doc_dict.
    """
    if file_path.suffix.lower() != ".pdf":
        log.info("  VLM rescue: skipped (not a PDF)")
        return doc_dict

    if not _check_vllm_reachable():
        log.warning(
            "  VLM rescue: vLLM not reachable at %s — skipping Stage 2",
            VLLM_BASE_URL,
        )
        return doc_dict

    pages = doc_dict.get("pages", {})
    rescued = 0

    for page_key, page_data in pages.items():
        confidence = page_data.get("ocr_confidence")

        # Trigger rescue if: confidence is known AND below threshold,
        # OR if confidence is None (page may be fully image-based)
        needs_rescue = (
            confidence is None
            or (isinstance(confidence, float) and confidence < VLM_CONFIDENCE_THRESHOLD)
        )

        if needs_rescue:
            page_index = int(page_key) - 1   # Docling pages are 1-based
            reason     = f"confidence={confidence}" if confidence is not None else "no text layer"
            log.info(
                "  Page %s: %s → Qwen2.5-VL rescue", page_key, reason
            )
            vlm_text = _qwen_vl_extract_page(file_path, page_index)
            if vlm_text:
                page_data["vlm_extraction"] = vlm_text
                page_data["vlm_model"]      = VLLM_MODEL
                rescued += 1
            else:
                page_data["vlm_extraction"] = None
                page_data["vlm_rescue_failed"] = True

    log.info("  VLM rescue: %d page(s) re-extracted", rescued)
    return doc_dict


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — LLM Enrichment (Ollama: gemma3:12b or qwen3)
# ══════════════════════════════════════════════════════════════════════════════

_ENRICHMENT_SYSTEM_PROMPT = """\
You are a technical specification analyst.
You will receive the full text of an engineering/product specification document.
Your job is to extract specific fields and return ONLY a valid JSON object.
Do not include markdown fences, preamble, or any text outside the JSON object.
If a field is not present in the document, set its value to null.
For list fields, return a JSON array of strings; for text fields return a string.
"""

def _build_enrichment_user_prompt(markdown_text: str) -> str:
    fields_spec = "\n".join(
        f'  "{f}": <string or list depending on field name>'
        for f in SPEC_FIELDS_TO_EXTRACT
    )
    return (
        f"Extract the following fields from the specification document below.\n"
        f"Return ONLY a JSON object with exactly these keys:\n"
        f"{{\n{fields_spec}\n}}\n\n"
        f"--- DOCUMENT START ---\n{markdown_text}\n--- DOCUMENT END ---"
    )


def _check_ollama_reachable() -> bool:
    import httpx  # type: ignore
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run_llm_enrichment(markdown_text: str) -> dict | None:
    """
    Send the full document Markdown to Ollama and extract structured fields.

    Returns a dict like:
    {
      "document_title": "...",
      "revision": "B",
      "key_requirements": ["req1", "req2"],
      ...
    }
    or None on failure.
    """
    import httpx  # type: ignore

    if not _check_ollama_reachable():
        log.warning(
            "  LLM enrichment: Ollama not reachable at %s — skipping Stage 3",
            OLLAMA_BASE_URL,
        )
        return None

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": _ENRICHMENT_SYSTEM_PROMPT},
            {"role": "user",   "content": _build_enrichment_user_prompt(markdown_text)},
        ],
    }

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()

        # Strip accidental markdown fences that some models add
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        enriched = json.loads(raw)
        log.info("  LLM enrichment: ✅ %d fields extracted", len(enriched))
        return enriched

    except json.JSONDecodeError as e:
        log.warning("  LLM enrichment: JSON parse failed (%s) — raw: %.120s", e, raw)
        return None
    except Exception as e:
        log.warning("  LLM enrichment: Ollama call failed: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  File processor — orchestrates all three stages for one file
# ══════════════════════════════════════════════════════════════════════════════

def process_file(file_path: Path, output_dir: Path, converter) -> dict:
    """
    Full pipeline for a single input file.

    Final JSON schema written to disk:
    {
      "schema_name":   "SpecParserDocument",
      "version":       "1.0",

      # ── Docling core (Stage 1) ──────────────────────────────────────────
      "name":          str,           # filename stem
      "body":          {...},         # hierarchical document tree
      "texts":         [{text, label, prov:[{page_no, bbox}]}],
      "tables":        [{data:{grid:[[{text}]]}, prov:[{page_no, bbox}]}],
      "pictures":      [{prov:[{page_no, bbox}]}],
      "pages": {
        "1": {
          "size": {"width": float, "height": float},
          "ocr_confidence": float | null,

          # ── VLM rescue (Stage 2, if triggered) ─────────────────────────
          "vlm_extraction": str | null,
          "vlm_model":      str | null,
        },
        ...
      },

      # ── LLM enrichment (Stage 3) ────────────────────────────────────────
      "spec_summary": {
        "document_title":           str | null,
        "document_number":          str | null,
        "revision":                 str | null,
        "issue_date":               str | null,
        "author_or_owner":          str | null,
        "scope_or_purpose":         str | null,
        "applicable_standards":     str | null,
        "key_requirements":         [str] | null,
        "materials_or_components":  [str] | null,
        "test_or_acceptance_criteria": str | null,
        "safety_notes":             str | null,
      },

      # ── Pipeline metadata ───────────────────────────────────────────────
      "_pipeline_meta": {
        "source_file":             str,
        "processed_at_utc":        str (ISO-8601),
        "processing_time_sec":     float,
        "ocr_engine":              str,
        "ocr_languages":           [str],
        "force_full_ocr":          bool,
        "table_structure_enabled": bool,
        "vlm_rescue_enabled":      bool,
        "vlm_model":               str | null,
        "vlm_confidence_threshold": float | null,
        "llm_enrichment_enabled":  bool,
        "llm_enrichment_model":    str | null,
      }
    }
    """
    t0   = time.perf_counter()
    stem = file_path.stem
    json_path = output_dir / f"{stem}.json"
    md_path   = output_dir / f"{stem}.md"

    log.info("─" * 60)
    log.info("File : %s", file_path.name)

    try:
        # ── Stage 1: Docling ──────────────────────────────────────────────
        log.info("  Stage 1 : Docling OCR + layout")
        doc_dict, markdown = run_docling(file_path, converter)

        # Save intermediate Markdown for human inspection / debugging
        md_path.write_text(markdown, encoding="utf-8")
        log.info("  Markdown : saved → %s", md_path.name)

        # ── Stage 2: VLM rescue ───────────────────────────────────────────
        if ENABLE_VLM_RESCUE:
            log.info("  Stage 2 : VLM rescue (threshold=%.2f)", VLM_CONFIDENCE_THRESHOLD)
            doc_dict = apply_vlm_rescue(file_path, doc_dict)
        else:
            log.info("  Stage 2 : skipped (ENABLE_VLM_RESCUE=False)")

        # ── Stage 3: LLM enrichment ───────────────────────────────────────
        spec_summary: dict | None = None
        if ENABLE_LLM_ENRICHMENT:
            log.info("  Stage 3 : LLM enrichment via Ollama (%s)", OLLAMA_MODEL)
            spec_summary = run_llm_enrichment(markdown)
        else:
            log.info("  Stage 3 : skipped (ENABLE_LLM_ENRICHMENT=False)")

        # ── Assemble final JSON ───────────────────────────────────────────
        elapsed = round(time.perf_counter() - t0, 2)

        # Rewrite schema_name to make it clear this is our enriched version
        doc_dict["schema_name"] = "SpecParserDocument"
        doc_dict["version"]     = "1.0"

        doc_dict["spec_summary"] = spec_summary or {
            f: None for f in SPEC_FIELDS_TO_EXTRACT
        }

        doc_dict["_pipeline_meta"] = {
            "source_file":              str(file_path.resolve()),
            "processed_at_utc":         datetime.now(timezone.utc).isoformat(),
            "processing_time_sec":      elapsed,
            "ocr_engine":               OCR_ENGINE,
            "ocr_languages":            OCR_LANGUAGES,
            "force_full_ocr":           FORCE_FULL_OCR,
            "table_structure_enabled":  ENABLE_TABLE_STRUCTURE,
            "vlm_rescue_enabled":       ENABLE_VLM_RESCUE,
            "vlm_model":                VLLM_MODEL if ENABLE_VLM_RESCUE else None,
            "vlm_confidence_threshold": VLM_CONFIDENCE_THRESHOLD if ENABLE_VLM_RESCUE else None,
            "llm_enrichment_enabled":   ENABLE_LLM_ENRICHMENT,
            "llm_enrichment_model":     OLLAMA_MODEL if ENABLE_LLM_ENRICHMENT else None,
        }

        json_path.write_text(
            json.dumps(doc_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("  JSON     : saved → %s  (%.1fs)", json_path.name, elapsed)

        return {
            "source":       str(file_path),
            "status":       "success",
            "json_output":  str(json_path),
            "md_output":    str(md_path),
            "elapsed_sec":  elapsed,
            "page_count":   len(doc_dict.get("pages", {})),
            "text_blocks":  len(doc_dict.get("texts",  [])),
            "table_count":  len(doc_dict.get("tables", [])),
            "spec_summary": doc_dict["spec_summary"],
        }

    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 2)
        log.error("  ❌ FAILED after %.1fs: %s", elapsed, exc, exc_info=True)
        return {
            "source":      str(file_path),
            "status":      "error",
            "error":       str(exc),
            "elapsed_sec": elapsed,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> list[dict]:
    """
    Discover input files, run the full pipeline on each, write a summary manifest.
    Returns the list of per-file result dicts.
    """
    input_path = Path(INPUT_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    if input_path.is_file():
        files = [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    elif input_path.is_dir():
        files = sorted(
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        raise FileNotFoundError(f"INPUT_PATH does not exist: {input_path}")

    if not files:
        log.warning("No supported files found in: %s", input_path)
        return []

    log.info("═" * 60)
    log.info("spec_parser  |  %d file(s) to process", len(files))
    log.info("OCR engine   :  %s", OCR_ENGINE)
    log.info("VLM rescue   :  %s  (%s)", ENABLE_VLM_RESCUE, VLLM_MODEL)
    log.info("LLM enrich   :  %s  (%s)", ENABLE_LLM_ENRICHMENT, OLLAMA_MODEL)
    log.info("Output dir   :  %s", output_dir.resolve())
    log.info("═" * 60)

    # Build Docling converter once — shared across all files
    pipeline_options = _build_pipeline_options()
    converter        = _build_converter(pipeline_options)

    results: list[dict] = []
    for f in files:
        result = process_file(f, output_dir, converter)
        results.append(result)

    # Write run manifest
    summary_path = output_dir / "_pipeline_summary.json"
    manifest = {
        "run_at_utc":  datetime.now(timezone.utc).isoformat(),
        "total":       len(results),
        "succeeded":   sum(1 for r in results if r["status"] == "success"),
        "failed":      sum(1 for r in results if r["status"] == "error"),
        "files":       results,
    }
    summary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ok  = manifest["succeeded"]
    err = manifest["failed"]
    log.info("═" * 60)
    log.info("Done  ✅ %d succeeded  ❌ %d failed", ok, err)
    if err:
        for r in results:
            if r["status"] == "error":
                log.warning("  FAILED: %s → %s", r["source"], r.get("error"))
    log.info("Manifest: %s", summary_path)
    log.info("═" * 60)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  How to consume the output JSON in your downstream LLM pipeline
# ══════════════════════════════════════════════════════════════════════════════
#
#   import json
#   doc = json.load(open("output/my_spec.json"))
#
#   # ── Quick access ─────────────────────────────────────────────────────────
#   summary   = doc["spec_summary"]               # Stage 3 structured fields
#   title     = summary["document_title"]
#   reqs      = summary["key_requirements"]       # list[str]
#
#   # ── All text blocks (paragraphs, headings, captions) ─────────────────────
#   texts = [t["text"] for t in doc.get("texts", [])]
#
#   # ── All tables ────────────────────────────────────────────────────────────
#   for table in doc.get("tables", []):
#       grid = table["data"]["grid"]              # list of rows
#       for row in grid:
#           cells = [cell.get("text", "") for cell in row]
#
#   # ── Per-page (incl. VLM rescue text where applicable) ────────────────────
#   for page_num, page in doc.get("pages", {}).items():
#       ocr_text = page.get("vlm_extraction")    # present only if rescue ran
#       confidence = page.get("ocr_confidence")
#
#   # ── Pass to Qwen3 / any LLM ──────────────────────────────────────────────
#   # The JSON is self-contained. For RAG, chunk doc["texts"] by page_no.
#   # For direct Q&A, serialize doc["spec_summary"] + doc["texts"] as context.
#
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    run_pipeline()
