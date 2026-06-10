"""
telemetry_enricher.py
─────────────────────
Reads:
  • SSD Telemetry Specification (Excel, all sheets)
  • DB attribute CSV  [TABLE_NAME, COLUMN_NAME, ...]

Queries Qwen3 (vLLM OpenAI-compatible endpoint) to produce
  [TABLE_NAME, COLUMN_NAME, table_description, column_description]

Outputs: enriched_attributes.csv
"""

import os
import re
import json
import time
import logging
import textwrap
import pandas as pd
from openai import OpenAI
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
VLLM_BASE_URL   = "http://localhost:8000/v1"   # change to your vLLM host
VLLM_API_KEY    = "EMPTY"                       # vLLM default
MODEL_NAME      = "Qwen/Qwen3-7B"              # or Qwen3-6B-Instruct etc.
EXCEL_SPEC_PATH = "SSD_Telemetry_Specification.xlsx"
ATTRIBUTES_CSV  = "db_attributes.csv"
OUTPUT_CSV      = "enriched_attributes.csv"
MAX_RETRIES     = 3
RETRY_DELAY     = 2   # seconds between retries
MAX_SPEC_CHARS  = 12_000  # cap spec context sent per request to avoid OOM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_telemetry_spec(path: str) -> str:
    """Flatten all sheets of the Excel spec into a single text blob."""
    log.info("Loading telemetry spec from %s", path)
    xl = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    chunks = []
    for sheet_name, df in xl.items():
        df = df.fillna("")
        text = df.to_csv(sep="\t", index=False, header=False)
        chunks.append(f"=== Sheet: {sheet_name} ===\n{text}")
    full = "\n\n".join(chunks)
    log.info("Spec loaded: %d chars across %d sheets", len(full), len(xl))
    return full


def load_attributes(path: str) -> pd.DataFrame:
    log.info("Loading attribute CSV from %s", path)
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"TABLE_NAME", "COLUMN_NAME"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}. Found: {set(df.columns)}")
    return df


def trim_spec(spec: str, table: str, column: str, max_chars: int) -> str:
    """
    Return a context-relevant slice of the spec.
    First try lines that contain the column/table name (case-insensitive).
    Fall back to a head truncation if nothing relevant found.
    """
    col_lower = column.lower().replace("_", " ")
    tbl_lower = table.lower().replace("_", " ")
    relevant, rest = [], []
    for line in spec.splitlines():
        ll = line.lower()
        if col_lower in ll or tbl_lower in ll or any(
            tok in ll for tok in col_lower.split() if len(tok) > 3
        ):
            relevant.append(line)
        else:
            rest.append(line)

    combined = "\n".join(relevant) + "\n" + "\n".join(rest)
    return combined[:max_chars]


def build_prompt(spec_excerpt: str, table: str, column: str) -> str:
    return textwrap.dedent(f"""
        You are a precise technical documentation assistant for SSD firmware telemetry.
        Below is an excerpt from the SSD Telemetry Specification document.

        <telemetry_spec>
        {spec_excerpt}
        </telemetry_spec>

        Task:
        1. Find the description for the TABLE "{table}" — what this telemetry table represents.
        2. Find the description for the COLUMN "{column}" within that table — what this field measures or records.

        IMPORTANT rules:
        - If you cannot find a description, you MUST say so explicitly — do NOT invent one.
          Missing descriptions are critical gaps that need human review.
        - Keep descriptions concise (1–3 sentences each).
        - Respond ONLY with valid JSON in this exact structure (no markdown fences, no extra text):
        {{
          "table_description": "<description or 'NOT FOUND IN SPEC — requires manual review'>",
          "column_description": "<description or 'NOT FOUND IN SPEC — requires manual review'>"
        }}
    """).strip()


def query_llm(client: OpenAI, prompt: str) -> dict:
    """Call vLLM and parse the JSON response with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip accidental markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning("Attempt %d — JSON parse error: %s | raw: %s", attempt, e, raw[:200])
        except Exception as e:
            log.warning("Attempt %d — API error: %s", attempt, e)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    log.error("All retries exhausted for prompt excerpt.")
    return {
        "table_description": "ERROR: LLM call failed after retries",
        "column_description": "ERROR: LLM call failed after retries",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    spec     = load_telemetry_spec(EXCEL_SPEC_PATH)
    attrs_df = load_attributes(ATTRIBUTES_CSV)
    client   = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    results = []
    total = len(attrs_df)

    for idx, row in attrs_df.iterrows():
        table  = row["TABLE_NAME"].strip()
        column = row["COLUMN_NAME"].strip()
        log.info("[%d/%d] Processing %s.%s", idx + 1, total, table, column)

        spec_excerpt = trim_spec(spec, table, column, MAX_SPEC_CHARS)
        prompt       = build_prompt(spec_excerpt, table, column)
        llm_out      = query_llm(client, prompt)

        result = {
            "TABLE_NAME":         table,
            "COLUMN_NAME":        column,
            "table_description":  llm_out.get("table_description", ""),
            "column_description": llm_out.get("column_description", ""),
        }
        # Preserve any extra columns from the original CSV
        for col in attrs_df.columns:
            if col not in result:
                result[col] = row[col]

        results.append(result)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved enriched attributes to %s (%d rows)", OUTPUT_CSV, len(out_df))

    # Summary: flag NOT FOUND entries
    not_found = out_df[
        out_df["column_description"].str.contains("NOT FOUND", na=False) |
        out_df["table_description"].str.contains("NOT FOUND", na=False)
    ]
    if not not_found.empty:
        log.warning(
            "⚠  %d attribute(s) had descriptions NOT FOUND in spec — "
            "manual review required:\n%s",
            len(not_found),
            not_found[["TABLE_NAME", "COLUMN_NAME"]].to_string(index=False)
        )
    else:
        log.info("✓ All attributes resolved successfully.")


if __name__ == "__main__":
    main()
