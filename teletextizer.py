"""
telemetry_enricher.py
─────────────────────
Reads:
  • SSD Telemetry Specification  (Excel, all sheets)
  • DB attribute CSV             [TABLE_NAME, COLUMN_NAME, ...]

For each attribute:
  1. Uses Ollama embeddings (mxbai-embed-large) to semantically retrieve the
     most relevant chunks from the spec (no direct string matching needed).
  2. Queries Qwen3 on vLLM to produce column_description.

After all columns in a table are done:
  3. Concatenates all column descriptions and asks Qwen3 to write a single
     coherent table_description sentence from them.

Output: enriched_attributes.csv  [TABLE_NAME, COLUMN_NAME,
                                   table_description, column_description]
"""

import os, re, json, time, logging, textwrap
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
VLLM_BASE_URL   = "http://localhost:8000/v1"   # ← your vLLM host
VLLM_API_KEY    = "EMPTY"
MODEL_NAME      = "Qwen/Qwen3-7B"             # ← exact model name on vLLM
EXCEL_SPEC_PATH = "SSD_Telemetry_Specification.xlsx"
ATTRIBUTES_CSV  = "db_attributes.csv"
OUTPUT_CSV      = "enriched_attributes.csv"

# Ollama embedding config
# mxbai-embed-large chosen: highest MTEB score of the four options, excellent
# on technical/domain vocabulary — outperforms nomic-embed-text, embedding-gemma
# and all-minilm for niche firmware spec retrieval.
OLLAMA_BASE_URL = "http://localhost:11434"     # ← your Ollama host
EMBED_MODEL     = "mxbai-embed-large"

# How many spec chunks to feed the LLM per attribute
TOP_K_CHUNKS    = 6
# Lines per chunk when splitting the spec
CHUNK_SIZE      = 15
# LLM call settings
MAX_RETRIES     = 3
RETRY_DELAY_SEC = 3
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 512
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────── Spec loading ────────────────────────────────────

def load_spec_chunks(path: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Load all Excel sheets, convert each to text, split into overlapping chunks."""
    log.info("Loading spec: %s", path)
    xl = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    raw_lines: list[str] = []
    for sheet_name, df in xl.items():
        df = df.fillna("")
        raw_lines.append(f"[Sheet: {sheet_name}]")
        for _, row in df.iterrows():
            line = "\t".join(str(c) for c in row if str(c).strip())
            if line.strip():
                raw_lines.append(line)

    # Sliding window with 50 % overlap
    chunks, step = [], max(1, chunk_size // 2)
    for i in range(0, len(raw_lines), step):
        chunk = "\n".join(raw_lines[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    log.info("Spec → %d lines → %d chunks (size=%d, step=%d)",
             len(raw_lines), len(chunks), chunk_size, step)
    return chunks


# ──────────────────────────── Embeddings / retrieval ─────────────────────────

def _ollama_embed(texts: list[str], model: str, base_url: str) -> np.ndarray:
    """
    Call Ollama /api/embed for a batch of texts.
    Returns a float32 matrix of shape (len(texts), dim), L2-normalised.
    Ollama's /api/embed accepts a list under the 'input' key (Ollama ≥ 0.3).
    Falls back to single-call loop for older versions automatically.
    """
    url = f"{base_url.rstrip('/')}/api/embed"
    try:
        resp = requests.post(url, json={"model": model, "input": texts}, timeout=120)
        resp.raise_for_status()
        vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
    except (KeyError, requests.HTTPError):
        # Older Ollama: /api/embeddings, single string per call
        url_legacy = f"{base_url.rstrip('/')}/api/embeddings"
        vecs = []
        for t in texts:
            r = requests.post(url_legacy,
                              json={"model": model, "prompt": t}, timeout=60)
            r.raise_for_status()
            vecs.append(r.json()["embedding"])
        vecs = np.array(vecs, dtype=np.float32)

    # L2 normalise for cosine similarity via dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def build_chunk_index(chunks: list[str],
                      embed_model_name: str = EMBED_MODEL,
                      ollama_url: str = OLLAMA_BASE_URL,
                      batch_size: int = 32):
    """
    Embed all spec chunks once via Ollama and return the normalised matrix.
    Returns (embed_model_name, matrix) — the 'model' is just the name string,
    passed into retrieve_chunks so it can call Ollama for query vectors.
    """
    log.info("Building embedding index via Ollama '%s' …", embed_model_name)

    # Connectivity check
    try:
        requests.get(f"{ollama_url}/api/tags", timeout=5).raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {ollama_url}. "
            f"Is Ollama running and is '{embed_model_name}' pulled?\n"
            f"  ollama pull {embed_model_name}\nError: {exc}"
        ) from exc

    all_vecs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        log.info("  Embedding chunks %d–%d / %d …",
                 i + 1, min(i + batch_size, len(chunks)), len(chunks))
        all_vecs.append(_ollama_embed(batch, embed_model_name, ollama_url))

    matrix = np.vstack(all_vecs)
    log.info("Index ready: %s", matrix.shape)
    return embed_model_name, matrix


def retrieve_chunks(query: str, embed_model_name: str,
                    chunk_matrix: np.ndarray,
                    chunks: list[str],
                    top_k: int = TOP_K_CHUNKS,
                    ollama_url: str = OLLAMA_BASE_URL) -> str:
    """Return the top-k spec chunks most semantically similar to `query`."""
    q_vec = _ollama_embed([query], embed_model_name, ollama_url)   # (1, dim)
    scores = (chunk_matrix @ q_vec.T).ravel()
    top_idx = np.argsort(scores)[::-1][:top_k]
    selected = [chunks[i] for i in sorted(top_idx)]   # preserve doc order
    return "\n---\n".join(selected)


# ──────────────────────────── LLM helpers ────────────────────────────────────

def call_llm(client: OpenAI, prompt: str, label: str = "") -> str:
    """
    Call vLLM and return the raw text response.
    Surfaces the real exception on each attempt so you can diagnose failures.
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            last_err = exc
            log.warning("[%s] Attempt %d/%d failed: %s: %s",
                        label, attempt, MAX_RETRIES, type(exc).__name__, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    # All retries exhausted — raise with full detail
    raise RuntimeError(
        f"[{label}] vLLM call failed after {MAX_RETRIES} attempts. "
        f"Last error → {type(last_err).__name__}: {last_err}\n\n"
        f"Check: is vLLM running at {VLLM_BASE_URL}? "
        f"Is model '{MODEL_NAME}' loaded? (curl {VLLM_BASE_URL}/v1/models)"
    ) from last_err


def parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean).strip()
    return json.loads(clean)


# ──────────────────────────── Per-attribute prompt ───────────────────────────

def prompt_column_description(table: str, column: str,
                               spec_excerpt: str) -> str:
    return textwrap.dedent(f"""
        You are a precise technical documentation assistant for SSD firmware telemetry.
        The following excerpts are from the SSD Telemetry Specification and were
        retrieved as the most semantically relevant sections for the attribute below.

        <spec_excerpts>
        {spec_excerpt}
        </spec_excerpts>

        Task:
        Provide a concise 1–2 sentence description for the column "{column}"
        in telemetry table "{table}".

        Rules:
        - Base your answer ONLY on the spec excerpts above.
        - If the information is genuinely absent, say exactly:
          "NOT FOUND IN SPEC — requires manual review"
          Do NOT invent or infer beyond what is written.
        - Respond ONLY with valid JSON — no markdown, no preamble:
        {{"column_description": "<your description>"}}
    """).strip()


# ──────────────────────────── Table summary prompt ───────────────────────────

def prompt_table_description(table: str,
                              col_descriptions: list[tuple[str, str]]) -> str:
    col_block = "\n".join(
        f"  • {col}: {desc}" for col, desc in col_descriptions
        if "NOT FOUND" not in desc
    )
    return textwrap.dedent(f"""
        You are writing concise technical documentation for an SSD telemetry database.

        The table "{table}" contains these columns and their descriptions:
        {col_block}

        Task:
        Write ONE coherent sentence (max 40 words) that describes what the
        table "{table}" as a whole captures or represents.
        Focus on the collective purpose, not individual columns.

        Respond ONLY with valid JSON — no markdown, no preamble:
        {{"table_description": "<one sentence>"}}
    """).strip()


# ──────────────────────────── Main pipeline ──────────────────────────────────

def main():
    # 1. Load inputs
    chunks = load_spec_chunks(EXCEL_SPEC_PATH)
    embed_model_name, chunk_matrix = build_chunk_index(chunks)

    attrs_df = pd.read_csv(ATTRIBUTES_CSV, dtype=str).fillna("")
    required = {"TABLE_NAME", "COLUMN_NAME"}
    if not required.issubset(attrs_df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Got: {set(attrs_df.columns)}")

    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    # Quick connectivity check before the main loop
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        log.info("vLLM reachable. Available models: %s", available)
        if MODEL_NAME not in available:
            log.warning("Model '%s' not in available list %s — proceeding anyway.",
                        MODEL_NAME, available)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach vLLM at {VLLM_BASE_URL}. "
            f"Start vLLM first.\nError: {exc}"
        ) from exc

    # 2. Process each attribute
    records = []          # [{ TABLE_NAME, COLUMN_NAME, column_description, ...orig }]
    total = len(attrs_df)

    for idx, row in attrs_df.iterrows():
        table  = row["TABLE_NAME"].strip()
        column = row["COLUMN_NAME"].strip()
        log.info("[%d/%d] %s.%s", idx + 1, total, table, column)

        # Semantic retrieval — query is the natural-language attribute description
        query        = f"telemetry attribute {column} in table {table} SSD firmware"
        spec_excerpt = retrieve_chunks(query, embed_model_name, chunk_matrix, chunks)

        prompt = prompt_column_description(table, column, spec_excerpt)
        raw    = call_llm(client, prompt, label=f"{table}.{column}")

        try:
            col_desc = parse_json_response(raw).get("column_description", "")
        except json.JSONDecodeError:
            log.warning("JSON parse failed for %s.%s — storing raw LLM output.", table, column)
            col_desc = raw  # keep whatever came back rather than lose it

        rec = {"TABLE_NAME": table, "COLUMN_NAME": column,
               "column_description": col_desc, "table_description": ""}
        for col in attrs_df.columns:
            if col not in rec:
                rec[col] = row[col]
        records.append(rec)

    result_df = pd.DataFrame(records)

    # 3. Generate table descriptions by summarising all column descriptions per table
    log.info("Generating table descriptions …")
    table_descs: dict[str, str] = {}

    for table, grp in result_df.groupby("TABLE_NAME", sort=False):
        col_descs = list(zip(grp["COLUMN_NAME"], grp["column_description"]))
        valid = [(c, d) for c, d in col_descs if "NOT FOUND" not in d and d.strip()]

        if not valid:
            log.warning("Table '%s' has no resolved column descriptions — skipping summary.", table)
            table_descs[table] = "NOT FOUND — all columns unresolved; requires manual review"
            continue

        prompt   = prompt_table_description(table, valid)
        raw      = call_llm(client, prompt, label=f"TABLE:{table}")
        try:
            t_desc = parse_json_response(raw).get("table_description", "")
        except json.JSONDecodeError:
            t_desc = raw

        table_descs[table] = t_desc
        log.info("  Table '%s': %s", table, t_desc[:80])

    result_df["table_description"] = result_df["TABLE_NAME"].map(table_descs)

    # 4. Reorder columns and save
    front_cols = ["TABLE_NAME", "COLUMN_NAME", "table_description", "column_description"]
    extra_cols = [c for c in result_df.columns if c not in front_cols]
    result_df  = result_df[front_cols + extra_cols]
    result_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved → %s  (%d rows)", OUTPUT_CSV, len(result_df))

    # 5. Report gaps
    not_found = result_df[
        result_df["column_description"].str.contains("NOT FOUND", na=False) |
        result_df["table_description"].str.contains("NOT FOUND", na=False)
    ]
    if not not_found.empty:
        log.warning(
            "⚠  %d attribute(s) not found in spec — manual review needed:\n%s",
            len(not_found),
            not_found[["TABLE_NAME", "COLUMN_NAME"]].to_string(index=False),
        )
    else:
        log.info("✓  All attributes resolved.")


if __name__ == "__main__":
    main()
