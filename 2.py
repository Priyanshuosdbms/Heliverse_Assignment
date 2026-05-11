"""
JSON to RDL Converter — Ollama + qwen3-coder
=============================================
Handles large JSON by splitting into chunks and processing in parallel.

Requirements:
    pip install requests

Usage:
    python json_to_rdl_ollama.py
    python json_to_rdl_ollama.py --system-prompt system.txt --user-prompt-template user.txt
"""

import argparse
import json
import sys
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ──────────────────────────────────────────────────────────────────────────────
# ✏️  EDIT THESE
# ──────────────────────────────────────────────────────────────────────────────

JSON_INPUT_PATH = "data.json"   # Path to your input JSON file
OUTPUT_PATH     = None          # e.g. "output.rdl" — or None for auto (<input>.rdl)

CHUNK_SIZE  = 10   # Number of JSON keys / array items per chunk
MAX_WORKERS = 4    # Parallel threads (keep low for Ollama — it runs single-threaded)

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "qwen3-coder"

TEMPERATURE = 0.1
MAX_TOKENS  = 4096   # Max tokens per chunk response

# ──────────────────────────────────────────────────────────────────────────────


DEFAULT_SYSTEM_PROMPT = """You are an expert at converting structured JSON data into RDL (Register Description Language).
Convert the provided JSON chunk accurately into valid RDL format.
Follow RDL specification strictly.
Output only the RDL content — no explanation, no markdown, no extra text."""

DEFAULT_USER_PROMPT_TEMPLATE = """Convert the following JSON chunk (chunk {chunk_index} of {total_chunks}) into RDL format:

{json_content}

Output only the RDL."""


# ──────────────────────────────────────────────────────────────────────────────
# OLLAMA CLIENT
# ──────────────────────────────────────────────────────────────────────────────

def call_ollama(system_prompt: str, user_prompt: str, retries: int = 3) -> str:
    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            sys.exit(f"[ERROR] Cannot connect to Ollama at {OLLAMA_URL}. Is Ollama running?")
        except requests.exceptions.HTTPError as e:
            if attempt == retries:
                sys.exit(f"[ERROR] Ollama API failed after {retries} attempts: {e}\n{response.text}")
            print(f"[WARN] Attempt {attempt} failed, retrying...")
            time.sleep(2 ** attempt)


# ──────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ──────────────────────────────────────────────────────────────────────────────

def chunk_json(data) -> list:
    """
    Split JSON into chunks of CHUNK_SIZE.
      - list  → sublists of CHUNK_SIZE items
      - dict  → dicts of CHUNK_SIZE top-level keys
      - other → single-element list (no chunking possible)
    """
    if isinstance(data, list):
        return [data[i:i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]

    if isinstance(data, dict):
        keys = list(data.keys())
        return [
            {k: data[k] for k in keys[i:i + CHUNK_SIZE]}
            for i in range(0, len(keys), CHUNK_SIZE)
        ]

    return [data]


# ──────────────────────────────────────────────────────────────────────────────
# PARALLEL CONVERSION
# ──────────────────────────────────────────────────────────────────────────────

def process_chunk(
    chunk,
    chunk_index: int,
    total_chunks: int,
    system_prompt: str,
    user_prompt_template: str,
) -> tuple[int, str]:
    json_str    = json.dumps(chunk, indent=2)
    user_prompt = user_prompt_template.format(
        chunk_index  = chunk_index + 1,
        total_chunks = total_chunks,
        json_content = json_str,
    )

    print(f"  [chunk {chunk_index + 1}/{total_chunks}] Sending to Ollama...")
    rdl = call_ollama(system_prompt=system_prompt, user_prompt=user_prompt)
    print(f"  [chunk {chunk_index + 1}/{total_chunks}] Done.")
    return chunk_index, rdl


def convert(json_data, system_prompt: str, user_prompt_template: str) -> str:
    chunks       = chunk_json(json_data)
    total_chunks = len(chunks)
    workers      = min(MAX_WORKERS, total_chunks)

    print(f"[INFO] JSON split into {total_chunks} chunk(s) of up to {CHUNK_SIZE} items each.")
    print(f"[INFO] Running with {workers} parallel worker(s)...\n")

    results = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_chunk,
                chunk                = chunk,
                chunk_index          = idx,
                total_chunks         = total_chunks,
                system_prompt        = system_prompt,
                user_prompt_template = user_prompt_template,
            ): idx
            for idx, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            try:
                idx, rdl = future.result()
                results[idx] = rdl
            except Exception as e:
                sys.exit(f"[ERROR] Chunk {futures[future] + 1} failed: {e}")

    ordered   = [results[i] for i in range(total_chunks)]
    separator = "\n\n// " + "─" * 56 + "\n\n"
    return separator.join(ordered)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict | list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] Invalid JSON in {path}: {e}")


def load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI  (only prompt overrides remain as flags)
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert large JSON to RDL using Ollama + qwen3-coder."
    )
    parser.add_argument("--output", "-o", default=None,
                        help="Override output RDL file path.")
    parser.add_argument("--system-prompt", default=None,
                        help="Path to a custom system prompt text file.")
    parser.add_argument("--user-prompt-template", default=None,
                        help="Path to a custom user prompt template. "
                             "Placeholders: {json_content}, {chunk_index}, {total_chunks}.")
    return parser.parse_args()


def main():
    args = parse_args()

    system_prompt        = load_text(args.system_prompt)        if args.system_prompt        else DEFAULT_SYSTEM_PROMPT
    user_prompt_template = load_text(args.user_prompt_template) if args.user_prompt_template else DEFAULT_USER_PROMPT_TEMPLATE

    print(f"[INFO] Loading JSON from : {JSON_INPUT_PATH}")
    print(f"[INFO] Model             : {OLLAMA_MODEL}  |  chunk_size={CHUNK_SIZE}  |  workers={MAX_WORKERS}")
    json_data = load_json(JSON_INPUT_PATH)

    rdl = convert(json_data, system_prompt, user_prompt_template)

    output_path = args.output or OUTPUT_PATH or str(Path(JSON_INPUT_PATH).with_suffix(".rdl"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rdl)

    print(f"\n[OK] RDL written to: {output_path}")
    print("\n" + "─" * 60)
    print(rdl)


if __name__ == "__main__":
    main()
