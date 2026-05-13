"""
JSON → TOON Converter
=====================
Uses the OFFICIAL @toon-format/cli (Node.js) as the conversion engine,
which is the authoritative reference implementation of the TOON spec.

This avoids output inconsistencies from the many unofficial Python ports.

Requirements:
    Node.js >= 16  (https://nodejs.org)
    npx is bundled with Node.js

Usage:
    python json_to_toon.py               # uses hardcoded INPUT_PATH / OUTPUT_PATH
    python json_to_toon.py in.json       # writes in.toon next to in.json
    python json_to_toon.py in.json out.toon


vllm serve Qwen/Qwen3.6-27B-FP8 \
  --tensor-parallel-size 4 \
  --max-model-len 108000 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 108000 \
  --max-num-seqs 8 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
"""

import subprocess
import sys
import json
import shutil
import os
from pathlib import Path


# ──────────────────────────────────────────────
# CONFIGURE YOUR PATHS HERE
# ──────────────────────────────────────────────
INPUT_PATH  = "/path/to/your/input.json"   # <-- change me
OUTPUT_PATH = "/path/to/your/output.toon"  # <-- change me
# ──────────────────────────────────────────────


def check_node() -> None:
    """Ensure Node.js / npx is available."""
    if shutil.which("npx") is None:
        sys.exit(
            "ERROR: npx not found. Install Node.js from https://nodejs.org "
            "then re-run this script."
        )


def validate_json(path: Path) -> None:
    """Fail fast with a clear message if the input isn't valid JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: {path} is not valid JSON.\n  {e}")
    except FileNotFoundError:
        sys.exit(f"ERROR: Input file not found: {path}")


def convert(input_path: Path, output_path: Path) -> None:
    """
    Call the official @toon-format/cli via npx.

    npx will download the package on first run and cache it afterward.
    Pin a specific version (e.g. @toon-format/cli@1.x) for reproducibility.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "npx",
        "--yes",                     # auto-install without prompting
        "@toon-format/cli",          # official CLI — pin version if needed
        str(input_path),
        "--output", str(output_path),
    ]

    print(f"Converting: {input_path} → {output_path}")
    print(f"Running:    {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if result.returncode != 0:
        # Surface the CLI's own error message for easy debugging
        sys.exit(
            f"ERROR: @toon-format/cli exited with code {result.returncode}.\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )

    # Verify the output file was actually written
    if not output_path.exists() or output_path.stat().st_size == 0:
        sys.exit(f"ERROR: Output file was not created or is empty: {output_path}")

    size_kb = output_path.stat().st_size / 1024
    print(f"✓ Done.  Output: {output_path}  ({size_kb:.1f} KB)")

    # Print a preview of the first 10 lines
    with open(output_path, encoding="utf-8") as f:
        lines = [next(f, None) for _ in range(10)]
    preview = "".join(l for l in lines if l is not None).rstrip()
    if preview:
        print("\n── Preview (first 10 lines) ──────────────────")
        print(preview)
        print("──────────────────────────────────────────────")


def resolve_paths() -> tuple[Path, Path]:
    """
    Priority:
      1. CLI args: python json_to_toon.py <input> [output]
      2. Hardcoded INPUT_PATH / OUTPUT_PATH at the top of this file
    """
    args = sys.argv[1:]
    if args:
        inp = Path(args[0]).resolve()
        out = Path(args[1]).resolve() if len(args) > 1 else inp.with_suffix(".toon")
    else:
        inp = Path(INPUT_PATH).resolve()
        out = Path(OUTPUT_PATH).resolve()
    return inp, out


def main() -> None:
    check_node()

    input_path, output_path = resolve_paths()

    # Safety check: don't overwrite the input file
    if input_path == output_path:
        sys.exit("ERROR: Input and output paths must be different.")

    validate_json(input_path)
    convert(input_path, output_path)


if __name__ == "__main__":
    main()
