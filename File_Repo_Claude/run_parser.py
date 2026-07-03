"""
tools/run_parser.py

Executes the parser script the codegen sub-agent wrote, against the real
workbook, in a subprocess (isolation + easy stdout/stderr capture for the
validator loop).
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path


def run_generated_parser(script_path: str, xlsx_path: str, out_json_path: str, timeout: int = 120) -> dict:
    """Run `script_path xlsx_path out_json_path` and report what happened.

    The generated script is expected to accept exactly these two CLI args
    and write valid JSON to out_json_path. This contract is stated in the
    codegen sub-agent's prompt.
    """
    result = subprocess.run(
        [sys.executable, script_path, xlsx_path, out_json_path],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = {
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "output_exists": Path(out_json_path).exists(),
    }
    if out["output_exists"]:
        out["output_preview"] = Path(out_json_path).read_text(encoding="utf-8")[:3000]
    return out