"""
agent/main.py

Orchestrator: deepagents-based pipeline that converts a register-map
xlsx/xlsm into JSON, regenerating its own parsing logic per input file
shape instead of relying on one static parser.

Usage:
    python agent/main.py /path/to/registers.xlsm /path/to/out.json

Requires a running vLLM OpenAI-compatible server, e.g.:
    vllm serve Qwen/Qwen3.6-XXB-Instruct --port 8000
(adjust model name to whatever Qwen3.6 checkpoint you're actually serving)
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tools.inspect_excel import inspect_workbook
from tools.run_parser import run_generated_parser
from tools.validate_json import validate_register_json

# ---------------------------------------------------------------------------
# Model: point this at your vLLM server. ChatOpenAI works against any
# OpenAI-compatible endpoint, which vLLM's `--api-key` / openai server mode
# provides out of the box.
# ---------------------------------------------------------------------------
MODEL = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM doesn't check this unless you set --api-key
    model="qwen3.6",        # must match the --served-model-name on your vLLM server
    temperature=0,
)

WORK_DIR = Path("/home/claude/xlsx2json_agent/output")
WORK_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Tools exposed to the agent (thin wrappers around the python helpers)
# ---------------------------------------------------------------------------
def inspect_excel_tool(xlsx_path: str, sheets: str = "") -> str:
    """Inspect an xlsx/xlsm register-map workbook and return a compact JSON
    summary of its sheets, header rows, a sample of data rows (with cell
    fill colors and merged ranges), so you can figure out the row grammar
    (e.g. which rows are register/group headers vs. individual bitfields).

    `sheets`: optional comma-separated list of sheet names to restrict to
    (e.g. "Conf-1,Conf-2"). Leave empty to inspect all sheets."""
    sheet_list = [s.strip() for s in sheets.split(",") if s.strip()] or None
    return inspect_workbook(xlsx_path, sheets=sheet_list)


def write_parser_script(code: str, script_name: str = "generated_parser.py") -> str:
    """Write the Python parser script you've designed to disk. The script
    MUST accept exactly two CLI args: <xlsx_path> <out_json_path>, read the
    workbook with openpyxl, and write the final register-map JSON to
    out_json_path. Returns the path it was written to."""
    path = WORK_DIR / script_name
    path.write_text(code, encoding="utf-8")
    return str(path)


def run_parser_tool(script_path: str, xlsx_path: str, out_json_path: str) -> str:
    """Run a previously-written parser script against the real workbook and
    capture stdout/stderr/output preview."""
    return json.dumps(run_generated_parser(script_path, xlsx_path, out_json_path))


def validate_json_tool(json_path: str) -> str:
    """Run structural validation (bitfield overlap, missing addresses,
    empty field lists, etc.) on a produced register-map JSON. Returns
    errors/warnings you should fix by patching the parser script."""
    return json.dumps(validate_register_json(json_path))


TOOLS = [inspect_excel_tool, write_parser_script, run_parser_tool, validate_json_tool]


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------
SCHEMA_SNIFFER_PROMPT = """\
You are a row-grammar analyst for hardware register-map spreadsheets.
You will be given a JSON dump (from inspect_excel_tool) of a sheet's header
rows and a sample of data rows, including each row's raw cell values and
fill color, plus merged-cell ranges.

These workbooks follow a pattern like this example (column names vary,
language may be Japanese or English, column order may change):
  - A "row type" column whose value is something like "Group" or "Reg".
  - A "Group" row marks the start of a new register: it carries the
    register's base/offset address and its name. It usually has no
    MSB/LSB and a distinct fill color (e.g. shaded yellow/orange).
  - One or more "Reg" rows immediately follow, each describing ONE
    bitfield of that same register: MSB, LSB, field/mnemonic name, reset
    value, access type (RW/RO/Reserved/...), and a description. All "Reg"
    rows under a "Group" share that group's address, until the next
    "Group" row starts a new register.
  - A register can have just one field spanning the whole word, or many
    narrow bitfields.

Your job: produce a JSON "grammar spec" describing, for THIS workbook
specifically:
  - which column (by index, 0-based, after the header) holds the row type,
    and what values mean "new register" vs "field row"
  - which columns hold: address/offset, end-address (if present), msb,
    lsb, register name, field name, reset value, access type, description
  - which row index the real data starts on
  - any quirks you noticed (merged cells, multi-sheet layout, reserved
    rows to special-case, etc.)

Output ONLY the JSON grammar spec, no prose.
"""

CODEGEN_PROMPT = """\
You are a Python codegen agent. You receive a JSON grammar spec describing
how to read a specific register-map workbook (produced by the schema
sniffer), and you write a standalone Python script using openpyxl that:

  1. Accepts exactly two CLI args: input xlsx/xlsm path, output json path
     (sys.argv[1], sys.argv[2]).
  2. Walks the rows per the grammar spec, accumulating fields under their
     parent register until the next register-start row, mirroring the
     spec's column mapping (do not hardcode column letters from any other
     file — use what the grammar spec says for THIS file).
  3. Emits JSON shaped like:
     {
       "registers": [
         {
           "name": "...",
           "address": "0x000",
           "fields": [
             {"name": "...", "msb": 31, "lsb": 6, "reset": "0",
              "type": "RW", "description": "..."},
             ...
           ]
         },
         ...
       ]
     }
  4. Skips pure header/instruction rows; handles merged cells (a value
     merged across rows applies to all rows in that merge).
  5. Is defensive: skips rows that don't match the expected row-type
     values rather than crashing, but logs a one-line warning to stderr
     for anything skipped so problems are visible.

Use write_parser_script to save the script, then run_parser_tool to run it
against the real workbook, then validate_json_tool to check the result.
If validation reports errors, inspect the relevant raw rows again with
inspect_excel_tool if needed, fix the script, and re-run. Iterate up to 5
times. Report the final validation result and the output path.
"""

subagents = [
    {
        "name": "schema-sniffer",
        "description": "Analyzes a workbook's row layout and produces a grammar spec for it.",
        "prompt": SCHEMA_SNIFFER_PROMPT,
        "tools": ["inspect_excel_tool"],
    },
    {
        "name": "codegen-and-validate",
        "description": "Writes, runs, and iteratively fixes a parser script for a given grammar spec, producing validated JSON.",
        "prompt": CODEGEN_PROMPT,
        "tools": ["inspect_excel_tool", "write_parser_script", "run_parser_tool", "validate_json_tool"],
    },
]

MAIN_PROMPT = """\
You convert hardware register-map Excel workbooks (xlsx/xlsm) into
structured JSON. The input file's column layout and language can change
between files, so you must NOT assume a fixed format. Workflow:

1. Call the schema-sniffer sub-agent on the input file to get a grammar
   spec for this specific workbook's row layout.
2. Call the codegen-and-validate sub-agent with that grammar spec to
   produce, run, and validate a parser script, iterating until validation
   passes or you've made a reasonable number of attempts.
3. Report the final output JSON path and a short summary (register count,
   any remaining warnings) to the user.
"""

agent = create_deep_agent(
    tools=TOOLS,
    instructions=MAIN_PROMPT,
    model=MODEL,
    subagents=subagents,
)


def convert(xlsx_path: str, out_json_path: str, sheets: list[str] | None = None):
    sheet_note = (
        f" Only process these sheets: {', '.join(sheets)}. Ignore all other tabs."
        if sheets else ""
    )
    user_msg = (
        f"Convert the workbook at {xlsx_path} into register-map JSON and "
        f"write the final validated output to {out_json_path}.{sheet_note}"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
    return result


if __name__ == "__main__":
    xlsx_in = sys.argv[1]
    json_out = sys.argv[2] if len(sys.argv) > 2 else str(WORK_DIR / "result.json")
    sheets_arg = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    res = convert(xlsx_in, json_out, sheets=sheets_arg)
    print(res["messages"][-1].content)