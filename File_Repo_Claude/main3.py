"""
agent/main.py

Register-map xlsx/xlsm → JSON converter using deepagents.

Usage:
    python agent/main.py <xlsx_path> <out_json_path> [sheet1,sheet2,...]

Requires a running vLLM server in OpenAI-compatible mode, e.g.:
    vllm serve Qwen/Qwen3-8B-Instruct --served-model-name qwen3 --port 8000
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tools.inspect_excel import inspect_workbook
from tools.run_parser import run_generated_parser
from tools.validate_json import validate_register_json
from tools.resolve_address import resolve_address as _resolve_address

# ---------------------------------------------------------------------------
# Model — vLLM runs an OpenAI-compatible server; ChatOpenAI points there.
# Change base_url / model to match your vllm serve --served-model-name.
# ---------------------------------------------------------------------------
model = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="qwen3",
    temperature=0,
)

WORK_DIR = Path(__file__).resolve().parent.parent / "output"
WORK_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Tools — plain functions; the docstring is what the agent sees as the
# tool description. Keep docstrings precise: the agent uses them to decide
# which tool to call and with which arguments.
# ---------------------------------------------------------------------------

@tool
def inspect_excel(xlsx_path: str, sheets: str = "") -> str:
    """Inspect an xlsx/xlsm register-map workbook. Returns a compact JSON
    summary of sheet structure: header rows, a sample of data rows with
    raw cell values and fill colours, and merged-cell ranges. This lets
    you figure out the row grammar (which column holds the row type,
    where the address/MSB/LSB/name/description columns live, etc.).

    xlsx_path: absolute or relative path to the workbook.
    sheets: optional comma-separated sheet names to restrict to
            (e.g. "Conf-1,Conf-2"). Leave empty to inspect all sheets.
    """
    sheet_list = [s.strip() for s in sheets.split(",") if s.strip()] or None
    return inspect_workbook(xlsx_path, sheets=sheet_list)


@tool
def write_parser_script(code: str) -> str:
    """Write the Python parser script you have designed to disk and return
    its path. The script MUST:
      - accept exactly two CLI args: sys.argv[1]=xlsx_path, sys.argv[2]=out_json_path
      - read the workbook with openpyxl (data_only=True)
      - walk rows per the grammar you discovered, grouping all 'Reg' bitfield
        rows that follow a 'Group' row into a single register's fields list
        until the next 'Group' row starts a new register
      - write a JSON file shaped like:
          {"registers": [{"name": "...", "address": "0x...",
                          "fields": [{"name":"...", "msb":31, "lsb":0,
                                      "reset":"0", "type":"RW",
                                      "description":"..."}]}]}
      - handle merged cells (a value merged across rows applies to all rows
        in that merge range)
      - print one-line warnings to stderr for rows it skips, not crash

    code: the complete Python source of the parser script.
    """
    path = WORK_DIR / "generated_parser.py"
    path.write_text(code, encoding="utf-8")
    return str(path)


@tool
def run_parser(script_path: str, xlsx_path: str, out_json_path: str) -> str:
    """Run a parser script against the real workbook and return a JSON
    summary of the result: returncode, stdout (last 4k), stderr (last 4k),
    and the first 3k of the output JSON if it was written successfully.

    script_path: path returned by write_parser_script.
    xlsx_path: path to the original workbook.
    out_json_path: where the script should write its JSON output.
    """
    return json.dumps(run_generated_parser(script_path, xlsx_path, out_json_path))


@tool
def validate_json(json_path: str) -> str:
    """Validate a produced register-map JSON for structural correctness:
    checks that every register has at least one field, every field has
    msb/lsb integers with msb >= lsb, no bitfields overlap within a
    register, and every register has an address. Returns a JSON object
    with keys: ok (bool), errors (list), warnings (list), register_count.
    Fix any errors by patching the parser script and re-running.
    """
    return json.dumps(validate_register_json(json_path))


@tool
def resolve_address(expression: str, description: str = "", variable: str = "N") -> str:
    """Resolve a dynamic register address expression that contains a variable
    (typically 'N') into a list of concrete hex addresses.

    Use this tool whenever you encounter an address cell whose value is an
    expression rather than a plain hex number — for example:
        "0x018+0x20*N",  "0x100+N*4",  "BASE+0x10*N"

    The tool will:
      1. Search `description` (the register's or group's description text)
         for a statement that pins the variable, e.g. "N = 0 to 3".
      2. If found automatically, expand the expression for each value.
      3. If NOT found, pause and prompt the user on the CLI to supply a
         value or range (e.g. "0-3" or "0,1,2,3"), then expand.

    Returns JSON:
      {
        "expression": "0x018+0x20*N",
        "variable": "N",
        "n_values": [0, 1, 2, 3],
        "n_source": "description" | "user",
        "addresses": ["0x18", "0x38", "0x58", "0x78"]
      }

    The generated parser script must call this tool for every non-numeric
    address it encounters, and then emit one separate register JSON object
    per address in the returned list, each with the same fields but its
    own concrete address value.

    expression: the raw address cell value, e.g. "0x018+0x20*N"
    description: the description text of the register or group row.
    variable: the variable name in the expression (default "N").
    """
    return _resolve_address(expression, description, variable)


# ---------------------------------------------------------------------------
# Sub-agents
# Note: `system_prompt` (not `prompt`) is the correct key in the dict.
# `tools` is a list of actual @tool-decorated function references — NOT strings.
# deepagents calls .name on each entry internally, so strings will fail.
# ---------------------------------------------------------------------------

schema_sniffer_subagent = {
    "name": "schema-sniffer",
    "description": (
        "Analyses a register-map workbook's row layout and produces a grammar "
        "spec (JSON) that describes which columns hold row-type, address, MSB, "
        "LSB, field name, reset value, access type, and description, plus where "
        "data rows start and any structural quirks (merged cells, reserved rows, "
        "multi-sheet layout). Call this first on any new workbook."
    ),
    "system_prompt": (
        "You are a row-grammar analyst for hardware register-map spreadsheets.\n\n"
        "You will receive a compact JSON dump of a workbook's sheets via the "
        "inspect_excel tool. Each sheet entry shows: column letters, merged-cell "
        "ranges, and a sample of data rows with raw cell values and fill colour.\n\n"
        "These workbooks follow a two-row-type pattern (column names and language "
        "vary — may be Japanese or English):\n"
        "  • A 'Group' row starts a new register. It carries the register's "
        "base/offset address (often as a hex string) and the register name. "
        "It usually has NO MSB/LSB values and a distinct fill colour "
        "(e.g. yellow/orange).\n"
        "  • One or more 'Reg' rows follow immediately. Each describes ONE "
        "bitfield of that same register: MSB, LSB, field/mnemonic name, reset "
        "value, access type (RW / RO / Reserved / …), and a description. All "
        "'Reg' rows under the same 'Group' share that group's address, until "
        "the next 'Group' row begins a new register.\n"
        "  • A single register can have anywhere from one to many bitfields.\n\n"
        "Your output must be a JSON grammar spec with these keys:\n"
        "  row_type_col (0-based column index of the 'Group'/'Reg' column),\n"
        "  group_value (exact cell value that means 'new register'),\n"
        "  reg_value (exact cell value that means 'bitfield row'),\n"
        "  address_col, register_name_col, msb_col, lsb_col,\n"
        "  field_name_col, reset_col, access_type_col, description_col "
        "(all 0-based column indices),\n"
        "  data_start_row (1-based row index where real data begins, after headers),\n"
        "  sheets_to_parse (list of sheet names that contain register data),\n"
        "  quirks (free-text list of anything unusual you noticed).\n\n"
        "Output ONLY the JSON grammar spec, no prose, no markdown fences."
    ),
    "tools": [inspect_excel],
}

codegen_subagent = {
    "name": "codegen-and-validate",
    "description": (
        "Given a grammar spec from schema-sniffer, writes a Python openpyxl "
        "parser script, runs it against the real workbook, validates the output "
        "JSON, and iterates (up to 5 times) until validation passes. Reports the "
        "final output path and validation summary."
    ),
    "system_prompt": (
        "You are a Python codegen agent that turns a grammar spec into a working "
        "register-map parser.\n\n"
        "You will be given a JSON grammar spec describing the column layout of a "
        "specific workbook. Use it to write a Python script with these properties:\n\n"
        "1. Accepts sys.argv[1] (xlsx path) and sys.argv[2] (output json path).\n"
        "2. Uses openpyxl with data_only=True.\n"
        "3. Only iterates over the sheets listed in grammar_spec['sheets_to_parse'].\n"
        "4. Walks rows per the spec: when it sees a row_type_col value matching "
        "   group_value, it starts a new register object (saving name + address) "
        "   and resets the current fields list. When it sees reg_value, it appends "
        "   a field dict {name, msb, lsb, reset, type, description} to the current "
        "   register's fields. When a new group_value row is seen, the previous "
        "   register (if it has fields) is appended to the output list.\n"
        "5. DYNAMIC ADDRESSES — when an address cell contains an expression "
        "   with a variable (e.g. '0x018+0x20*N', '0x100+N*4') instead of a "
        "   plain hex number, the generated parser script must call the "
        "   resolve_address tool with (expression, description_text). The tool "
        "   returns a JSON object with 'addresses' AND 'name_suffixes' lists of "
        "   equal length (e.g. addresses=['0x18','0x38'], name_suffixes=['_00','_01']). "
        "   The parser must emit one separate register object per entry, appending "
        "   the suffix to BOTH the register name AND every field name — e.g. for "
        "   register 'EDPQ' with field 'ADDRESS', instance 1 becomes register "
        "   'EDPQ_01' with field 'ADDRESS_01'. Use the suffix exactly as returned "
        "   (already zero-padded to at least 2 digits). If N cannot be determined "
        "   from the description the tool will prompt the user on the CLI — this "
        "   is expected behaviour, not an error.\n"
        "6. Handles merged cells: call ws.merged_cells.ranges once and build a "
        "   lookup {(row, col): merged_cell_value} so merged values propagate "
        "   to all cells in the merge range.\n"
        "6. Converts address values to '0x{hex}' strings, msb/lsb to int.\n"
        "7. Skips rows where row_type_col is None or doesn't match either value; "
        "   prints a one-line warning to stderr for skipped rows.\n"
        "8. Writes output JSON: {\"registers\": [ ... ]} to sys.argv[2].\n\n"
        "Workflow:\n"
        "  a. Call write_parser_script with your code.\n"
        "  b. Call run_parser with the script path, xlsx path, and a temp output path.\n"
        "  c. Call validate_json on the output.\n"
        "  d. If validate_json returns errors, inspect the relevant raw rows again "
        "     with inspect_excel if needed, fix the script, and repeat from (a).\n"
        "  e. Stop when validate_json returns ok=true, or after 5 attempts "
        "     (report remaining errors in that case).\n"
        "  f. Return the final output path and register count."
    ),
    "tools": [inspect_excel, write_parser_script, run_parser, validate_json, resolve_address],
}


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
agent = create_deep_agent(
    model=model,
    tools=[inspect_excel, write_parser_script, run_parser, validate_json, resolve_address],
    system_prompt=(
        "You convert hardware register-map Excel workbooks (xlsx/xlsm) into "
        "structured JSON. The column layout and language change between files, "
        "so you must re-derive the parsing logic for each input file.\n\n"
        "Workflow:\n"
        "1. Call the schema-sniffer sub-agent on the input file with the "
        "   target sheet list (if provided) to get a grammar spec for this "
        "   specific workbook.\n"
        "2. Call the codegen-and-validate sub-agent with that grammar spec "
        "   plus the xlsx path and desired output path; it will write, run, "
        "   and validate the parser, iterating until output is correct.\n"
        "3. Report the final output JSON path, register count, and any "
        "   remaining warnings to the user."
    ),
    subagents=[schema_sniffer_subagent, codegen_subagent],
)


def convert(xlsx_path: str, out_json_path: str, sheets: list[str] | None = None):
    """Convert a single register-map workbook to JSON.

    xlsx_path: path to the .xlsx or .xlsm file.
    out_json_path: where to write the final JSON.
    sheets: optional list of sheet names to process (e.g. ["Conf-1", "Conf-2"]).
            If omitted, all sheets are considered.
    """
    sheet_note = (
        f" Only process these sheets: {', '.join(sheets)}. Ignore all other tabs."
        if sheets else ""
    )
    user_msg = (
        f"Convert the register-map workbook at '{xlsx_path}' to JSON "
        f"and write the output to '{out_json_path}'.{sheet_note}"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
    return result


def convert_folder(
    folder_path: str,
    out_dir: str | None = None,
    sheets: list[str] | None = None,
):
    """Convert all .xlsx and .xlsm files in a folder to JSON.

    folder_path: path to the folder containing the workbooks.
    out_dir: directory to write output JSON files to.
             Defaults to a subfolder named 'json_output' inside folder_path.
    sheets: optional sheet filter applied to every file in the folder.

    Each workbook produces one JSON file named <workbook_stem>.json in out_dir.
    Returns a dict mapping each input path to its result or error string.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"'{folder_path}' is not a directory")

    files = sorted(folder.glob("*.xlsx")) + sorted(folder.glob("*.xlsm"))
    if not files:
        raise ValueError(f"No .xlsx or .xlsm files found in '{folder_path}'")

    output_dir = Path(out_dir) if out_dir else folder / "json_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for f in files:
        out_json = str(output_dir / f"{f.stem}.json")
        print(f"\n{'='*60}\nProcessing: {f.name} → {out_json}\n{'='*60}")
        try:
            convert(str(f), out_json, sheets=sheets)
            results[str(f)] = out_json
        except Exception as e:
            results[str(f)] = f"ERROR: {e}"
            print(f"  Failed: {e}")

    print(f"\nDone. {len([v for v in results.values() if not v.startswith('ERROR')])} "
          f"of {len(files)} files converted successfully.")
    print(f"Output directory: {output_dir}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:  python agent/main.py <xlsx_path> <out_json_path> [sheet1,sheet2,...]")
        print("  Folder:       python agent/main.py <folder_path> [out_dir] [sheet1,sheet2,...]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    sheets_arg = None

    if input_path.is_dir():
        out_dir_arg = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].endswith((".xlsx", ".xlsm")) else None
        sheets_raw = sys.argv[3] if len(sys.argv) > 3 else (sys.argv[2] if len(sys.argv) > 2 and "," in sys.argv[2] else None)
        sheets_arg = sheets_raw.split(",") if sheets_raw else None
        convert_folder(str(input_path), out_dir=out_dir_arg, sheets=sheets_arg)
    else:
        json_out = sys.argv[2] if len(sys.argv) > 2 else str(WORK_DIR / f"{input_path.stem}.json")
        sheets_arg = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        res = convert(str(input_path), json_out, sheets=sheets_arg)
        print(res["messages"][-1].content)
