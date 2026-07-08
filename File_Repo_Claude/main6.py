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
import time
from pathlib import Path
from typing import Any
from uuid import UUID
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from deepagents import create_deep_agent

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tools.inspect_excel import inspect_workbook
from tools.run_parser import run_generated_parser
from tools.validate_json import validate_register_json
from tools.resolve_address import resolve_address as _resolve_address
from tools.pattern_cache import (
    build_fingerprint,
    get_buckets,
    push_bucket,
    mark_bucket_success,
    mark_bucket_failure,
    print_pattern_table,
)


# ---------------------------------------------------------------------------
# Live terminal callback — streams tokens, tool calls, and sub-agent
# transitions to stdout as they happen so you can follow the agent live.
# ---------------------------------------------------------------------------

# ANSI colour helpers (no external deps)
_R  = "\033[0m"       # reset
_B  = "\033[1m"       # bold
_DIM = "\033[2m"      # dim
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_MAGENTA = "\033[35m"
_BLUE   = "\033[34m"


def _bar(char: str = "─", width: int = 70) -> str:
    return char * width


def _truncate(text: str, limit: int = 400) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"  {_DIM}[…+{len(text)-limit} chars]{_R}"


class LiveConsoleCallback(BaseCallbackHandler):
    """Prints the agent's reasoning tokens, tool invocations, and results
    to the terminal in real time so you can follow exactly what is happening
    during a conversion run."""

    def __init__(self):
        super().__init__()
        self._in_token_stream = False   # True while streaming LLM tokens
        self._tool_start_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # LLM events
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # Extract sub-agent / chain name if available
        name = serialized.get("name") or serialized.get("id", ["?"])[-1]
        print(f"\n{_CYAN}{_B}{_bar()}{_R}")
        print(f"{_CYAN}{_B}  🤖  LLM CALL  [{name}]{_R}")
        print(f"{_CYAN}{_bar()}{_R}")
        # Print last user message so context is clear
        for msg_group in messages:
            for msg in (msg_group if isinstance(msg_group, list) else [msg_group]):
                role = getattr(msg, "type", "?")
                if role in ("human", "user"):
                    content = getattr(msg, "content", "")
                    if isinstance(content, list):   # multi-part message
                        content = " ".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    print(f"{_DIM}  USER ▶ {_truncate(content, 300)}{_R}")
        print(f"{_YELLOW}  THINKING ▶ {_R}", end="", flush=True)
        self._in_token_stream = True

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Stream each token to stdout as it arrives."""
        print(token, end="", flush=True)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self._in_token_stream:
            print()   # newline after streamed tokens
            self._in_token_stream = False
        # Print token usage if available
        usage = getattr(response, "llm_output", {}) or {}
        token_info = usage.get("token_usage") or usage.get("usage")
        if token_info:
            prompt_t = token_info.get("prompt_tokens", "?")
            completion_t = token_info.get("completion_tokens", "?")
            print(f"{_DIM}  [tokens: prompt={prompt_t}  completion={completion_t}]{_R}")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        if self._in_token_stream:
            print()
            self._in_token_stream = False
        print(f"\n{_RED}{_B}  ✖  LLM ERROR: {error}{_R}")

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if self._in_token_stream:
            print()
            self._in_token_stream = False
        name = serialized.get("name", "?")
        self._tool_start_times[str(run_id)] = time.time()
        print(f"\n{_GREEN}{_B}  ⚙  TOOL CALL → {name}{_R}")
        print(f"{_GREEN}  INPUT  ▶ {_truncate(input_str, 300)}{_R}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed = time.time() - self._tool_start_times.pop(str(run_id), time.time())
        out_str = output if isinstance(output, str) else str(output)
        print(f"{_GREEN}  OUTPUT ▶ {_truncate(out_str, 400)}{_R}")
        print(f"{_DIM}  [{elapsed:.1f}s]{_R}")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_start_times.pop(str(run_id), None)
        print(f"{_RED}{_B}  ✖  TOOL ERROR: {error}{_R}")

    # ------------------------------------------------------------------
    # Chain / sub-agent transitions
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name") or serialized.get("id", ["?"])[-1]
        # Only print for named sub-agents, not every internal chain step
        if name and name not in ("RunnableSequence", "RunnableLambda",
                                  "RunnableMap", "ChannelWrite", "ChannelRead",
                                  "CompiledStateGraph", "LangGraph"):
            print(f"\n{_MAGENTA}{_B}  ▶▶  CHAIN START  [{name}]{_R}")

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        pass   # too noisy if printed for every internal chain


class LoopGuardCallback(BaseCallbackHandler):
    """Detects when the agent is calling the same tool with the same input
    repeatedly (a sure sign it's stuck) and raises StopIteration to break
    the loop early rather than letting it spin until recursion_limit.

    Also enforces a hard cap on total tool calls per run.
    """

    def __init__(self, max_total_tool_calls: int = 40, max_repeat: int = 3):
        super().__init__()
        self.max_total_tool_calls = max_total_tool_calls
        self.max_repeat = max_repeat
        self._call_history: list[tuple[str, str]] = []   # (tool_name, input_str)
        self._total = 0

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "?")
        self._total += 1
        key = (name, input_str)
        self._call_history.append(key)

        # Hard cap on total tool calls
        if self._total > self.max_total_tool_calls:
            msg = (
                f"\n{_RED}{_B}  ✖  LOOP GUARD: hard cap of {self.max_total_tool_calls} "
                f"tool calls reached. Stopping.{_R}"
            )
            print(msg)
            raise StopIteration(
                f"Agent exceeded {self.max_total_tool_calls} total tool calls. "
                "Increase max_total_tool_calls in convert() if this workbook "
                "genuinely needs more steps."
            )

        # Repeated identical call detection
        recent = self._call_history[-self.max_repeat * 2:]
        repeat_count = recent.count(key)
        if repeat_count >= self.max_repeat:
            msg = (
                f"\n{_RED}{_B}  ✖  LOOP GUARD: '{name}' called with identical "
                f"input {repeat_count} times. Agent is stuck. Stopping.{_R}"
            )
            print(msg)
            raise StopIteration(
                f"Agent called '{name}' with identical inputs {repeat_count} times — "
                "circular loop detected. Use the notes= parameter to hint at the "
                "structural quirk causing the confusion."
            )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="qwen3",
    temperature=0,
    streaming=True,          # enables on_llm_new_token live token stream
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


@tool
def lookup_cache(inspection_json: str) -> str:
    """Look up the Redis pattern cache for a workbook that has already been
    seen with the same structural layout.

    Builds a fingerprint from the raw inspect_excel output (sheet names,
    header cell values, column counts, row-type values, fill colours) and
    checks if any validated parser scripts are stored under that fingerprint.

    Returns a JSON object:
    {
      "cache_hit": bool,
      "fingerprint_hash": str,          # 16-char hex, use this in store_to_cache
      "fingerprint_summary": { ... },   # human-readable structural signature
      "buckets": [                       # empty list on miss
        {
          "id": str,
          "script": str,                 # full parser Python source — try this first
          "grammar_spec": { ... },
          "success_count": int,
          "fail_count": int,
          "validated": bool
        }, ...
      ]
    }

    Buckets are sorted best-first (highest success_count). Try each bucket
    script with run_parser + validate_json before falling back to the full
    schema-sniff + codegen pipeline. If a bucket passes, call
    store_to_cache with status="success". If it fails, call store_to_cache
    with status="failure" for that bucket_id so its fail_count is updated.

    inspection_json: the raw string returned by inspect_excel.
    """
    fp = build_fingerprint(inspection_json)
    fp_hash = fp["hash"]
    buckets = get_buckets(fp_hash)

    # Strip the hash from the summary shown to the agent (keep it readable)
    fp_summary = {k: v for k, v in fp.items() if k != "hash"}

    result = {
        "cache_hit": len(buckets) > 0,
        "fingerprint_hash": fp_hash,
        "fingerprint_summary": fp_summary,
        "buckets": buckets,
    }

    status = f"HIT — {len(buckets)} bucket(s)" if buckets else "MISS"
    print(f"\n\033[33m  [cache] {status}  fingerprint={fp_hash}\033[0m")
    return json.dumps(result, ensure_ascii=False)


@tool
def store_to_cache(
    fingerprint_hash: str,
    script: str,
    grammar_spec_json: str,
    validated: bool,
    bucket_id: str = "",
    status: str = "",
) -> str:
    """Update the Redis pattern cache after running a parser.

    Two use-cases — pick the right one:

    1. NEW BUCKET (after schema-sniff + codegen produced a new script):
       Pass script + grammar_spec_json + validated=True/False.
       Leave bucket_id and status empty.
       The cache stores the script as a new bucket under fingerprint_hash.

    2. EXISTING BUCKET RESULT (after trying a cached script):
       Pass bucket_id (from lookup_cache) and status="success" or "failure".
       Leave script and grammar_spec_json empty ("").
       The cache updates success_count / fail_count for that bucket.

    Returns a JSON confirmation: {action, bucket_id, fingerprint_hash}

    fingerprint_hash:  from lookup_cache's "fingerprint_hash" field.
    script:            full Python source of the parser (new bucket only).
    grammar_spec_json: JSON string of the grammar spec (new bucket only).
    validated:         whether validate_json returned ok=True (new bucket).
    bucket_id:         id of existing bucket to update (existing bucket only).
    status:            "success" or "failure" (existing bucket only).
    """
    if bucket_id and status:
        # Update existing bucket counters
        if status == "success":
            mark_bucket_success(fingerprint_hash, bucket_id)
        else:
            mark_bucket_failure(fingerprint_hash, bucket_id)
        print(f"\n\033[33m  [cache] marked bucket {bucket_id} as {status}\033[0m")
        return json.dumps({"action": "updated", "bucket_id": bucket_id,
                           "fingerprint_hash": fingerprint_hash})

    # Store new bucket
    try:
        grammar_spec = json.loads(grammar_spec_json) if grammar_spec_json else {}
    except json.JSONDecodeError:
        grammar_spec = {"raw": grammar_spec_json}

    new_id = push_bucket(fingerprint_hash, script, grammar_spec, validated)
    return json.dumps({"action": "stored", "bucket_id": new_id,
                       "fingerprint_hash": fingerprint_hash, "validated": validated})


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
        "  f. Return the final output path and register count.\n"
        "  g. After a successful validation, ALWAYS call store_to_cache to persist\n"
        "     the working script so future files with the same pattern skip codegen.\n"
        "     Pass fingerprint_hash (from lookup_cache), the script source, the\n"
        "     grammar spec as a JSON string, and validated=True.\n"
        "     If all 5 attempts fail, still store with validated=False so the\n"
        "     pattern is recorded even without a working script."
    ),
    "tools": [inspect_excel, write_parser_script, run_parser, validate_json,
              resolve_address, lookup_cache, store_to_cache],
}


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
agent = create_deep_agent(
    model=model,
    tools=[inspect_excel, write_parser_script, run_parser, validate_json,
           resolve_address, lookup_cache, store_to_cache],
    system_prompt=(
        "You convert hardware register-map Excel workbooks (xlsx/xlsm) into "
        "structured JSON. The column layout and language change between files, "
        "so you re-derive parsing logic per file — BUT you cache every working "
        "parser and reuse it when the same structural pattern is seen again.\n\n"
        "WORKFLOW (follow in order):\n\n"
        "STEP 1 — Inspect and check cache:\n"
        "  a. Call inspect_excel on the input file.\n"
        "  b. Call lookup_cache with the raw inspection JSON.\n"
        "  c. If cache_hit=true, go to STEP 2. Otherwise go to STEP 3.\n\n"
        "STEP 2 — Try cached scripts (cache hit path):\n"
        "  For each bucket returned by lookup_cache (best-first order):\n"
        "    i.  Write the bucket's script to disk with write_parser_script.\n"
        "   ii.  Run it with run_parser.\n"
        "  iii.  Validate with validate_json.\n"
        "   iv.  If ok=true: call store_to_cache with bucket_id + status='success'.\n"
        "        Report success and the output path. STOP.\n"
        "    v.  If ok=false: call store_to_cache with bucket_id + status='failure'.\n"
        "        Try the next bucket.\n"
        "  If all buckets fail, fall through to STEP 3.\n\n"
        "STEP 3 — Full pipeline (cache miss or all buckets failed):\n"
        "  a. Call the schema-sniffer sub-agent to produce a grammar spec.\n"
        "  b. Call the codegen-and-validate sub-agent with the grammar spec;\n"
        "     it writes, runs, and validates a parser (up to 5 attempts).\n"
        "  c. Call store_to_cache with the new script, grammar spec JSON,\n"
        "     fingerprint_hash (from lookup_cache), and validated=True/False.\n\n"
        "STEP 4 — Report:\n"
        "  Print the output path, register count, cache action taken\n"
        "  (hit/miss/stored), and any remaining warnings."
    ),
    subagents=[schema_sniffer_subagent, codegen_subagent],
)


def convert(xlsx_path: str, out_json_path: str, sheets: list[str] | None = None, notes: str | None = None):
    """Convert a single register-map workbook to JSON.

    xlsx_path: path to the .xlsx or .xlsm file.
    out_json_path: where to write the final JSON.
    sheets: optional list of sheet names to process (e.g. ["Conf-1", "Conf-2"]).
            If omitted, all sheets are considered.
    notes: optional free-text hints about this file's quirks, passed verbatim
           to both sub-agents (e.g. "column order is reversed on sheet Conf-2",
           "N ranges 0-7 for all dynamic registers", "address column may be
           empty for Reg rows — inherit from the Group row above").
    """
    sheet_note = (
        f" Only process these sheets: {', '.join(sheets)}. Ignore all other tabs."
        if sheets else ""
    )
    notes_note = (
        f"\n\nAdditional notes about this specific file (treat these as "
        f"higher-priority hints that override general assumptions):\n{notes}"
        if notes else ""
    )
    user_msg = (
        f"Convert the register-map workbook at '{xlsx_path}' to JSON "
        f"and write the output to '{out_json_path}'.{sheet_note}{notes_note}"
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config={
            "callbacks": [LiveConsoleCallback(), LoopGuardCallback()],
            "recursion_limit": 100,   # hard LangGraph graph-step cap
        },
    )
    return result


def convert_folder(
    folder_path: str,
    out_dir: str | None = None,
    sheets: list[str] | None = None,
    notes: dict[str, str] | str | None = None,
):
    """Convert all .xlsx and .xlsm files in a folder to JSON.

    folder_path: path to the folder containing the workbooks.
    out_dir: directory to write output JSON files to.
             Defaults to a subfolder named 'json_output' inside folder_path.
    sheets: optional sheet filter applied to every file in the folder.
    notes: optional hints. Two forms accepted:
           - str: applied to every file in the folder.
           - dict: maps filename stem (without extension) to a hint string,
                   so each file gets only its own note, e.g.:
                   {"registers_v2": "N ranges 0-7", "conf_special": "address col shifted right by 1"}
                   Files not mentioned in the dict get no note.

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
        file_notes = (
            notes.get(f.stem)      # per-file dict
            if isinstance(notes, dict)
            else notes             # global string or None
        )
        print(f"\n{'='*60}\nProcessing: {f.name} → {out_json}")
        if file_notes:
            print(f"  Notes: {file_notes}")
        print("=" * 60)
        try:
            convert(str(f), out_json, sheets=sheets, notes=file_notes)
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
        print("  Single file:  python agent/main.py <xlsx_path> <out_json_path> [sheet1,sheet2,...] [\"notes\"]")
        print("  Folder:       python agent/main.py <folder_path> [out_dir] [sheet1,sheet2,...] [\"notes\"]")
        print("  Show cache:   python agent/main.py --patterns")
        sys.exit(1)

    if sys.argv[1] == "--patterns":
        print_pattern_table()
        sys.exit(0)

    input_path = Path(sys.argv[1])

    if input_path.is_dir():
        out_dir_arg  = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].endswith((".xlsx", ".xlsm")) else None
        sheets_raw   = sys.argv[3] if len(sys.argv) > 3 else None
        notes_arg    = sys.argv[4] if len(sys.argv) > 4 else None
        sheets_arg   = sheets_raw.split(",") if sheets_raw else None
        convert_folder(str(input_path), out_dir=out_dir_arg, sheets=sheets_arg, notes=notes_arg)
    else:
        json_out   = sys.argv[2] if len(sys.argv) > 2 else str(WORK_DIR / f"{input_path.stem}.json")
        sheets_arg = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        notes_arg  = sys.argv[4] if len(sys.argv) > 4 else None
        res = convert(str(input_path), json_out, sheets=sheets_arg, notes=notes_arg)
        print(res["messages"][-1].content)
