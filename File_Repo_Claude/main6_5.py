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

_LAST_INSPECTION_PATH = WORK_DIR / "_last_inspection.json"


@tool
def inspect_excel(xlsx_path: str, sheets: str = "") -> str:
    """Inspect an xlsx/xlsm register-map workbook. Returns a compact JSON
    summary of sheet structure: header rows, a sample of data rows with
    raw cell values and fill colours, and merged-cell ranges. This lets
    you figure out the row grammar (which column holds the row type,
    where the address/MSB/LSB/name/description columns live, etc.).

    Also saves the result to disk so review_column_mapping can read it
    without needing it passed as an argument.

    xlsx_path: absolute or relative path to the workbook.
    sheets: optional comma-separated sheet names to restrict to
            (e.g. "Conf-1,Conf-2"). Leave empty to inspect all sheets.
    """
    sheet_list = [s.strip() for s in sheets.split(",") if s.strip()] or None
    result = inspect_workbook(xlsx_path, sheets=sheet_list)
    # Save to disk — review_column_mapping reads from here
    _LAST_INSPECTION_PATH.write_text(result, encoding="utf-8")
    return result


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
def review_column_mapping(grammar_spec_json: str) -> str:
    """HUMAN CHECKPOINT 1 — show the user a table of detected column mappings
    and let them approve or correct before codegen runs.

    Reads the saved inspection output (written by inspect_excel) from disk
    and combines it with the grammar spec to render a table of every sampled
    column, its header label, and its detected usage. The user can type 'y'
    to approve or provide JSON corrections like {"msb_col": 3, "lsb_col": 4}.

    Returns a JSON string of the (possibly corrected) grammar spec. Always
    use this return value — not the original grammar_spec_json — when calling
    the codegen sub-agent.

    grammar_spec_json: JSON string returned by the schema-sniffer sub-agent.
    """
    try:
        grammar_spec = json.loads(grammar_spec_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"grammar_spec parse error: {e}"})

    # Load inspection from disk (written by inspect_excel)
    inspection: dict = {}
    if _LAST_INSPECTION_PATH.exists():
        try:
            inspection = json.loads(_LAST_INSPECTION_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    # Map field names → col indices from grammar spec
    col_fields = {
        "row_type":    grammar_spec.get("row_type_col"),
        "address":     grammar_spec.get("address_col"),
        "reg_name":    grammar_spec.get("register_name_col"),
        "msb":         grammar_spec.get("msb_col"),
        "lsb":         grammar_spec.get("lsb_col"),
        "field_name":  grammar_spec.get("field_name_col"),
        "reset":       grammar_spec.get("reset_col"),
        "access_type": grammar_spec.get("access_type_col"),
        "description": grammar_spec.get("description_col"),
    }
    usage_map: dict[int, list[str]] = {}
    for field, idx in col_fields.items():
        if idx is not None:
            usage_map.setdefault(int(idx), []).append(field)

    # Pull header values and column letters from first sheet in inspection
    header_vals: dict[int, str] = {}
    col_letters: list[str] = []
    first_sheet = next(iter(inspection.get("sheets", {}).values()), {})
    col_letters = first_sheet.get("column_letters", [])
    for row in first_sheet.get("rows", [])[:3]:
        for ci, v in enumerate(row.get("values") or []):
            if ci not in header_vals and v is not None:
                header_vals[ci] = str(v).strip()

    max_col = max(
        max(header_vals.keys(), default=0),
        max(usage_map.keys(),   default=0),
    ) + 1

    W = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    CYAN = "\033[36m"; YEL = "\033[33m"; GRN = "\033[32m"; MAG = "\033[35m"

    print(f"\n{B}{CYAN}{'─'*72}{W}")
    print(f"{B}{CYAN}  COLUMN MAPPING REVIEW{W}  —  approve or correct before codegen runs")
    print(f"{CYAN}{'─'*72}{W}\n")
    print(f"  {B}{'COL':<5} {'LETTER':<8} {'HEADER VALUE':<28} {'DETECTED USAGE'}{W}")
    print(f"  {'─'*64}")

    for ci in range(max_col):
        letter  = col_letters[ci] if ci < len(col_letters) else f"col{ci}"
        header  = header_vals.get(ci, "")[:26]
        usages  = usage_map.get(ci, [])
        usage_s = ", ".join(usages) if usages else f"{DIM}[unused]{W}"
        colour  = GRN if usages else DIM
        print(f"  {colour}{ci:<5} {letter:<8} {header:<28} {usage_s}{W}")

    print(f"\n  {B}Grammar spec extras:{W}")
    print(f"  {DIM}data_start_row : {grammar_spec.get('data_start_row')}{W}")
    print(f"  {DIM}group_value    : {grammar_spec.get('group_value')!r}{W}")
    print(f"  {DIM}reg_value      : {grammar_spec.get('reg_value')!r}{W}")
    print(f"  {DIM}sheets         : {grammar_spec.get('sheets_to_parse')}{W}")
    if grammar_spec.get("quirks"):
        print(f"  {YEL}quirks         : {grammar_spec.get('quirks')}{W}")

    print(f"\n{CYAN}{'─'*72}{W}")
    print(f"  {B}Approve?{W}  Type  {GRN}y{W}  to proceed,")
    print(f"            or enter corrections as JSON, e.g.")
    print(f"            {MAG}{{\"msb_col\": 3, \"lsb_col\": 4, \"group_value\": \"Group\"}}{W}")
    print(f"{CYAN}{'─'*72}{W}")

    while True:
        raw = input("  > ").strip()
        if raw.lower() in ("y", "yes", ""):
            print(f"  {GRN}✔  Column mapping approved.{W}\n")
            return json.dumps(grammar_spec)
        try:
            overrides = json.loads(raw)
            if not isinstance(overrides, dict):
                raise ValueError("expected a JSON object")
            grammar_spec.update(overrides)
            print(f"  {GRN}✔  Applied {len(overrides)} override(s). Moving to codegen.{W}\n")
            return json.dumps(grammar_spec)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  {YEL}⚠  Could not parse — try again. ({e}){W}")

    # Map field names → col indices from grammar spec
    col_fields = {
        "row_type":    grammar_spec.get("row_type_col"),
        "address":     grammar_spec.get("address_col"),
        "reg_name":    grammar_spec.get("register_name_col"),
        "msb":         grammar_spec.get("msb_col"),
        "lsb":         grammar_spec.get("lsb_col"),
        "field_name":  grammar_spec.get("field_name_col"),
        "reset":       grammar_spec.get("reset_col"),
        "access_type": grammar_spec.get("access_type_col"),
        "description": grammar_spec.get("description_col"),
    }
    # Invert: col_idx → list of usages
    usage_map: dict[int, list[str]] = {}
    for field, idx in col_fields.items():
        if idx is not None:
            usage_map.setdefault(int(idx), []).append(field)

    # Pull header values from inspection (first non-empty value per column)
    header_vals: dict[int, str] = {}
    first_sheet = next(iter(inspection.get("sheets", {}).values()), {})
    col_letters = first_sheet.get("column_letters", [])
    for row in first_sheet.get("rows", [])[:3]:
        for ci, v in enumerate(row.get("values") or []):
            if ci not in header_vals and v is not None:
                header_vals[ci] = str(v).strip()

    max_col = max(
        max(header_vals.keys(), default=0),
        max(usage_map.keys(),   default=0),
    ) + 1

    # ── Print table ──────────────────────────────────────────────────────────
    W = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    CYAN = "\033[36m"; YEL = "\033[33m"; GRN = "\033[32m"; MAG = "\033[35m"

    print(f"\n{B}{CYAN}{'─'*72}{W}")
    print(f"{B}{CYAN}  COLUMN MAPPING REVIEW{W}  —  approve or correct before codegen runs")
    print(f"{CYAN}{'─'*72}{W}\n")
    print(f"  {B}{'COL':<5} {'LETTER':<8} {'HEADER VALUE':<28} {'DETECTED USAGE'}{W}")
    print(f"  {'─'*64}")

    for ci in range(max_col):
        letter  = col_letters[ci] if ci < len(col_letters) else f"col{ci}"
        header  = header_vals.get(ci, "")[:26]
        usages  = usage_map.get(ci, [])
        usage_s = ", ".join(usages) if usages else f"{DIM}[unused]{W}"
        colour  = GRN if usages else DIM
        print(f"  {colour}{ci:<5} {letter:<8} {header:<28} {usage_s}{W}")

    print(f"\n  {B}Grammar spec extras:{W}")
    print(f"  {DIM}data_start_row : {grammar_spec.get('data_start_row')}{W}")
    print(f"  {DIM}group_value    : {grammar_spec.get('group_value')!r}{W}")
    print(f"  {DIM}reg_value      : {grammar_spec.get('reg_value')!r}{W}")
    print(f"  {DIM}sheets         : {grammar_spec.get('sheets_to_parse')}{W}")
    if grammar_spec.get("quirks"):
        print(f"  {YEL}quirks         : {grammar_spec.get('quirks')}{W}")

    print(f"\n{CYAN}{'─'*72}{W}")
    print(f"  {B}Approve?{W}  Type  {GRN}y{W}  to proceed,")
    print(f"            or enter corrections as JSON, e.g.")
    print(f"            {MAG}{{\"msb_col\": 3, \"lsb_col\": 4, \"group_value\": \"Group\"}}{W}")
    print(f"{CYAN}{'─'*72}{W}")

    while True:
        raw = input("  > ").strip()
        if raw.lower() in ("y", "yes", ""):
            print(f"  {GRN}✔  Column mapping approved.{W}\n")
            return json.dumps(grammar_spec)
        try:
            overrides = json.loads(raw)
            if not isinstance(overrides, dict):
                raise ValueError("expected a JSON object")
            grammar_spec.update(overrides)
            print(f"  {GRN}✔  Applied {len(overrides)} override(s). Moving to codegen.{W}\n")
            return json.dumps(grammar_spec)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  {YEL}⚠  Could not parse — try again. ({e}){W}")


@tool
def review_json_output(json_path: str) -> str:
    """HUMAN CHECKPOINT 2 — show the user a preview of the generated JSON
    and let them approve or describe what is wrong.

    Prints the first 3 registers (with all their fields) in a readable
    format, plus a summary count. The user types 'y' to approve, or
    describes what is incorrect (free text). That feedback is returned
    to the agent so the codegen sub-agent can do a targeted fix.

    Returns a JSON object:
    {
      "approved": bool,
      "feedback": str   # empty string if approved, user description if not
    }

    json_path: path to the output JSON file produced by run_parser.
    """
    W = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    CYAN = "\033[36m"; YEL = "\033[33m"; GRN = "\033[32m"; RED = "\033[31m"
    BLU = "\033[34m"

    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception as e:
        return json.dumps({"approved": False, "feedback": f"Could not read output JSON: {e}"})

    registers = data if isinstance(data, list) else data.get("registers", [])
    total = len(registers)
    preview = registers[:3]

    print(f"\n{B}{CYAN}{'─'*72}{W}")
    print(f"{B}{CYAN}  OUTPUT JSON REVIEW{W}  —  {total} register(s) found")
    print(f"{CYAN}{'─'*72}{W}\n")

    for reg in preview:
        name    = reg.get("name", "?")
        address = reg.get("address", "?")
        fields  = reg.get("fields", [])
        print(f"  {B}{BLU}{name}{W}  {DIM}@ {address}{W}  —  {len(fields)} field(s)")
        for f in fields:
            msb  = f.get("msb", "?")
            lsb  = f.get("lsb", "?")
            fname = f.get("name", "?")
            ftype = f.get("type", "?")
            desc  = (f.get("description") or "")[:55]
            print(f"    {DIM}[{msb}:{lsb}]{W}  {fname:<22} {DIM}{ftype:<8}{W}  {DIM}{desc}{W}")
        print()

    if total > 3:
        print(f"  {DIM}… and {total - 3} more register(s) not shown.{W}\n")

    print(f"{CYAN}{'─'*72}{W}")
    print(f"  {B}Approve?{W}  Type  {GRN}y{W}  to finalise,")
    print(f"            or describe what is wrong (free text):")
    print(f"            e.g. {YEL}\"addresses are off by one\"{W}")
    print(f"                 {YEL}\"field names are missing the suffix\"{W}")
    print(f"{CYAN}{'─'*72}{W}")

    raw = input("  > ").strip()
    if raw.lower() in ("y", "yes", ""):
        print(f"  {GRN}✔  Output approved. Done.{W}\n")
        return json.dumps({"approved": True, "feedback": ""})

    print(f"  {YEL}⚠  Feedback noted — sending back to codegen.{W}\n")
    return json.dumps({"approved": False, "feedback": raw})


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
        "  e. If the user provided feedback from review_json_output (passed in your "
        "     instructions), treat it as the highest-priority fix directive and address "
        "     it specifically in the next script revision.\n"
        "  f. Stop when validate_json returns ok=true, or after 5 attempts "
        "     (report remaining errors in that case).\n"
        "  g. Return the final output path and register count."
    ),
    "tools": [inspect_excel, write_parser_script, run_parser, validate_json, resolve_address],
}


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
agent = create_deep_agent(
    model=model,
    tools=[inspect_excel, write_parser_script, run_parser, validate_json,
           resolve_address, review_column_mapping, review_json_output],
    system_prompt=(
        "You convert hardware register-map Excel workbooks (xlsx/xlsm) into "
        "structured JSON. Follow this exact workflow — do not skip any step.\n\n"

        "STEP 1 — Inspect:\n"
        "  Call inspect_excel on the input file (pass sheet filter if provided).\n\n"

        "STEP 2 — Sniff schema:\n"
        "  Call the schema-sniffer sub-agent with the inspection JSON to get a "
        "  grammar spec (column indices, row-type values, data start row, quirks).\n\n"

        "STEP 3 — Human column review (MANDATORY, never skip):\n"
        "  Call review_column_mapping with ONLY the grammar spec JSON string.\n"
        "  It reads the inspection data from disk automatically.\n"
        "  Use the JSON it returns (not the original grammar spec) for all subsequent steps.\n\n"

        "STEP 4 — Generate and validate:\n"
        "  Call the codegen-and-validate sub-agent with the approved grammar spec, "
        "  xlsx path, and output path. It writes, runs, and validates the parser.\n\n"

        "STEP 5 — Human output review (MANDATORY, never skip):\n"
        "  Call review_json_output with the output JSON path.\n"
        "  a. If approved=true: report success, output path, and register count. DONE.\n"
        "  b. If approved=false: pass the feedback string back to the codegen-and-"
        "     validate sub-agent as an additional instruction to fix the specific "
        "     issue the user described. Then call review_json_output again.\n"
        "     Repeat up to 3 times. If still not approved after 3 rounds, report "
        "     the remaining issues and the best output path produced so far."
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
