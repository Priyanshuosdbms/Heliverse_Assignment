# Low-Level Design (LLD)
# xlsx2json Register-Map Agent

---

## 1. Repository Layout

```
xlsx2json_agent/
├── agent/
│   └── main.py              # deepagents orchestrator, all @tool functions,
│                            # sub-agent dicts, convert() / convert_folder()
├── tools/
│   ├── inspect_excel.py     # pure-Python workbook sampler
│   ├── run_parser.py        # subprocess runner for generated script
│   ├── validate_json.py     # structural validator for output JSON
│   └── resolve_address.py   # dynamic address expression resolver
├── output/                  # generated_parser.py and result JSON written here
├── requirements.txt
├── HLD.md
└── LLD.md
```

---

## 2. Module: `tools/inspect_excel.py`

### Purpose
Give the LLM a bounded, structure-rich view of an unknown workbook without
sending the entire file to the model.

### Key function: `inspect_workbook(xlsx_path, sample_rows=40, sheets=None)`

**Inputs**
| Parameter | Type | Description |
|---|---|---|
| xlsx_path | str | Path to the workbook |
| sample_rows | int | Max rows to include per sheet (default 40) |
| sheets | list[str] \| None | Whitelist of sheet names; None = all sheets |

**Processing steps**
1. `load_workbook(xlsx_path, data_only=True)` — formulas already evaluated.
2. For each target sheet, cap columns at `min(ws.max_column, 20)`.
3. For each row up to `sample_rows`:
   - Collect raw cell values.
   - Extract `fgColor.rgb` from the first non-None fill found in that row.
4. Collect up to 50 merged-cell range strings.
5. Return a JSON string:

```json
{
  "file": "registers.xlsm",
  "sheets": {
    "Conf-1": {
      "max_row": 412,
      "max_col": 18,
      "columns_sampled": 18,
      "column_letters": ["A","B",...,"R"],
      "merged_ranges_sample": ["A3:B4", ...],
      "rows": [
        {"row": 1, "values": [...], "fill": "FFFFFF00"},
        ...
      ]
    }
  }
}
```

**Why fill colour matters:** Group rows (register headers) are typically
shaded a distinct colour (yellow/orange in the file shown). The schema sniffer
uses this as a secondary signal alongside the row-type column value.

---

## 3. Module: `tools/run_parser.py`

### Purpose
Execute the agent-generated parser script in an isolated subprocess, capturing
all output for the validation loop.

### Key function: `run_generated_parser(script_path, xlsx_path, out_json_path, timeout=600)`

**Subprocess contract (enforced via codegen prompt):**
- The generated script accepts exactly: `sys.argv[1]` = xlsx path, `sys.argv[2]` = output JSON path.
- It writes valid JSON to `sys.argv[2]` on success.
- Non-fatal row-skip warnings go to `stderr`.

**Return dict**
```python
{
    "returncode": int,
    "stdout": str,          # last 4000 chars
    "stderr": str,          # last 4000 chars
    "output_exists": bool,
    "output_preview": str   # first 3000 chars of JSON, if file was written
}
```

`output_preview` is what the validate sub-agent uses to spot-check content
before calling `validate_json`.

---

## 4. Module: `tools/validate_json.py`

### Purpose
File-shape-agnostic structural validation of the produced register-map JSON.
Checks rules that hold universally, regardless of this workbook's column order.

### Key function: `validate_register_json(json_path)`

**Checks performed (in order)**

| Check | Error condition |
|---|---|
| JSON parseable | File missing or malformed |
| Top-level structure | Not a list and no `registers` key |
| At least one register | Empty list |
| Address present | `address`, `offset`, or `base_address` key missing |
| Fields list non-empty | `fields` key missing or `[]` |
| msb/lsb present | Either key absent on a field dict |
| msb/lsb are integers | TypeError or ValueError on `int()` cast |
| msb >= lsb | msb < lsb on any field |
| No bitfield overlap | Sorted interval check: `cur_lsb <= prev_msb` |
| Duplicate addresses | Warning (not error) — valid for aliased registers |

**Return dict**
```python
{
    "ok": bool,
    "errors": ["EDPQ.ADDRESS: missing msb/lsb", ...],
    "warnings": ["address 0x18 used by multiple registers: [...]"],
    "register_count": int
}
```

When `ok=False`, the codegen sub-agent receives the `errors` list and patches
the generated script, then re-runs.

---

## 5. Module: `tools/resolve_address.py`

### Purpose
Expand a dynamic address expression (`0x018+0x20*N`) into a concrete list of
hex addresses, together with zero-padded name suffixes for register/field naming.

### Internal functions

#### `_find_n_in_text(text) -> list[int] | None`
Tries regex patterns against the description in this order:
1. Range patterns: `N = 0 to 3`, `N: 0...7`, `N ranges from 0 to 3`, `where N is 0 to 3`
2. Comma-list: `N = 0, 1, 2, 3`, `for N in {0,1,2,3}`
3. Single value: `N = 2`, `where N is 2`

Returns `None` if nothing matches.

#### `_parse_user_input(raw) -> list[int]`
Accepts three formats typed by the user:
- `"3"` → `[3]`
- `"0-3"` or `"0..3"` → `[0, 1, 2, 3]`
- `"0,1,2,3"` → `[0, 1, 2, 3]`

#### `_evaluate_expression(expr, variable, value) -> str`
1. Replaces the variable (word-boundary, case-insensitive) with `str(value)`.
2. Validates that the substituted string contains only `[0-9a-fA-FxX+\-*/() \t]`.
3. Calls `eval()` on the sanitised string.
4. Returns `hex(result)` (e.g. `"0x38"`).

### Key function: `resolve_address(expression, description="", variable="N")`

**Flow**
```
expression = "0x018+0x20*N"
description = "This register is used for N = 0 to 3"

1. _find_n_in_text(description) → [0, 1, 2, 3]   (found)
   OR
   print prompt → input() → _parse_user_input()   (not found)

2. _evaluate_expression for each n_value

3. Build name_suffixes:
   pad = max(len(str(max(n_values))), 2)   # minimum 2 digits
   suffixes = [f"_{v:0{pad}d}" for v in n_values]
   # → ["_00", "_01", "_02", "_03"]

4. Return JSON:
{
  "expression": "0x018+0x20*N",
  "variable": "N",
  "n_values": [0, 1, 2, 3],
  "n_source": "description",
  "addresses": ["0x18", "0x38", "0x58", "0x78"],
  "name_suffixes": ["_00", "_01", "_02", "_03"]
}
```

**How the generated parser uses this output:**
For each `(address, suffix)` pair, emit one register object:
```python
for addr, sfx in zip(result["addresses"], result["name_suffixes"]):
    registers.append({
        "name": f"{group_name}{sfx}",          # e.g. "EDPQ_01"
        "address": addr,
        "fields": [
            {**field, "name": f"{field['name']}{sfx}"}  # e.g. "ADDRESS_01"
            for field in current_fields
        ]
    })
```

---

## 6. Module: `agent/main.py`

### 6.1 Model configuration

```python
model = ChatOpenAI(
    base_url="http://localhost:8000/v1",   # vLLM OpenAI-compatible endpoint
    api_key="not-needed",
    model="qwen3",                         # must match --served-model-name
    temperature=0,                         # deterministic output
)
```

### 6.2 Tool registration

All four tools are decorated with `@tool` from `langchain_core.tools`. This
gives each function a `.name` attribute that deepagents accesses internally when
building sub-agent tool lists. Plain functions without `@tool` will raise
`AttributeError: 'function' object has no attribute 'name'`.

```python
@tool
def inspect_excel(xlsx_path: str, sheets: str = "") -> str: ...

@tool
def write_parser_script(code: str) -> str: ...

@tool
def run_parser(script_path: str, xlsx_path: str, out_json_path: str) -> str: ...

@tool
def validate_json(json_path: str) -> str: ...

@tool
def resolve_address(expression: str, description: str = "", variable: str = "N") -> str: ...
```

The tool docstrings are what the LLM reads to decide when and how to call each
tool — they must be precise and complete.

### 6.3 Sub-agent specification

Sub-agents are plain Python dicts. Critical keys:

| Key | Type | Notes |
|---|---|---|
| `name` | str | Identifier used in agent routing |
| `description` | str | How the orchestrator decides which sub-agent to call |
| `system_prompt` | str | The sub-agent's instructions (**not** `prompt` — that key is ignored) |
| `tools` | list | Must be `@tool`-decorated function references, **not** name strings |

**schema-sniffer sub-agent**
- Tools: `[inspect_excel]`
- Input: xlsx path, optional sheet list, optional user notes
- Output: grammar spec JSON (row indices, column indices, value patterns, quirks)

**codegen-and-validate sub-agent**
- Tools: `[inspect_excel, write_parser_script, run_parser, validate_json, resolve_address]`
- Input: grammar spec + xlsx path + output path
- Output: path to validated JSON; up to 5 script revision cycles internally

### 6.4 `create_deep_agent` call

```python
agent = create_deep_agent(
    model=model,
    tools=[inspect_excel, write_parser_script, run_parser, validate_json, resolve_address],
    system_prompt="...",          # orchestrator instructions
    subagents=[schema_sniffer_subagent, codegen_subagent],
)
```

### 6.5 `convert()` — user message construction

```python
user_msg = (
    f"Convert the register-map workbook at '{xlsx_path}' to JSON "
    f"and write the output to '{out_json_path}'."
    + (f" Only process these sheets: {', '.join(sheets)}." if sheets else "")
    + (f"\n\nAdditional notes ... higher-priority hints ...\n{notes}" if notes else "")
)
```

Notes are injected into the user message (not the system prompt) so they are
scoped to this single invocation and do not persist across files in batch mode.

### 6.6 `convert_folder()` — batch mode

```
for each .xlsx / .xlsm in folder:
    file_notes = notes[f.stem]   # if notes is dict
                 OR notes        # if notes is str
                 OR None
    convert(f, out_json, sheets=sheets, notes=file_notes)
    catch Exception → log, continue
```

Notes dict keying by filename stem means `{"registers_v2": "..."}` applies
only to `registers_v2.xlsx` or `registers_v2.xlsm`, not to other files.

---

## 7. Generated Parser Contract

The codegen sub-agent is prompted to produce a script that satisfies:

1. `sys.argv[1]` = input xlsx path, `sys.argv[2]` = output JSON path
2. Uses `openpyxl.load_workbook(data_only=True)`
3. Builds a merged-cell lookup once: `{(row, col): value}` for all merge ranges
4. Row-walk state machine:

```
state = {current_group_name, current_address, current_fields}

for row in ws.iter_rows():
    row_type = cell(row_type_col)
    if row_type == group_value:
        flush current register to output (if fields exist)
        reset state with new group name + address
        if address is expression → call resolve_address tool → expand
    elif row_type == reg_value:
        append field to current_fields
    else:
        stderr warning, skip
flush final register
```

5. Output:
```json
{"registers": [
  {"name": "EDPQ_00", "address": "0x18",
   "fields": [{"name": "ADDRESS_00", "msb": 31, "lsb": 6,
               "reset": "0", "type": "RW", "description": "..."}]}
]}
```

---

## 8. Error Handling Summary

| Location | Failure | Behaviour |
|---|---|---|
| `inspect_workbook` | Sheet name not in workbook | Raises `ValueError` with available sheet list |
| `run_generated_parser` | Subprocess exceeds timeout | `subprocess.TimeoutExpired` propagates to agent |
| `run_generated_parser` | Script crashes (non-zero exit) | `returncode != 0` + stderr returned to agent for diagnosis |
| `validate_register_json` | JSON not parseable | Returns `ok=False`, single error message |
| `resolve_address` | Unsafe chars in expression after substitution | Raises `ValueError` — not caught, surfaces to agent |
| `_parse_user_input` | Non-numeric user input | `ValueError` from `int()` — user prompted to retry (not handled automatically) |
| `convert_folder` | Single file fails | Caught, logged, processing continues for remaining files |
