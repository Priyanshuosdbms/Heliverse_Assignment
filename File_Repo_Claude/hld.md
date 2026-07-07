# High-Level Design (HLD)
# xlsx2json Register-Map Agent

---

## 1. Purpose

Convert hardware register-map Excel workbooks (`.xlsx` / `.xlsm`) into structured
JSON without relying on a static parser. Because column layouts, languages, and
structural conventions change between files and versions, the system re-derives its
own parsing logic per input file using an LLM-based agentic pipeline.

---

## 2. System Context

```
User / CI Script
      │
      │  xlsx path(s), sheet filter, notes
      ▼
┌─────────────────────────────┐
│     xlsx2json Agent         │   (deepagents + Qwen3 via vLLM)
│                             │
│  ┌──────────────────────┐   │
│  │  Schema Sniffer      │   │   Analyses workbook structure
│  │  Sub-Agent           │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │  Codegen & Validate  │   │   Writes, runs, and fixes parser
│  │  Sub-Agent           │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
      │
      │  registers JSON
      ▼
  Output Directory
```

**External dependencies:**
- vLLM server serving Qwen3 on an OpenAI-compatible endpoint (`localhost:8000`)
- Input workbooks on local disk
- Python 3.11+ with `deepagents`, `langchain-openai`, `openpyxl`, `jsonschema`

---

## 3. Key Design Decisions

### 3.1 Adaptive parsing over static parsing
A static parser breaks the moment a column moves or a header label changes language.
Instead, the agent inspects each file and writes a tailored openpyxl script for it.
That generated script is then executed, validated, and patched in a loop — no
human intervention needed for format drift.

### 3.2 LLM sees a compact sample, not the whole file
Feeding a multi-thousand-row workbook into the LLM context would be expensive and
slow. The `inspect_excel` tool extracts only the first ~40 rows (per sheet) plus
structural metadata (merged ranges, fill colours, column count). This is enough
to infer the row grammar while keeping token usage bounded.

### 3.3 Two-phase sub-agent split
The schema-sniffing and code-generation concerns are deliberately separated into
two sub-agents so each has a focused, shorter context. The grammar spec output by
the schema sniffer is a small, unambiguous JSON contract — easier to verify and
pass between sub-agents than raw prose reasoning.

### 3.4 Validation loop inside the agent
The codegen sub-agent runs the generated script in a subprocess and validates its
output before reporting success. Errors (bitfield overlap, missing address, empty
field list) are fed back so the agent can self-correct, up to 5 attempts. This
replaces manual debugging of generated parsers.

### 3.5 Dynamic address expansion at tool level
Addresses expressed as formulas (`0x018+0x20*N`) are resolved by a dedicated tool
(`resolve_address`) that extracts N from the description text or prompts the user
interactively. The tool returns concrete addresses and zero-padded name suffixes so
the generated parser can emit multiple register instances without the LLM doing
arithmetic.

### 3.6 User notes as a priority override
Free-text notes passed at call time are appended to the agent's user message with
explicit instruction to treat them as higher-priority than general assumptions. This
lets users correct known quirks in specific files without modifying any code.

---

## 4. Data Flow

```
Input xlsx/xlsm
      │
      ▼
inspect_excel (pure Python, no LLM)
      │  compact JSON: headers, 40-row sample,
      │  merged ranges, fill colours, per sheet
      ▼
Schema Sniffer sub-agent (Qwen3)
      │  grammar spec JSON:
      │  {row_type_col, group_value, reg_value,
      │   address_col, msb_col, lsb_col,
      │   field_name_col, reset_col, type_col,
      │   description_col, data_start_row,
      │   sheets_to_parse, quirks}
      ▼
Codegen & Validate sub-agent (Qwen3)
      │
      ├── write_parser_script → generated_parser.py on disk
      │
      ├── run_parser → subprocess → stdout/stderr/output preview
      │
      ├── validate_json → structural checks
      │       │
      │       └── errors? → patch script → re-run (up to 5×)
      │
      └── (if address is a formula) resolve_address
              │  → search description for N
              │  → or prompt user on CLI
              └── addresses[], name_suffixes[]
                        │
                        └── one register entry per address
      │
      ▼
Output JSON
{"registers": [
  {"name": "EDPQ_00", "address": "0x18",
   "fields": [{"name": "ADDRESS_00", "msb": 31, "lsb": 6,
               "reset": "0", "type": "RW", "description": "..."},
              ...]},
  ...
]}
```

---

## 5. Folder / Batch Mode

When given a folder path, `convert_folder()` iterates all `.xlsx` and `.xlsm` files,
calling `convert()` for each. Errors on one file are caught and logged; processing
continues for the rest. Per-file notes can be passed as a `dict` keyed by filename
stem so quirk annotations are scoped to the right file.

---

## 6. Constraints and Limitations

- Only columns 1–20 are sampled during inspection (register sheets rarely need more).
- The 40-row sample may miss structural patterns that only appear deep in large files;
  the codegen loop can re-call `inspect_excel` during validation if needed.
- `resolve_address` evaluates expressions with `eval()` but sanitises input to digits,
  hex characters, and arithmetic operators — no arbitrary code execution.
- Subprocess timeout for the generated parser defaults to 600 s; adjust in `run_parser.py`.
- Batch mode is sequential, not parallel — one agent invocation at a time.
