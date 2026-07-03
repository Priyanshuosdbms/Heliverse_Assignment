xlsx2json_agent/
├── agent/
│   └── main.py
├── tools/
│   ├── inspect_excel.py
│   ├── run_parser.py
│   └── validate_json.py
├── requirements.txt
└── README.md

# xlsx2json register-map agent

Converts register-map Excel workbooks (xlsx/xlsm) to JSON using a
deepagents pipeline that *re-derives its parsing logic per input file*
instead of relying on one static script. Built against vLLM-served
Qwen3.6 (OpenAI-compatible endpoint).

## Setup

```bash
pip install -r requirements.txt

# In another terminal, serve the model:
vllm serve <your-qwen3.6-checkpoint> --port 8000 --served-model-name qwen3.6
```

Edit `agent/main.py` if your served model name or port differs.

## Run

```bash
python agent/main.py /path/to/registers.xlsm /path/to/out.json
```

## Why this isn't a static parser

The `schema-sniffer` sub-agent looks at column order, header labels, and
row-type values *for each input file* and emits a grammar spec; the
`codegen-and-validate` sub-agent writes a fresh openpyxl script against
that spec, runs it, and validates the structural correctness (bitfield
overlap, missing addresses, empty field lists) — patching and re-running
up to 5 times if validation fails. If next file moves columns around,
swaps language, or renames "Group"/"Reg" to something else, the same
pipeline adapts without you touching code.

## Files

- `tools/inspect_excel.py` — bounded, structure-aware workbook sampler (no LLM).
- `tools/run_parser.py` — runs the generated script, captures errors.
- `tools/validate_json.py` — file-shape-agnostic output validator.
- `agent/main.py` — deepagents orchestrator + sub-agent prompts.