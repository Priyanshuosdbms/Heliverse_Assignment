# nvme-wiki-agent

A deepagents harness serving Qwen3.6 via vLLM, wired to the ingest/lint/query
workflow and folder taxonomy defined in `AGENTS.md`.

## Setup

1. Put your NVMe 2.0 spec, already parsed to JSON, under `raw/` (one file per
   logical chunk — section, command group, whatever granularity you already
   have). This directory is read-only to the agent.
2. Put `AGENTS.md` (already drafted) at the root of your wiki directory,
   e.g. `nvme-wiki/AGENTS.md`. Create empty `nvme-wiki/index.md` and
   `nvme-wiki/log.md` to start.
3. `pip install -r requirements.txt`
4. vLLM is assumed to **already be running** (started separately — not by
   this harness). `serve_qwen.sh` is kept only as a reference for the flags
   that server was likely started with; it's not invoked by anything here.
5. Set environment variables to match however that server was actually
   launched — these are the two most likely to drift out of sync since the
   server and this harness are started independently:
   ```
   export NVME_RAW_DIR=./raw
   export NVME_WIKI_DIR=./nvme-wiki
   export VLLM_BASE_URL=http://localhost:8000/v1   # host/port of the running server
   export NVME_MODEL=Qwen/Qwen3.6-27B-FP8           # must match the served model id exactly
   ```
   Every CLI command runs a connectivity + model-name check against
   `VLLM_BASE_URL` before doing anything else (`agent.py`'s
   `check_vllm_connection`), and exits with a clear message if the server
   isn't reachable or is serving a different model than `NVME_MODEL` — so a
   mismatch here fails immediately instead of surfacing later as an opaque
   error from inside a subagent call.

## Usage

```bash
# First pass: batch-ingest a whole category, minimal supervision, auto-lints after
python -m nvme_agent.main ingest-batch "raw/admin_commands/*.json"

# Spot-check a sample of what came out by reading the pages directly,
# or ask the query agent:
python -m nvme_agent.main query "What pages were created for admin commands and do they look complete?"

# Going forward: one new source/errata at a time, fully reviewed, no auto-lint
python -m nvme_agent.main ingest-one raw/errata/2026-07-revision.json

# Run lint manually whenever you want, outside the batch cadence
python -m nvme_agent.main lint

# Ask questions against the accumulated wiki
python -m nvme_agent.main query "What changed in the Identify command between 1.4 and 2.0?"
```

## What's genuinely load-bearing vs. what you'll need to adapt

- `tools.py`'s `read_raw_chunk` / `list_raw_chunks` assume your raw JSON is
  organized as one file per chunk under `raw/`. If your NVMe JSON is actually
  one giant file, you'll want a `read_raw_chunk(path, node_id)`-style tool
  that indexes into it instead — the rest of the harness doesn't care either way.
- The **snapshot-before-edit** and **append_log** tools are the two places
  AGENTS.md procedure is enforced by code rather than by hoping the model
  remembers — the ingest-agent's prompt tells it to call these, but nothing
  stops it from skipping them except your own spot-checks early on.
- The **lint-after-every-batch** rule is enforced in `main.py` itself (two
  sequential `agent.invoke` calls), not left to the model's prompt — that's
  deliberate, since it's a hard invariant you decided on, not a judgment call.
- `interrupt_on` isn't set here, i.e. no human-in-the-loop pause on
  write_file/edit_file. Since you've already chosen "batch with less
  supervision" for the first pass, that's consistent — but if early spot-checks
  turn up bad entity splits, the fastest fix is adding
  `interrupt_on={"write_file": True, "edit_file": True}` in `agent.py` to pause
  before every write until you trust the taxonomy more.
