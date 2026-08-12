# LLM Wiki Capability Comparison — nashsu/llm_wiki vs. lucasastorian/llmwiki vs. langchain-ai/deepagents

> Note on `deepagents` up front: it is **not an LLM-wiki application**. It's LangChain's
> general-purpose "agent harness" (planning, filesystem tools, sub-agent delegation,
> memory — built on LangGraph). It doesn't know what a "wiki" is out of the box; you'd
> be *building* an LLM-wiki tool on top of it, not installing one. It's included here
> because its flexibility and native vLLM support make it a legitimate third option if
> the other two don't fit — just a structurally different kind of option.

---

## Comparison table

| | **nashsu/llm_wiki** | **lucasastorian/llmwiki** | **langchain-ai/deepagents** |
|---|---|---|---|
| **What it actually is** | A finished desktop app with a built-in ingest pipeline | Workspace + MCP tools; wiki-writing delegated to your MCP client (Claude/Codex) | A code library/framework for building an agent — no wiki logic exists until you write it |
| **Out-of-the-box wiki capability** | ✅ High — ingest, retrieval, dedup, lint, graph all implemented and working the moment you install it | ⚠️ Medium — the page taxonomy/citation rules are pre-written (`GUIDE_TEXT`), but the actual "read this, write that" reasoning depends entirely on your connected agent's session behavior each time | ❌ None — `create_deep_agent()` gives you planning + filesystem tools + sub-agents; you must write the system prompt, ingest logic, and page conventions yourself before it does anything wiki-like |
| **Effort to get a working LLM-wiki** | Low — download, configure LLM provider, drop files in `raw/` | Low-medium — `pip install` + `npm install`, then connect an MCP client and tell it to "read the guide and start" | High — you're writing an application (system prompt design, file-write conventions, chunking/retrieval strategy, dedup/lint logic) using the harness's primitives |
| **vLLM / Qwen3.6 support (for the actual writing task)** | ✅ Native — explicit "Custom" OpenAI-compatible provider setting | ❌ None — wiki-writing model is whatever MCP client you connect (Claude/Codex); no config surface in this repo for it | ✅ Native and explicit — docs state self-hosted models via **Ollama, vLLM, or llama.cpp** are supported directly, same as any LangChain chat model |
| **PDF/document extraction built in?** | ✅ Yes — bundled PDFium + optional MinerU OCR | ✅ Yes — local `opendataloader-pdf` + optional Mistral OCR | ❌ No — no document loaders/OCR included; you'd wire in your own (e.g. LangChain's document loaders, or the same PDFium/MinerU/Mistral options the other two use) |
| **Folder structure — fixed or modifiable?** | **Semi-fixed.** `wiki-schema.ts`/`wiki-page-types.ts` hardcode the page-type system (source/entity/concept/etc.) and `entities/`/`concepts/`/`sources/` layout. Changing it means editing and rebuilding the TypeScript/Rust app yourself. | **Fixed by convention, technically editable.** The taxonomy (`overview.md`, `plan.md`, `concepts/`, etc.) lives in one plain-text `GUIDE_TEXT` constant in `mcp/tools/guide.py` — editable without touching build tooling, but still requires opening the source and it's the same guide for every workspace. | **Fully open — this is the point of the tool.** There is no default wiki folder structure at all. You define it entirely in your own system prompt and filesystem-tool usage — one file per "project" or a deep taxonomy, your call, changeable per run with no code rebuild. |
| **Multi-step reasoning depth** | Two-stage (analysis → generation) per file, hardcoded flow | Full open-ended agentic session — the connected AI can take as many tool-calling turns as it wants | Full open-ended agentic session, plus **sub-agent delegation** — can spin off isolated sub-tasks (e.g. "summarize this PDF" as its own sub-agent) which the other two don't offer |
| **Deterministic safety nets** (index integrity, path validation, structural lint) | ✅ Yes — `updateWikiIndexDeterministically()`, `isSafeIngestPath()`, Web-Worker structural lint all run outside the LLM | ⚠️ Partial — `lint.py`/`quiz_lint.py` exist as tools, but nothing runs automatically; relies on the agent choosing to invoke them | ❌ None by default — "trust the LLM" is the harness's documented philosophy; you'd have to build any deterministic guardrails yourself |
| **Setup complexity** | Low — single `.deb`/`.AppImage`/binary | Medium — Python 3.11+, Node 20+, venv, two `pip install` targets, `npm install` | Medium-high for a wiki use case specifically — `pip install deepagents` is trivial, but you're then writing real application code, not just configuring one |
| **License** | GPLv3 | Apache 2.0 | MIT |
| **Best fit** | You want a working LLM-wiki today, with minimal setup, self-hosted model support built in | You already live in Claude Code/Codex/Cowork daily and want your wiki to be another thing those tools maintain for you | You want to design your *own* wiki behavior/folder convention from scratch and are comfortable writing the orchestration code, with vLLM/Qwen3.6 as a first-class citizen |

---

## On "folder structure being modifiable," specifically

This is worth expanding since it's the sharpest differentiator of the three:

- **nashsu/llm_wiki**: the page-type system and default folder layout are **compiled
  into the app** (TypeScript enums/schema files, Rust for the file I/O). You can't
  reconfigure this from Settings — changing it means forking the repo, editing
  `wiki-schema.ts`/`wiki-page-types.ts`, and rebuilding the Tauri app yourself.

- **lucasastorian/llmwiki**: the taxonomy is **one plain-text constant**
  (`GUIDE_TEXT` in `mcp/tools/guide.py`) that a connected AI reads as instructions
  rather than enforced code. You *could* edit this file to change the conventions
  wiki-wide (no compiling required, it's just a string the MCP server serves) — but
  it's still one global guide file, not something you configure per-workspace from a UI,
  and nothing enforces the AI actually follows your edits every session.

- **deepagents**: there is genuinely **no built-in folder structure** — you're handed
  filesystem tools (read/write/edit) and a blank system prompt. The "wiki" folder
  convention is 100% something you specify yourself, and it can be different for every
  project you build, changed at any time by just editing your own prompt text, with no
  fixed schema fighting you. This is the most flexible of the three by a wide margin —
  at the direct cost of nothing being built for you yet.

**Practical read:** if you want to *use* an LLM-wiki today with your vLLM/Qwen3.6 setup,
`nashsu/llm_wiki` remains the most direct path. If you specifically want full control
over the wiki's folder/page conventions and are willing to build the ingest logic
yourself, `deepagents` is the only one of the three that gives you that without fighting
an existing schema — at the cost of having to build the actual "wiki" behavior from
scratch.
