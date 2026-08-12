# lucasastorian/llmwiki — Architecture & Design Overview

> Independent documentation based on direct inspection of the repository source
> (`api/`, `mcp/`, `web/`, `extension/`, `converter/`, `shared/`, and the `llmwiki` CLI).

---

## 1. What it is, and how it differs fundamentally from nashsu/llm_wiki

This is **not a self-contained "app + built-in LLM calls" tool** the way `nashsu/llm_wiki`
is. Instead, `llmwiki` is closer to **infrastructure + an MCP tool surface**: it stores
your source documents, indexes/chunks/extracts them, and exposes a set of MCP tools
(`search`, `read`, `create`, `edit`, `append`, `delete`, `list`, `lint`, `comments`,
`references`, `reply`) plus a written **GUIDE** (a long instruction document,
`mcp/tools/guide.py`) that tells a connected AI *how* to behave as the wiki's maintainer.

**The actual "thinking" — deciding what pages to create, how to synthesize sources,
how to cross-link concepts — is not done by this codebase at all.** It's done by
whatever MCP client you connect: Claude Code, Claude Desktop, Claude Cowork, Codex, or
any other MCP-compatible agent. This repo supplies the workspace, the tools, and the
instructions; the connected agent supplies the intelligence and does the actual
reading/writing via tool calls.

This is a meaningfully different design philosophy from `nashsu/llm_wiki`'s two-step
`streamChat()` ingest pipeline with a configurable "Custom" LLM endpoint — there is no
equivalent settings screen here for "point wiki-generation at any model," because
wiki-generation isn't something this codebase performs.

---

## 2. Directory map

```
llmwiki/
├── api/                       # FastAPI backend (Python)
│   ├── routes/                  # documents, files, graph, quiz, knowledge_bases, ws (websocket)...
│   ├── services/                 # chunker, ocr, pdf_extract, parsers, graph, quiz_grader, s3...
│   ├── domain/                    # core domain models
│   ├── html_parser/                # web-clip HTML → structured content
│   └── infra/                      # infrastructure glue (DB, storage)
├── mcp/                        # The MCP server — the actual integration point for AI agents
│   ├── tools/                     # search.py, read.py, write.py, ingest.py, lint.py, guide.py...
│   ├── services/
│   └── vaultfs/                    # filesystem abstraction for the local workspace
├── web/                        # Next.js web app — browse the wiki, view graph, sources
│   └── src/
├── extension/                  # Chrome extension — clip webpages/PDFs, highlight, comment
│   └── src/
├── converter/                  # Standalone PDF/Office → structured content microservice
├── shared/
│   └── sqlite_schema.sql          # Local-mode SQLite schema (index.db)
├── supabase/migrations/        # Hosted-mode Postgres schema (for llmwiki.app / self-hosted remote)
├── llmwiki                     # CLI entrypoint (Python script): open/init/serve/mcp/mcp-config/reindex
└── tests/                      # unit + integration tests
```

---

## 3. Two deployment modes

| | **Local mode** | **Hosted/remote mode** |
|---|---|---|
| Storage | SQLite (`shared/sqlite_schema.sql`), files on your own disk under a workspace folder | Postgres via Supabase (`supabase/migrations/`), S3 for file storage |
| Networking | API binds to `127.0.0.1` only — the CLI comment is explicit: *"intentionally loopback-only... does not support LAN or remote binding"* | Full hosted service at llmwiki.app, or self-hosted remotely |
| PDF/Office extraction | Runs in-process locally (`services/pdf_extract.py`, `opendataloader-pdf`) | Routed through an authenticated `converter/` microservice (`CONVERTER_URL`/`CONVERTER_SECRET`) so parsing stays isolated from the main API |
| Setup | `./llmwiki open <folder>` — one command, no accounts | Requires Supabase project, S3 bucket, and associated credentials |

---

## 4. The CLI (`./llmwiki`)

A single Python script, no separate install step beyond `pip install -r requirements`:

```
llmwiki open <workspace>         Init if needed + serve + open browser
llmwiki init <workspace>         Create .llmwiki/ + wiki/, index files
llmwiki serve <workspace>        Start API + web on localhost
llmwiki mcp <workspace>          Run stdio MCP server (for Claude config)
llmwiki mcp-config <workspace>   Print claude_desktop_config.json snippet
llmwiki reindex <workspace>      Force full rebuild of index.db
```

`cmd_init()` creates `.llmwiki/` (with a `cache/` subfolder and a SQLite `index.db`) and
a `wiki/` folder inside whatever workspace directory you point it at — notably, this
workspace is **outside the repo itself**, e.g. `~/research`. The repo's own directory is
never touched by your data.

---

## 5. Design features, module by module

### 5.1 The Guide — the "prompt" that drives everything
`mcp/tools/guide.py` contains a large `GUIDE_TEXT` constant: a structured instruction
document describing the wiki's required page taxonomy (Overview hub page, optional Plan
tracker with task-status glyphs, Concepts, and presumably Entities/Sources categories
below what was inspected), citation conventions, and image-handling rules
(`include_images=true` on the `read` tool for MCP native image blocks). This is
functionally the equivalent of `nashsu/llm_wiki`'s hardcoded ingest prompts — except
here it's a single, human-readable markdown document a connected AI reads as a system
resource, rather than being embedded across a compiled two-stage pipeline.

### 5.2 MCP tools — the AI's entire interface
`mcp/tools/` exposes:
- `search.py`, `read.py`, `list.py` — retrieval-side tools
- `create.py`/`write.py`, `edit.py`, `delete.py`, `append.py` (implied by guide text) —
  mutation-side tools for the wiki layer
- `ingest.py` — notably **hosted-mode only**: `add_source_from_url` pulls a public PDF
  in directly by URL (arXiv links work as-is)
- `lint.py`, `quiz_lint.py` — health-check tooling exposed as callable tools, not an
  internal automatic pass
- `comments.py`, `references.py`, `reply.py` — supports margin notes/comments captured
  via the Chrome extension and threaded replies, exposed to the connected AI as readable
  context

### 5.3 PDF/Office extraction — two tiers, same as nashsu's split
- **Baseline (always available, local):** `api/services/pdf_extract.py`, using the
  `opendataloader-pdf` Python package — a local, no-API-key text/structure extractor.
- **Higher quality (optional):** `api/services/ocr.py` calls the **Mistral OCR API**
  (`MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"`), gated behind a `MISTRAL_API_KEY`
  env var. The README explicitly calls this out as needed for *"higher-quality PDF OCR."*
  Also handles Office formats (`.pptx`, `.docx`) and images through the same OCR path.
- `services/chunker.py`, `services/highlight_chunks.py`, `services/highlight_merge.py` —
  chunk extracted content and merge in extension-captured highlights/comments so they
  travel with the source content the AI reads.

### 5.4 Graph & visualization
- `api/services/graph.py` + `api/routes/graph.py` / `local_graph.py` — builds and serves
  a concept/entity relationship graph, rendered in the Next.js web app's graph viewer.
- README also advertises support for **Mermaid diagrams and SVGs** as content the
  connected AI can embed directly into wiki pages — visualizations become first-class
  wiki content, not just a separate graph-view feature.

### 5.5 Quiz / spaced-recall feature
- `mcp/tools/quiz_lint.py`, `api/services/quiz_grader.py`, `api/routes/quiz.py` — a
  distinct feature where the wiki can generate quiz questions and grade free-form
  answers. Grading runs through **Cloudflare Workers AI** with a fixed hardcoded model
  (`@cf/google/gemma-4-26b-a4b-it`), separate from whatever agent is doing the writing —
  this is the one place in the codebase with its own baked-in model choice.

### 5.6 Chrome extension
`extension/src/` — clips webpages **and PDFs**, supports highlighting and leaving
comments inline, which then sync into the workspace as sources the connected AI can see
via MCP (`comments.py`/`references.py`).

### 5.7 Real-time updates
`api/routes/ws.py` — a websocket route, presumably powering live UI updates in the
Next.js app as the connected AI writes/edits pages during a session.

### 5.8 "Claude Routines" — the self-maintenance mechanism
Not code in this repo, but a documented usage pattern: the README recommends scheduling
a recurring prompt (via Claude Code Routines or a Desktop scheduled task) that tells
Claude to check the workspace for anything new and update the wiki accordingly — this is
how the "autonomous, self-maintaining" claim in the README is actually realized: an
external scheduler triggers an external agent, which then uses this repo's MCP tools.

---

## 6. End-to-end flow (local mode)

```
./llmwiki open ~/research
        ↓
cmd_init(): creates .llmwiki/ (index.db, cache/) + wiki/ inside ~/research
        ↓
cmd_serve(): starts FastAPI (api/) + Next.js (web/) on localhost, both loopback-only
        ↓
_index_existing_files(): scans ~/research, extracts/chunks any files already present
        ↓
./llmwiki mcp-config ~/research → prints MCP server config JSON
        ↓
User pastes config into Claude Desktop / Claude Code settings
        ↓
User (in Claude): "Read the guide, then ingest my sources and start building the wiki."
        ↓
Claude reads GUIDE_TEXT (mcp/tools/guide.py) via MCP
        ↓
Claude calls list/search/read tools to see what's in the workspace
        ↓
For each source: Claude decides what wiki pages to create/update,
calls create/edit/append tools accordingly (all of this reasoning
happens in Claude itself, not in this codebase)
        ↓
Pages written to ~/research/wiki/, indexed in .llmwiki/index.db
        ↓
Next.js web app (localhost:3000) reflects the changes, live via websocket
        ↓
(Optional, ongoing) A scheduled Claude Routine re-runs the same
"check for new sources, update the wiki" prompt nightly
```

---

## 7. License

Apache 2.0 (per repo `LICENSE` and the README badge) — notably more permissive than
`nashsu/llm_wiki`'s GPLv3, including for closed-source/commercial derivative use.
