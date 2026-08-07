# nashsu/llm_wiki — Architecture & Design Deep Dive

> A detailed technical breakdown of how `nashsu/llm_wiki` is built, based on direct
> inspection of the repository source (frontend `src/`, native backend `src-tauri/`,
> `mcp-server/`, and `extension/`). This is independent documentation, not a copy of the
> project's own README.

---

## 1. What it is, at a systems level

`llm_wiki` is a **desktop application** (not a web app, not a CLI tool) that watches a
folder of raw source documents and incrementally builds a structured, cross-linked
Obsidian-compatible markdown wiki out of them using an LLM. It's built as:

- **Frontend:** React + TypeScript + Vite, styled with Tailwind + shadcn/ui components
- **Native shell/backend:** Rust, via **Tauri v2** — gives the web UI access to the real
  filesystem, a bundled PDF renderer, subprocess control, and a local HTTP server
- **Optional companion pieces:** a standalone **MCP server** (Node/TS) and a **Chrome
  extension** (vanilla JS web clipper) that both talk to the desktop app over a local
  HTTP API rather than embedding any logic themselves

The single most important architectural decision: **almost all "intelligence" lives in
the TypeScript frontend (`src/lib/`), not in Rust.** Rust is used for privileged/native
operations (real file I/O, PDF parsing, spawning CLI subprocesses, running a local
server) — but prompt construction, retrieval logic, graph algorithms, and ingest
orchestration are all plain TypeScript, calling out to whichever LLM endpoint you've
configured over HTTP. This is *why* it wasn't extractable as a lightweight script
earlier in this conversation: the "logic" is real application code, deeply wired into
Tauri's `invoke()` bridge for every filesystem read/write, not a self-contained module.

---

## 2. Directory map

```
llm_wiki/
├── src/                        # React/TS frontend — where almost all logic lives
│   ├── lib/                    # ~100 modules: ingest, retrieval, graph, dedup, lint...
│   ├── components/              # UI: settings panels, chat, graph view, wiki browser
│   ├── stores/                  # App state (Zustand-style stores): wiki, chat, review...
│   ├── commands/                 # Thin wrappers around Tauri invoke() calls (fs.ts, file-sync.ts)
│   ├── i18n/                     # Multi-language UI strings
│   └── App.tsx                   # Root component or app shell
├── src-tauri/                   # Rust native backend
│   ├── src/commands/             # Tauri command handlers (fs, project, search, ebook, extract_images...)
│   ├── src/agent/                 # Native runtime for the in-app coding-agent chat mode
│   ├── src/api_server.rs          # Local HTTP server (serves MCP server + Chrome extension)
│   ├── src/clip_server.rs         # Endpoint the Chrome extension posts clipped pages to
│   └── pdfium/                    # Bundled Google PDFium library for PDF rendering/text extraction
├── mcp-server/                  # Standalone Node MCP server — exposes the wiki as MCP tools
├── extension/                   # Chrome extension (Readability.js + Turndown.js web clipper)
├── llm-wiki.md                  # The original design-pattern doc this whole app is based on
└── scripts/debug_ollama_tokens.py  # Dev utility for debugging Ollama token counting
```

---

## 3. Design features, module by module

### 3.1 Project & state management
- `project-store.ts`, `project-identity.ts`, `project-mutex.ts`, `persist.ts` — a
  "project" is a folder on disk with its own identity/lockfile (the mutex prevents two
  instances of the app editing the same project simultaneously) and persisted settings.
- `reset-project-state.ts` — explicit teardown/reset path, implying the app treats
  project state as something that can get corrupted and needs a clean-slate recovery.

### 3.2 Ingest pipeline (the core loop)
This is the biggest subsystem — `ingest.ts` alone is ~3,400 lines. Supporting modules:

| File | Role |
|---|---|
| `ingest-queue.ts` | Queues files for processing; handles ordering, retries, cancellation |
| `ingest-cache.ts` | Hash-based cache so unchanged files are skipped on re-ingest |
| `ingest-sanitize.ts` | Cleans/normalizes raw content before it hits the LLM |
| `source-lifecycle.ts`, `source-identity.ts` | Tracks a source's identity across renames/moves so wiki pages stay linked to the right origin file even if it's relocated |
| `source-watch-config.ts`, `source-watch-defaults.json` | Config for which folders/patterns get auto-watched vs. ignored |
| `scheduled-import.ts` | Periodic/scheduled re-scan of watched sources |
| `raw-source-resolver.ts` | Resolves a source reference back to its actual file on disk |
| `frontmatter.ts` | Reads/writes YAML frontmatter on generated wiki pages |
| `text-chunker.ts` | Splits long documents into LLM-context-sized chunks |
| `parseFileBlocks()` (in `ingest.ts`) | Parses the LLM's structured output back into discrete file writes |
| `isSafeIngestPath()` (in `ingest.ts`) | Path-traversal guard — validates LLM-proposed output paths before writing to disk (cross-platform, including Windows-unsafe segment checks) |

**Flow:** a file lands in the watched folder → `ingest-queue` picks it up → cache check
(skip if hash unchanged) → sanitize/chunk → LLM call(s) via `llm-client.ts` → response
parsed by `parseFileBlocks` → paths validated (`isSafeIngestPath`) → files written via
`commands/fs.ts` → `updateWikiIndexDeterministically()` updates the index **without**
another LLM call (deterministic, not LLM-generated) → activity/log entry recorded.

### 3.3 LLM abstraction layer
| File | Role |
|---|---|
| `llm-client.ts` | Core `streamChat()` — the actual HTTP call, with streaming callbacks |
| `llm-providers.ts` | Defines provider shapes/types (OpenAI, Anthropic, Azure, Ollama, Custom) |
| `llm-task-routing.ts` | Lets different operations (ingest vs. chat vs. lint) use different configured models/providers |
| `endpoint-normalizer.ts` | Normalizes user-entered base URLs (trailing slashes, `/v1` suffixes, etc.) — this is the code path your vLLM URL goes through |
| `azure-openai.ts` | Azure-specific auth/request quirks, kept separate from the generic OpenAI-compatible path |
| `connection-tests.ts` | "Test connection" button logic in settings |
| `has-usable-llm.ts` | Gatekeeper check before allowing ingest/chat to run |
| `context-budget.ts` | Token budget allocation — decides how much retrieved context fits in a given model's window |
| `components/settings/llm-presets.ts` | The actual list of provider presets, including the `"ollama-local"` preset and the generic `"custom"` (OpenAI-compatible `chat_completions`) mode — **this is what you're using for vLLM/Qwen3.6** |

### 3.4 Retrieval (how the app decides what context to feed the LLM)
This is more sophisticated than plain keyword search:
- `search.ts` / `anytxt-search.ts` — text/keyword search, with CJK-aware tokenization
- `embedding.ts`, `dedup_embedding.ts` — vector embeddings for semantic search
- `graph-relevance.ts` — scores wiki pages by a multi-signal relevance model that walks
  the **link graph** (in-links/out-links between pages), not just text similarity — a
  `RetrievalGraph` of `RetrievalNode`s with `outLinks`/`inLinks` sets
- `graph-search.ts`, `graph-filters.ts`, `graph-visibility.ts` — querying and filtering
  that graph for the UI's graph view and for retrieval
- `context-budget.ts` (see above) — takes the ranked results and fits as many as possible
  into the target model's context window

### 3.5 Knowledge graph & insights
- `wiki-graph.ts` — builds the graph structure from all pages' `[[wikilinks]]`
- `graph-insights.ts` — surfaces "surprising connections" (pages that are semantically
  related but not yet linked) and likely does community/cluster detection
- Rendered in the UI as an interactive graph view (see `assets/3-knowledge_graph.jpg`,
  `assets/kg_community.jpg`, `assets/kg_insights.jpg` in the repo)

### 3.6 Deduplication system
A separate background subsystem, not part of the main ingest call:
- `dedup.ts` — core duplicate-detection logic
- `dedup-queue.ts`, `dedup-runner.ts` — queued, throttled background job runner (keeps
  dedup from blocking ingest or freezing the UI)
- `dedup-storage.ts` — persists dedup state/results between runs

### 3.7 Wikilinks & page lifecycle
- `wikilink-transform.ts` — parses/rewrites `[[Page Title]]` syntax
- `enrich-wikilinks.ts` — a separate LLM pass that adds links a first-pass ingest missed
- `wiki-page-resolver.ts` — resolves a wikilink to an actual file path
- `wiki-page-delete.ts`, `sources-tree-delete.ts`, `source-delete-decision.ts` —
  cascade-deletion logic (what happens to dependent pages when a source is removed)
- `page-merge.ts`, `sources-merge.ts` — merging duplicate/near-duplicate pages
- `wiki-schema.ts`, `wiki-page-types.ts`, `wiki-type-style.ts` — defines page "types"
  (source/entity/concept/etc.) and their visual styling
- `wiki-filename.ts`, `wiki-cleanup.ts` — filename normalization and orphan cleanup
- `review-create-page.ts`, `review-utils.ts`, `sweep-reviews.ts`, `stores/review-store.ts`
  — a **review queue**: rather than silently trusting every LLM-generated page, changes
  can be routed through a review step before being finalized

### 3.8 Lint / health-check system
- `lint.ts` — orchestrates the audit
- `lint-structural-core.ts` + `lint-structural.worker.ts` — structural checks (broken
  links, orphan pages, schema violations) run in a **web worker** so linting a large
  wiki doesn't block the UI thread
- `lint-fixes.ts` — auto-fix suggestions/application for issues the structural linter finds
- This is a real hybrid: cheap deterministic structural checks run first (no LLM cost),
  then an LLM pass layers on semantic review (contradictions, staleness)

### 3.9 Deep Research
- `deep-research.ts`, `optimize-research-topic.ts`, `web-search.ts` — a mode where the
  app can go out to the web, gather sources on a topic, and ingest them the same way as
  local files (see `assets/1-deepresearch.jpg`)

### 3.10 Multimodal / images / PDFs
- `mineru.ts` — integrates **MinerU** (an external layout-aware PDF/OCR parser) for
  PDFs the bundled PDFium can't cleanly extract (scanned docs, complex layouts)
- `src-tauri/pdfium/` + `src-tauri/src/commands/extract_images.rs` — the Rust-side
  bundled PDFium renderer, used for the common case (native-text PDFs)
- `extract-source-images.ts`, `image-caption-pipeline.ts`, `vision-caption.ts` — pulls
  embedded images out of sources and captions them with a vision-capable LLM call
- `markdown-image-resolver.ts`, `chat-image-utils.ts` — resolving/displaying images
  referenced in generated markdown and chat
- `src-tauri/src/commands/ebook.rs` — separate handling for ebook formats (epub, etc.)

### 3.11 Chat / agent mode
- `stores/chat-store.ts`, `lib/chat-agent-types.ts`, `chat-save-to-wiki.ts` — an in-app
  chat interface that can save its own outputs back into the wiki
- `claude-cli-transport.ts`, `codex-cli-transport.ts` — notably, chat can be routed
  through an **installed Claude Code or Codex CLI as a subprocess** (via
  `src-tauri/src/commands/cli_resolver.rs`), not just direct API calls — giving it
  access to those tools' own agentic capabilities (file edits, tool use) inside the
  wiki app
- `src-tauri/src/agent/` (Rust: `router.rs`, `tools.rs`, `permissions.rts`, `skills.rs`,
  `session.rs`, `workspace.rs`) — a **native in-app agent runtime**, a third path
  alongside "plain LLM chat" and "CLI passthrough," with its own permission/tool system

### 3.12 Local API server, MCP server, and Chrome extension
- `src-tauri/src/api_server.rs` + `cors.rs` + `proxy.rs` + `server_bind.rs` — the desktop
  app runs a **local HTTP server**. This is the integration point for everything outside
  the app itself.
- `mcp-server/` is a **separate Node package** you run independently; its
  `api-client.ts` just calls this local HTTP server and exposes the results as MCP
  tools — meaning the MCP server has no ingest/retrieval logic of its own, it's a thin
  protocol adapter in front of the desktop app.
- `extension/` (Chrome) similarly has no LLM logic — it uses `Readability.js` (strip
  page chrome, extract article content) and `Turndown.js` (HTML→Markdown) client-side,
  then POSTs the result to `clip_server.rs`, which enqueues it into the same ingest
  pipeline as any other raw source.
- **Design implication:** both companion tools are "dumb" clients of the one real
  brain (the desktop app's ingest/retrieval logic) — nothing is duplicated between them.

### 3.13 Internationalization & language handling
- `detect-language.ts`, `language-metadata.ts` — detects source document language
- `output-language.ts`, `output-language-options.ts` — lets you set wiki output language
  independent of source language (the project-creation prompt you saw earlier)
- `languageRule()` (in `ingest.ts`) — injects a language instruction directly into the
  ingest prompt based on this setting
- `greeting-detector.ts` — likely used to keep chat responses appropriately localized/toned

---

## 4. End-to-end code flow

### 4.1 App startup
1. Tauri boots the Rust process (`main.rs` → `lib.rs`), which starts the local API
   server (`api_server.rs`) and system tray (`tray.rs`).
2. The React app (`main.tsx` → `App.tsx`) mounts in the Tauri webview.
3. `project-store.ts` loads the last-open project (or shows the create/open dialog you
   hit initially) via `persist.ts`.
4. `has-usable-llm.ts` checks whether a working LLM provider is configured; if not,
   ingest/chat actions are disabled until Settings → LLM Provider is filled in.

### 4.2 Path A — Ingesting a local file (the flow you're using)
```
File appears/changes in watched folder
        ↓
source-watch-config.ts decides if it's in scope
        ↓
ingest-queue.ts enqueues it
        ↓
ingest-cache.ts checks sha-equivalent hash → skip if unchanged
        ↓
ingest-sanitize.ts + text-chunker.ts prep the content
        ↓
context-budget.ts + graph-relevance.ts/embedding.ts pull related existing
wiki context (so the LLM knows what already exists, to link/reuse instead
of duplicating)
        ↓
llm-client.ts → streamChat() → your configured endpoint
(this is where your vLLM/Qwen3.6 "Custom" provider is called)
        ↓
ingest.ts: parseFileBlocks() parses the structured response into
{path, content} writes
        ↓
isSafeIngestPath() validates every proposed path
        ↓
commands/fs.ts → Tauri invoke() → Rust fs command → actual disk write
        ↓
updateWikiIndexDeterministically() updates index.md (no LLM call)
        ↓
wikilink-transform.ts / enrich-wikilinks.ts resolve/backfill [[links]]
        ↓
dedup-queue.ts enqueues the new content for background dedup checking
        ↓
wiki-graph.ts updates the in-memory graph; UI graph view + activity log refresh
```

### 4.3 Path B — PDF specifically
```
PDF added to watched folder
        ↓
Rust: extract_images.rs + bundled pdfium/ attempt direct text/image extraction
        ↓
   if layout is too complex / scanned (heuristic or explicit user choice):
        ↓
mineru.ts hands the file to external MinerU for OCR + layout-aware markdown
        ↓
        (rejoins the same ingest.ts flow as 4.2 from "ingest-sanitize.ts" onward)
```

### 4.4 Path C — Web clip (Chrome extension)
```
User clicks extension icon on a webpage
        ↓
Readability.js strips chrome/ads, isolates article content (in-browser, no LLM)
        ↓
Turndown.js converts that HTML to Markdown (in-browser, no LLM)
        ↓
popup.js POSTs the markdown to clip_server.rs (local Rust HTTP server)
        ↓
Treated as a new raw source → same ingest.ts flow as 4.2
```

### 4.5 Path D — Deep Research
```
User gives a research topic
        ↓
optimize-research-topic.ts refines/expands the topic via LLM
        ↓
web-search.ts fetches candidate sources
        ↓
Each result funneled through deep-research.ts → same core ingest as 4.2
```

### 4.6 Path E — Chat, three sub-paths
```
User sends a chat message
        ↓
chat-store.ts routes based on configured mode:
   ├─ Direct LLM: llm-client.ts → streamChat() straight to your provider
   ├─ CLI passthrough: claude-cli-transport.ts / codex-cli-transport.ts →
   │     cli_resolver.rs spawns the installed `claude`/`codex` binary as a
   │     subprocess and streams its output back
   └─ Native agent: src-tauri/src/agent/router.rs dispatches to tools.rs
         under permissions.rs, using skills.rs — an in-app agent loop
        ↓
chat-save-to-wiki.ts can persist any chat output as a new wiki page
(re-entering the ingest flow)
```

### 4.7 Path F — Lint
```
User triggers lint
        ↓
lint-structural-core.ts runs cheap deterministic checks in a Web Worker
(lint-structural.worker.ts) — no LLM cost, catches broken links/schema issues
        ↓
lint.ts layers an LLM pass on top for semantic issues (contradictions, staleness)
        ↓
lint-fixes.ts proposes/applies fixes, some auto-applicable, some routed to
the review queue (review-store.ts) for user confirmation
```

### 4.8 Path G — MCP server access
```
External MCP client (e.g. Claude Code, Claude Desktop) calls a tool
        ↓
mcp-server/src/index.ts (MCP protocol handler)
        ↓
api-client.ts makes an HTTP call to the already-running desktop app's
local server (api_server.rs)
        ↓
Desktop app runs the exact same ingest/search/retrieval code as any other
path — MCP is just another door into the same house
```

---

## 5. Key design principles worth naming explicitly

1. **One brain, many doors.** The Chrome extension, MCP server, chat, deep research,
   and file-watch ingest all converge on the same `ingest.ts`/retrieval pipeline rather
   than each having their own logic — reduces duplication and inconsistency risk.
2. **Deterministic where possible, LLM only where necessary.** Index updates, structural
   lint, hash-based caching, and path safety checks are all plain code — the LLM is only
   invoked for the genuinely generative/semantic steps.
3. **Layered PDF strategy.** Cheap bundled renderer first (PDFium), heavier external OCR
   (MinerU) only when needed — avoids paying OCR cost on every PDF.
4. **Provider-agnostic by design.** The `llm-providers.ts` / `llm-presets.ts` /
   `endpoint-normalizer.ts` layer treats OpenAI, Anthropic, Azure, Ollama, and arbitrary
   custom OpenAI-compatible endpoints (like your vLLM/Qwen3.6 server) as interchangeable
   — nothing about ingest/retrieval logic is hardcoded to a specific provider.
5. **Review as a safety valve, not a gate.** Most content flows straight through, but
   `review-store.ts`/`sweep-reviews.ts` exist for cases (dedup conflicts, risky
   deletions, lint fixes) where auto-applying an LLM's decision is risky enough to want
   a human check first.
6. **Background work is explicitly queued and throttled**, not fire-and-forget — dedup,
   ingest, and structural lint all have dedicated queue/runner modules and (for lint) a
   Web Worker, so heavy operations don't block the UI or race each other.

---

## 6. What this means for your vLLM + Qwen3.6 setup specifically

Every path above (ingest, chat, dedup's optional embedding step, deep research, lint's
LLM pass) ultimately funnels through `llm-client.ts`'s `streamChat()`, which respects
whatever provider you selected in Settings. Since you've set the provider to **Custom**
pointing at your vLLM server, *all* of these subsystems — not just basic ingest — are
running against Qwen3.6. The only exception worth knowing: `dedup_embedding.ts` and the
semantic side of `embedding.ts` call a separate embeddings endpoint — if your vLLM
server isn't also serving an embedding model, that specific feature may fall back to
text-only search rather than vector search (worth checking in Settings if dedup/
retrieval quality seems off).
