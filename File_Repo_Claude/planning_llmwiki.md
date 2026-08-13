# Planning Doc: LLM Wiki on `deepagents`

> Scope discipline: this document only includes pieces that are part of the **core
> wiki-building logic** — ingest, retrieval, dedup, linking, lint, index integrity,
> extraction quality. It deliberately excludes anything that was GUI/platform-specific
> in the two reference repos (Tauri shell, React components, Chrome extension UI,
> Next.js web app, VNC/desktop packaging) since those aren't part of the *logic*, just
> a delivery surface for it. Every ported concept cites the exact file it's based on.

---

## 1. Goal

Build an LLM-wiki tool as an application on top of `langchain-ai/deepagents`, running
against a self-hosted vLLM server serving Qwen3.6, that reaches functional parity with
the *logical* capabilities of `nashsu/llm_wiki` and `lucasastorian/llmwiki` — without
inheriting either one's platform lock-in (Tauri/GPL for nashsu; Claude/Codex-only
authorship for lucas).

---

## 2. Core logic to port from `nashsu/llm_wiki`

### 2.1 Two-stage ingest (analysis → generation)
Ingest is split into a separate **analysis** call (reason about the source, decide what
pages it affects) and a **generation** call (produce the actual page content), rather
than one combined call.
**Source:** `src/lib/ingest.ts` — distinct `"Analysis stream failed"` /
`"Generation stream failed"` error paths confirm the two-call structure.
**Port as:** a two-step sub-agent invocation in deepagents — first a planning turn
against the retrieved context, then a generation turn constrained to that plan. This
maps naturally onto deepagents' built-in **planning tool**, rather than needing custom
orchestration code.

### 2.2 Structured, parseable page-output format with path validation
The LLM's output must be parsed into discrete `{path, content}` writes, and every
proposed path is validated before it touches disk.
**Source:** `src/lib/ingest.ts` — `parseFileBlocks()` and `isSafeIngestPath()` (rejects
`..`, absolute paths, unsafe path segments).
**Port as:** a custom `write_wiki_page` tool (not the raw filesystem-write tool) that
enforces path-safety validation server-side, independent of whether the model's output
is well-formed — do not trust the model to self-police path safety, per deepagents'
own documented "trust the LLM, constrain at the tool level" philosophy.

### 2.3 Deterministic index maintenance (not LLM-generated)
The wiki's `index.md` catalog is updated with plain code after a write, not by asking
the LLM to summarize/rewrite the index itself.
**Source:** `src/lib/ingest.ts` — `updateWikiIndexDeterministically()`.
**Port as:** the `write_wiki_page` tool itself appends/updates a structured index
(e.g. `index.json` or `index.md`) as a side effect of every successful write — this
should be host code, never a separate LLM call, to avoid index drift/hallucination.

### 2.4 Hash-based ingest caching
Unchanged source files are skipped on re-ingest via content hashing.
**Source:** `src/lib/ingest-cache.ts`.
**Port as:** a pre-agent-invocation check (outside the LLM loop entirely) — compute
sha256 per source file, skip invoking the agent at all for unchanged files. This is the
single cheapest win for cost/latency and should not be reimplemented as an agent tool;
it belongs in the harness code that decides whether to invoke the agent per file.

### 2.5 Multi-signal retrieval (link-graph relevance, not just text/vector similarity)
Existing-wiki context fed into ingest is chosen using a relevance score that walks the
page link graph (in-links/out-links), not purely keyword or embedding similarity.
**Source:** `src/lib/graph-relevance.ts` (`RetrievalGraph`, `RetrievalNode` with
`outLinks`/`inLinks`).
**Port as:** a `search_wiki` tool that combines (a) keyword match, (b) optional vector
similarity if an embedding endpoint is configured, and (c) a graph-walk boost for pages
linked to already-matched pages. Rank-merge these signals rather than relying on one.

### 2.6 Context budgeting
Retrieved context is fit into a token budget appropriate to the target model, not
dumped in unbounded.
**Source:** `src/lib/context-budget.ts`.
**Port as:** a budget function invoked before each generation turn, sized to Qwen3.6's
actual context window (not hardcoded to whatever nashsu tuned for its default
providers) — this needs to be a first-class config value since local models vary widely
in usable context length in practice vs. advertised.

### 2.7 Structural lint separated from semantic lint
Cheap, deterministic checks (broken `[[links]]`, orphan pages, schema violations) run
as plain code; a separate, optional LLM pass handles semantic issues (contradictions,
staleness).
**Source:** `src/lib/lint-structural-core.ts` (worker-based deterministic checks) vs.
`src/lib/lint.ts` (LLM-based semantic pass).
**Port as:** a `lint_wiki` tool with two modes — `structural` (host code, always cheap
to run, can run after every ingest automatically) and `semantic` (an explicit agent
sub-task, run on demand or scheduled, not after every single file).

### 2.8 Wikilink resolution and backfill
`[[Page Title]]` syntax is parsed, resolved to real paths, and a separate pass can add
links a first-generation pass missed.
**Source:** `src/lib/wikilink-transform.ts`, `src/lib/enrich-wikilinks.ts`.
**Port as:** part of the same `write_wiki_page` tool — resolve links at write time
against the existing page index; expose a separate `enrich_links` tool the agent (or a
scheduled routine) can invoke as a follow-up pass over recently written pages.

### 2.9 Deduplication as a distinct, queued background concern
Duplicate/near-duplicate content detection runs separately from the main ingest call,
not inline with every write.
**Source:** `src/lib/dedup.ts`, `src/lib/dedup-queue.ts`.
**Port as:** a `check_duplicates` tool the agent can call before finalizing a new page,
plus an optional standalone dedup sub-agent that periodically sweeps the wiki — using
deepagents' **sub-agent delegation** primitive so this doesn't block the main ingest
turn's context/latency.

### 2.10 Tiered PDF/document extraction (cheap default, optional high-quality tier)
A fast local extractor is the default; a heavier OCR/layout-aware parser is optional
and explicitly gated (cost/privacy tradeoff), since the default tier is known to produce
poor results on scanned/complex-layout documents.
**Source:** `src-tauri/pdfium/` + `src/lib/mineru.ts` (nashsu: bundled PDFium default,
optional MinerU); `api/services/pdf_extract.py` + `api/services/ocr.py` (lucas:
`opendataloader-pdf` default, optional Mistral OCR).
**Port as:** a `extract_document` tool with an explicit `quality: "fast" | "accurate"`
parameter, defaulting to fast, so the same failure mode that caused your earlier PDF
ingest problem is a visible, deliberate choice rather than a silent default. This is the
single highest-leverage fix given your prior experience with both reference repos.

---

## 3. Core logic to port from `lucasastorian/llmwiki`

### 3.1 The Guide as an explicit, editable instruction document
Wiki conventions (page taxonomy, citation rules, how to handle images) are defined in
one human-readable instruction document the agent reads, rather than being scattered
across compiled parsing/schema code.
**Source:** `mcp/tools/guide.py` (`GUIDE_TEXT`).
**Port as:** this maps almost directly onto deepagents' `systemPrompt`/`system_prompt`
parameter to `create_deep_agent()` — keep the wiki conventions as an editable markdown
document loaded into the system prompt at startup, not hardcoded logic. This is
strictly better-suited to deepagents than to nashsu's compiled-schema approach (see
§4.3 for why this should be the primary way folder structure is defined).

### 3.2 Highlight/annotation merging into source content
User-added highlights and margin notes on a source are merged into what the agent
reads, so annotations travel with the content rather than existing in a separate
system the agent never sees.
**Source:** `api/services/highlight_chunks.py`, `api/services/highlight_merge.py`.
**Port as:** if/when a capture mechanism is added (see §4.4), merge any captured
annotations into the source text before it's handed to the ingest sub-agent — treat
annotations as part of the source content, not wiki-adjacent metadata.

### 3.3 Tool-scoped mutation surface (search/read/create/edit/append/delete)
Wiki mutation happens through a small, well-defined set of named tools rather than
generic filesystem write access, giving a natural place to add validation per operation
type.
**Source:** `mcp/tools/` (`search.py`, `read.py`, `create.py`/`write.py`, `edit.py`,
`append.py`, `delete.py`).
**Port as:** define exactly these six tool verbs for deepagents' `FilesystemBackend`
usage instead of exposing raw file read/write — each verb gets its own validation
(e.g. `delete` triggers the cascade-deletion logic in §4.2, `create`/`edit` both funnel
through the path-safety + index-update logic from §2.2/§2.3).

---

## 4. New — not present (or not adequate) in either reference repo, to add fresh

### 4.1 Native vLLM/Qwen3.6 model wiring (deepagents-specific)
Neither reference repo's approach applies directly to deepagents' model configuration.
**Basis:** deepagents' documented model interface accepts `provider:model` strings or a
configured LangChain chat-model instance, and explicitly supports self-hosted models
via Ollama, vLLM, or llama.cpp.
**To add:** configure `create_deep_agent(model=...)` using `ChatOpenAI` pointed at your
vLLM server's `base_url` (vLLM speaks the OpenAI-compatible protocol) with the Qwen3.6
model name — this is genuinely simpler than either reference repo's provider-abstraction
layer, since deepagents/LangChain already treat this as a first-class case.

### 4.2 Cascade-aware deletion (needs to be built — neither repo's Rust/Python delete logic is portable)
When a source is removed, dependent pages should be cleaned up, but shared
entities/concepts referenced by *other* still-existing sources should be preserved.
**Basis:** nashsu references this concept (`source-delete-decision.ts`,
`wiki-page-delete.ts`) but the actual decision logic is Rust/Tauri-specific and not
extractable; it needs a fresh, explicit reference-counting implementation.
**To add:** track inbound-reference counts per page (derived from the wikilink graph,
§2.5/2.8) so `delete_source` can safely auto-remove pages with zero remaining
references and flag (not silently delete) pages that are still referenced elsewhere.

### 4.3 A single canonical folder-structure spec, versioned and swappable
Neither repo lets you change the wiki's folder convention without touching source code
(nashsu: compiled TS schema; lucas: one global `GUIDE_TEXT`, same for every workspace).
**To add:** since §3.1 already puts conventions in an editable system-prompt document,
formalize this as a per-project `WIKI_SPEC.md` file living inside each project's
workspace (not the app's own source tree) — loaded at agent startup the same way
`GUIDE_TEXT` is, but scoped per-project so different projects can use different
taxonomies without forking anything. This directly answers the "folder structure
modifiable" requirement from earlier in this conversation, in a way neither reference
repo actually provides.

### 4.4 Explicit, visible extraction-quality logging
Both reference repos can silently produce poor wiki pages from PDFs without surfacing
*why* (§2.10) — this was the root cause of your earlier problem with both tools.
**To add:** `extract_document` (§2.10) should always log which extraction tier it used
and a confidence/quality signal (e.g. extracted-character-count vs. page-count as a
crude sanity check, flagging suspiciously low-yield extractions) directly into the
per-file ingest log — do not let a bad extraction pass through to generation silently.

### 4.5 Sub-agent isolation per source, using deepagents' delegation primitive
**Basis:** deepagents' sub-agent delegation for isolated task execution is a capability
neither reference repo has (nashsu and lucas's "agent" both run as one continuous
session/call chain).
**To add:** ingest each source as an isolated sub-agent task (own context window, own
scratch space) that reports back a structured result to the main wiki-maintainer agent
— prevents one large/messy source from polluting the main agent's context with
irrelevant intermediate reasoning, and allows parallelizing ingest of multiple sources.

### 4.6 Deterministic pre-flight cache check outside the agent loop (extends §2.4)
**To add:** explicitly architect the harness so hash-checking and skip-if-unchanged
(§2.4) happens in host code *before* any sub-agent is even spawned for a file — this
must be a harness-level decision, not something the agent decides via a tool call,
since it should cost zero tokens for unchanged files.

---

## 5. Explicit exclusions (deliberately not carried over)

- Tauri/Rust native shell, React UI, system tray — delivery mechanism only, not logic
- Chrome extension's Readability.js/Turndown.js clipping — a capture mechanism, not
  wiki-building logic; can be added later as its own concern if needed, out of scope here
- Next.js web viewer / knowledge-graph visualization UI — presentation, not logic
- Quiz/spaced-recall feature (`quiz_grader.py`) — unrelated to core wiki-building
- MCP server/API server wrapper layers — integration plumbing, not wiki logic itself;
  deepagents' own tool-calling loop replaces the need for a separate MCP hop entirely

---

## 6. Summary mapping table

| Capability | Ported from | deepagents primitive used |
|---|---|---|
| Two-stage ingest | nashsu `ingest.ts` §2.1 | Planning tool + constrained generation turn |
| Path-safe writes | nashsu `ingest.ts` §2.2 | Custom `write_wiki_page` tool |
| Deterministic index | nashsu `ingest.ts` §2.3 | Host-code side effect of `write_wiki_page` |
| Ingest caching | nashsu `ingest-cache.ts` §2.4 | Harness pre-check, outside agent loop |
| Graph-aware retrieval | nashsu `graph-relevance.ts` §2.5 | Custom `search_wiki` tool |
| Context budgeting | nashsu `context-budget.ts` §2.6 | Pre-generation budget function |
| Structural + semantic lint | nashsu `lint-structural-core.ts`/`lint.ts` §2.7 | `lint_wiki` tool, two modes |
| Wikilink resolution | nashsu `wikilink-transform.ts` §2.8 | Part of `write_wiki_page` + `enrich_links` tool |
| Dedup | nashsu `dedup.ts` §2.9 | `check_duplicates` tool + dedup sub-agent |
| Tiered extraction | nashsu `mineru.ts` / lucas `ocr.py` §2.10 | `extract_document` tool, explicit quality param |
| Guide-as-prompt | lucas `guide.py` §3.1 | `system_prompt` parameter |
| Annotation merging | lucas `highlight_merge.py` §3.2 | Pre-ingest content merge step |
| Scoped mutation tools | lucas `mcp/tools/*` §3.3 | Six named deepagents tools |
| vLLM/Qwen3.6 wiring | new, §4.1 | `ChatOpenAI(base_url=...)` passed to `create_deep_agent` |
| Cascade deletion | new, §4.2 | Reference-counted `delete_source` tool |
| Per-project folder spec | new, §4.3 | `WIKI_SPEC.md` loaded into system prompt per project |
| Extraction quality logging | new, §4.4 | Structured ingest log entries |
| Per-source sub-agent isolation | new, §4.5 | deepagents sub-agent delegation |
| Pre-flight cache in harness | new, §4.6 | Harness-level check before agent spawn |
