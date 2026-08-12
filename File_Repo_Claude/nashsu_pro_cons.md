# nashsu/llm_wiki — Pros & Cons (with evidence from the actual code)

> This is written specifically to explain **why your PDF ingest didn't produce the wiki
> you expected**, using facts pulled directly from the source (not speculation). Each
> point below cites the actual file/mechanism responsible, with a concrete example.

---

## TL;DR — most likely reason your PDFs didn't work

**MinerU (the high-quality PDF/OCR parser) is OFF by default.** Unless you explicitly
enabled it in Settings, your PDFs were only run through the bundled PDFium text
extractor — which works fine for simple, native-text PDFs, but produces poor or garbled
output for scanned documents, multi-column layouts, tables, and image-heavy PDFs. That
poor extraction is then what gets sent to Qwen3.6 to build wiki pages from — garbage in,
garbage out. See §2.1 for the fix.

A second, independent likely cause: the ingest pipeline expects the LLM to return output
in a strict `---FILE:...---END FILE---` block format, and the code's own comments
document multiple ways this can fail and get **silently dropped with no page created**.
See §2.2.

---

## 1. Pros

### 1.1 Provider-agnostic — genuinely works with vLLM + Qwen3.6
**Evidence:** `src/components/settings/llm-presets.ts` defines an explicit `"custom"`
provider mode alongside OpenAI/Anthropic/Ollama presets, feeding into
`endpoint-normalizer.ts` and `llm-client.ts`'s `streamChat()`, which just POSTs to
whatever OpenAI-compatible `base_url` you configure.
**Example:** this is exactly why pointing it at `http://localhost:8000/v1` with Qwen3.6
worked as a connection at all — nothing in the ingest/retrieval code is hardcoded to a
specific provider's SDK.

### 1.2 Deterministic bookkeeping, not LLM-dependent
**Evidence:** `updateWikiIndexDeterministically()` in `ingest.ts` updates `index.md`
with plain code, not another LLM call.
**Example:** even if a generation call produces a mediocre page, your `index.md` won't
drift out of sync or get corrupted by a bad LLM summarization of itself — it's computed
from what was actually written to disk.

### 1.3 Structural lint runs without LLM cost
**Evidence:** `lint-structural-core.ts` + `lint-structural.worker.ts` run in a Web
Worker and catch broken links/orphan pages/schema violations with plain code, before
any LLM-based semantic lint pass runs.
**Example:** you can lint a 500-page wiki for broken `[[links]]` in milliseconds with
zero tokens spent, and only pay for an LLM call on the smaller semantic-review pass.

### 1.4 Path-traversal safety on every LLM-proposed write
**Evidence:** `isSafeIngestPath()`, called on every parsed FILE block before any write
reaches disk (see `ingest.ts` line ~540: rejects `..`, absolute paths, unsafe Windows
filenames).
**Example:** if a prompt-injected or malformed document somehow got the model to
propose writing to `../../etc/something`, this rejects it rather than trusting the LLM
output blindly.

### 1.5 Markdown-aware chunking preserves structure — for markdown/text sources
**Evidence:** `text-chunker.ts`'s docstring specifies it never splits inside a fenced
code block or a table, and prefers splitting at heading boundaries first.
**Example:** ingesting a well-formatted `.md` design doc keeps each `## Section` and
any embedded tables intact as one chunk, rather than tearing a table row in half across
two chunks (this quality only applies to sources that already have markdown
headings/tables — see §2.1 for why this doesn't help most PDFs).

### 1.6 Single pipeline for every entry point
**Evidence:** file-watch ingest, the Chrome-clipper (`clip_server.rs`), MCP server
(`api_server.rs` → same `ingest.ts`), and Deep Research (`deep-research.ts`) all funnel
into the same `ingest.ts` logic rather than duplicating it.
**Example:** a fix or prompt improvement you'd make to ingest quality benefits every
entry point at once — you don't need to separately debug "why does web-clipped content
look different from locally-ingested content," because it's the same code path.

---

## 2. Cons (with concrete failure examples)

### 2.1 PDF quality is a landmine because the good parser is opt-in, not default
**Evidence:** `src/components/settings/sections/mineru-section.tsx` — `mineruEnabled`
defaults to off; the description in-app literally says *"Use MinerU cloud or a
self-hosted local service for **higher quality** PDF parsing (tables, formulas, complex
layouts)"* — implying the default path is lower quality for exactly those cases.

**Concrete example of what this looks like in practice:**
- A scanned PDF (image-only pages, no embedded text layer) run through bundled PDFium
  alone often extracts to **empty or near-empty text**, since PDFium reads embedded text
  objects — it doesn't OCR. If your source content vanishes before it even reaches the
  LLM, no amount of Qwen3.6 quality fixes that; you'll get a thin, near-empty wiki page
  or nothing at all.
- A two-column academic-paper-style PDF often gets extracted by naive text-layer
  readers as interleaved garbage — reading left-column-line-1, right-column-line-1,
  left-column-line-2, right-column-line-2 out of visual order, because the text objects
  in the PDF aren't necessarily stored in reading order. The LLM then receives
  semantically scrambled text and produces a wiki page that reads as nonsensical or
  wrong, even though the model itself did nothing wrong — it was fed bad input.
- Tables extracted via plain PDFium usually collapse into flat, unaligned text (columns
  smashed together with no delimiters), so a page that should become a clean data table
  in the wiki instead becomes an unusable wall of numbers.

**The fix:** Settings → MinerU PDF Parser → enable it, and pick a backend:
- **Cloud** — needs a MinerU API token, and the in-app privacy notice is explicit:
  *"contents are uploaded to MinerU cloud for parsing"* — don't use this for sensitive
  documents.
- **Self-hosted local** — points at `http://127.0.0.1:8000` by default
  (`DEFAULT_LOCAL_MINERU_ENDPOINT` in `mineru.ts`), meaning you'd need to run MinerU's
  own service separately alongside vLLM. This is extra infrastructure most people don't
  expect to need for "just point it at some PDFs."

### 2.2 Malformed LLM output can silently drop entire pages — no error shown
**Evidence:** the docstring directly above `parseFileBlocks()` in `ingest.ts` lists,
in the developers' own words, hazard classes their regex-based parser has had to fight:
- *H2 — Stream truncation: "the entire block was silently dropped with no logging"*
- *H5 — a literal `---END FILE---` string appearing inside a code block (e.g. the model
  writing a page **about** the ingest format itself) causes the parser to stop early,
  *"truncating the page and dumping all subsequent real content into no-man's-land"**
- *H6 — Empty path: "block matched but was silently dropped by a downstream check"*

The comment says most of these are now fixed/surfaced as warnings — **except H2**,
which it explicitly calls *"fundamentally a stream-budget problem."*

**Concrete example of what this looks like in practice:**
Your ingest is a **two-stage** process — a separate "analysis" LLM call, then a separate
"generation" LLM call (`ingest.ts` throws distinct errors: `"Analysis stream failed"`
and `"Generation stream failed"` for each). Both stages expect the model to follow a
specific `---FILE: path---\n...content...\n---END FILE---` block convention in its
output. Smaller or less rigorously instruction-tuned models are more likely than
GPT-4/Claude-class models to:
- Truncate a long generation mid-block if `max_tokens` runs out before `---END FILE---`
  is emitted → **H2, silently dropped, no page written, no error surfaced to you.**
- Slightly vary the marker format (extra whitespace, different casing) — the parser
  claims to now handle these variants, but any *novel* variant a different model
  produces that wasn't in their test fixtures could still slip through.

If you fed in a long or structurally complex PDF, the generation step had to describe
possibly many new pages in one streamed response — the longer that response, the more
likely it hits a truncation boundary before properly closing the last block, which is
exactly the one failure mode the code admits it can't fully fix.

### 2.3 Two LLM calls per file means two chances to fail, and cost/latency doubles
**Evidence:** as above — ingest is explicitly two-stage (analysis, then generation),
not one call.
**Example:** for local inference via vLLM (as opposed to a fast hosted API), this means
each ingested file waits on two full sequential generations from Qwen3.6, which can be
slow on modest hardware — and if the analysis stage's output already reasons poorly
about a badly-OCR'd source, the generation stage inherits that bad reasoning even if it
executes its own formatting perfectly.

### 2.4 Chunking's structural awareness disappears for non-markdown sources
**Evidence:** `text-chunker.ts` prioritizes splitting at markdown heading boundaries
(`## `, `### `) first, falling back to paragraphs/sentences only when no headings exist.
**Example:** raw PDFium-extracted text has no markdown syntax at all — no `##`
headings, no `|` table delimiters. So for PDF sources specifically, the "smart" part of
the chunker is inactive; it silently falls back to sentence/paragraph splitting, which
can sever a logical section (e.g. a multi-paragraph explanation of one concept) right
in the middle, weakening the context each ingest call receives.

### 2.5 GPLv3 license
**Evidence:** repository root `LICENSE` file (GPLv3).
**Example:** if you ever wanted to embed this into a closed-source internal tool or
redistribute a modified version commercially without open-sourcing your changes, GPLv3
would require you to release your modifications under the same license — worth knowing
before investing heavily in customizing it.

### 2.6 Review queue can hide output you expect to see immediately
**Evidence:** `review-store.ts`, `sweep-reviews.ts`, `review-create-page.ts` — some
LLM-generated changes (dedup conflicts, risky page merges/deletions) are routed to a
separate review queue rather than landing directly in the visible wiki.
**Example:** if your PDF ingest triggered dedup logic (e.g. content overlapping with an
already-ingested page) or a risky merge decision, the resulting page(s) may be sitting
in the review queue rather than in the main wiki view you were checking — worth looking
there specifically if pages seem to have "vanished" rather than never having been
created at all.

### 2.7 Embeddings/vector search silently degrades without a separate embedding model
**Evidence:** `embedding.ts` / `dedup_embedding.ts` call a distinct embeddings endpoint,
separate from your chat-completions endpoint.
**Example:** if your vLLM server is only serving Qwen3.6 for chat completions and isn't
also serving an embedding model, semantic retrieval and dedup quietly fall back to
text-only matching — you won't get an error, just quieter/weaker cross-linking between
pages than the "knowledge graph" features imply are possible.

---

## 3. Practical checklist for your specific PDF issue

Given the above, in order of likely impact:

1. **Settings → MinerU PDF Parser → enable it.** Even the cloud free tier will likely
   outperform bundled PDFium on anything beyond simple single-column text PDFs. If your
   documents are sensitive, set up the self-hosted local backend instead.
2. **Check the review queue**, not just the main wiki view — pages may have been
   generated but routed there instead of surfacing directly.
3. **Check `wiki/log.md` and the in-app Activity panel** for the specific PDF — if
   ingest threw `"Analysis stream failed"` or `"Generation stream failed"`, or logged a
   dropped-block warning, that confirms §2.2/§2.3 rather than a MinerU/extraction issue.
4. **Try a smaller/simpler PDF first** (single column, native text, no scanned images)
   to isolate whether the issue is extraction quality (§2.1) vs. generation-format
   compliance (§2.2) — if a simple PDF works fine and a complex one doesn't, that
   points squarely at extraction; if even a simple PDF fails, that points at the
   ingest-format/model-compliance issue instead.
5. **Increase `max_tokens`/output length on your vLLM generation config if configurable
   in the app's model settings**, since long generation responses truncating mid-block
   (H2 in §2.2) is the one failure mode the developers say they can't fully prevent.
