# NVMe LLM-Wiki — README

A self-maintaining knowledge base over the NVMe Base Specification, built as an
[LLM-wiki](https://github.com/SamurAIGPT/llm-wiki-agent)-pattern application on
[`deepagents`](https://github.com/langchain-ai/deepagents), running against a
self-hosted vLLM server (Qwen3.6-FP8) with local Ollama embeddings.

## Files

| File | Role |
|---|---|
| `wiki_agent_deepagents.py` | The agent + all tools + CLI (ingest/query/lint/etc.) |
| `nvme-wiki-schema.md` | The wiki's spec/taxonomy/filing rules — loaded as the system prompt |
| `build_graph.py` | Generates the interactive HTML graph view (also runs automatically after ingest) |

## Core idea

Rather than RAG-style retrieval that re-derives everything from raw sources on
every question, the agent incrementally builds and maintains a **persistent
wiki** of interlinked markdown pages — a structured layer between you and the
raw spec. Knowledge compounds: cross-references, contradictions, and synthesis
accumulate across ingests instead of being rebuilt from scratch each time.

## How we got here (planning trail)

The pipeline went through several rounds of deliberate design, each addressing
a gap surfaced by actually trying to use the previous version:

1. **OKF conformance.** The wiki format follows the
   [Open Knowledge Format spec](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) —
   every page is a "concept" with required `type` frontmatter, `index.md`/
   `log.md` have defined formats, and links are bundle-relative. OKF only
   asserts *that* two concepts relate (via prose links), not *how* — so
   everything below `relates_to` is a house extension on top of OKF core,
   not part of the spec itself.

2. **The correlation weak point.** Early on it became clear the hardest
   problem wasn't ingestion — it was *finding relationships the source text
   doesn't spell out explicitly*. This produced three deliberate mechanisms:
   - **`aliases`** — every concept can declare alternate names/acronyms,
     aggregated into a registry the pipeline re-scans on every ingest.
   - **Ambiguity protection** — an alias mapping to 2+ concepts is never
     auto-resolved; it's surfaced and left for a human or better context.
   - **`confidence` on every `relates_to` entry** — computed *deterministically*
     (regex-detected citation language → `explicit`; alias-registry hit with
     no citation → `alias-matched`; neither → `llm-inferred`), never
     self-reported by the model. This improves *precision* (labels you can
     trust), not *recall* (it can't force the model to notice a relationship
     it wasn't primed to look for) — that distinction is documented
     explicitly in the schema and this README, not glossed over.
   - **Coverage tooling** — `audit_alias_coverage` finds concepts with zero
     declared aliases (invisible to the pre-scan), `fix_coverage` proposes
     aliases for them (checking for new ambiguity before writing), and a
     `llm-inferred` confidence-share metric is tracked over time
     (`_metrics.jsonl`) as a trend signal, not proof of any specific miss.

3. **The graph view.** An interactive HTML graph (vis-network) lets you
   search a node, focus on it, and filter to N degrees of separation —
   edges are directional, labeled with the relationship `kind`, and clicking
   one shows the exact `description` plus its `confidence` tier. Regenerated
   automatically at the end of every ingest.

4. **The architecture rewrite.** A planning doc (see `Planning Doc: LLM Wiki
   on deepagents` in project history) audited two reference implementations —
   [`nashsu/llm_wiki`](https://github.com/nashsu/llm_wiki) and
   [`lucasastorian/llmwiki`](https://github.com/lucasastorian/llmwiki) — and
   specified which of their *logical* patterns (not their GUI/platform code)
   were worth porting onto `deepagents`. That's the shape of the current
   `wiki_agent_deepagents.py`.

## What was ported from the planning doc, and what it maps to

| Capability | Ported from | Implementation here |
|---|---|---|
| Path-safe writes + deterministic index | nashsu `ingest.ts` | `write_wiki_page` — host-enforced, not LLM-trusted |
| Ingest caching | nashsu `ingest-cache.ts` | `wiki/_ingest-cache.json`, checked before any agent call |
| Graph-aware + embedding retrieval | nashsu `graph-relevance.ts` | `search_wiki` — keyword + link-graph-walk + Ollama embeddings |
| Context budgeting | nashsu `context-budget.ts` | `chunk_sections()` sized from `QWEN_CONTEXT_TOKENS` |
| Structural vs. semantic lint | nashsu `lint-structural-core.ts` / `lint.ts` | `lint_wiki()` dispatcher — structural is tool-callable, semantic is the top-level `lint` command's LLM pass only |
| Wikilink backfill | nashsu `enrich-wikilinks.ts` | `find_missing_links` (deterministic) + `enrich_links` (LLM pass) |
| Dedup | nashsu `dedup.ts` | `check_duplicates` (pre-write) + `dedup_sweep` (wiki-wide) — both warn only, never auto-merge |
| Tiered extraction | nashsu `mineru.ts` / lucas `ocr.py` | `extract_document` — fast (JSON/text) tier only; PDF and "accurate" explicitly refused and logged, never silently degraded |
| Guide-as-prompt | lucas `guide.py` | `nvme-wiki-schema.md` loaded as `system_prompt`, swappable per project (`WIKI_SPEC_PATH`) |
| vLLM/Qwen wiring | new | `ChatOpenAI(base_url=...)` into `create_deep_agent` |
| Cascade-aware deletion | new | `delete_concept` — reference-counted, refuses if still linked unless forced |
| Per-entity sub-agent isolation | new | `entity-ingest-agent`, one subagent call per command/entity via deepagents' `task` tool |
| Pre-flight cache in harness | new | Hash-checked before any agent spawn — zero tokens for unchanged input |

Deliberately **not** ported: Tauri/React/Next.js shells, Chrome-extension
clipping, quiz features, MCP server wrapper layers — delivery mechanism and
unrelated features, not wiki-building logic.

## Known, open limitations (stated plainly, not hidden)

- **Recall, not just precision.** Two truly related concepts with *no shared
  vocabulary at all* still depend entirely on the model's own domain
  knowledge — no deterministic mechanism catches that class of miss.
- **`write_wiki_page` enforcement is prompt-level, not hard-sandboxed.**
  deepagents' generic `write_file`/`edit_file` are still present alongside
  it by default; fully closing this needs building off deepagents'
  middleware directly instead of `create_deep_agent()`'s default stack.
- **PDF extraction is out of scope for now** — the pipeline is JSON/text-source
  only until an OCR/layout-aware extractor is deliberately wired in.
- **The citation-language regex is conservative, not exhaustive** — an
  unusually phrased spec cross-reference can fall through to `alias-matched`
  confidence rather than `explicit`. This is the safer failure direction
  (under- rather than over-confident), but worth knowing about.

## Quick start

```bash
pip install deepagents langchain-openai pyyaml requests

# Model server
vllm serve <qwen3.6-fp8-model> --served-model-name qwen3.6-fp8 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3

# Embeddings (optional but recommended for search_wiki)
ollama pull embeddinggemma && ollama serve

python wiki_agent_deepagents.py ingest-large full_nvme_spec.json
python wiki_agent_deepagents.py query "what does Abort return if the limit is exceeded?"
python wiki_agent_deepagents.py lint
```

Full command list, config env vars, and per-function detail are documented in
the module docstring at the top of `wiki_agent_deepagents.py`.
