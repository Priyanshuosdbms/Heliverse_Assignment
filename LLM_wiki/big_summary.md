# LLM-Wiki
### Evaluation, Architecture, and a Path to an In-House Implementation

**Prepared for:** Department Director
**Prepared by:** [Your Name]
**Date:** 13 August 2026
**Status:** For Review — Detailed Technical Report

---

## 1. Background

Over the past several days I have been experimenting with the "LLM-Wiki" pattern, a way of using large language models to build and continuously maintain a structured, interlinked knowledge base rather than repeatedly re-deriving answers from raw source documents. I first came across the idea on Andrej Karpathy's GitHub page, where he published an "idea file" (`llm-wiki.md`) describing the pattern and inviting engineers to adapt it with their own coding agents.

The core insight is simple: most people's experience with LLMs and documents looks like Retrieval-Augmented Generation (RAG) — upload files, retrieve relevant chunks at query time, and generate an answer. This works, but the LLM re-discovers the same knowledge from scratch on every question; nothing accumulates. An LLM-Wiki instead has the model compile what it learns into durable, cross-linked pages once, and update those pages as new material arrives — so the knowledge base compounds over time instead of decaying like a hand-maintained wiki typically does.

I used our NVMe technical documentation set — the NVMe base specification, the OCP (Open Compute Project) datacenter NVMe SSD specification, and the NVMe-MI (Management Interface) specification — as the test corpus, since it is dense, cross-referential, and representative of the kind of internal documentation this approach is meant to help with.

## 2. Advantages of LLM-Wiki

### 2.1 LLM-Wiki vs. RAG for NVMe-Style Documentation

For a corpus like ours — base spec, OCP spec, and NVMe-MI, which together are large but not unbounded, and which reference each other constantly (e.g., OCP extends behaviors defined in the base spec; NVMe-MI commands tie back to management structures defined elsewhere) — the wiki approach has a structural advantage: relationships between documents are captured once, explicitly, and reused on every future query, rather than being re-inferred (or missed) on each retrieval.

| Aspect | RAG | LLM-Wiki |
|---|---|---|
| Knowledge accumulation | Re-derived from raw chunks on every query; no memory across sessions. | Compiled once into durable pages; each new ingest updates and cross-links existing knowledge. |
| Cross-document relationships | Chunks are retrieved independently; relationships between e.g. base spec and OCP spec are not explicit. | Relationships are written directly into wiki pages as links and notes — dependency chains are explicit. |
| Infrastructure | Requires a vector database / embedding pipeline and retrieval tuning. | Plain files (markdown/JSON); no vector DB needed for small-to-mid corpora (roughly under ~100 documents / ~80k tokens). |
| Best fit | Very large, fast-changing, loosely related document sets where targeted retrieval beats reading everything. | Bounded, high-value, cross-referential corpora — like a spec family — where relationships matter more than raw recall. |
| Answer quality on repeat questions | Same cost and same risk of drift each time. | Improves over time; the answer is already synthesized and curated. |
| Failure mode | Irrelevant or fragmented chunks retrieved; "lost in the middle" effects. | Upfront compilation cost; risk of staleness if sources change and are not re-ingested. |

Net for our use case: given three tightly related specs rather than thousands of loosely related documents, LLM-Wiki is the better fit. RAG remains preferable once a corpus becomes very large or heterogeneous, or when the priority is finding a needle in a haystack rather than understanding how a bounded set of documents relate to one another.

### 2.2 Version Control

**LLM-Wiki:** the wiki itself is plain markdown/JSON, so it is a first-class, git-native artifact. Every compiled page can be diffed, reviewed in a pull request, blamed to a source ingest, and branched — the knowledge base evolves under the same version control discipline as code.

**RAG:** version control naturally applies to the raw source documents, but not to what the system actually produces — embeddings and vector indexes are not human-diffable, must be regenerated whenever sources change, and the retrieval output itself is never a stored, versioned artifact.

### 2.3 Readability and Dependency Identification

Because LLM-Wiki pages are interlinked prose written for both humans and agents, dependencies between specs become navigable rather than implicit. For example, a page on an NVMe-MI command can link directly to the base-spec management structure it relies on, and to the OCP requirement that extends it — giving a reader (or another agent) a readable dependency chain instead of a set of disconnected, retrieved fragments, which is what RAG typically returns without additional engineering on top.

## 3. OKF (Open Knowledge Format) and Its Importance

OKF is an open, vendor-neutral specification published by the Google Cloud Data team (v0.1, 12 June 2026) that formalizes the LLM-Wiki pattern into a portable, interoperable format. It is intentionally minimal: a "bundle" is a directory of markdown files with YAML frontmatter, and the only mandatory field on any concept page is `type`. Everything else — what types exist, what additional fields to use, how the body is structured — is left to the producer. There is no required SDK or runtime.

**Why it matters for us:**

- **Portability:** an OKF-conformant wiki is not locked to Claude, a specific vector store, or a specific viewer — any OKF-aware tool can read, browse, or serve it.
- **Producer/consumer independence:** a bundle we hand-author can be consumed by an agent; a bundle an agent writes can be reviewed by a human in a visualizer; a bundle one LLM writes can be queried by another.
- **No lock-in:** the format is "just markdown, just files, just YAML frontmatter" — shippable in git, readable on GitHub, indexable by any search tool.
- **Forward compatibility:** OKF v0.2 adds a provenance layer (`generated`, `verified`, a status lifecycle of draft/stable/deprecated, `stale_after` dates, and a `sources` field) — directly relevant for a regulated, spec-driven domain like NVMe where knowing when a page was last validated against the spec matters.

Adopting OKF as the bundle format for our wiki means we build once against a public, evolving standard rather than a bespoke internal schema — lowering the cost of adopting future tooling (visualizers, catalog integrations, other agent frameworks) built against the same spec.

## 4. Core Principles, Architecture, and Integration with OKF

### 4.1 Core Principles

- **Compile-first, not just Q&A** — the agent's job is not to answer and forget, but to write its conclusions into the wiki.
- **Writeback is mandatory** — every decision or synthesized fact goes back into a wiki page, not just into a chat transcript.
- **Wiki before RAG** — below a practical size threshold (roughly under 100 documents or ~80k tokens), the agent reads the wiki directly; a vector store is an optimization to add later if recall measurably degrades, not a starting requirement.
- **The paradigm outranks the tool** — Obsidian, a particular editor, or a particular viewer is replaceable; the durable engine is "LLM + filesystem + markdown."
- **The knowledge outranks the code it produces** — the wiki of curated decisions and cross-references is the asset; anything generated from it (code, reports) is a downstream, regenerable artifact.

### 4.2 Architecture

Across the pattern's implementations, the same three-layer architecture recurs:

| Layer | Contents | Mutability |
|---|---|---|
| Raw sources | Original documents — in our case the NVMe base spec, OCP spec, and NVMe-MI spec, pre-processed into sectioned JSON. | Immutable — the agent reads from here but never edits it. |
| Wiki | LLM-generated, interlinked markdown pages: an index, a glossary, per-topic concept pages, and an activity log. | Owned and maintained by the agent; the primary artifact. |
| Schema / config | Rules the agent follows — page-creation thresholds, naming conventions, linking conventions, when to log vs. create a page. | Human-authored, occasionally revised. |

### 4.3 Integration with OKF

Concretely, our wiki layer is structured as an OKF bundle: each concept page is a markdown file with YAML frontmatter carrying at minimum a `type` (e.g., spec-section, command, dependency-note, glossary-term), plus `title`, `description`, `resource`, `tags`, and a `timestamp`. The reserved filenames (`index.md` as the master catalog, `log.md` as the chronological activity record) follow the OKF convention, and cross-references between pages use bundle-relative links so they survive file moves. This means the wiki our agent builds for NVMe documentation is, from day one, consumable by any other OKF-aware tool — a visualizer, a catalog, or a different agent — without a bespoke integration.

## 5. Why JSON (Sectionally Divided) Instead of Raw PDFs

Rather than feeding raw PDFs directly, I pre-processed the NVMe base spec into JSON, split along the specification's own section and subsection structure. This was a deliberate choice, with benefits beyond "JSON is easy to handle":

- **Structure preservation:** the spec's native section/subsection hierarchy becomes explicit, addressable keys rather than something the model has to re-infer from page layout.
- **Cleaner input:** avoids PDF-parsing artifacts — broken tables, footnotes, headers/footers, and page-break noise that commonly corrupt LLM ingestion of technical PDFs.
- **Incremental, targeted feeding:** individual sections (e.g., the Abort command) can be fed to the model on their own, keeping context windows small and outputs focused, instead of dumping the entire spec at once.
- **Programmatic diffing:** when a new spec revision is released, sectioned JSON can be diffed section-by-section to see exactly what changed, which maps directly onto which wiki pages need updating.
- **Coverage validation:** it is straightforward to programmatically check that every section in the JSON has a corresponding wiki page, catching gaps automatically.
- **Tool-agnostic intermediate format:** JSON is easy to generate from PDF once, and easy to feed into any framework or agent afterward — it decouples "parsing the spec" from "building the wiki."

## 6. Experiment Walkthrough

- Fed the entire NVMe base specification, as sectioned JSON, to Claude.
- Claude proposed a page/section structure to follow for compiling the base spec into a wiki.
- Adopted that structure and seeded the pipeline using the Abort command as the pilot section.
- Claude produced a first set of wiki pages from that pilot, validating the ingest → compile → wiki-page flow end-to-end.

This pilot confirmed the core mechanics work on real spec content: the model can take a structured section, propose a sensible wiki organization, and produce readable, cross-linkable pages from it — the remaining work is turning this manual, single-command pilot into a repeatable, agent-driven pipeline (Section 8).

## 7. Survey of Open-Source LLM-Wiki Implementations

Before building our own pipeline, we surveyed existing open-source implementations of the pattern to understand what design choices others have already made, and which of those choices are worth carrying into our implementation.

### 7.1 Nash Su's LLM-Wiki (nashsu/llm_wiki)

A cross-platform desktop application implementing Karpathy's pattern faithfully, with substantial extensions. It follows the same three-layer architecture (raw sources → wiki → schema) as a packaged, standalone product rather than an agent skill.

### 7.2 Lucas Astorian's LLM-Wiki (lucasastorian/llmwiki)

An open-source implementation that connects to a Claude account via MCP and runs a nightly "Claude Routine" to autonomously synthesize captured documents, notes, and web clippings into a persistent wiki. Its clipper captures highlights and margin notes alongside the source, so the wiki records not just what was read but what the reader thought about it.

### 7.3 LangChain's LLM-Wiki — OpenWiki (langchain-ai/openwiki)

A CLI tool purpose-built for codebases rather than general documents: it scans a repository and generates AGENTS.md/CLAUDE.md-style documentation describing architecture, modules, and conventions, then regenerates it automatically via a scheduled GitHub Actions workflow whenever the code changes, so an agent is always reading current documentation instead of stale notes.

### 7.4 Obsidian & Hermes Agent's LLM-Wiki

Hermes Agent (Nous Research) provides an Obsidian-integrated skill/plugin that turns a vault into an agent workspace: the agent reads, writes, and searches notes directly, and can build a "smart graph" of semantic relationships (shared topics, prerequisites, contradictions) that goes beyond Obsidian's native explicit wikilinks. The recommended pattern keeps judgment human-owned while the agent owns coverage and maintenance.

| Implementation | Approach | Key Feature |
|---|---|---|
| Nash Su — nashsu/llm_wiki | Standalone cross-platform desktop app | Faithful 3-layer architecture (raw → wiki → schema) with production-grade extensions |
| Lucas Astorian — lucasastorian/llmwiki | MCP-connected, Claude-account based | Nightly autonomous "Claude Routine" compilation; captures reader's own notes alongside sources |
| LangChain — openwiki | CLI for codebases, CI-integrated | Scheduled auto-regeneration via GitHub Actions keeps docs current with code |
| Obsidian + Hermes Agent | Obsidian vault as agent workspace | Agent-built semantic "smart graph" beyond explicit wikilinks; human-judgment / agent-coverage split |

## 8. Next Steps: Proposed Roadmap

We would like to move ahead with our own implementation, built on **deepagents** (LangChain's open-source agent harness providing planning, sub-agent delegation, a virtual filesystem, and persistent memory), while deliberately incorporating the strongest design choices observed in the surveyed repos above.

### Design elements we plan to adopt

- **From OpenWiki:** scheduled, automatic regeneration whenever source specs are updated — keeping the wiki current without manual re-runs.
- **From Lucas's LLM-Wiki:** an MCP-native, routine-style compilation step rather than a purely on-demand chat flow.
- **From Nash Su's LLM-Wiki:** a strictly enforced three-layer separation (immutable raw / agent-owned wiki / human-owned schema).
- **From Hermes Agent:** a clear human-judgment vs. agent-coverage split, with human review gating anything written back to the wiki.

### Proposed phases

| Phase | Scope | Output |
|---|---|---|
| Phase 1 — Pilot | NVMe base spec only, sectioned JSON, single deepagents pipeline | OKF-conformant wiki bundle for the base spec, human-reviewed |
| Phase 2 — Expand corpus | Add OCP and NVMe-MI specs; build cross-spec linking | Interlinked multi-spec wiki with explicit dependency chains |
| Phase 3 — Standardize | Formalize schema, adopt OKF v0.2 provenance fields (status, stale_after, sources) | Auditable, versioned wiki bundle usable by any OKF-aware tool |
| Phase 4 — Automate | deepagents-driven scheduled re-ingest on spec revisions, review-gated writeback | Self-updating internal knowledge base with human sign-off |

### Asks for the department

- Approval to proceed with a Phase 1–2 pilot using the existing NVMe documentation set.
- A dedicated git repository to host the wiki bundle so it benefits from standard version control and review workflows.
- Time allocation to extend the pilot to OCP and NVMe-MI, and to formalize the schema against OKF v0.2.

---

*Happy to walk through any of the above in more detail, or to demo the pilot wiki pages generated from the Abort command.*