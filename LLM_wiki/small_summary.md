# LLM-Wiki
### Executive Summary — Evaluation and Proposed Next Steps

**Prepared for:** Department Director
**Prepared by:** [Your Name]
**Date:** 13 August 2026

---

## Background

I have spent the last several days evaluating "LLM-Wiki" — a pattern, originating from a GitHub idea file by Andrej Karpathy, in which an LLM compiles what it learns from source documents into a durable, cross-linked wiki once, and keeps that wiki current — rather than re-deriving answers from scratch on every query, which is how RAG works. I tested it against our NVMe documentation (base spec, OCP spec, NVMe-MI spec).

## Why LLM-Wiki Over RAG for Our NVMe Docs

| Aspect | RAG | LLM-Wiki |
|---|---|---|
| Knowledge accumulation | Re-derived on every query | Compiled once, improves over time |
| Cross-doc relationships (base spec ↔ OCP ↔ NVMe-MI) | Implicit, easy to miss | Explicit links, readable dependency chains |
| Infrastructure | Vector DB required | Plain files; no vector DB for a bounded corpus like ours |
| Version control | Applies to raw sources only | The wiki itself is a git-diffable, versioned artifact |

For a bounded, cross-referential corpus like our spec family, LLM-Wiki is the better fit; RAG remains preferable for very large, loosely related document sets.

## OKF (Open Knowledge Format)

Google Cloud published OKF (v0.1, June 2026) to formalize the LLM-Wiki pattern into a vendor-neutral spec: markdown files with YAML frontmatter, one mandatory field (`type`), no SDK or proprietary runtime required. Building to OKF means our wiki is portable across tools and future-proofed against lock-in — which is why we're adopting it as our bundle format.

## Core Principles & Architecture

- **Compile-first:** the agent writes conclusions into the wiki, not just into a chat reply.
- **Three layers:** Raw sources (immutable) → Wiki (LLM-owned, interlinked) → Schema (human-owned rules).
- **Wiki before RAG:** for a corpus this size, direct reads beat vector search.
- **OKF integration:** every wiki page is an OKF concept (typed, cross-linked), so it's consumable by any OKF-aware tool.

## Why Sectioned JSON Instead of Raw PDFs

We pre-process specs into JSON split along the spec's own section structure. This preserves structure, avoids PDF-parsing noise, allows targeted feeding section-by-section, supports automated diffing between spec revisions, and lets us validate wiki coverage programmatically.

## Pilot Result

We fed the full NVMe base spec (as sectioned JSON) to Claude, which proposed a wiki structure. We used that structure, piloted it on the Abort command, and Claude generated the first set of wiki pages — validating the ingest → compile → wiki pipeline end to end.

## Open-Source Landscape

| Implementation | Approach | Key Feature |
|---|---|---|
| Nash Su — llm_wiki | Standalone desktop app | Faithful 3-layer architecture, production extensions |
| Lucas Astorian — llmwiki | MCP + Claude account | Nightly autonomous compile "Routine"; captures reader's own notes |
| LangChain — openwiki | CLI, CI-integrated | Auto-regenerates docs via GitHub Actions on code change |
| Obsidian + Hermes Agent | Obsidian vault as agent workspace | Agent-built semantic graph; human-judgment/agent-coverage split |

## Proposed Roadmap

We propose building our own pipeline on **deepagents** (LangChain's agent harness), adopting: scheduled auto-regeneration (from OpenWiki), routine-style MCP compilation (from Lucas's implementation), strict layer separation (from Nash Su's), and a human-review gate over agent writeback (from Hermes Agent).

| Phase | Scope |
|---|---|
| 1. Pilot | NVMe base spec only, deepagents pipeline, OKF-bundle output |
| 2. Expand | Add OCP + NVMe-MI, cross-spec linking |
| 3. Standardize | Adopt OKF v0.2 provenance fields (status, stale_after, sources) |
| 4. Automate | Scheduled re-ingest on spec revisions, review-gated writeback |

### Asks

- Approval to run a Phase 1–2 pilot on the existing NVMe doc set.
- A dedicated git repository for the wiki bundle.
- Time allocation to extend to OCP/NVMe-MI and formalize the OKF schema.

---

*A detailed technical report with the full evaluation and architecture is available on request.*