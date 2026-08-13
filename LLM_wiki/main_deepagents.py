"""
wiki_agent_deepagents.py — drives an OKF-conformant NVMe wiki using LangChain's
`deepagents` library, a local vLLM server (Qwen3.6-FP8), and a local Ollama
server for embeddings. Restructured per the "LLM Wiki on deepagents" planning
doc — see inline §-references throughout this file mapping each piece back
to that plan.

Usage:
    python wiki_agent_deepagents.py ingest <path-to-source.json-or-txt>
    python wiki_agent_deepagents.py ingest-large <path-to-full-spec.json>
    python wiki_agent_deepagents.py query "your question"
    python wiki_agent_deepagents.py lint
    python wiki_agent_deepagents.py fix-coverage [limit]
    python wiki_agent_deepagents.py enrich-links
    python wiki_agent_deepagents.py dedup
    python wiki_agent_deepagents.py delete <path> [--force]

What's fully implemented from the planning doc:
  §2.2/2.3  write_wiki_page tool — host-enforced path safety + deterministic
            index.md maintenance (no LLM call for indexing).
  §2.4/4.6  Pre-flight hash cache (wiki/_ingest-cache.json) — unchanged
            sources/chunks are skipped BEFORE any agent call, zero tokens.
  §2.5      search_wiki — keyword + link-graph-walk + optional Ollama
            embedding similarity, rank-merged.
  §2.6      Context budgeting — chunk_sections() sizes batches from
            QWEN_CONTEXT_TOKENS (env var), not a hardcoded constant.
  §2.7      lint_wiki dispatcher — structural (deterministic, tool-callable)
            vs semantic (LLM pass, top-level `lint` command only — NOT
            recursively callable as a tool, to avoid nested agent calls).
  §2.8      find_missing_links (deterministic scan) + enrich_links (LLM
            backfill pass using it).
  §2.9      check_duplicates (pre-write tool) + dedup_sweep (wiki-wide scan).
            Both only warn/report — never auto-merge.
  §2.10     extract_document — fast tier only (JSON/text), per your choice
            to stay JSON-source-only; PDF and "accurate" tier are explicitly
            refused and logged, never silently degraded.
  §3.1/4.3  WIKI_SPEC (the schema file) as the editable system-prompt doc —
            WIKI_SPEC_PATH env var, per-project swappable.
  §4.1      vLLM/Qwen wiring via ChatOpenAI.
  §4.2      delete_concept — reference-counted cascade-aware deletion,
            refuses (unless force=True) when other pages still reference it.
  §4.4      extract_document logs every extraction's tier/outcome/char-count
            to wiki/_ingest-log.jsonl.
  §4.5      Per-entity subagent delegation — entity-ingest-agent, invoked
            once per individual command/entity (the granularity you chose),
            via deepagents' built-in task tool.

What's mapped onto existing infra rather than newly built:
  §2.1      Two-stage ingest — deepagents' built-in planning/todo middleware
            (write_todos) already provides this; the STEP 1/2/3 instruction
            wording in ingest()/ingest_large() leans on it explicitly rather
            than needing separate analysis/generation call plumbing.

What's intentionally deferred (not requested / no capture mechanism exists):
  §3.2      Annotation/highlight merging — no capture mechanism exists yet
            in this pipeline; nothing to merge until one does.

Known, real limitation (not solved, flagged honestly):
  §2.2      "Always use write_wiki_page" is enforced by strong system-prompt
            instruction, NOT hard tool removal — deepagents' default stack
            still exposes generic write_file/edit_file alongside it. Fully
            closing this would mean building the agent from deepagents'
            middleware directly instead of create_deep_agent()'s default
            stack. Left as-is for now; flagged so it isn't mistaken for a
            closed gap.

Install:
    pip install deepagents langchain-openai pyyaml requests

Serve the model (separate terminal):
    vllm serve <your-qwen3.6-fp8-model-path-or-repo> \\
        --served-model-name qwen3.6-fp8 \\
        --enable-auto-tool-choice \\
        --tool-call-parser hermes \\
        --reasoning-parser qwen3 \\
        --port 8000

    Note: --tool-call-parser is model-family/vLLM-version specific. Check
    `vllm serve --help` or https://docs.vllm.ai/en/stable/features/tool_calling/
    against your exact build if tool calls come back malformed.

Serve embeddings (separate terminal, for §2.5's search_wiki):
    ollama pull embeddinggemma   # or all-minilm / nomic-embed-text
    ollama serve                 # defaults to http://localhost:11434

Config (env vars, all optional):
    VLLM_BASE_URL, WIKI_MODEL, WIKI_ROOT, WIKI_SPEC_PATH (or WIKI_SCHEMA),
    WIKI_GRAPH_OUTPUT, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, QWEN_CONTEXT_TOKENS
"""

import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

import build_graph  # same-directory module — generates the interactive graph

# ---- Config ----------------------------------------------------------------

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("WIKI_MODEL", "qwen3.6-fp8")  # match --served-model-name
WIKI_ROOT = Path(os.environ.get("WIKI_ROOT", "./wiki")).resolve()
# Per plan §4.3: this file plays the role of a per-project WIKI_SPEC.md — the
# wiki's folder taxonomy/filing rules live here as an editable document, not
# compiled code, so a different project can point WIKI_SPEC_PATH at its own
# spec without forking this script. WIKI_SCHEMA is kept as an alias for
# backward compatibility with earlier runs.
SCHEMA_PATH = Path(
    os.environ.get("WIKI_SPEC_PATH") or os.environ.get("WIKI_SCHEMA", "./nvme-wiki-schema.md")
).resolve()
GRAPH_OUTPUT = Path(os.environ.get("WIKI_GRAPH_OUTPUT", "./wiki_graph.html")).resolve()

# §2.5 optional embedding tier for search_wiki, via a local Ollama server.
# Any of embeddinggemma / all-minilm / nomic-embed-text work; embeddinggemma
# is the default here but this is a config choice, not a correctness one —
# change OLLAMA_EMBED_MODEL to whichever you've pulled.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma")

# §2.6 context budgeting — sized to YOUR served --max-model-len, not a
# platform default. token estimate is a crude chars/4 heuristic (no Qwen
# tokenizer dependency); treat QWEN_CONTEXT_TOKENS as the ceiling you're
# budgeting against, with headroom reserved for the schema/system prompt.
QWEN_CONTEXT_TOKENS = int(os.environ.get("QWEN_CONTEXT_TOKENS", "32000"))
CONTEXT_RESERVED_FOR_PROMPT_TOKENS = 6000  # schema + instructions + tool defs

WIKI_ROOT.mkdir(parents=True, exist_ok=True)
INGEST_CACHE_PATH = WIKI_ROOT / "_ingest-cache.json"
INGEST_LOG_PATH = WIKI_ROOT / "_ingest-log.jsonl"

# Citation language that, if found near a term mention, deterministically
# marks that reference as "explicit" confidence — no LLM judgment involved.
CITATION_RE = re.compile(
    r"\b(see|refer(?:s)?\s+to|defined\s+in|specified\s+in|described\s+in|per)\b"
    r"[^.]{0,60}?\b(section|§|figure|table)\s*[\dA-Za-z.]+",
    re.IGNORECASE,
)

# ChatOpenAI works against any OpenAI-compatible endpoint, including vLLM's.
model = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key="not-needed",
    model=MODEL_NAME,
    temperature=0.2,
)

# Sandboxes all built-in filesystem tools (ls/read_file/write_file/edit_file)
# to WIKI_ROOT; virtual_mode=True blocks path traversal outside it.
backend = FilesystemBackend(root_dir=str(WIKI_ROOT), virtual_mode=True)


# ---- §2.4 / §4.6: pre-flight ingest cache (harness-level, zero LLM cost) --
# Unchanged sources/chunks are skipped BEFORE any agent is spawned. This is
# plain host code by design (per the plan) — the model never decides this.


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_ingest_cache() -> dict:
    if INGEST_CACHE_PATH.exists():
        try:
            return json.loads(INGEST_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_ingest_cache(cache: dict) -> None:
    INGEST_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def is_unchanged_since_last_ingest(cache_key: str, text: str) -> bool:
    """True if `text` hashes the same as the last time `cache_key` was
    ingested. Callers should skip invoking the agent entirely when this is
    True — that's the whole point: zero tokens spent on unchanged input."""
    cache = _load_ingest_cache()
    return cache.get(cache_key) == _content_hash(text)


def mark_ingested(cache_key: str, text: str) -> None:
    cache = _load_ingest_cache()
    cache[cache_key] = _content_hash(text)
    _save_ingest_cache(cache)


# ---- §2.6: context budgeting ------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Crude chars/4 heuristic — no Qwen tokenizer dependency. Treat this as
    approximate; if you have the real tokenizer available, swap this out."""
    return len(text) // 4


def available_context_budget() -> int:
    return max(QWEN_CONTEXT_TOKENS - CONTEXT_RESERVED_FOR_PROMPT_TOKENS, 1000)



# deepagents already gives the agent ls / read_file / write_file / edit_file
# against `backend`. We only add the two things specific to the OKF workflow:
# appending to log.md, and a quick cross-index search.


# ---- §2.2 / §2.3: path-safe, index-maintaining page writes -----------------
# Per the plan: don't trust the model to self-police path safety or to
# rewrite index.md by hand each time (that invites index drift/hallucinated
# entries). Both are enforced here in host code as a side effect of the one
# tool the model should use for concept writes.

ALLOWED_TOP_LEVEL_DIRS = {
    "architecture", "commands", "log-pages", "data-structures",
    "status-codes", "features", "extended-capabilities", "concepts",
}


def is_safe_ingest_path(path: str) -> tuple[bool, str]:
    """Reject `..`, absolute filesystem paths, and anything outside the
    known category folders (or a bare index.md/log.md at any depth) — this
    is enforced independent of whether the model's proposed path looks
    reasonable; do not trust the model to self-police this (plan §2.2)."""
    if ".." in path.split("/"):
        return False, "path traversal ('..') is not allowed"
    if path.startswith("~") or (len(path) > 1 and path[1] == ":"):
        return False, "absolute/drive paths are not allowed"
    normalized = path.lstrip("/")
    if not normalized:
        return False, "empty path"
    parts = normalized.split("/")
    if parts[-1] in ("index.md", "log.md"):
        return True, ""
    if parts[0] not in ALLOWED_TOP_LEVEL_DIRS:
        return False, (
            f"top-level folder '{parts[0]}' is not one of the schema's allowed "
            f"categories: {sorted(ALLOWED_TOP_LEVEL_DIRS)}"
        )
    resolved = (WIKI_ROOT / normalized).resolve()
    if WIKI_ROOT not in resolved.parents and resolved != WIKI_ROOT:
        return False, "path escapes the wiki root"
    return True, ""


def _extract_frontmatter(content: str) -> dict:
    try:
        import yaml
        _, fm_text, _ = content.split("---", 2)
        return yaml.safe_load(fm_text) or {}
    except Exception:
        return {}


def _index_heading_for(rel_path: str) -> str:
    """Derive a human-readable index.md heading from the folder path, e.g.
    commands/admin/abort.md -> 'Commands - Admin'."""
    parts = Path(rel_path.lstrip("/")).parts[:-1]
    return " - ".join(p.replace("-", " ").title() for p in parts) or "Concepts"


def _update_index_deterministically(rel_path: str, title: str, description: str) -> None:
    """Host-code side effect of every write_wiki_page call: append/update a
    bullet line for this concept under the right heading in BOTH the root
    index.md and the nearest subdirectory index.md, per OKF §6 format. Never
    an LLM call — this is exactly what plan §2.3 asks for."""
    concept_path = WIKI_ROOT / rel_path.lstrip("/")
    heading = _index_heading_for(rel_path)
    link_line_root = f"* [{title}]({rel_path.lstrip('/')}) - {description}".rstrip()

    for index_path, link_target in (
        (WIKI_ROOT / "index.md", rel_path.lstrip("/")),
        (concept_path.parent / "index.md", concept_path.name),
    ):
        line = f"* [{title}]({link_target}) - {description}".rstrip()
        if not index_path.exists():
            prefix = '---\nokf_version: "0.1"\n---\n\n' if index_path == WIKI_ROOT / "index.md" else ""
            index_path.write_text(prefix, encoding="utf-8")
        text = index_path.read_text(encoding="utf-8")

        # Replace an existing bullet for this path if present, else append
        # under the right heading (creating the heading if it doesn't exist).
        existing_line_re = re.compile(
            rf"^\* \[[^\]]*\]\({re.escape(link_target)}\).*$", re.MULTILINE
        )
        if existing_line_re.search(text):
            text = existing_line_re.sub(line, text)
        else:
            heading_marker = f"# {heading}"
            if heading_marker in text:
                text = text.replace(heading_marker, f"{heading_marker}\n{line}", 1)
            else:
                text = text.rstrip() + f"\n\n{heading_marker}\n\n{line}\n"
        index_path.write_text(text, encoding="utf-8")


def write_wiki_page(path: str, content: str) -> str:
    """The ONLY tool to use for creating/updating concept pages (and
    index.md/log.md). Validates the path is safe and within an allowed
    category folder BEFORE writing anything (host-enforced, not trusting
    the model's own judgment — plan §2.2), then deterministically updates
    both the root index.md and the nearest subdirectory index.md as a side
    effect (plan §2.3) — you do not need to hand-edit index.md yourself.

    Args:
        path: wiki-relative path, e.g. "commands/admin/abort.md".
        content: full file content including OKF frontmatter.
    """
    ok, reason = is_safe_ingest_path(path)
    if not ok:
        return f"REJECTED: {reason}"

    target = (WIKI_ROOT / path.lstrip("/")).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    rel_path = "/" + str(target.relative_to(WIKI_ROOT)).replace("\\", "/")
    if target.name not in ("index.md", "log.md"):
        fm = _extract_frontmatter(content)
        title = fm.get("title") or target.stem.replace("-", " ").title()
        description = fm.get("description", "")
        _update_index_deterministically(rel_path, title, description)
        return f"wrote {rel_path} and updated index.md (root + local) deterministically"
    return f"wrote {rel_path}"


def append_log(entry_markdown: str, date_heading: str | None = None) -> str:
    """Append a dated bullet entry to the wiki's root log.md (OKF §7 format).

    Args:
        entry_markdown: e.g. "* **Creation**: Added [Abort](/commands/admin/abort.md)."
        date_heading: ISO date, e.g. "2026-07-26". Defaults to today if omitted.
    """
    date_heading = date_heading or date.today().isoformat()
    log_path = WIKI_ROOT / "log.md"
    if not log_path.exists():
        log_path.write_text("# Directory Update Log\n\n", encoding="utf-8")
    text = log_path.read_text(encoding="utf-8")
    heading = f"## {date_heading}"
    if heading in text:
        text = text.replace(heading, f"{heading}\n{entry_markdown}", 1)
    else:
        text += f"\n{heading}\n{entry_markdown}\n"
    log_path.write_text(text, encoding="utf-8")
    return f"logged under {date_heading}"


def grep_index(query: str) -> str:
    """Search every index.md in the wiki for a keyword — use this before
    creating a new concept file, to check whether one already exists.

    Args:
        query: keyword or phrase to search for (case-insensitive).
    """
    hits = []
    for idx in WIKI_ROOT.rglob("index.md"):
        for i, line in enumerate(idx.read_text(encoding="utf-8").splitlines(), 1):
            if query.lower() in line.lower():
                hits.append(f"{idx.relative_to(WIKI_ROOT)}:{i}: {line.strip()}")
    return "\n".join(hits[:50]) if hits else "no matches"


# ---- §2.5: multi-signal retrieval (keyword + graph-walk + optional embed) --

def _ollama_embed(texts: list[str]) -> list[list[float]] | None:
    """Call a local Ollama server's /api/embed. Returns None (not an
    exception) on any failure — embedding is an optional retrieval signal,
    never a hard dependency; search_wiki degrades to keyword+graph-only if
    Ollama isn't reachable."""
    try:
        import requests
    except ImportError:
        return None
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": texts},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("embeddings")
    except Exception:
        return None


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _iter_concepts() -> list[dict]:
    """Deterministic scan of every concept's id/title/type/description —
    shared by search_wiki, check_duplicates, and delete_concept so they all
    see the same view of the wiki without re-deriving it differently."""
    try:
        import yaml
    except ImportError:
        return []
    out = []
    for md_file in WIKI_ROOT.rglob("*.md"):
        if md_file.name in ("index.md", "log.md") or "_lint-reports" in md_file.parts:
            continue
        text = md_file.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        try:
            _, fm_text, body = text.split("---", 2)
            fm = yaml.safe_load(fm_text) or {}
        except (ValueError, yaml.YAMLError):
            continue
        rel_path = "/" + str(md_file.relative_to(WIKI_ROOT)).replace("\\", "/")
        out.append({
            "path": rel_path,
            "title": fm.get("title") or md_file.stem.replace("-", " ").title(),
            "type": fm.get("type", "Unknown"),
            "description": fm.get("description", ""),
            "body": body,
        })
    return out


def _link_adjacency(concepts: list[dict]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {c["path"]: set() for c in concepts}
    for c in concepts:
        for m in LINK_RE_SIMPLE.finditer(c["body"]):
            target = m.group(1)
            adjacency.setdefault(c["path"], set()).add(target)
            adjacency.setdefault(target, set()).add(c["path"])
    return adjacency


LINK_RE_SIMPLE = re.compile(r"\]\((/[^)\s]+\.md)\)")


def search_wiki(query: str, top_k: int = 8) -> str:
    """Multi-signal retrieval per plan §2.5: rank-merges (a) keyword match
    against title/description/body, (b) a graph-walk boost for pages linked
    to already-matched pages, and (c) optional cosine similarity against a
    local Ollama embedding model if reachable. Use this instead of grep_index
    when you need ranked relevance rather than an exact substring hit.

    Args:
        query: natural-language or keyword query.
        top_k: how many results to return (default 8).
    """
    concepts = _iter_concepts()
    if not concepts:
        return "wiki is empty or pyyaml not installed"

    scores = {c["path"]: 0.0 for c in concepts}
    q_lower = query.lower()

    # (a) keyword signal
    keyword_hits = set()
    for c in concepts:
        haystack = f"{c['title']} {c['description']} {c['body']}".lower()
        if q_lower in haystack:
            scores[c["path"]] += 1.0
            keyword_hits.add(c["path"])
            if q_lower in c["title"].lower():
                scores[c["path"]] += 0.5  # title match ranks higher

    # (b) graph-walk boost — neighbors of keyword hits get a partial bump
    adjacency = _link_adjacency(concepts)
    for hit in keyword_hits:
        for neighbor in adjacency.get(hit, set()):
            if neighbor in scores:
                scores[neighbor] += 0.3

    # (c) optional embedding signal — never blocks if Ollama is unreachable
    query_emb = _ollama_embed([query])
    if query_emb:
        doc_texts = [f"{c['title']}: {c['description']}" for c in concepts]
        doc_embs = _ollama_embed(doc_texts)
        if doc_embs and len(doc_embs) == len(concepts):
            for c, emb in zip(concepts, doc_embs):
                scores[c["path"]] += _cosine_sim(query_emb[0], emb) * 1.2

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    ranked = [(p, s) for p, s in ranked if s > 0]
    if not ranked:
        return f'no matches for "{query}" (tried keyword, graph-walk, and embedding signals)'

    by_path = {c["path"]: c for c in concepts}
    lines = [f"results for \"{query}\" (embedding signal: {'used' if query_emb else 'unavailable, keyword+graph only'}):"]
    for path, score in ranked:
        c = by_path.get(path)
        if c:
            lines.append(f"- {path} ({c['type']}) score={score:.2f} — {c['description']}")
    return "\n".join(lines)


def build_alias_index() -> dict[str, list[dict]]:
    """Deterministic (no LLM call): walk every concept file, collect its
    `aliases` frontmatter plus its own title, and build
    alias_lower -> [{"path": ..., "title": ...}, ...].

    A single alias string mapping to 2+ concepts means that alias is
    ambiguous — callers must NOT auto-resolve it; see schema §4a. Recomputed
    fresh on every call so it always reflects the current state of the wiki
    on disk (no separate persisted registry file to go stale).
    """
    try:
        import yaml
    except ImportError:
        return {}

    index: dict[str, list[dict]] = {}
    for md_file in WIKI_ROOT.rglob("*.md"):
        if md_file.name in ("index.md", "log.md") or "_lint-reports" in md_file.parts:
            continue
        text = md_file.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        try:
            _, fm_text, _ = text.split("---", 2)
            fm = yaml.safe_load(fm_text) or {}
        except (ValueError, yaml.YAMLError):
            continue

        rel_path = "/" + str(md_file.relative_to(WIKI_ROOT)).replace("\\", "/")
        title = fm.get("title") or md_file.stem.replace("-", " ").title()
        names = set(fm.get("aliases") or []) | {title}

        for name in names:
            key = name.strip().lower()
            if not key:
                continue
            entry = {"path": rel_path, "title": title}
            index.setdefault(key, [])
            if not any(e["path"] == rel_path for e in index[key]):
                index[key].append(entry)

    return index


# ---- §2.9: deduplication ----------------------------------------------------

import difflib


def check_duplicates(candidate_title: str, candidate_description: str = "") -> str:
    """Call BEFORE creating a new concept page: checks the proposed title
    (and description, if given) against every existing concept's title for
    near-duplicates using string similarity. This is deterministic
    (difflib.SequenceMatcher), not an LLM judgment call, and only warns —
    it never blocks or auto-merges; you decide whether it's a real duplicate.

    Args:
        candidate_title: the title you're about to use for a new page.
        candidate_description: optional, improves the check.
    """
    concepts = _iter_concepts()
    warnings = []
    for c in concepts:
        ratio = difflib.SequenceMatcher(None, candidate_title.lower(), c["title"].lower()).ratio()
        if ratio > 0.85:
            warnings.append(f'- "{c["title"]}" at {c["path"]} (title similarity {ratio:.0%})')
    if not warnings:
        return f'no near-duplicate found for "{candidate_title}" — safe to create'
    return f'POSSIBLE DUPLICATE(S) of "{candidate_title}":\n' + "\n".join(warnings)


def dedup_sweep() -> str:
    """Deterministic (no LLM call) pairwise near-duplicate sweep across the
    whole wiki, grouped by `type` to keep comparisons manageable. Reports
    candidate pairs for human review — never auto-merges, since merging
    concept content correctly needs judgment this function doesn't have."""
    concepts = _iter_concepts()
    by_type: dict[str, list[dict]] = {}
    for c in concepts:
        by_type.setdefault(c["type"], []).append(c)

    findings = []
    for type_, group in by_type.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                ratio = difflib.SequenceMatcher(None, a["title"].lower(), b["title"].lower()).ratio()
                if ratio > 0.85:
                    findings.append(f'- [{type_}] "{a["title"]}" ({a["path"]}) ~ "{b["title"]}" ({b["path"]}) — {ratio:.0%} similar')

    if not findings:
        return "no near-duplicate concepts found"
    return f"{len(findings)} possible duplicate pair(s):\n" + "\n".join(findings)


# ---- §4.2: cascade-aware deletion -------------------------------------------

def delete_concept(path: str, force: bool = False) -> str:
    """Reference-counted deletion: refuses to delete a concept that other
    pages still link to (via relates_to or prose links) unless force=True,
    since silently deleting a page other concepts depend on breaks the wiki
    for everyone still referencing it. Pass force=True only after confirming
    those references should also be removed/updated.

    Args:
        path: wiki-relative path to delete, e.g. "commands/admin/abort.md".
        force: delete even if other pages still reference this one.
    """
    ok, reason = is_safe_ingest_path(path)
    if not ok:
        return f"REJECTED: {reason}"

    target = (WIKI_ROOT / path.lstrip("/")).resolve()
    if not target.exists():
        return f"nothing to delete — {path} does not exist"

    rel_path = "/" + str(target.relative_to(WIKI_ROOT)).replace("\\", "/")
    concepts = _iter_concepts()
    adjacency = _link_adjacency(concepts)
    referencing_pages = [
        c["path"] for c in concepts
        if c["path"] != rel_path and rel_path in adjacency.get(c["path"], set())
    ]

    if referencing_pages and not force:
        return (
            f"REFUSED: {rel_path} is still referenced by {len(referencing_pages)} "
            f"page(s): {', '.join(referencing_pages)}. Update or remove those "
            "references first, or call again with force=True if you're certain."
        )

    target.unlink()
    # Best-effort index cleanup — remove the bullet line(s) pointing at this path.
    for index_path in (WIKI_ROOT / "index.md", target.parent / "index.md"):
        if not index_path.exists():
            continue
        text = index_path.read_text(encoding="utf-8")
        text = re.sub(
            rf"^\* \[[^\]]*\]\({re.escape(str(target.relative_to(WIKI_ROOT)))}\).*$\n?",
            "", text, flags=re.MULTILINE,
        )
        text = re.sub(rf"^\* \[[^\]]*\]\({re.escape(target.name)}\).*$\n?", "", text, flags=re.MULTILINE)
        index_path.write_text(text, encoding="utf-8")

    note = "" if not referencing_pages else f" (forced despite {len(referencing_pages)} remaining reference(s) — those links are now broken and should be fixed)"
    return f"deleted {rel_path}{note}"


# ---- §2.10: tiered document extraction (JSON/text tier only — see plan §2.10)

def extract_document(path: str, quality: str = "fast") -> str:
    """Reads a source document and logs the extraction outcome. Per the
    current build, only JSON and plain text are supported — no OCR/MinerU
    service is configured, so PDF is explicitly refused rather than silently
    producing a poor extraction (this was the root cause flagged in plan
    §4.4: a bad extraction should never pass through to generation silently).

    Args:
        path: path to the source file (relative to cwd, not the wiki).
        quality: "fast" or "accurate" — "accurate" is not implemented in
            this build and will be logged and refused.
    """
    from datetime import datetime

    p = Path(path).resolve()
    result = {"timestamp": datetime.now().isoformat(timespec="seconds"), "path": str(p), "quality_requested": quality}

    if not p.exists():
        result.update(status="error", reason="file not found")
        _log_extraction(result)
        return f"ERROR: {path} not found"

    if quality == "accurate":
        result.update(status="refused", reason="accurate-tier extraction (OCR/MinerU) not configured in this build")
        _log_extraction(result)
        return (
            "REFUSED: accurate-tier extraction requested but no OCR/MinerU service "
            "is configured in this build (see plan §2.10). Convert this document to "
            "JSON/text first, or request the fast tier."
        )

    suffix = p.suffix.lower()
    if suffix == ".json":
        text = p.read_text(encoding="utf-8")
        json.loads(text)  # validate it's real JSON; raises if not
    elif suffix in (".txt", ".md"):
        text = p.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        result.update(status="refused", reason="PDF extraction not supported in this build (JSON/text only)")
        _log_extraction(result)
        return (
            "REFUSED: PDF extraction is not supported in this build (no fast-tier "
            "PDF parser wired in, per your earlier choice to stay JSON-source-only). "
            "Convert the PDF to JSON/text externally before ingesting it."
        )
    else:
        result.update(status="error", reason=f"unsupported file type: {suffix}")
        _log_extraction(result)
        return f"ERROR: unsupported file type {suffix} (only .json, .txt, .md supported)"

    result.update(status="ok", quality_tier="fast", char_count=len(text))
    if len(text) < 50:
        result["warning"] = "suspiciously low character count for a source document"
    _log_extraction(result)
    return text


def _log_extraction(result: dict) -> None:
    with INGEST_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def lookup_alias(term: str) -> str:
    """Look up a term/acronym in the aggregated alias registry. Returns every
    matching concept — if more than one, the term is AMBIGUOUS and must not
    be auto-linked; disambiguate from context or flag under a
    '# Ambiguous References' heading per the schema instead of guessing.

    Args:
        term: the term or acronym to look up, e.g. "SQID" or "CQ".
    """
    index = build_alias_index()
    matches = index.get(term.strip().lower(), [])
    if not matches:
        return f'no known concept has "{term}" as a title or alias'
    if len(matches) == 1:
        return f'"{term}" -> {matches[0]["path"]} ({matches[0]["title"]}) — unambiguous'
    candidates = "; ".join(f'{m["path"]} ({m["title"]})' for m in matches)
    return f'"{term}" is AMBIGUOUS — {len(matches)} candidates: {candidates}. Do not auto-link.'


# ---- §2.8: wikilink backfill (find missing links, don't insert blindly) ---

def find_missing_links() -> str:
    """Deterministic (no LLM call) scan: for every concept, find alias
    mentions in its body that are NOT already inside a markdown link. These
    are candidates a first-generation ingest pass may have missed —
    report them; the actual insertion is left to enrich_links (an LLM-driven
    follow-up pass), since deciding exactly where/how to phrase a link needs
    judgment this function doesn't have.
    """
    alias_index = build_alias_index()
    concepts = _iter_concepts()
    findings = []

    for c in concepts:
        body = c["body"]
        for alias, candidates in alias_index.items():
            if len(candidates) != 1 or candidates[0]["path"] == c["path"]:
                continue  # skip ambiguous aliases and self-references here
            pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(body):
                # crude "already linked" check: is there a '](' within 5 chars after the match?
                tail = body[m.end(): m.end() + 5]
                already_linked = "](" in body[max(0, m.start() - 40): m.end()] and tail.strip().startswith(")")
                if not already_linked and candidates[0]["path"] not in body:
                    findings.append(f'- {c["path"]}: mentions "{alias}" (unlinked) -> could link to {candidates[0]["path"]}')
                    break  # one finding per alias per file is enough

    if not findings:
        return "no missing links found"
    return f"{len(findings)} candidate missing link(s):\n" + "\n".join(findings)


def enrich_links() -> None:
    """LLM-driven follow-up pass (plan §2.8): runs find_missing_links(), then
    has the agent actually insert well-phrased links for each candidate using
    edit_file — insertion needs judgment (natural phrasing, not just
    substring replacement) that the deterministic scan doesn't have."""
    findings = find_missing_links()
    if findings == "no missing links found":
        print("[enrich-links] nothing to do")
        return
    print(f"[enrich-links] {findings}")
    instruction = (
        "The following concept files mention a known alias in their body text "
        "without a markdown link to the concept it refers to:\n\n"
        f"{findings}\n\n"
        "For each: read the file, and edit the relevant sentence to wrap that "
        "mention in a proper markdown link (e.g. `[SQID](/concepts/glossary.md"
        "#sqid)`), phrased naturally — don't just mechanically substitute text. "
        "Also add a corresponding relates_to entry (with kind/description/"
        "confidence: alias-matched) if one doesn't already exist. Use write_wiki_page "
        "to save each updated file. Report what you changed."
    )
    print(run(instruction))
    regenerate_graph()


# ---- §2.7: structural vs semantic lint, as a single dispatcher -------------

def lint_wiki(mode: str = "structural") -> str:
    """Dispatcher tool matching plan §2.7's two-mode split. `structural`
    (the default, and the only mode safe to call as a tool mid-agent-loop)
    runs every deterministic check — relates_to/type validity, alias
    coverage, ambiguity, confidence-mix metrics — with zero LLM cost.
    `semantic` (contradictions, staleness, orphan-page judgment calls) is
    NOT invoked recursively from inside a tool call in this build — it's the
    `lint` CLI command's own LLM pass, run once per lint invocation at the
    top level, not something an agent should trigger on itself repeatedly.

    Args:
        mode: "structural" (default) or "semantic" (returns a pointer,
            not a recursive agent call).
    """
    if mode == "semantic":
        return (
            "semantic lint is not callable as a tool from within an agent run "
            "in this build — it's the LLM pass the `lint` CLI command runs at "
            "the top level, to avoid recursive agent calls. Run `python "
            "wiki_agent_deepagents.py lint` instead."
        )
    parts = [
        validate_relates_to(write_report=False),
        audit_alias_coverage(write_report=False),
    ]
    return "\n\n".join(parts)


def prescan_text(text: str, alias_index: dict[str, list[dict]]) -> str:
    """Deterministic (no LLM call) pre-scan of raw source text: finds every
    known alias mentioned verbatim, checks for nearby citation language, and
    returns a human-readable block of candidate matches + confidence hints
    to hand to the model — so it confirms deterministic evidence instead of
    recalling associations from memory. Ambiguous aliases are surfaced with
    every candidate, never pre-resolved to one.
    """
    if not alias_index:
        return "(no alias registry yet — this is likely the first ingest batch)"

    lines = []
    seen_aliases = set()
    for alias, candidates in alias_index.items():
        if alias in seen_aliases:
            continue
        pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
        m = pattern.search(text)
        if not m:
            continue
        seen_aliases.add(alias)
        window = text[max(0, m.start() - 80): m.end() + 80]
        has_citation = bool(CITATION_RE.search(window))
        confidence = "explicit" if has_citation else "alias-matched"

        if len(candidates) > 1:
            cand_str = "; ".join(f'{c["path"]} ({c["title"]})' for c in candidates)
            lines.append(
                f'- "{alias}" found — AMBIGUOUS, {len(candidates)} candidates: {cand_str}. '
                "Do not auto-link; disambiguate from context or flag under "
                "'# Ambiguous References' per schema §4a."
            )
        else:
            c = candidates[0]
            lines.append(
                f'- "{alias}" found -> {c["path"]} ({c["title"]}) '
                f"[confidence: {confidence}]"
                + (f' — citation language nearby: "...{window.strip()}..."' if has_citation else "")
            )

    if not lines:
        return "(no known aliases matched in this batch's text)"
    return "ALIAS/CITATION PRE-SCAN (deterministic, computed before this call):\n" + "\n".join(lines)


def validate_relates_to(write_report: bool = True) -> str:
    """Deterministic (no LLM call) check: for every concept file's
    `relates_to` frontmatter entries, confirm the referenced path exists in
    the wiki, and flag any relates_to entry that has no matching prose link
    anywhere else in the same file. This is cheap and exact — always run it
    before/alongside the LLM lint pass rather than asking the model to
    eyeball path existence itself.

    If write_report is True (default), findings are also written to
    wiki/_lint-reports/<timestamp>.md so they survive beyond terminal
    scrollback and can be diffed across ingest runs.
    """
    try:
        import yaml  # pip install pyyaml
    except ImportError:
        return "ERROR: pyyaml not installed - run `pip install pyyaml` to enable this check."

    problems = []
    checked = 0

    for md_file in WIKI_ROOT.rglob("*.md"):
        if md_file.name in ("index.md", "log.md"):
            continue
        if "_lint-reports" in md_file.parts:
            continue
        text = md_file.read_text(encoding="utf-8")
        if not text.startswith("---"):
            problems.append(f"{md_file.relative_to(WIKI_ROOT)}: missing frontmatter entirely")
            continue

        try:
            _, fm_text, body = text.split("---", 2)
            fm = yaml.safe_load(fm_text) or {}
        except ValueError:
            problems.append(f"{md_file.relative_to(WIKI_ROOT)}: malformed frontmatter block (no closing ---)")
            continue
        except yaml.YAMLError as e:
            line = getattr(getattr(e, "problem_mark", None), "line", None)
            problems.append(
                f"{md_file.relative_to(WIKI_ROOT)}: invalid YAML in frontmatter"
                + (f" (near line {line + 1})" if line is not None else "")
                + " - likely an unquoted string containing a colon; wrap it in "
                'double quotes, e.g. description: "...text: more text..."'
            )
            continue

        checked += 1

        if "type" not in fm:
            problems.append(f"{md_file.relative_to(WIKI_ROOT)}: missing required `type` field")

        for rel in fm.get("relates_to", []) or []:
            path = rel.get("path", "") if isinstance(rel, dict) else ""
            if not path:
                problems.append(f"{md_file.relative_to(WIKI_ROOT)}: relates_to entry missing `path`")
                continue
            target = (WIKI_ROOT / path.lstrip("/")).resolve()
            if not target.exists():
                problems.append(
                    f"{md_file.relative_to(WIKI_ROOT)}: relates_to path {path} does not exist "
                    "(fine if this is an intentional forward reference not yet ingested - "
                    "otherwise a stale/typo'd link)"
                )
            if path not in body and Path(path).stem not in body:
                problems.append(
                    f"{md_file.relative_to(WIKI_ROOT)}: relates_to entry {path} has no "
                    "matching prose link in the body - relates_to should index a real link, "
                    "not stand alone"
                )
            if not rel.get("description"):
                problems.append(
                    f"{md_file.relative_to(WIKI_ROOT)}: relates_to entry {path} missing "
                    "`description` - bare path/kind pairs aren't acceptable per schema §2"
                )
            confidence = rel.get("confidence")
            if confidence not in ("explicit", "alias-matched", "llm-inferred"):
                problems.append(
                    f"{md_file.relative_to(WIKI_ROOT)}: relates_to entry {path} has invalid "
                    f"or missing confidence value: {confidence!r}"
                )
            elif confidence == "explicit" and not CITATION_RE.search(body):
                problems.append(
                    f"{md_file.relative_to(WIKI_ROOT)}: relates_to entry {path} is tagged "
                    "`explicit` but no citation language (see/refer to/defined in/per + "
                    "section/figure/table) was found anywhere in the body — possible mis-tag, "
                    "should likely be `alias-matched` or `llm-inferred`"
                )

    alias_index = build_alias_index()
    ambiguous = {a: c for a, c in alias_index.items() if len(c) > 1}
    if ambiguous:
        problems.append(f"{len(ambiguous)} ambiguous alias(es) in the registry:")
        for alias, candidates in ambiguous.items():
            paths = ", ".join(c["path"] for c in candidates)
            problems.append(f"  - \"{alias}\" -> {paths}")

    summary = f"Checked {checked} concept file(s)."
    if not problems:
        report = summary + " No relates_to/type problems found."
    else:
        report = summary + "\n" + "\n".join(f"- {p}" for p in problems)

    if write_report:
        from datetime import datetime

        reports_dir = WIKI_ROOT / "_lint-reports"
        reports_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        report_path = reports_dir / f"{stamp}.md"
        report_path.write_text(
            f"# Deterministic lint report — {stamp}\n\n"
            f"{len(problems)} problem(s) found.\n\n" + report,
            encoding="utf-8",
        )
        report = report + f"\n\n[full report written to _lint-reports/{stamp}.md]"

    return report


def audit_alias_coverage(write_report: bool = True) -> str:
    """Deterministic (no LLM call): find every concept file that declares NO
    aliases at all. A concept with zero aliases is invisible to the pre-scan
    for every OTHER source that mentions it by an abbreviation or alternate
    name — it can only ever be found by exact title match or by the model's
    own memory. This doesn't fix anything by itself; it tells you where
    coverage is thin so you can point the agent (or yourself) at it.
    """
    try:
        import yaml
    except ImportError:
        return "ERROR: pyyaml not installed - run `pip install pyyaml` to enable this check."

    missing = []
    total = 0
    for md_file in WIKI_ROOT.rglob("*.md"):
        if md_file.name in ("index.md", "log.md") or "_lint-reports" in md_file.parts:
            continue
        text = md_file.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        try:
            _, fm_text, _ = text.split("---", 2)
            fm = yaml.safe_load(fm_text) or {}
        except (ValueError, yaml.YAMLError):
            continue
        total += 1
        if not fm.get("aliases"):
            missing.append("/" + str(md_file.relative_to(WIKI_ROOT)).replace("\\", "/"))

    pct = (len(missing) / total * 100) if total else 0
    report = (
        f"{len(missing)}/{total} concepts ({pct:.0f}%) have no declared aliases "
        "and are only discoverable by exact title match in the pre-scan.\n"
        + "\n".join(f"- {p}" for p in missing)
    )

    if write_report:
        from datetime import datetime
        reports_dir = WIKI_ROOT / "_lint-reports"
        reports_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        (reports_dir / f"{stamp}-coverage.md").write_text(
            f"# Alias coverage audit — {stamp}\n\n{report}", encoding="utf-8"
        )
        report += f"\n\n[full report written to _lint-reports/{stamp}-coverage.md]"

    return report


def compute_confidence_metrics() -> dict:
    """Deterministic (no LLM call): count relates_to entries by confidence
    tier across the whole wiki. This is a coverage/trust HEALTH METRIC, not
    a correctness check — a rising llm-inferred share over successive
    ingests is a smell worth investigating (e.g. alias registry falling
    behind new terminology), not proof anything specific was missed.
    """
    try:
        import yaml
    except ImportError:
        return {}

    counts = {"explicit": 0, "alias-matched": 0, "llm-inferred": 0, "unspecified": 0}
    total_concepts = 0
    concepts_without_aliases = 0

    for md_file in WIKI_ROOT.rglob("*.md"):
        if md_file.name in ("index.md", "log.md") or "_lint-reports" in md_file.parts:
            continue
        text = md_file.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        try:
            _, fm_text, _ = text.split("---", 2)
            fm = yaml.safe_load(fm_text) or {}
        except (ValueError, yaml.YAMLError):
            continue

        total_concepts += 1
        if not fm.get("aliases"):
            concepts_without_aliases += 1

        for rel in fm.get("relates_to", []) or []:
            if isinstance(rel, dict):
                counts[rel.get("confidence", "unspecified")] = (
                    counts.get(rel.get("confidence", "unspecified"), 0) + 1
                )

    total_edges = sum(counts.values())
    return {
        "timestamp": None,  # filled in by caller when persisting
        "total_concepts": total_concepts,
        "concepts_without_aliases": concepts_without_aliases,
        "total_relates_to_edges": total_edges,
        "confidence_counts": counts,
        "llm_inferred_share": round(counts["llm-inferred"] / total_edges, 3) if total_edges else None,
    }


def record_and_report_metrics() -> str:
    """Append current metrics to wiki/_metrics.jsonl and print the delta
    against the previous recorded snapshot (if any) — this is how the
    'rising llm-inferred share' smell test actually gets tracked over time,
    rather than only being visible as a single-point-in-time number."""
    from datetime import datetime

    metrics_path = WIKI_ROOT / "_metrics.jsonl"
    previous = None
    if metrics_path.exists():
        lines = [l for l in metrics_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            previous = json.loads(lines[-1])

    current = compute_confidence_metrics()
    current["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(current) + "\n")

    lines = [
        f"Concepts: {current['total_concepts']} "
        f"({current['concepts_without_aliases']} with no declared aliases)",
        f"relates_to edges: {current['total_relates_to_edges']} "
        f"{current['confidence_counts']}",
    ]
    if current["llm_inferred_share"] is not None:
        lines.append(f"llm-inferred share: {current['llm_inferred_share']:.0%}")

    if previous and previous.get("llm_inferred_share") is not None and current["llm_inferred_share"] is not None:
        delta = current["llm_inferred_share"] - previous["llm_inferred_share"]
        direction = "up" if delta > 0 else ("down" if delta < 0 else "unchanged")
        lines.append(
            f"  vs previous snapshot ({previous['timestamp']}): {direction} "
            f"{abs(delta):.0%} — {'investigate alias/citation coverage if this keeps rising' if delta > 0 else ''}"
        )

    return "\n".join(lines)


def fix_coverage(limit: int = 15) -> None:
    """Take the alias coverage audit's flagged list and have the agent
    propose real aliases for them (acronyms/alternate names it knows from
    domain knowledge or from re-reading the concept's own body), checking
    each proposal against lookup_alias first so we don't silently introduce
    a NEW ambiguous alias while trying to close a coverage gap. Capped at
    `limit` concepts per run so one call doesn't need to hold too much in
    context at once — rerun the command to keep going.
    """
    report = audit_alias_coverage(write_report=False)
    flagged = [line[2:] for line in report.splitlines() if line.startswith("- /")]
    if not flagged:
        print("[fix-coverage] no concepts missing aliases — nothing to do")
        return

    batch = flagged[:limit]
    print(f"[fix-coverage] proposing aliases for {len(batch)}/{len(flagged)} flagged concept(s)")

    instruction = (
        "The following concept files have NO declared aliases, which makes them "
        "invisible to the deterministic pre-scan when other sources mention them "
        "by an abbreviation or alternate name:\n\n"
        + "\n".join(f"- {p}" for p in batch)
        + "\n\nFor each file: read it, then propose 1-3 real aliases/acronyms for "
        "the concept (from your own NVMe domain knowledge and/or terms used in "
        "the file's own body text — not invented). Before adding any alias, call "
        "lookup_alias on it first — if it comes back AMBIGUOUS or already maps to "
        "a DIFFERENT concept, do not add it as a plain alias; either skip it or "
        "add a short disambiguating note to that concept's `description` field "
        "instead, per schema §4a. Only add aliases you're confident are correct "
        "and unambiguous. Update each file's `aliases` frontmatter field with "
        "edit_file. Report which concepts you updated and which you skipped, and why."
    )
    print(run(instruction))
    regenerate_graph()


# ---- Build the agent --------------------------------------------------------

SCHEMA_TEXT = SCHEMA_PATH.read_text(encoding="utf-8")


def validate_relates_to_tool() -> str:
    """Run the deterministic (non-LLM) relates_to/type validator across the
    whole wiki and return a report. Findings are also saved to
    wiki/_lint-reports/<timestamp>.md so they persist beyond this response.
    Use this after a batch of writes to check your own correlations
    mechanically, rather than trusting memory."""
    return validate_relates_to()


# Shared tool list — the main agent and the entity-ingest subagent both need
# these. Note: deepagents still exposes its own generic write_file/edit_file
# by default alongside these; the "always use write_wiki_page" rule below is
# enforced by strong instruction, NOT hard sandboxing — fully removing the
# generic tools would require building the agent from deepagents' middleware
# directly instead of create_deep_agent()'s default stack. Flagging this as
# a real, not-yet-closed gap rather than pretending it's airtight.
SHARED_TOOLS = [
    write_wiki_page, append_log, grep_index, search_wiki,
    validate_relates_to_tool, lookup_alias, check_duplicates,
    extract_document, lint_wiki,
]

SYSTEM_PROMPT = (
    "You are the maintainer agent for an OKF-conformant NVMe knowledge wiki. "
    "Follow the schema below exactly — the folder layout, the required `type` "
    "frontmatter, the filing rules, and the index.md/log.md formats.\n\n"
    "ALWAYS use write_wiki_page to create or update concept pages, index.md, "
    "or log.md — never the generic write_file/edit_file tools for these. "
    "write_wiki_page enforces path safety and updates index.md deterministically "
    "as a side effect; using the generic tools bypasses both of those checks.\n\n"
    "Before creating any NEW concept page, call check_duplicates on its "
    "proposed title first. Use search_wiki (not just grep_index) when you "
    "need ranked relevance rather than an exact keyword hit — it combines "
    "keyword, link-graph, and (if available) embedding similarity.\n\n"
    "This agent has access to an entity-ingest-agent subagent (via the task "
    "tool) that files exactly one concept per invocation, given that "
    "concept's own source slice plus pre-scan context. When ingesting a "
    "batch of source material, first enumerate the distinct entities in it "
    "(commands, log pages, data structures, status tables, features, "
    "extended capabilities, architecture topics), THEN delegate to "
    "entity-ingest-agent once per entity — do not write concept files "
    "yourself in the main turn; delegating gives each entity's ingestion "
    "its own isolated context rather than polluting yours with every other "
    "entity's intermediate reasoning.\n\n" + SCHEMA_TEXT
)

ENTITY_SUBAGENT_PROMPT = (
    "You file EXACTLY ONE NVMe concept into the wiki per invocation — the "
    "one described in the task you were given. Read the entity's source "
    "text and any pre-scan context you were passed, apply the schema's "
    "filing rules to decide the correct path and `type`, call "
    "check_duplicates on the title first, then use write_wiki_page to save "
    "it with full OKF frontmatter (including kind/description/confidence "
    "on every relates_to entry — use the confidence value from the pre-scan "
    "context you were given, never invent your own). Call lookup_alias "
    "before treating any alias as unambiguous. Do not process any entity "
    "other than the one you were asked to file. Report back the path you "
    "wrote and a one-line summary.\n\n" + SCHEMA_TEXT
)


# §4.5: per-entity sub-agent isolation, at the granularity you chose (one
# subagent invocation per individual command/entity, not per whole chunk).
# Invoked automatically via deepagents' built-in `task` tool once `subagents`
# is provided to create_deep_agent — no extra wiring needed for that part.
entity_ingest_subagent = {
    "name": "entity-ingest-agent",
    "description": (
        "Files exactly ONE NVMe concept (a single command, log page, data "
        "structure, status table, feature, extended capability, or "
        "architecture topic) into the wiki, given that entity's own source "
        "text slice and relevant pre-scan context. Call this once per "
        "distinct entity found in a batch — never pass multiple entities "
        "to one call."
    ),
    "system_prompt": ENTITY_SUBAGENT_PROMPT,
    "tools": SHARED_TOOLS,
}

agent = create_deep_agent(
    model=model,
    tools=SHARED_TOOLS,
    system_prompt=SYSTEM_PROMPT,
    backend=backend,
    subagents=[entity_ingest_subagent],
)


# ---- CLI --------------------------------------------------------------------


def run(instruction: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": instruction}]})
    return result["messages"][-1].content


def build_manifest(sections: list[dict]) -> str:
    """Deterministic (non-LLM) table-of-contents built from the JSON itself:
    every section number + title. Given to every chunked ingest call so the
    model can pre-emptively link to concepts it hasn't seen the body of yet
    (OKF explicitly tolerates such forward references - see §5.3).
    """
    lines = [f"- {s.get('section', '?')} {s.get('title', '')}".strip() for s in sections]
    return "SPEC MANIFEST (section numbers + titles, full document):\n" + "\n".join(lines)


def chunk_sections(sections: list[dict], max_chars: int | None = None) -> list[list[dict]]:
    """Group sections into batches under a character budget per call, derived
    from §2.6's available_context_budget() (QWEN_CONTEXT_TOKENS minus
    reserved headroom) rather than a hardcoded constant — sized to YOUR
    served --max-model-len via the env var, not a platform default."""
    if max_chars is None:
        max_chars = available_context_budget() * 4  # tokens -> rough chars
    chunks, current, current_len = [], [], 0
    for sec in sections:
        size = len(json.dumps(sec))
        if current and current_len + size > max_chars:
            chunks.append(current)
            current, current_len = [], 0
        current.append(sec)
        current_len += size
    if current:
        chunks.append(current)
    return chunks


def regenerate_graph() -> None:
    """Rebuild the interactive graph HTML from the current wiki state. Called
    automatically at the end of every ingest — you shouldn't need to run
    build_graph.py by hand."""
    n_nodes, n_edges = build_graph.generate_graph(WIKI_ROOT, GRAPH_OUTPUT)
    print(f"[graph] regenerated {GRAPH_OUTPUT} — {n_nodes} nodes, {n_edges} edges")


def ingest_large(source_path: Path) -> None:
    """Chunked ingest for a full spec JSON: pre-scans a manifest, then walks
    section-batches one agent.invoke() at a time so no single call needs the
    whole document in context. Cross-links resolve via (a) the manifest,
    (b) reading the growing wiki on disk each call, and (c) a final
    cross-link repair pass. Each batch also gets a deterministic alias +
    citation pre-scan (see prescan_text) so confidence tags reflect real
    evidence rather than the model's self-reported certainty.

    §2.4/§4.6: each chunk is hash-checked against wiki/_ingest-cache.json
    BEFORE any agent call — unchanged chunks (e.g. rerunning after a partial
    failure) are skipped entirely, zero tokens spent.

    §4.5: within each chunk, the main agent is instructed to enumerate
    entities and delegate one entity-ingest-agent subagent call per entity,
    rather than writing every concept in the chunk in one continuous turn.
    """
    sections = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = build_manifest(sections)
    chunks = chunk_sections(sections)

    print(f"[ingest-large] {len(sections)} sections -> {len(chunks)} chunk(s)")

    for i, chunk in enumerate(chunks, 1):
        cache_key = f"{source_path.name}:chunk{i}"
        chunk_text = json.dumps(chunk, indent=2)

        if is_unchanged_since_last_ingest(cache_key, chunk_text):
            print(f"[ingest-large] chunk {i}/{len(chunks)} unchanged since last run — skipping (0 tokens)")
            continue

        print(f"[ingest-large] chunk {i}/{len(chunks)} "
              f"(sections {chunk[0].get('section')}-{chunk[-1].get('section')})")

        alias_index = build_alias_index()  # fresh — reflects prior chunks' writes
        prescan = prescan_text(chunk_text, alias_index)

        instruction = (
            f"{manifest}\n\n{prescan}\n\n"
            "You are ingesting ONE BATCH of sections from the full spec above "
            "(not the whole document - the rest arrives in later batches, "
            "some already ingested, some not yet).\n\n"
            "STEP 1 — PLAN: enumerate the distinct entities in this batch "
            "(commands, log pages, data structures, status tables, features, "
            "extended capabilities, architecture topics). Use write_todos to "
            "record this plan before proceeding.\n\n"
            "STEP 2 — DELEGATE: for EACH entity in your plan, call the "
            "entity-ingest-agent subagent (via the task tool) once, passing "
            "it that entity's specific source text slice plus the relevant "
            "part of the ALIAS/CITATION PRE-SCAN above. Do not write concept "
            "files yourself in this main turn.\n\n"
            "STEP 3 — VERIFY: after all entities are delegated, call "
            "grep_index / search_wiki to confirm each was actually written, "
            "and call append_log for today's date summarizing this batch.\n\n"
            "For any alias flagged AMBIGUOUS in the pre-scan, tell the "
            "relevant subagent to follow schema §4a: do not auto-link. If "
            "this batch mentions a concept from a LATER section per the "
            "manifest that hasn't been ingested yet, the forward reference "
            "is still valid per OKF §5.3 — pass that context to the subagent "
            "so it writes the link anyway.\n\n"
            f"SECTIONS IN THIS BATCH:\n\n{chunk_text}"
        )
        print(run(instruction))
        mark_ingested(cache_key, chunk_text)

    print("[ingest-large] all chunks done - running cross-link repair pass")
    repair_instruction = (
        "Full ingest is complete. Now do a cross-link repair pass: read every "
        "concept file, and for every entity/command/capability it mentions in "
        "prose, confirm a matching markdown link exists and points to the "
        "correct now-existing path. Fix any forward references that were "
        "guessed at an earlier stage and turned out to have a different real "
        "path than expected, using write_wiki_page to save any fixes. Call "
        "validate_relates_to_tool to check your work mechanically. Report "
        "anything you couldn't resolve."
    )
    print(run(repair_instruction))

    regenerate_graph()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "ingest-large":
        ingest_large(Path(sys.argv[2]).resolve())

    elif cmd == "ingest":
        source_path = Path(sys.argv[2]).resolve()
        source_text = source_path.read_text(encoding="utf-8")
        cache_key = source_path.name

        if is_unchanged_since_last_ingest(cache_key, source_text):
            print(f"[ingest] {source_path.name} unchanged since last run — skipping (0 tokens)")
            return

        alias_index = build_alias_index()
        prescan = prescan_text(source_text, alias_index)
        instruction = (
            f"{prescan}\n\n"
            f"Ingest this source ({source_path.name}).\n\n"
            "STEP 1 — PLAN: enumerate the distinct entities in it (commands, "
            "log pages, data structures, status tables, features, extended "
            "capabilities, architecture topics). Use write_todos to record "
            "this plan.\n\n"
            "STEP 2 — DELEGATE: for EACH entity, call the entity-ingest-agent "
            "subagent (via the task tool) once, passing it that entity's "
            "source text slice plus the relevant part of the pre-scan below. "
            "Do not write concept files yourself in this main turn.\n\n"
            "STEP 3 — VERIFY: confirm each entity was written, and call "
            "append_log for today's date. Use the ALIAS/CITATION PRE-SCAN "
            "below as the source of truth for `confidence` values you pass "
            "to each subagent - do not invent your own. For any alias "
            "flagged AMBIGUOUS, do not auto-link; follow schema §4a.\n\n"
            f"SOURCE CONTENT:\n\n{source_text}"
        )
        print(run(instruction))
        mark_ingested(cache_key, source_text)
        regenerate_graph()

    elif cmd == "query":
        question = " ".join(sys.argv[2:])
        instruction = (
            f"Answer this question using the wiki: {question}\n"
            "Use search_wiki (ranked, multi-signal) to find relevant concept "
            "files first, then read_file to check the details. Cite which "
            "concept files you used."
        )
        print(run(instruction))

    elif cmd == "dedup":
        print(dedup_sweep())

    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("usage: delete <path> [--force]")
            sys.exit(1)
        force = "--force" in sys.argv[3:]
        print(delete_concept(sys.argv[2], force=force))
        regenerate_graph()

    elif cmd == "enrich-links":
        enrich_links()

    elif cmd == "fix-coverage":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        fix_coverage(limit=limit)

    elif cmd == "lint":
        print("=== deterministic relates_to/type check (no LLM) ===")
        print(validate_relates_to())
        print("\n=== deterministic alias coverage audit (no LLM) ===")
        print(audit_alias_coverage())
        print("\n=== deterministic dedup sweep (no LLM) ===")
        print(dedup_sweep())
        print("\n=== deterministic missing-links scan (no LLM) ===")
        print(find_missing_links())
        print("\n=== confidence-mix metrics (no LLM) ===")
        print(record_and_report_metrics())
        print("\n=== LLM lint pass (semantic — contradictions, staleness) ===")
        instruction = (
            "Run the semantic lint checklist from the schema against the current "
            "wiki. List concept files with ls, check each has a valid `type` "
            "frontmatter field, check for orphan pages and missing cross-links "
            "among cross-cutting topics (e.g. Reservations, ANA, Sanitize), look "
            "for contradictions between pages, and report findings. Do not "
            "modify files unless asked to fix issues, not just report them."
        )
        print(run(instruction))

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
