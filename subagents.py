"""
Subagent specs. Each subagent is a context-quarantined worker: it gets a
narrow toolset and a prompt seeded with the relevant part of AGENTS.md, and it
reports back a summary rather than flooding the orchestrator's context with
every intermediate tool call.

Note: subagents automatically get the built-in filesystem tools (ls,
read_file, write_file, edit_file, glob, grep) against the same FilesystemBackend
as the parent — we only need to hand them the custom tools from tools.py.
"""

import os
from pathlib import Path

from .tools import list_raw_chunks, read_raw_chunk, snapshot_page, append_log

WIKI_DIR = os.environ.get("NVME_WIKI_DIR", "./nvme-wiki")
_AGENTS_MD_PATH = Path(WIKI_DIR) / "AGENTS.md"


def _schema_text() -> str:
    if _AGENTS_MD_PATH.exists():
        return _AGENTS_MD_PATH.read_text()
    return (
        "[AGENTS.md not found at expected path — subagent has no schema to "
        "follow. Do not proceed; report this back instead of guessing "
        "conventions.]"
    )


INGEST_SUBAGENT = {
    "name": "ingest-agent",
    "description": (
        "Ingests one or more raw NVMe 2.0 spec chunks into the wiki: extracts "
        "distinct entities, creates or edits their canonical pages, updates "
        "index.md files, snapshots overwritten pages, and appends a log.md "
        "entry. Use this for any ingest task."
    ),
    "prompt": (
        "You are the ingest agent for an OKF-conformant NVMe 2.0 knowledge "
        "wiki. Follow this schema exactly:\n\n"
        f"{_schema_text()}\n\n"
        "--- Ingest procedure for this task (AGENTS.md §13) ---\n"
        "1. Read the assigned raw chunk(s) with read_raw_chunk.\n"
        "2. Identify every distinct entity described (commands, data "
        "structures, registers, features, error codes) — a single chunk "
        "may touch several entities, or an entity across several chunks.\n"
        "3. For each entity: ls/grep the relevant category directory and its "
        "index.md to decide new-page vs. edit-in-place (§9). Prefer editing "
        "a near-match over creating a near-duplicate.\n"
        "4. If editing an EXISTING page with a substantive (factual/"
        "behavioral) change, call snapshot_page FIRST, before writing.\n"
        "5. Write or edit the page using the §4 frontmatter and §5 body "
        "template, including a Version History table entry (§6) for any "
        "version-specific detail. New pages: status: unverified.\n"
        "6. Update every index.md you touched (§11), and create "
        "concepts/open-questions.md entries for anything ambiguous or "
        "contradictory instead of guessing.\n"
        "7. Call append_log with ONE entry summarizing the whole task "
        "(§12), listing every page created or edited.\n"
        "Never invent a field, value, or behavior not present in the raw "
        "chunk. If the raw source is ambiguous, say so in "
        "concepts/open-questions.md rather than filling the gap."
    ),
    "tools": [read_raw_chunk, list_raw_chunks, snapshot_page, append_log],
}


LINT_SUBAGENT = {
    "name": "lint-agent",
    "description": (
        "Runs a health-check pass over the wiki: contradictions, orphan "
        "pages, stale unverified claims, missing cross-references. Invoked "
        "automatically after every ingest batch, and available on demand."
    ),
    "prompt": (
        "You are the lint agent for this NVMe 2.0 wiki. Follow this schema:\n\n"
        f"{_schema_text()}\n\n"
        "--- Lint procedure (AGENTS.md §13) ---\n"
        "Walk the wiki (ls/glob/grep — you do not have read/write access to "
        "raw/ from this role) and check for:\n"
        "- Contradictions between pages describing the same entity/behavior.\n"
        "- Orphan pages with no inbound links from any other page.\n"
        "- Pages referenced by a markdown link that don't actually exist yet.\n"
        "- `status: unverified` pages that are old relative to this run and "
        "haven't been checked against raw/.\n"
        "- Missing cross-references between clearly related pages (e.g. a "
        "command page that doesn't link to a data structure it clearly uses).\n"
        "Append every substantive finding as a bullet to "
        "concepts/open-questions.md (create it with a '# Open Questions' "
        "header if it doesn't exist). You MAY fix purely mechanical breakage "
        "yourself without approval (a dead relative link, a missing "
        "index.md line) — but never silently resolve a contradiction or "
        "change a factual claim; that goes to open-questions.md for a human "
        "or the ingest-agent (with sign-off) to resolve.\n"
        "Call append_log with a summary of what you found and fixed."
    ),
    "tools": [append_log],
}


QUERY_SUBAGENT = {
    "name": "query-agent",
    "description": (
        "Answers a question against the existing wiki, with citations back "
        "to raw/ via each page's `resource:` field."
    ),
    "prompt": (
        "You are the query agent for this NVMe 2.0 wiki. Follow this schema:\n\n"
        f"{_schema_text()}\n\n"
        "--- Query procedure (AGENTS.md §13) ---\n"
        "Read the root index.md, drill into the relevant category index.md, "
        "then read the specific page(s) — do not guess from memory. Answer "
        "citing each page's `resource:` frontmatter field. If a claim you'd "
        "rely on lives only on an `unverified` page, you may still surface "
        "it, but say explicitly that it's unverified rather than presenting "
        "it as settled. You may use read_raw_chunk to check something "
        "directly against raw/ if a wiki page seems incomplete or suspect."
    ),
    "tools": [read_raw_chunk],
}
