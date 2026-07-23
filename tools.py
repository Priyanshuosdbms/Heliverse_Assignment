"""
Custom tools layered on top of deepagents' built-in filesystem tools
(ls, read_file, write_file, edit_file, glob, grep).

These exist for the parts of AGENTS.md that a generic file tool won't enforce
on its own: keeping raw/ read-only and separate from the wiki, snapshotting a
page to old_data/ before a substantive overwrite (§8), and appending to
log.md in the expected format (§12).

Everything else (creating/editing wiki pages, reading index.md, grepping for
duplicates) goes through deepagents' built-in file tools directly against the
FilesystemBackend rooted at NVME_WIKI_DIR — no custom tool needed for that.
"""

import os
import glob
import datetime

from langchain_core.tools import tool

RAW_DIR = os.environ.get("NVME_RAW_DIR", "./raw")
WIKI_DIR = os.environ.get("NVME_WIKI_DIR", "./nvme-wiki")


def _safe_raw_path(path: str) -> str:
    """Resolve a path under RAW_DIR and refuse anything that escapes it."""
    candidate = path if os.path.isabs(path) else os.path.join(RAW_DIR, path)
    raw_abs = os.path.abspath(RAW_DIR)
    candidate_abs = os.path.abspath(candidate)
    if not candidate_abs.startswith(raw_abs):
        raise ValueError(
            f"Refusing to read '{path}' — raw/ is the only source-of-truth "
            f"layer this tool may read, and it must never be edited."
        )
    return candidate_abs


@tool
def list_raw_chunks(pattern: str = "**/*.json") -> list[str]:
    """
    List available raw NVMe 2.0 spec source files under raw/, matching a glob
    pattern (default: all .json files, recursive). Use this to discover what's
    available to ingest before picking specific chunks.
    """
    matches = glob.glob(os.path.join(RAW_DIR, pattern), recursive=True)
    return sorted(os.path.relpath(m, RAW_DIR) for m in matches)


@tool
def read_raw_chunk(path: str) -> str:
    """
    Read one raw source JSON chunk by path (relative to raw/, or absolute
    but must resolve inside raw/). This is the ONLY source of ground truth —
    never treat wiki pages as a substitute for re-reading raw/ during
    verification (§7).
    """
    full = _safe_raw_path(path)
    with open(full, "r") as f:
        return f.read()


@tool
def snapshot_page(relative_wiki_path: str) -> str:
    """
    Archive the CURRENT contents of an existing wiki page to old_data/ before
    a substantive edit, per AGENTS.md §8. Call this immediately before editing
    an existing page with a factual/behavioral/version-history change.

    Skip this for pure typo/formatting fixes — those don't need a snapshot.
    If the page doesn't exist yet (you're creating it, not editing it), this
    is a no-op and tells you so.

    relative_wiki_path: path relative to the wiki root, e.g.
        "commands/admin/identify.md"
    """
    src = os.path.join(WIKI_DIR, relative_wiki_path)
    if not os.path.exists(src):
        return (
            f"No existing page at {relative_wiki_path} — nothing to snapshot "
            f"(this is a new page, not an edit)."
        )
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    dst = os.path.join(WIKI_DIR, "old_data", f"{relative_wiki_path}__{ts}.md")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "r") as f_in, open(dst, "w") as f_out:
        f_out.write(f_in.read())
    return f"Snapshotted {relative_wiki_path} -> old_data/{relative_wiki_path}__{ts}.md"


@tool
def append_log(entry: str) -> str:
    """
    Append one entry to the wiki's log.md, per AGENTS.md §12.

    entry should be pre-formatted starting with a header line like:
        "## [2026-07-22] ingest | NVMe 2.0 spec Admin Commands"
    followed by a line (or lines) naming the pages touched, e.g.:
        "Pages touched: commands/admin/identify.md (new), ..."
    """
    log_path = os.path.join(WIKI_DIR, "log.md")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a") as f:
        f.write("\n" + entry.strip() + "\n")
    return "Logged to log.md."
