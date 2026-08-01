"""
wiki_agent_deepagents.py — drives the OKF-conformant NVMe wiki using LangChain's
`deepagents` library, with a local vLLM server (Qwen3.6-FP8 or similar) as the model.

Usage:
    python wiki_agent_deepagents.py ingest <path-to-source.json-or-txt>
    python wiki_agent_deepagents.py ingest-large <path-to-full-spec.json>
    python wiki_agent_deepagents.py query "your question"
    python wiki_agent_deepagents.py lint

ingest-large is for a full-spec JSON too big to fit in one context window:
it expects a JSON array of section objects (matching the sample you shared,
each with "section", "title", "content"), builds a cheap non-LLM manifest of
every section number + title, chunks the array into batches under a rough
character budget, and calls the agent once per batch - so cross-references
resolve via the growing wiki on disk (and the manifest, for forward
references) rather than needing the whole document in context at once. It
finishes with an automatic cross-link repair pass over the whole wiki.

Every ingest (both `ingest` and `ingest-large`) also:
  - runs a deterministic alias + citation-language pre-scan (prescan_text)
    before each LLM call, so `relates_to.confidence` values are grounded in
    real evidence (explicit citation match, alias registry match, or neither)
    rather than the model self-reporting how sure it feels. Ambiguous
    aliases (same name, multiple concepts) are surfaced but never
    auto-resolved — see the schema's §4a.
  - automatically regenerates the interactive graph (build_graph.py) at the
    end, writing to WIKI_GRAPH_OUTPUT (default ./wiki_graph.html) — you
    don't need to run build_graph.py by hand after an ingest.

Install:
    pip install deepagents langchain-openai pyyaml

Serve the model first (separate terminal):
    vllm serve <your-qwen3.6-fp8-model-path-or-repo> \\
        --served-model-name qwen3.6-fp8 \\
        --enable-auto-tool-choice \\
        --tool-call-parser hermes \\
        --reasoning-parser qwen3 \\
        --port 8000

    Note: --tool-call-parser is model-family/vLLM-version specific (Qwen3 chat
    variants generally use `hermes`; Qwen3-Coder variants use `qwen3_coder` or
    `qwen3_xml`, and this has shifted across vLLM releases). Check
    `vllm serve --help` or https://docs.vllm.ai/en/stable/features/tool_calling/
    against your exact build if tool calls come back malformed.

Why deepagents instead of a hand-rolled loop:
    deepagents already ships filesystem tools (ls, read_file, write_file,
    edit_file), a planning/todo tool, and context management, wired onto a
    real LangGraph agent loop. We only need to add two domain-specific tools
    (log.md appending, index.md search) and point it at the wiki directory
    via FilesystemBackend(root_dir=..., virtual_mode=True), which sandboxes
    all file access to that directory (blocks `..` traversal).
"""

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
SCHEMA_PATH = Path(os.environ.get("WIKI_SCHEMA", "./nvme-wiki-schema.md")).resolve()
GRAPH_OUTPUT = Path(os.environ.get("WIKI_GRAPH_OUTPUT", "./wiki_graph.html")).resolve()

WIKI_ROOT.mkdir(parents=True, exist_ok=True)

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


# ---- Domain-specific tools (additive on top of deepagents' built-ins) ------
# deepagents already gives the agent ls / read_file / write_file / edit_file
# against `backend`. We only add the two things specific to the OKF workflow:
# appending to log.md, and a quick cross-index search.


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


# ---- Build the agent --------------------------------------------------------

SCHEMA_TEXT = SCHEMA_PATH.read_text(encoding="utf-8")
SYSTEM_PROMPT = (
    "You are the maintainer agent for an OKF-conformant NVMe knowledge wiki. "
    "Follow the schema below exactly — the folder layout, the required `type` "
    "frontmatter, the filing rules, and the index.md/log.md formats. Use your "
    "filesystem tools (ls, read_file, write_file, edit_file) plus grep_index "
    "and append_log to actually read and write the wiki files — never just "
    "describe what you would do.\n\n" + SCHEMA_TEXT
)

def validate_relates_to_tool() -> str:
    """Run the deterministic (non-LLM) relates_to/type validator across the
    whole wiki and return a report. Findings are also saved to
    wiki/_lint-reports/<timestamp>.md so they persist beyond this response.
    Use this after a batch of writes to check your own correlations
    mechanically, rather than trusting memory."""
    return validate_relates_to()


agent = create_deep_agent(
    model=model,
    tools=[append_log, grep_index, validate_relates_to_tool, lookup_alias],
    system_prompt=SYSTEM_PROMPT,
    backend=backend,
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


def chunk_sections(sections: list[dict], max_chars: int = 20_000) -> list[list[dict]]:
    """Group sections into batches under a rough character budget per call.
    Tune max_chars against your served --max-model-len; leave generous
    headroom for the schema system prompt, the manifest, and tool-call
    round trips, which all also consume context."""
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
    evidence rather than the model's self-reported certainty."""
    sections = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = build_manifest(sections)
    chunks = chunk_sections(sections)

    print(f"[ingest-large] {len(sections)} sections -> {len(chunks)} chunk(s)")

    for i, chunk in enumerate(chunks, 1):
        print(f"[ingest-large] chunk {i}/{len(chunks)} "
              f"(sections {chunk[0].get('section')}-{chunk[-1].get('section')})")

        chunk_text = json.dumps(chunk, indent=2)
        alias_index = build_alias_index()  # fresh — reflects prior chunks' writes
        prescan = prescan_text(chunk_text, alias_index)

        instruction = (
            f"{manifest}\n\n{prescan}\n\n"
            "You are ingesting ONE BATCH of sections from the full spec above "
            "(not the whole document - the rest arrives in later batches, "
            "some already ingested, some not yet). Before writing any concept:\n"
            "1. Call grep_index / read_file to check what already exists in the "
            "wiki from earlier batches, and link to it.\n"
            "2. Use the ALIAS/CITATION PRE-SCAN above as your source of truth "
            "for `confidence` values on any relates_to entry you write - do "
            "not invent your own confidence rating. For any alias flagged "
            "AMBIGUOUS, follow schema §4a: do not auto-link, disambiguate "
            "from context if genuinely clear, otherwise flag under "
            "'# Ambiguous References' and set review_status: needs-review.\n"
            "3. If this batch mentions a concept from a LATER section per the "
            "manifest that hasn't been ingested yet, still write the "
            "cross-link at its expected path (e.g. /extended-capabilities/"
            "reservations.md) even though the file doesn't exist yet - this "
            "is a valid forward reference per OKF section 5.3, not an error.\n"
            "4. Apply the filing rules from the schema, write full OKF "
            "frontmatter (including kind/description/confidence on every "
            "relates_to entry) + citations, update index.md, and append_log "
            "for today's date.\n\n"
            f"SECTIONS IN THIS BATCH:\n\n{chunk_text}"
        )
        print(run(instruction))

    print("[ingest-large] all chunks done - running cross-link repair pass")
    repair_instruction = (
        "Full ingest is complete. Now do a cross-link repair pass: read every "
        "concept file, and for every entity/command/capability it mentions in "
        "prose, confirm a matching markdown link exists and points to the "
        "correct now-existing path. Fix any forward references that were "
        "guessed at an earlier stage and turned out to have a different real "
        "path than expected. Call validate_relates_to_tool to check your work "
        "mechanically. Report anything you couldn't resolve."
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
        alias_index = build_alias_index()
        prescan = prescan_text(source_text, alias_index)
        instruction = (
            f"{prescan}\n\n"
            f"Ingest this source ({source_path.name}). Walk it section by section, "
            "apply the filing rules from the schema, write/update concept files "
            "with full OKF frontmatter and citations, update the relevant "
            "index.md files, and call append_log for today's date. Use the "
            "ALIAS/CITATION PRE-SCAN above as your source of truth for "
            "`confidence` values - do not invent your own. For any alias "
            "flagged AMBIGUOUS, do not auto-link; follow schema §4a.\n\n"
            f"SOURCE CONTENT:\n\n{source_text}"
        )
        print(run(instruction))
        regenerate_graph()

    elif cmd == "query":
        question = " ".join(sys.argv[2:])
        instruction = (
            f"Answer this question using the wiki: {question}\n"
            "Use grep_index and read_file to find relevant concept files first. "
            "Cite which concept files you used."
        )
        print(run(instruction))

    elif cmd == "lint":
        print("=== deterministic relates_to/type check (no LLM) ===")
        print(validate_relates_to())
        print("\n=== LLM lint pass ===")
        instruction = (
            "Run the lint checklist from the schema against the current wiki. "
            "List concept files with ls, check each has a valid `type` frontmatter "
            "field, check for orphan pages and missing cross-links among "
            "cross-cutting topics (e.g. Reservations, ANA, Sanitize), and report "
            "findings. Do not modify files unless asked to fix issues, not just "
            "report them."
        )
        print(run(instruction))

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
