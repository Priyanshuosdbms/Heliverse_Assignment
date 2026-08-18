"""
compare_json_to_wiki.py — coverage comparator between a raw NVMe spec JSON
source and the wiki generated from it.

Purpose: after an ingest, answer "did anything from the source JSON get
dropped on the way into the wiki?" — NOT "is the wiki correct" (that's what
lint/validate_relates_to are for) and NOT "did the wiki add anything extra"
(expected and fine — relates_to, aliases, confidence, tags, review_status,
etc. are pipeline additions with no JSON counterpart and are deliberately
EXCLUDED from the comparison, per your instruction).

What gets compared:
  - Every JSON section  -> is it claimed by at least one wiki concept's
    `source_sections` frontmatter at all? (the coarsest, highest-value check)
  - Every table row      -> does its content appear (verbatim or near-exact)
    somewhere in the wiki page(s) covering that section? Tables carry
    concrete field/bit/status values that should survive ingestion close to
    verbatim, so this is a high-confidence check.
  - Every table caption   -> same idea, one level up.
  - Every paragraph       -> paragraphs get paraphrased during ingestion, so
    exact matching isn't meaningful here. Instead: extract "significant
    terms" (acronyms, capitalized multi-word phrases) from the paragraph and
    check what fraction survive into the wiki text. Low overlap is flagged
    as "possibly under-represented," not a hard miss — paraphrase is
    expected and fine; the flag is for terms that seem to have vanished
    entirely, which is a different, more concerning thing.

What's deliberately IGNORED on the wiki side (per your instruction — this is
pipeline-added content with no JSON source to compare against):
  - relates_to (path/kind/description/confidence/alias_used)
  - aliases
  - tags
  - review_status, spec, spec_revision, timestamp
  - source_sections / source_pages themselves (used for section MAPPING,
    not compared as content)
Only `title`, `description`, and the markdown body are treated as
comparable content — those are the fields expected to trace back to JSON.

Usage:
    python compare_json_to_wiki.py <source.json> [--wiki ./wiki] [--out report.md]
                                    [--term-threshold 0.5]
"""

import argparse
import json
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


# ---- wiki-side loading -------------------------------------------------------

# Frontmatter keys deliberately excluded from the comparable text blob — all
# pipeline/correlation additions, never sourced from the raw JSON.
IGNORED_FRONTMATTER_KEYS = {
    "relates_to", "aliases", "tags", "review_status", "spec",
    "spec_revision", "timestamp", "type", "resource",
}


def load_wiki_concepts(wiki_root: Path) -> list[dict]:
    """Parse every concept file into {path, source_sections, comparable_text}.
    comparable_text = title + description + body, deliberately excluding
    IGNORED_FRONTMATTER_KEYS (see module docstring)."""
    if yaml is None:
        raise RuntimeError("pyyaml is required: pip install pyyaml")

    concepts = []
    for md_file in wiki_root.rglob("*.md"):
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

        comparable_parts = [
            str(fm.get("title", "")),
            str(fm.get("description", "")),
            body,
        ]
        comparable_text = _normalize(" ".join(comparable_parts))

        concepts.append({
            "path": "/" + str(md_file.relative_to(wiki_root)).replace("\\", "/"),
            "source_sections": [str(s) for s in (fm.get("source_sections") or [])],
            "comparable_text": comparable_text,
        })
    return concepts


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---- JSON-side atoms ---------------------------------------------------------

SIGNIFICANT_TERM_RE = re.compile(
    r"\b(?:[A-Z]{2,}(?:[a-z]*[A-Z]+[a-z]*)*|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}))\b"
)


def extract_significant_terms(text: str) -> set[str]:
    """Pull out acronyms (SQID, CID, PRP...) and capitalized multi-word
    phrases (Submission Queue Identifier) — the terms most likely to survive
    paraphrase intact, and therefore the most useful thing to check for."""
    return {m.strip() for m in SIGNIFICANT_TERM_RE.findall(text) if len(m.strip()) > 2}


def build_atoms(sections: list[dict]) -> list[dict]:
    """Flatten the JSON into checkable units: one per section (coarse
    coverage), one per table row, one per table caption, one per paragraph."""
    atoms = []
    for sec in sections:
        section_id = str(sec.get("section", ""))
        atoms.append({
            "kind": "section",
            "section": section_id,
            "title": sec.get("title", ""),
            "page": None,
        })
        for block in sec.get("content", []):
            btype = block.get("type")
            page = block.get("page")
            if btype == "paragraph":
                atoms.append({
                    "kind": "paragraph", "section": section_id, "page": page,
                    "text": block.get("text", ""),
                })
            elif btype == "table":
                caption = block.get("caption", "")
                if caption:
                    atoms.append({
                        "kind": "table_caption", "section": section_id, "page": page,
                        "text": caption,
                    })
                for row in block.get("table_data", []) or []:
                    atoms.append({
                        "kind": "table_row", "section": section_id, "page": page,
                        "caption": caption, "row": row,
                    })
    return atoms


# ---- comparison logic --------------------------------------------------------

def _row_search_snippet(row: dict) -> str:
    """Build the most distinctive searchable phrase from a table row — the
    Description field if present (richest text), else all values joined."""
    if "Description" in row:
        return str(row["Description"])
    return " ".join(str(v) for v in row.values())


def _word_overlap(a: set[str], b_text: str) -> float:
    if not a:
        return 1.0
    hits = sum(1 for term in a if _normalize(term) in b_text)
    return hits / len(a)


def compare(sections: list[dict], concepts: list[dict], term_threshold: float) -> dict:
    section_to_files: dict[str, list[str]] = {}
    for c in concepts:
        for s in c["source_sections"]:
            section_to_files.setdefault(s, []).append(c["path"])

    # Whole-wiki text, used only to distinguish "genuinely missing" from
    # "present, but filed under a different page than source_sections claims."
    whole_wiki_text = " ".join(c["comparable_text"] for c in concepts)

    atoms = build_atoms(sections)

    missing_sections = []
    missing_table_rows = []
    missing_table_captions = []
    underrepresented_paragraphs = []

    for atom in atoms:
        section_id = atom["section"]
        covering_files = section_to_files.get(section_id, [])

        if atom["kind"] == "section":
            if not covering_files:
                missing_sections.append(atom)
            continue

        if not covering_files:
            continue  # already reported at the section level; don't double-report every atom in it

        combined_text = " ".join(
            c["comparable_text"] for c in concepts if c["path"] in covering_files
        )

        if atom["kind"] == "table_row":
            snippet = _normalize(_row_search_snippet(atom["row"]))
            key_fragment = " ".join(snippet.split()[:8])  # first ~8 words, most distinctive
            exact_hit = bool(key_fragment) and key_fragment in combined_text

            # Two-tier: exact substring first; if that fails, fall back to
            # significant-term overlap so a paraphrased-but-present row
            # isn't reported as a false-positive miss.
            row_terms = extract_significant_terms(_row_search_snippet(atom["row"]))
            overlap = _word_overlap(row_terms, combined_text) if row_terms else 1.0

            if not exact_hit and overlap < term_threshold:
                found_elsewhere = key_fragment and key_fragment in whole_wiki_text
                missing_table_rows.append({
                    **atom, "covering_files": covering_files,
                    "term_overlap": round(overlap, 2),
                    "found_in_a_different_page": found_elsewhere,
                })

        elif atom["kind"] == "table_caption":
            snippet = _normalize(atom["text"])
            if snippet not in combined_text:
                found_elsewhere = snippet in whole_wiki_text
                missing_table_captions.append({
                    **atom, "covering_files": covering_files,
                    "found_in_a_different_page": found_elsewhere,
                })

        elif atom["kind"] == "paragraph":
            terms = extract_significant_terms(atom["text"])
            overlap = _word_overlap(terms, combined_text)
            if overlap < term_threshold:
                underrepresented_paragraphs.append({
                    **atom, "covering_files": covering_files,
                    "term_overlap": round(overlap, 2), "terms": sorted(terms),
                })

    return {
        "total_sections": sum(1 for a in atoms if a["kind"] == "section"),
        "missing_sections": missing_sections,
        "missing_table_rows": missing_table_rows,
        "missing_table_captions": missing_table_captions,
        "underrepresented_paragraphs": underrepresented_paragraphs,
    }


# ---- optional LLM verification pass -----------------------------------------
# Runs ONLY on what the deterministic tiers above still flag as ambiguous —
# not the whole document. This is the part that can catch what term-overlap
# structurally cannot: a paragraph that keeps every keyword but flips the
# meaning (e.g. drops a "shall NOT"). Deterministic matching can't see that;
# an LLM reading both texts can.

import os as _os

VLLM_BASE_URL = _os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = _os.environ.get("WIKI_MODEL", "qwen3.6-fp8")

VERIFY_PROMPT = """You are checking whether SOURCE TEXT from a technical spec is \
represented in WIKI TEXT derived from it. Paraphrase is expected and fine — \
only flag a real problem.

Classify as exactly one of:
- PRESENT: the wiki text conveys the same facts/claims as the source, even if reworded.
- PARTIAL: some of the source's facts are represented, but something specific \
is missing or changed (e.g. a condition, a negation, a numeric value, an exception).
- MISSING: the source's content is not represented at all.

Respond in exactly this format, nothing else:
VERDICT: <PRESENT|PARTIAL|MISSING>
NOTE: <one sentence — empty if PRESENT>

SOURCE TEXT:
{source}

WIKI TEXT (from the covering page(s)):
{wiki}
"""


def _get_llm_client():
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")


def llm_verify(source_text: str, wiki_text: str, client) -> tuple[str, str]:
    """Returns (verdict, note). Falls back to ('UNVERIFIED', reason) on any
    failure — verification is a bonus signal, never a hard dependency; the
    deterministic finding stands either way if this fails."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": VERIFY_PROMPT.format(
                source=source_text[:2000], wiki=wiki_text[:4000],
            )}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        verdict = "UNVERIFIED"
        note = ""
        for line in text.splitlines():
            if line.strip().upper().startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().upper()
            elif line.strip().upper().startswith("NOTE:"):
                note = line.split(":", 1)[1].strip()
        return verdict, note
    except Exception as e:
        return "UNVERIFIED", f"verification call failed: {e}"


def run_llm_verification(result: dict, concepts: list[dict]) -> None:
    """Mutates result in place: adds 'llm_verdict'/'llm_note' to each table
    row and paragraph finding, and DROPS items the LLM confirms are actually
    PRESENT (i.e. the deterministic pass's false positives) rather than
    reporting them as findings."""
    client = _get_llm_client()
    if client is None:
        print("[llm-verify] `openai` package not installed (pip install openai) — skipping verification")
        return

    by_path = {c["path"]: c for c in concepts}

    def wiki_text_for(covering_files: list[str]) -> str:
        return " ".join(by_path[p]["comparable_text"] for p in covering_files if p in by_path)

    for key, get_source_text in (
        ("missing_table_rows", lambda a: _row_search_snippet(a["row"])),
        ("underrepresented_paragraphs", lambda a: a["text"]),
    ):
        kept = []
        for a in result[key]:
            wiki_text = wiki_text_for(a["covering_files"])
            verdict, note = llm_verify(get_source_text(a), wiki_text, client)
            a["llm_verdict"] = verdict
            a["llm_note"] = note
            if verdict == "PRESENT":
                continue  # deterministic false positive — drop it
            kept.append(a)
        result[key] = kept


# ---- reporting ----------------------------------------------------------------

def format_report(result: dict) -> str:
    lines = ["# JSON -> Wiki coverage report", ""]

    lines.append(f"Sections in source: {result['total_sections']}")
    lines.append(f"Sections with NO wiki page at all: {len(result['missing_sections'])}")
    lines.append(f"Table rows apparently missing from their section's wiki page(s): {len(result['missing_table_rows'])}")
    lines.append(f"Table captions apparently missing: {len(result['missing_table_captions'])}")
    lines.append(f"Paragraphs with low term overlap (possibly under-represented): {len(result['underrepresented_paragraphs'])}")
    lines.append("")

    if result["missing_sections"]:
        lines.append("## Sections with zero wiki coverage")
        lines.append("(no concept file's `source_sections` frontmatter references these at all — the")
        lines.append("highest-value finding in this report; everything below only applies to sections")
        lines.append("that WERE at least partially ingested.)")
        lines.append("")
        for a in result["missing_sections"]:
            lines.append(f"- {a['section']} — {a['title']}")
        lines.append("")

    if result["missing_table_rows"]:
        lines.append("## Table rows not found in their covering wiki page(s)")
        lines.append("(field/bit/status definitions are expected to survive ingestion close to")
        lines.append("verbatim or with high term overlap — flagged here means BOTH checks failed.)")
        lines.append("")
        for a in result["missing_table_rows"]:
            tag = " [FOUND ELSEWHERE IN WIKI — possibly misfiled, not dropped]" if a.get("found_in_a_different_page") else ""
            lines.append(f"- section {a['section']} (p.{a['page']}), table \"{a['caption']}\"{tag}")
            lines.append(f"  row: {a['row']}")
            lines.append(f"  term overlap: {a['term_overlap']:.0%} — covered by: {', '.join(a['covering_files'])}")
            if "llm_verdict" in a:
                lines.append(f"  LLM verdict: {a['llm_verdict']}" + (f" — {a['llm_note']}" if a["llm_note"] else ""))
        lines.append("")

    if result["missing_table_captions"]:
        lines.append("## Table captions not found in their covering wiki page(s)")
        lines.append("")
        for a in result["missing_table_captions"]:
            tag = " [FOUND ELSEWHERE IN WIKI]" if a.get("found_in_a_different_page") else ""
            lines.append(f"- section {a['section']} (p.{a['page']}): \"{a['text']}\"{tag} — covered by {', '.join(a['covering_files'])}")
        lines.append("")

    if result["underrepresented_paragraphs"]:
        lines.append("## Paragraphs with low significant-term overlap")
        lines.append("(paraphrase is expected and fine — this flags paragraphs where the")
        lines.append("distinctive terms/acronyms seem to have vanished, not just been reworded.)")
        lines.append("")
        for a in result["underrepresented_paragraphs"]:
            lines.append(f"- section {a['section']} (p.{a['page']}), term overlap {a['term_overlap']:.0%}")
            lines.append(f"  terms expected: {a['terms']}")
            lines.append(f"  source text: \"{a['text'][:160]}{'...' if len(a['text']) > 160 else ''}\"")
            lines.append(f"  covered by: {', '.join(a['covering_files'])}")
            if "llm_verdict" in a:
                lines.append(f"  LLM verdict: {a['llm_verdict']}" + (f" — {a['llm_note']}" if a["llm_note"] else ""))
        lines.append("")

    if not any([
        result["missing_sections"], result["missing_table_rows"],
        result["missing_table_captions"], result["underrepresented_paragraphs"],
    ]):
        lines.append("No coverage gaps found.")

    return "\n".join(lines)


# ---- CLI ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_json", help="path to the raw spec JSON that was ingested")
    ap.add_argument("--wiki", default="./wiki", help="path to the wiki root (default ./wiki)")
    ap.add_argument("--out", default=None, help="write the report to this file (also prints to stdout)")
    ap.add_argument("--term-threshold", type=float, default=0.5,
                     help="minimum fraction of a paragraph's significant terms that must "
                          "appear in the wiki before it's NOT flagged (default 0.5)")
    ap.add_argument("--llm-verify", action="store_true",
                     help="run a second pass, using the same vLLM server as the ingest "
                          "pipeline, that reviews only the deterministically-flagged table "
                          "rows and paragraphs — drops confirmed false positives and can "
                          "catch semantic issues (e.g. a dropped negation) that keyword "
                          "overlap structurally cannot. Requires `pip install openai` and "
                          "a running vLLM server (VLLM_BASE_URL / WIKI_MODEL env vars).")
    args = ap.parse_args()

    sections = json.loads(Path(args.source_json).read_text(encoding="utf-8"))
    concepts = load_wiki_concepts(Path(args.wiki).resolve())
    result = compare(sections, concepts, args.term_threshold)

    if args.llm_verify:
        print("[llm-verify] reviewing flagged candidates against the vLLM server...")
        run_llm_verification(result, concepts)

    report = format_report(result)

    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
