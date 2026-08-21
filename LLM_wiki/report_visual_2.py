"""
visualize_report.py — turns the markdown coverage report from
compare_json_to_wiki.py into a searchable, collapsible HTML dashboard.

Takes ONLY the .md report file as input — it doesn't re-run any comparison,
re-read the wiki, or touch the JSON source. It just parses the markdown
structure compare_json_to_wiki.py already produces (headings, bullet
entries, summary count lines) and renders it as something you can actually
navigate instead of scrolling a long wall of text.

Usage:
    python compare_json_to_wiki.py full_spec.json --wiki ./wiki --out report.md
    python visualize_report.py report.md --out report.html
"""

import argparse
import html
import json
import re
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
BULLET_RE = re.compile(r"^- (.*)$")
CONTINUATION_RE = re.compile(r"^  (.*)$")  # two-space indented continuation line
SUMMARY_LINE_RE = re.compile(r"^([A-Za-z][^:]*):\s*(\d+)\s*$")

# Patterns used to recover which page(s)/section(s) an entry is talking
# about, purely from the rendered report text — this script is fed only
# the .md file (plus, now, the JSON), it never re-runs the comparison.
PAGE_WORD_RE = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)
PAGE_PAREN_RE = re.compile(r"\(p\.(\d+)\)")
SECTION_WORD_RE = re.compile(r"\bsection\s+([A-Za-z0-9.]+)\b", re.IGNORECASE)
SECTION_BULLET_PREFIX_RE = re.compile(r"^([A-Za-z0-9.]+)\s+—")  # "5.2 — 5.2 Async..." bullets
WIKI_PATH_RE = re.compile(r"(/[A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-]+)*\.md)")  # e.g. /commands/admin/abort.md

MAX_PREVIEW_CHARS = 4000  # cap embedded JSON/wiki content per entry so the HTML file stays sane


def parse_report(md_text: str) -> dict:
    """Parses the specific markdown shape compare_json_to_wiki.py produces:
    - `#` top title
    - `##`/`###` headings, each starting a collapsible section
    - `- ` bullets, with following two-space-indented lines treated as part
      of that same entry (compare_json_to_wiki.py writes multi-line entries
      this way for table rows/paragraphs)
    - bare "Label: N" lines (anywhere) are pulled out as overview counts,
      regardless of which section they appear under, since their position
      in the source isn't always cleanly nested under one heading.
    """
    top_title = "Report"
    sections: list[dict] = []
    overview_counts: list[tuple[str, str]] = []

    current = None
    current_entry_lines: list[str] | None = None

    def flush_entry():
        nonlocal current_entry_lines
        if current is not None and current_entry_lines:
            current["entries"].append("\n".join(current_entry_lines))
        current_entry_lines = None

    for line in md_text.splitlines():
        m_head = HEADING_RE.match(line)
        if m_head:
            flush_entry()
            level = len(m_head.group(1))
            title = m_head.group(2).strip()
            if level == 1:
                top_title = title
                current = None
                continue
            current = {"level": level, "title": title, "description": [], "entries": []}
            sections.append(current)
            continue

        m_bullet = BULLET_RE.match(line)
        if m_bullet:
            flush_entry()
            current_entry_lines = [m_bullet.group(1)]
            continue

        m_cont = CONTINUATION_RE.match(line)
        if m_cont and current_entry_lines is not None:
            current_entry_lines.append(m_cont.group(1))
            continue

        stripped = line.strip()
        if not stripped:
            flush_entry()
            continue

        m_summary = SUMMARY_LINE_RE.match(stripped)
        if m_summary:
            flush_entry()
            overview_counts.append((m_summary.group(1).strip(), m_summary.group(2)))
            continue

        # Plain descriptive prose line (e.g. the parenthetical explanations
        # compare_json_to_wiki.py writes under each heading).
        flush_entry()
        if current is not None:
            current["description"].append(stripped)

    flush_entry()
    return {"title": top_title, "overview_counts": overview_counts, "sections": sections}


# ---- JSON source lookup ------------------------------------------------------
# Deliberately self-contained (not imported from compare_json_to_wiki.py) so
# this script stays a standalone file fed only inputs, no cross-file coupling.

def blocks_for_page(sections_data, page: int) -> list[dict]:
    """Recursively find every dict node whose "page" equals `page`, tagged
    with its enclosing section context. Same idea as
    compare_json_to_wiki.get_blocks_for_page, reimplemented here standalone."""
    found = []

    def walk(node, ctx):
        if isinstance(node, dict):
            new_ctx = ctx
            if "section" in node and "title" in node:
                new_ctx = {"section": node.get("section"), "title": node.get("title")}
            node_page = node.get("page")
            if node_page is not None:
                try:
                    if int(node_page) == page:
                        found.append({"context": new_ctx, "block": node})
                except (TypeError, ValueError):
                    pass
            for value in node.values():
                walk(value, new_ctx)
        elif isinstance(node, list):
            for item in node:
                walk(item, ctx)

    walk(sections_data, None)
    return found


def build_section_index(sections_data) -> dict:
    """Top-level section id -> full section dict, for entries that reference
    a whole section (e.g. 'Sections with zero wiki coverage') rather than a
    specific page."""
    index = {}
    if isinstance(sections_data, list):
        for sec in sections_data:
            if isinstance(sec, dict) and "section" in sec:
                index[str(sec["section"])] = sec
    return index


def extract_ids_from_entry(entry_text: str) -> tuple[list[int], list[str]]:
    """Recover which page(s)/section(s) an entry is talking about, from the
    rendered report text alone."""
    pages = {int(m.group(1)) for m in PAGE_WORD_RE.finditer(entry_text)}
    pages |= {int(m.group(1)) for m in PAGE_PAREN_RE.finditer(entry_text)}

    sections = {m.group(1) for m in SECTION_WORD_RE.finditer(entry_text)}
    m = SECTION_BULLET_PREFIX_RE.match(entry_text)
    if m:
        sections.add(m.group(1))

    return sorted(pages), sorted(sections)


def build_source_preview(entry_text: str, sections_data, section_index: dict) -> list[dict]:
    """For one report entry, find the actual JSON content it's referring to
    — every content block on any page number mentioned, plus the full
    section for any section id mentioned (useful for section-level misses
    that have no specific page). Returns [] if nothing matched or no JSON
    was provided."""
    if sections_data is None:
        return []

    pages, section_ids = extract_ids_from_entry(entry_text)
    parts = []

    for page in pages:
        blocks = blocks_for_page(sections_data, page)
        if blocks:
            parts.append({"label": f"page {page}", "data": blocks})

    for sid in section_ids:
        sec = section_index.get(sid)
        if sec:
            parts.append({"label": f"section {sid} (full)", "data": sec})

    return parts


def extract_wiki_paths_from_entry(entry_text: str) -> list[str]:
    """Recover which wiki page path(s) an entry mentions (e.g. from a
    'covered by: /commands/admin/abort.md' or 'claimed by ...' line)."""
    return sorted(set(WIKI_PATH_RE.findall(entry_text)))


def read_wiki_page(wiki_root: Path, wiki_path: str) -> str | None:
    """Reads a wiki-relative path safely — resolves against wiki_root and
    refuses anything that would escape it, since the path comes from
    regex-matched report text rather than a trusted internal source. Returns
    None if the file doesn't exist or the path is unsafe, rather than
    raising, so one bad match doesn't break the whole render."""
    try:
        target = (wiki_root / wiki_path.lstrip("/")).resolve()
        if wiki_root.resolve() not in target.parents and target != wiki_root.resolve():
            return None
        if not target.exists() or not target.is_file():
            return None
        return target.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None


def build_wiki_preview(entry_text: str, wiki_root) -> list[dict]:
    """For one report entry, find every wiki page it mentions by path and
    read its current actual content — lets you directly compare 'what the
    JSON says' against 'what the wiki currently has' for the same finding."""
    if wiki_root is None:
        return []
    parts = []
    for wiki_path in extract_wiki_paths_from_entry(entry_text):
        content = read_wiki_page(wiki_root, wiki_path)
        if content is not None:
            parts.append({"label": wiki_path, "data": content})
    return parts




# ---- rendering ---------------------------------------------------------------

BADGE_PATTERNS = [
    (re.compile(r"LLM verdict:\s*MISSING"), "verdict-missing", "LLM: MISSING"),
    (re.compile(r"LLM verdict:\s*PARTIAL"), "verdict-partial", "LLM: PARTIAL"),
    (re.compile(r"LLM verdict:\s*PRESENT"), "verdict-present", "LLM: PRESENT"),
    (re.compile(r"LLM verdict:\s*UNVERIFIED"), "verdict-unverified", "LLM: UNVERIFIED"),
    (re.compile(r"\[FOUND ELSEWHERE IN WIKI[^\]]*\]"), "badge-misfiled", "possibly misfiled"),
]


def _entry_html(entry_text: str, sections_data=None, section_index=None, wiki_root=None) -> str:
    escaped = html.escape(entry_text)
    badges = []
    for pattern, css_class, label in BADGE_PATTERNS:
        if pattern.search(entry_text):
            badges.append(f'<span class="badge {css_class}">{label}</span>')
    badge_html = " ".join(badges)
    body = escaped.replace("\n", "<br>")

    def _render_toggle(button_label, css_prefix, parts, is_markdown_only):
        if not parts:
            return ""
        blocks_html = []
        for part in parts:
            if is_markdown_only:
                dumped = part["data"]
            else:
                dumped = json.dumps(part["data"], indent=2, ensure_ascii=False)
            if len(dumped) > MAX_PREVIEW_CHARS:
                dumped = dumped[:MAX_PREVIEW_CHARS] + "\n... [truncated]"
            blocks_html.append(
                f'<div class="preview-block">'
                f'<div class="preview-label">{html.escape(part["label"])}</div>'
                f'<pre>{html.escape(dumped)}</pre></div>'
            )
        return (
            f'<button class="toggle-source {css_prefix}" '
            "onclick=\"this.nextElementSibling.classList.toggle('open'); "
            "this.textContent = this.textContent.startsWith('▸') ? "
            "this.textContent.replace('▸','▾') : this.textContent.replace('▾','▸');\">"
            f"▸ {button_label}</button>"
            f'<div class="source-preview {css_prefix}">{"".join(blocks_html)}</div>'
        )

    json_preview_html = ""
    if sections_data is not None:
        json_preview_html = _render_toggle(
            "View JSON source", "json-preview",
            build_source_preview(entry_text, sections_data, section_index or {}),
            is_markdown_only=False,
        )

    wiki_preview_html = ""
    if wiki_root is not None:
        wiki_preview_html = _render_toggle(
            "View wiki page(s)", "wiki-preview",
            build_wiki_preview(entry_text, wiki_root),
            is_markdown_only=True,
        )

    return (
        f'<div class="entry" data-search="{escaped.lower()}">{badge_html}'
        f'<div class="entry-body">{body}</div>{json_preview_html}{wiki_preview_html}</div>'
    )


def _section_tone(title: str) -> str:
    t = title.lower()
    if "no coverage gaps" in t:
        return "tone-ok"
    if "missing" in t or "skipped" in t or "not found" in t:
        return "tone-warn"
    if "never seen" in t or "not in json" in t:
        return "tone-info"
    return "tone-neutral"


def render_html(parsed: dict, sections_data=None, wiki_root=None) -> str:
    section_index = build_section_index(sections_data) if sections_data is not None else {}

    overview_cards = "".join(
        f'<div class="stat-card"><div class="stat-value">{count}</div><div class="stat-label">{html.escape(label)}</div></div>'
        for label, count in parsed["overview_counts"]
    )

    sections_html = []
    for sec in parsed["sections"]:
        tone = _section_tone(sec["title"])
        desc = " ".join(sec["description"])
        entries_html = "".join(_entry_html(e, sections_data, section_index, wiki_root) for e in sec["entries"])
        count = len(sec["entries"])
        open_attr = "open" if count > 0 and tone == "tone-warn" else ""
        empty_note = "" if count else '<p class="empty-note">Nothing flagged here.</p>'
        sections_html.append(f"""
        <details class="section {tone}" {open_attr}>
          <summary>
            <span class="section-title">{html.escape(sec['title'])}</span>
            <span class="section-count">{count}</span>
          </summary>
          {f'<p class="section-desc">{html.escape(desc)}</p>' if desc else ''}
          {entries_html}
          {empty_note}
        </details>""")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(parsed['title'])}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #F4F5F7; color: #172B4D; }}
  #header {{ padding: 16px 24px; background: #172B4D; color: white; position: sticky; top: 0; z-index: 10; }}
  #header h1 {{ margin: 0; font-size: 18px; }}
  #search {{
    width: 100%; max-width: 480px; margin-top: 10px; padding: 8px 12px;
    border-radius: 4px; border: 1px solid #DFE1E6; font-size: 13px;
  }}
  #stats {{ display: flex; gap: 12px; flex-wrap: wrap; padding: 16px 24px; }}
  .stat-card {{
    background: white; border: 1px solid #DFE1E6; border-radius: 6px;
    padding: 10px 16px; min-width: 120px;
  }}
  .stat-value {{ font-size: 20px; font-weight: 700; }}
  .stat-label {{ font-size: 11px; color: #6B778C; }}
  #content {{ padding: 0 24px 40px; max-width: 900px; }}
  details.section {{
    background: white; border: 1px solid #DFE1E6; border-radius: 6px;
    margin-bottom: 12px; padding: 4px 16px;
  }}
  details.section summary {{
    cursor: pointer; padding: 10px 0; display: flex; align-items: center;
    gap: 10px; font-weight: 600; list-style: none;
  }}
  details.section summary::-webkit-details-marker {{ display: none; }}
  .section-count {{
    background: #DFE1E6; color: #172B4D; border-radius: 10px; padding: 1px 9px;
    font-size: 12px; font-weight: 700;
  }}
  .tone-warn .section-count {{ background: #FFEBE6; color: #BF2600; }}
  .tone-info .section-count {{ background: #FFF7E6; color: #974F0C; }}
  .tone-ok .section-count {{ background: #E3FCEF; color: #006644; }}
  .section-desc {{ color: #6B778C; font-size: 13px; margin: 0 0 10px; }}
  .entry {{
    border-top: 1px solid #F4F5F7; padding: 8px 0; font-size: 13px;
    font-family: -apple-system, sans-serif;
  }}
  .entry-body {{ white-space: normal; line-height: 1.5; margin-top: 4px; }}
  .empty-note {{ color: #6B778C; font-size: 13px; font-style: italic; }}
  .badge {{
    display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 7px;
    border-radius: 10px; margin-right: 6px; text-transform: uppercase;
  }}
  .verdict-missing {{ background: #FFEBE6; color: #BF2600; }}
  .verdict-partial {{ background: #FFF7E6; color: #974F0C; }}
  .verdict-present {{ background: #E3FCEF; color: #006644; }}
  .verdict-unverified {{ background: #EAECF0; color: #42526E; }}
  .badge-misfiled {{ background: #DEEBFF; color: #0052CC; }}
  .entry.hidden {{ display: none; }}
  details.section.all-hidden {{ display: none; }}
  .toggle-source {{
    background: none; border: none; color: #0052CC; font-size: 12px;
    cursor: pointer; padding: 4px 0; font-family: inherit;
  }}
  .toggle-source:hover {{ text-decoration: underline; }}
  .source-preview {{
    display: none; background: #0B0E11; border-radius: 6px;
    padding: 10px 14px; margin-top: 6px;
  }}
  .source-preview.open {{ display: block; }}
  .preview-label {{
    font-family: "SFMono-Regular", Consolas, monospace; font-size: 11px;
    color: #8A97A0; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.03em;
  }}
  .source-preview pre {{
    margin: 0 0 12px; white-space: pre-wrap; word-break: break-word;
    font-size: 12px; color: #CFD8DC; font-family: "SFMono-Regular", Consolas, monospace;
  }}
  .source-preview pre:last-child {{ margin-bottom: 0; }}
  .toggle-source.wiki-preview {{ color: #006644; }}
  .source-preview.wiki-preview {{ background: #06301F; }}
  .source-preview.wiki-preview .preview-label {{ color: #79E2B7; }}
  .source-preview.wiki-preview pre {{ color: #D3F5E6; }}
</style>
</head>
<body>
  <div id="header">
    <h1>{html.escape(parsed['title'])}</h1>
    <input id="search" type="text" placeholder="Filter findings (page number, section id, filename, term...)"/>
  </div>
  <div id="stats">{overview_cards}</div>
  <div id="content">
    {''.join(sections_html)}
  </div>
  <script>
    const searchBox = document.getElementById('search');
    searchBox.addEventListener('input', () => {{
      const q = searchBox.value.trim().toLowerCase();
      document.querySelectorAll('details.section').forEach(section => {{
        let visibleCount = 0;
        section.querySelectorAll('.entry').forEach(entry => {{
          const match = !q || entry.dataset.search.includes(q);
          entry.classList.toggle('hidden', !match);
          if (match) visibleCount++;
        }});
        const hasNoEntries = section.querySelectorAll('.entry').length === 0;
        section.classList.toggle('all-hidden', !hasNoEntries && q && visibleCount === 0);
        if (q && visibleCount > 0) section.open = true;
      }});
    }});
  </script>
</body>
</html>
"""


# ---- CLI ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_md", help="path to the .md report produced by compare_json_to_wiki.py")
    ap.add_argument("--json", default=None,
                     help="path to the same source JSON that was compared — if given, clicking "
                          "'View JSON source' on any finding shows the actual raw content it's "
                          "referring to, pulled straight from this file.")
    ap.add_argument("--wiki", default=None,
                     help="path to the wiki root directory — if given, clicking 'View wiki "
                          "page(s)' on any finding that mentions a wiki path shows that page's "
                          "current actual content, so you can compare it directly against the "
                          "JSON source for the same finding.")
    ap.add_argument("--out", default=None, help="output HTML path (default: same name, .html)")
    args = ap.parse_args()

    md_path = Path(args.report_md)
    out_path = Path(args.out) if args.out else md_path.with_suffix(".html")

    sections_data = None
    if args.json:
        sections_data = json.loads(Path(args.json).read_text(encoding="utf-8"))

    wiki_root = Path(args.wiki).resolve() if args.wiki else None

    parsed = parse_report(md_path.read_text(encoding="utf-8"))
    html_out = render_html(parsed, sections_data, wiki_root)
    out_path.write_text(html_out, encoding="utf-8")

    extras = []
    if sections_data is not None:
        extras.append("JSON source previews")
    if wiki_root is not None:
        extras.append("wiki page previews")
    suffix = f" (with {', '.join(extras)})" if extras else ""
    print(f"wrote {out_path}{suffix}")


if __name__ == "__main__":
    main()
