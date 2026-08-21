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
import re
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
BULLET_RE = re.compile(r"^- (.*)$")
CONTINUATION_RE = re.compile(r"^  (.*)$")  # two-space indented continuation line
SUMMARY_LINE_RE = re.compile(r"^([A-Za-z][^:]*):\s*(\d+)\s*$")


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


# ---- rendering ---------------------------------------------------------------

BADGE_PATTERNS = [
    (re.compile(r"LLM verdict:\s*MISSING"), "verdict-missing", "LLM: MISSING"),
    (re.compile(r"LLM verdict:\s*PARTIAL"), "verdict-partial", "LLM: PARTIAL"),
    (re.compile(r"LLM verdict:\s*PRESENT"), "verdict-present", "LLM: PRESENT"),
    (re.compile(r"LLM verdict:\s*UNVERIFIED"), "verdict-unverified", "LLM: UNVERIFIED"),
    (re.compile(r"\[FOUND ELSEWHERE IN WIKI[^\]]*\]"), "badge-misfiled", "possibly misfiled"),
]


def _entry_html(entry_text: str) -> str:
    escaped = html.escape(entry_text)
    badges = []
    for pattern, css_class, label in BADGE_PATTERNS:
        if pattern.search(entry_text):
            badges.append(f'<span class="badge {css_class}">{label}</span>')
    badge_html = " ".join(badges)
    # render as-is (preserves the multi-line indentation compare_json_to_wiki wrote)
    body = escaped.replace("\n", "<br>")
    return f'<div class="entry" data-search="{escaped.lower()}">{badge_html}<div class="entry-body">{body}</div></div>'


def _section_tone(title: str) -> str:
    t = title.lower()
    if "no coverage gaps" in t:
        return "tone-ok"
    if "missing" in t or "skipped" in t or "not found" in t:
        return "tone-warn"
    if "never seen" in t or "not in json" in t:
        return "tone-info"
    return "tone-neutral"


def render_html(parsed: dict) -> str:
    overview_cards = "".join(
        f'<div class="stat-card"><div class="stat-value">{count}</div><div class="stat-label">{html.escape(label)}</div></div>'
        for label, count in parsed["overview_counts"]
    )

    sections_html = []
    for sec in parsed["sections"]:
        tone = _section_tone(sec["title"])
        desc = " ".join(sec["description"])
        entries_html = "".join(_entry_html(e) for e in sec["entries"])
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
    ap.add_argument("--out", default=None, help="output HTML path (default: same name, .html)")
    args = ap.parse_args()

    md_path = Path(args.report_md)
    out_path = Path(args.out) if args.out else md_path.with_suffix(".html")

    parsed = parse_report(md_path.read_text(encoding="utf-8"))
    html_out = render_html(parsed)
    out_path.write_text(html_out, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
