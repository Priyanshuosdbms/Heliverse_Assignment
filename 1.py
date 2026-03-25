Got it. Clean, focused, no surrounding code:

```python
import re
import json

# Matches any TOC line: "Some Entry Title . . . . 153"
TOC_LINE_RE = re.compile(r"^(.+?)\s*[.\s]{2,}\s*(\d+)\s*$")

# Register names are ALL_CAPS_WITH_UNDERSCORES, 3+ chars
REGISTER_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


def is_register_name(text: str) -> bool:
    text = text.strip()
    return bool(REGISTER_NAME_RE.match(text)) and "_" in text


def parse_toc_to_json(pdf, config: dict) -> list[dict]:
    """
    Step 1 — Read TOC pages line by line and convert to intermediate JSON.
    Only keeps entries whose page falls within the register chapter range.

    Returns:
        [{"name": "IC_CON", "page": 161}, {"name": "IC_TAR", "page": 168}, ...]
    """
    chapter_start = config["chapter6_start_doc_page"]
    chapter_end   = config["chapter6_end_doc_page"]

    entries = []
    seen = set()

    for pidx in config["toc_pages"]:
        if pidx >= len(pdf.pages):
            continue
        text = pdf.pages[pidx].extract_text() or ""
        for line in text.splitlines():
            m = TOC_LINE_RE.match(line.strip())
            if not m:
                continue
            name = m.group(1).strip()
            try:
                page = int(m.group(2))
            except ValueError:
                continue
            # Only keep entries inside the chapter range
            if not (chapter_start <= page <= chapter_end):
                continue
            if name not in seen:
                seen.add(name)
                entries.append({"name": name, "page": page})

    # Sort by page number — TOC order is authoritative
    entries.sort(key=lambda x: x["page"])
    return entries


def filter_registers(toc_json: list[dict]) -> list[tuple[str, int]]:
    """
    Step 2 — Filter intermediate JSON to register entries only.
    Drops section headings like "Register Memory Map".

    Returns:
        [("IC_CON", 161), ("IC_TAR", 168), ...]
    """
    return [
        (entry["name"], entry["page"])
        for entry in toc_json
        if is_register_name(entry["name"])
    ]


def build_register_ranges(
    registers: list[tuple[str, int]],
    chapter_end: int
) -> list[tuple[str, int, int]]:
    """
    Step 3 — Derive (name, start_page, end_page) from consecutive page numbers.
    Register N owns pages [page_N, page_{N+1} - 1].
    Last register ends at chapter_end.

    Returns:
        [("IC_CON", 161, 167), ("IC_TAR", 168, 170), ...]
    """
    result = []
    for i, (name, start_page) in enumerate(registers):
        if i + 1 < len(registers):
            end_page = registers[i + 1][1] - 1
        else:
            end_page = chapter_end
        end_page = max(end_page, start_page)  # safety: end >= start
        result.append((name, start_page, end_page))
    return result
```

Three functions, three distinct steps. You call them in sequence:

```python
# Usage
toc_json   = parse_toc_to_json(pdf, config)          # raw TOC as JSON
registers  = filter_registers(toc_json)               # (name, page) tuples
ranges     = build_register_ranges(                   # (name, start, end) tuples
                registers,
                config["chapter6_end_doc_page"]
             )
```

The intermediate `toc_json` is deliberately kept as a separate step — you can inspect it, log it, or store it alongside your other outputs to debug what the TOC contained before filtering. That makes it easier to diagnose if `is_register_name` drops something it shouldn't.