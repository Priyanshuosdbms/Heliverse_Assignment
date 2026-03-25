def parse_toc_to_ranges(pdf, start_page: int, end_page: int) -> list[tuple[str, int, int]]:
    """
    Parses TOC once and returns (register_name, start_page, end_page) tuples.

    User provides the doc page range where registers are listed in the TOC.
    Every subsection heading (x.x.x format) within that range is treated
    as a register name — no prefix matching, no filtering.

    Args:
        pdf        : pdfplumber PDF object
        start_page : first doc page of the register TOC range (1-indexed)
        end_page   : last doc page of the register TOC range (1-indexed)

    Returns:
        [("IC_CON", 161, 167), ("IC_TAR", 168, 170), ...]
    """
    import re

    # Matches any TOC line with a subsection number x.x.x and a page at the end
    # e.g. "5.1.1 IC_CON . . . . 161"  or  "5.1.1 IC_CON 161"
    SUBSECTION_RE = re.compile(
        r"^\s*\d+\.\d+\.\d+\s+(.+?)\s*[.\s]*\s*(\d+)\s*$"
    )

    entries = []  # intermediate: [(name, page), ...]
    seen = set()

    # Convert 1-indexed doc pages to 0-indexed PDF pages
    for pidx in range(start_page - 1, end_page):
        if pidx >= len(pdf.pages):
            break
        text = pdf.pages[pidx].extract_text() or ""
        for line in text.splitlines():
            m = SUBSECTION_RE.match(line.strip())
            if not m:
                continue
            name = m.group(1).strip()
            try:
                page = int(m.group(2))
            except ValueError:
                continue
            if name not in seen:
                seen.add(name)
                entries.append((name, page))

    # Sort by page number — order is authoritative
    entries.sort(key=lambda x: x[1])

    # Build (name, start_page, end_page) from consecutive entries
    result = []
    for i, (name, page) in enumerate(entries):
        s = page
        e = entries[i + 1][1] - 1 if i + 1 < len(entries) else end_page
        e = max(e, s)
        result.append((name, s, e))

    return result