"""
build_graph.py — scans the OKF NVMe wiki and generates a self-contained
vis-network HTML visualization of concepts and their correlations.

Usage:
    python build_graph.py [wiki_root] [output.html]

    Defaults: wiki_root="./wiki", output="./wiki_graph.html"

Install:
    pip install pyyaml

What it does:
    - Walks every concept .md file (skips index.md/log.md).
    - Reads `type` and `title` from frontmatter -> becomes a graph node,
      colored by type.
    - Reads `relates_to` frontmatter entries -> becomes graph edges, labeled
      by `kind` (used-by, depends-on, etc.) — the structured correlation
      layer we added on top of OKF.
    - ALSO scans the markdown body for plain OKF-style prose links
      (`](/path/to/file.md)`) and adds those as edges too (dashed, since
      they weren't necessarily backed by a relates_to entry) — this means
      the graph reflects real correlations even for pages that only used
      OKF's native link mechanism and skipped the relates_to extension.
    - Embeds the resulting node/edge data directly into the HTML (no
      separate JSON fetch), so the file opens correctly via file:// with
      no CORS issues, and works fully offline except for the vis-network
      CDN script.

Open the output file in a browser afterward — no server needed.
"""

import json
import re
import sys
from pathlib import Path

import yaml

LINK_RE = re.compile(r"\]\(\s*(/[^)\s]+\.md)\s*\)")

TYPE_COLORS = {
    "Command": "#4C9AFF",
    "Log Page": "#57D9A3",
    "Data Structure": "#FFAB00",
    "Status Code Table": "#FF5630",
    "Feature": "#998DD9",
    "Extended Capability": "#00B8D9",
    "Architecture Concept": "#6554C0",
}
DEFAULT_COLOR = "#B3BAC5"


def parse_concept(md_path: Path, wiki_root: Path) -> dict | None:
    if md_path.name in ("index.md", "log.md"):
        return None
    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    try:
        _, fm_text, body = text.split("---", 2)
        fm = yaml.safe_load(fm_text) or {}
    except ValueError:
        return None

    rel_path = "/" + str(md_path.relative_to(wiki_root)).replace("\\", "/")
    node_type = fm.get("type", "Unknown")
    title = fm.get("title") or md_path.stem.replace("-", " ").title()
    description = fm.get("description", "")

    edges = []
    for rel in fm.get("relates_to", []) or []:
        if isinstance(rel, dict) and rel.get("path"):
            edges.append({"target": rel["path"], "label": rel.get("kind", ""), "structured": True})

    for match in LINK_RE.findall(body):
        target = match if match.startswith("/") else "/" + match
        if not any(e["target"] == target for e in edges):
            edges.append({"target": target, "label": "", "structured": False})

    return {
        "id": rel_path,
        "label": title,
        "type": node_type,
        "description": description,
        "edges": edges,
    }


def build_graph(wiki_root: Path) -> tuple[list[dict], list[dict]]:
    concepts = {}
    for md_file in wiki_root.rglob("*.md"):
        parsed = parse_concept(md_file, wiki_root)
        if parsed:
            concepts[parsed["id"]] = parsed

    nodes = []
    for cid, c in concepts.items():
        nodes.append({
            "id": cid,
            "label": c["label"],
            "title": f"{c['label']} ({c['type']})\n{c['description']}".strip(),
            "color": TYPE_COLORS.get(c["type"], DEFAULT_COLOR),
            "group": c["type"],
        })

    # Include forward-referenced targets that don't exist as files yet, so
    # broken/forward links (valid per OKF §5.3) still show up in the graph
    # as visibly distinct "not yet ingested" nodes rather than being dropped.
    seen_ids = set(concepts.keys())
    edges = []
    for cid, c in concepts.items():
        for e in c["edges"]:
            target = e["target"]
            if target not in seen_ids:
                nodes.append({
                    "id": target,
                    "label": Path(target).stem.replace("-", " ").title() + "\n(not yet ingested)",
                    "title": "Forward reference — concept not yet written",
                    "color": "#DFE1E6",
                    "group": "Not Yet Ingested",
                })
                seen_ids.add(target)
            edges.append({
                "from": cid,
                "to": target,
                "label": e["label"],
                "dashes": not e["structured"],
                "arrows": "to",
            })

    return nodes, edges


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NVMe Wiki Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; }}
    #header {{ padding: 12px 20px; background: #172B4D; color: white; }}
    #header h1 {{ margin: 0; font-size: 16px; font-weight: 600; }}
    #header p {{ margin: 4px 0 0; font-size: 12px; color: #B3BAC5; }}
    #legend {{ padding: 10px 20px; font-size: 12px; display: flex; gap: 16px; flex-wrap: wrap; background: #F4F5F7; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
    #graph {{ width: 100%; height: calc(100vh - 90px); }}
  </style>
</head>
<body>
  <div id="header">
    <h1>NVMe OKF Wiki — Concept Graph</h1>
    <p>{node_count} concepts, {edge_count} correlations. Dashed edges = prose-link-only; solid = structured relates_to. Gray nodes = forward references not yet ingested.</p>
  </div>
  <div id="legend">{legend_html}</div>
  <div id="graph"></div>
  <script>
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    const container = document.getElementById('graph');
    const data = {{ nodes, edges }};
    const options = {{
      nodes: {{ shape: 'dot', size: 14, font: {{ size: 12 }} }},
      edges: {{ font: {{ size: 10, align: 'middle' }}, smooth: {{ type: 'continuous' }} }},
      physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -6000, springLength: 140 }} }},
      groups: {{}},
      interaction: {{ hover: true, tooltipDelay: 100 }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""


def main():
    wiki_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("./wiki").resolve()
    output = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path("./wiki_graph.html").resolve()

    if not wiki_root.exists():
        print(f"error: wiki root not found: {wiki_root}")
        sys.exit(1)

    nodes, edges = build_graph(wiki_root)

    all_types = sorted({n["group"] for n in nodes})
    legend_html = "".join(
        f'<div class="legend-item"><span class="swatch" style="background:{TYPE_COLORS.get(t, "#DFE1E6")}"></span>{t}</div>'
        for t in all_types
    )

    html = HTML_TEMPLATE.format(
        node_count=len([n for n in nodes if n["group"] != "Not Yet Ingested"]),
        edge_count=len(edges),
        legend_html=legend_html,
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges),
    )
    output.write_text(html, encoding="utf-8")
    print(f"wrote {output} — {len(nodes)} nodes, {len(edges)} edges. Open it in a browser.")


if __name__ == "__main__":
    main()
