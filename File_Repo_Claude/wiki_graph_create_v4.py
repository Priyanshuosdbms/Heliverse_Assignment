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
            edges.append({
                "target": rel["path"],
                "kind": rel.get("kind", ""),
                "description": rel.get("description", ""),
                "confidence": rel.get("confidence", "unspecified"),
                "alias_used": rel.get("alias_used"),
                "structured": True,
            })

    for match in LINK_RE.findall(body):
        target = match if match.startswith("/") else "/" + match
        if not any(e["target"] == target for e in edges):
            edges.append({
                "target": target,
                "kind": "",
                "description": "",
                "confidence": "unspecified",  # no relates_to entry backs this link
                "alias_used": None,
                "structured": False,
            })

    return {
        "id": rel_path,
        "label": title,
        "type": node_type,
        "description": description,
        "edges": edges,
    }


CONFIDENCE_DASHES = {
    "explicit": False,           # solid — deterministic citation match
    "alias-matched": [6, 3],     # dashed — deterministic term match
    "llm-inferred": [1, 4],      # dotted — model judgment only, spot-check these
    "unspecified": [3, 3],       # plain prose link, no relates_to backing it
}
CONFIDENCE_COLOR = {
    "explicit": "#2C7A4B",
    "alias-matched": "#B08900",
    "llm-inferred": "#B33A3A",
    "unspecified": "#8C9BAB",
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

            confidence = e["confidence"]
            tooltip_lines = []
            if e["kind"]:
                tooltip_lines.append(f"kind: {e['kind']}")
            if e["description"]:
                tooltip_lines.append(e["description"])
            tooltip_lines.append(f"confidence: {confidence}")
            if e["alias_used"]:
                tooltip_lines.append(f"matched alias: \"{e['alias_used']}\"")

            edges.append({
                "id": len(edges),
                "from": cid,
                "to": target,
                "label": e["kind"] or "",
                "title": "\n".join(tooltip_lines),
                "dashes": CONFIDENCE_DASHES.get(confidence, [3, 3]),
                "color": {"color": CONFIDENCE_COLOR.get(confidence, "#8C9BAB")},
                "arrows": "to",
                "confidence": confidence,  # kept for the confidence filter in the UI
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
    #confidenceLegend {{ padding: 6px 20px 10px; font-size: 12px; display: flex; gap: 16px; flex-wrap: wrap; align-items: center; background: #F4F5F7; border-top: 1px solid #DFE1E6; }}
    #confidenceLegend .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
    #body-row {{ display: flex; }}
    #panel {{
      width: 260px; flex-shrink: 0; padding: 16px; background: #FAFBFC;
      border-right: 1px solid #DFE1E6; box-sizing: border-box;
    }}
    #panel label {{ display: block; font-size: 11px; color: #6B778C; margin: 12px 0 4px; text-transform: uppercase; letter-spacing: 0.04em; }}
    #panel label:first-child {{ margin-top: 0; }}
    #search {{ width: 100%; padding: 6px 8px; border: 1px solid #DFE1E6; border-radius: 4px; font-size: 13px; box-sizing: border-box; }}
    #degree {{ width: 100%; }}
    #degreeVal {{ font-weight: 600; color: #172B4D; }}
    #focusLabel {{ font-size: 12px; color: #172B4D; margin-top: 10px; min-height: 1.4em; word-break: break-word; }}
    #reset {{
      margin-top: 14px; width: 100%; padding: 7px; background: white;
      border: 1px solid #4C9AFF; color: #4C9AFF; border-radius: 4px;
      cursor: pointer; font-size: 12px; font-weight: 600;
    }}
    #reset:hover {{ background: #4C9AFF; color: white; }}
    #hint {{ font-size: 11px; color: #6B778C; margin-top: 16px; line-height: 1.4; }}
    #edgeDetail {{
      margin-top: 16px; padding: 10px 12px; background: white;
      border: 1px solid #DFE1E6; border-radius: 5px; font-size: 12px;
    }}
    .ed-rel {{ font-weight: 600; color: #172B4D; margin-bottom: 4px; }}
    .ed-kind {{
      display: inline-block; font-size: 10px; text-transform: uppercase;
      letter-spacing: 0.04em; color: #6B778C; background: #F4F5F7;
      padding: 2px 6px; border-radius: 3px; margin-bottom: 6px;
    }}
    .ed-desc {{ color: #253858; line-height: 1.4; }}
    #graph {{ flex: 1; height: calc(100vh - 90px); }}
  </style>
</head>
<body>
  <div id="header">
    <h1>NVMe OKF Wiki — Concept Graph</h1>
    <p>{node_count} concepts, {edge_count} correlations. Dashed edges = prose-link-only; solid = structured relates_to. Gray nodes = forward references not yet ingested.</p>
  </div>
  <div id="legend">{legend_html}</div>
  <div id="confidenceLegend">
    <span style="font-weight:600;color:#6B778C;margin-right:8px;">edge confidence:</span>
    {confidence_legend_html}
  </div>
  <div id="body-row">
    <div id="panel">
      <label for="search">Focus node</label>
      <input id="search" list="nodeList" placeholder="search a concept…" autocomplete="off"/>
      <datalist id="nodeList"></datalist>
      <div id="focusLabel"></div>

      <label for="degree">Degree of separation: <span id="degreeVal">2</span></label>
      <input id="degree" type="range" min="1" max="6" value="2"/>

      <button id="reset">Show whole graph</button>
      <div id="hint">Click any node to re-focus on it and keep exploring outward. Click an edge to see exactly how the two concepts relate. Degree controls how many hops from the focused node stay visible.</div>

      <div id="edgeDetail" style="display:none;"></div>
    </div>
    <div id="graph"></div>
  </div>
  <script>
    const nodesData = {nodes_json};
    const edgesData = {edges_json};
    const nodes = new vis.DataSet(nodesData);
    const edges = new vis.DataSet(edgesData);
    const container = document.getElementById('graph');
    const data = {{ nodes, edges }};
    const options = {{
      nodes: {{ shape: 'dot', size: 14, font: {{ size: 12 }} }},
      edges: {{ font: {{ size: 10, align: 'middle' }}, smooth: {{ type: 'continuous' }} }},
      physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -6000, springLength: 140 }} }},
      groups: {{}},
      interaction: {{ hover: true, tooltipDelay: 100 }}
    }};
    const network = new vis.Network(container, data, options);

    // ---- adjacency for BFS degree filtering ----
    const adjacency = {{}};
    nodesData.forEach(n => adjacency[n.id] = new Set());
    edgesData.forEach(e => {{
      if (adjacency[e.from]) adjacency[e.from].add(e.to);
      if (adjacency[e.to]) adjacency[e.to].add(e.from);
    }});

    function bfsWithinDegree(startId, maxDegree) {{
      const visited = new Map([[startId, 0]]);
      let frontier = [startId];
      for (let d = 1; d <= maxDegree; d++) {{
        const next = [];
        frontier.forEach(id => {{
          (adjacency[id] || new Set()).forEach(neigh => {{
            if (!visited.has(neigh)) {{ visited.set(neigh, d); next.push(neigh); }}
          }});
        }});
        frontier = next;
      }}
      return visited;
    }}

    // ---- search datalist ----
    const datalist = document.getElementById('nodeList');
    const idByLabel = {{}};
    nodesData.filter(n => n.group !== "Not Yet Ingested").forEach(n => {{
      const opt = document.createElement('option');
      opt.value = n.label;
      datalist.appendChild(opt);
      idByLabel[n.label] = n.id;
    }});

    let focusId = null;

    function applyFilter() {{
      const maxDegree = +document.getElementById('degree').value;
      document.getElementById('degreeVal').textContent = maxDegree;

      if (!focusId) {{
        nodes.update(nodesData.map(n => ({{ id: n.id, hidden: false }})));
        edges.update(edgesData.map(e => ({{ id: e.id, hidden: false }})));
        return;
      }}

      const within = bfsWithinDegree(focusId, maxDegree);
      nodes.update(nodesData.map(n => ({{ id: n.id, hidden: !within.has(n.id) }})));
      edges.update(edgesData.map(e => ({{
        id: e.id,
        hidden: !(within.has(e.from) && within.has(e.to))
      }})));
    }}

    function setFocus(id) {{
      focusId = id;
      const n = nodesData.find(n => n.id === id);
      document.getElementById('focusLabel').textContent = n ? ("Focused: " + n.label) : "";
      const searchInput = document.getElementById('search');
      if (n) searchInput.value = n.label;
      applyFilter();
      network.focus(id, {{ scale: 1.1, animation: {{ duration: 400 }} }});
    }}

    network.on("click", (params) => {{
      if (params.nodes.length > 0) {{
        setFocus(params.nodes[0]);
        document.getElementById('edgeDetail').style.display = 'none';
      }} else if (params.edges.length > 0) {{
        const e = edgesData.find(e => e.id === params.edges[0]);
        if (e) {{
          const fromLabel = (nodesData.find(n => n.id === e.from) || {{}}).label || e.from;
          const toLabel = (nodesData.find(n => n.id === e.to) || {{}}).label || e.to;
          const box = document.getElementById('edgeDetail');
          box.style.display = 'block';
          box.innerHTML =
            '<div class="ed-rel">' + fromLabel + ' \u2192 ' + toLabel + '</div>' +
            (e.label ? '<div class="ed-kind">' + e.label + '</div>' : '') +
            '<div class="ed-desc">' + (e.title || 'No description recorded.').replace(/\\n/g, '<br>') + '</div>';
        }}
      }}
    }});

    document.getElementById('search').addEventListener('change', function () {{
      const id = idByLabel[this.value];
      if (id) setFocus(id);
    }});

    document.getElementById('degree').addEventListener('input', applyFilter);

    document.getElementById('reset').addEventListener('click', () => {{
      focusId = null;
      document.getElementById('search').value = "";
      document.getElementById('focusLabel').textContent = "";
      applyFilter();
      network.fit({{ animation: {{ duration: 400 }} }});
    }});
  </script>
</body>
</html>
"""


def generate_graph(wiki_root: Path, output: Path) -> tuple[int, int]:
    """Build the graph and write the HTML file. Returns (node_count, edge_count)
    so callers (e.g. the ingest script, running this automatically at the end
    of an ingest) can report a summary without re-parsing anything."""
    nodes, edges = build_graph(wiki_root)

    all_types = sorted({n["group"] for n in nodes})
    legend_html = "".join(
        f'<div class="legend-item"><span class="swatch" style="background:{TYPE_COLORS.get(t, "#DFE1E6")}"></span>{t}</div>'
        for t in all_types
    )
    confidence_legend_html = "".join(
        f'<div class="legend-item"><span class="swatch" style="background:{color}"></span>{conf}</div>'
        for conf, color in CONFIDENCE_COLOR.items()
    )

    html = HTML_TEMPLATE.format(
        node_count=len([n for n in nodes if n["group"] != "Not Yet Ingested"]),
        edge_count=len(edges),
        legend_html=legend_html,
        confidence_legend_html=confidence_legend_html,
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges),
    )
    output.write_text(html, encoding="utf-8")
    return len(nodes), len(edges)


def main():
    wiki_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("./wiki").resolve()
    output = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path("./wiki_graph.html").resolve()

    if not wiki_root.exists():
        print(f"error: wiki root not found: {wiki_root}")
        sys.exit(1)

    n_nodes, n_edges = generate_graph(wiki_root, output)
    print(f"wrote {output} — {n_nodes} nodes, {n_edges} edges. Open it in a browser.")


if __name__ == "__main__":
    main()
