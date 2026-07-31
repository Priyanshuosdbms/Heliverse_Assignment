"""
build_graph.py — generates an interactive graph view of the NVMe wiki.

Reads every concept .md file's `type` + `relates_to` frontmatter (plus any
plain markdown links in the body, as a fallback for pages without relates_to
yet), builds a node/edge graph, and writes a single self-contained HTML file
you open in a browser.

Unlike a "show everything at once" force graph, this one lets you:
  - search/select a single node to focus on
  - pick a degree (1-6 hops) — only nodes within that many hops of the
    focused node stay fully visible; everything else dims out
  - click any visible node to re-focus on it and keep exploring outward
  - reset to see the whole graph again

Usage:
    python build_graph.py [--wiki ./wiki] [--out graph.html]
"""

import argparse
import json
import re
from pathlib import Path

import yaml  # pip install pyyaml

LINK_RE = re.compile(r"\[[^\]]*\]\((/[^)\s]+\.md)\)")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    try:
        _, fm_text, body = text.split("---", 2)
        fm = yaml.safe_load(fm_text) or {}
        return fm, body
    except (ValueError, yaml.YAMLError):
        return {}, text


def build_graph(wiki_root: Path) -> dict:
    nodes = {}
    edges = []
    seen_edges = set()

    concept_files = [
        p for p in wiki_root.rglob("*.md")
        if p.name not in ("index.md", "log.md") and "_lint-reports" not in p.parts
    ]

    for md_file in concept_files:
        rel_path = "/" + str(md_file.relative_to(wiki_root)).replace("\\", "/")
        text = md_file.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)

        nodes[rel_path] = {
            "id": rel_path,
            "label": fm.get("title") or md_file.stem.replace("-", " ").title(),
            "type": fm.get("type", "Unknown"),
            "description": fm.get("description", ""),
        }

        # Structured edges from relates_to frontmatter (kind is known)
        for rel in fm.get("relates_to", []) or []:
            if not isinstance(rel, dict):
                continue
            target = rel.get("path", "")
            kind = rel.get("kind", "related")
            if not target:
                continue
            key = (rel_path, target, kind)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({"source": rel_path, "target": target, "kind": kind})

        # Fallback edges from plain prose links (kind unknown -> "linked")
        for target in LINK_RE.findall(body):
            key = (rel_path, target, "linked")
            if key not in seen_edges and (rel_path, target) not in {(e["source"], e["target"]) for e in edges}:
                seen_edges.add(key)
                edges.append({"source": rel_path, "target": target, "kind": "linked"})

    # Ensure every edge endpoint exists as a node, even if only forward-referenced
    # (not yet ingested) — shown as a "stub" node so degree-filtering still works.
    for e in edges:
        for endpoint in (e["source"], e["target"]):
            if endpoint not in nodes:
                nodes[endpoint] = {
                    "id": endpoint,
                    "label": Path(endpoint).stem.replace("-", " ").title() + " (not yet ingested)",
                    "type": "Stub",
                    "description": "Forward-referenced but not yet ingested.",
                }

    return {"nodes": list(nodes.values()), "edges": edges}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>NVMe Wiki — Concept Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
  :root {
    --bg: #0b0e11;
    --panel: #12161b;
    --line: #232a31;
    --text: #cfd8dc;
    --muted: #5b6b73;
    --accent: #ff8a3d;
    --mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
    --sans: "IBM Plex Sans", system-ui, sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: var(--sans); overflow: hidden;
  }
  #graph { width: 100vw; height: 100vh; display: block; }

  #panel {
    position: fixed; top: 16px; left: 16px; width: 300px;
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 6px; padding: 14px 16px; z-index: 10;
  }
  #panel h1 {
    font-family: var(--mono); font-size: 12px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 12px;
  }
  #panel label {
    display: block; font-size: 11px; color: var(--muted);
    margin: 10px 0 4px; font-family: var(--mono);
  }
  #search {
    width: 100%; background: #0b0e11; border: 1px solid var(--line);
    color: var(--text); padding: 6px 8px; border-radius: 4px;
    font-family: var(--sans); font-size: 13px;
  }
  #degree { width: 100%; }
  #degreeVal { font-family: var(--mono); color: var(--accent); }
  #reset {
    margin-top: 12px; width: 100%; padding: 7px; background: transparent;
    border: 1px solid var(--accent); color: var(--accent); border-radius: 4px;
    cursor: pointer; font-family: var(--mono); font-size: 12px;
  }
  #reset:hover { background: var(--accent); color: #0b0e11; }
  #focusLabel {
    font-family: var(--mono); font-size: 12px; color: var(--text);
    margin-top: 10px; word-break: break-all; min-height: 1.2em;
  }
  #legend {
    position: fixed; bottom: 16px; left: 16px; background: var(--panel);
    border: 1px solid var(--line); border-radius: 6px; padding: 10px 14px;
    font-family: var(--mono); font-size: 11px; z-index: 10;
  }
  .legend-item { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
  .swatch { width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }

  #tooltip {
    position: fixed; pointer-events: none; background: var(--panel);
    border: 1px solid var(--line); border-radius: 5px; padding: 8px 10px;
    font-size: 12px; max-width: 260px; z-index: 20; display: none;
  }
  #tooltip .t-title { font-weight: 600; color: var(--text); margin-bottom: 2px; }
  #tooltip .t-type { font-family: var(--mono); color: var(--accent); font-size: 10px; }
  #tooltip .t-desc { color: var(--muted); margin-top: 4px; }

  .node-label {
    font-family: var(--sans); font-size: 10px; fill: var(--text);
    pointer-events: none;
  }
  #datalist-container { position: relative; }
</style>
</head>
<body>

<div id="panel">
  <h1>NVMe Wiki Graph</h1>
  <label for="search">Focus node</label>
  <input id="search" list="nodeList" placeholder="search a concept…" autocomplete="off"/>
  <datalist id="nodeList"></datalist>
  <div id="focusLabel"></div>

  <label for="degree">Degree of separation: <span id="degreeVal">2</span></label>
  <input id="degree" type="range" min="1" max="6" value="2"/>

  <button id="reset">Show whole graph</button>
</div>

<div id="legend"></div>
<div id="tooltip"></div>
<svg id="graph"></svg>

<script>
const GRAPH = __GRAPH_JSON__;

const TYPE_COLORS = {
  "Command": "#5ec8ff",
  "Log Page": "#8f7bff",
  "Data Structure": "#54d18a",
  "Status Code Table": "#ff5e7e",
  "Feature": "#ffc857",
  "Extended Capability": "#ff8a3d",
  "Architecture Concept": "#7bd3d8",
  "Unknown": "#8a97a0",
  "Stub": "#3a4149"
};

const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("#graph").attr("viewBox", [0, 0, width, height]);
const container = svg.append("g");

svg.call(d3.zoom().scaleExtent([0.15, 6]).on("zoom", (ev) => {
  container.attr("transform", ev.transform);
}));

// Build adjacency for BFS degree filtering
const adjacency = {};
GRAPH.nodes.forEach(n => adjacency[n.id] = new Set());
GRAPH.edges.forEach(e => {
  const s = typeof e.source === "object" ? e.source.id : e.source;
  const t = typeof e.target === "object" ? e.target.id : e.target;
  if (adjacency[s]) adjacency[s].add(t);
  if (adjacency[t]) adjacency[t].add(s);
});

function bfsWithinDegree(startId, maxDegree) {
  const visited = new Map([[startId, 0]]);
  let frontier = [startId];
  for (let d = 1; d <= maxDegree; d++) {
    const next = [];
    frontier.forEach(id => {
      (adjacency[id] || new Set()).forEach(neigh => {
        if (!visited.has(neigh)) {
          visited.set(neigh, d);
          next.push(neigh);
        }
      });
    });
    frontier = next;
  }
  return visited; // Map<nodeId, degreeFromFocus>
}

const simulation = d3.forceSimulation(GRAPH.nodes)
  .force("link", d3.forceLink(GRAPH.edges).id(d => d.id).distance(90).strength(0.5))
  .force("charge", d3.forceManyBody().strength(-220))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collide", d3.forceCollide(22));

const link = container.append("g")
  .selectAll("line")
  .data(GRAPH.edges)
  .join("line")
  .attr("stroke", "#2a323a")
  .attr("stroke-width", 1.1);

const node = container.append("g")
  .selectAll("circle")
  .data(GRAPH.nodes)
  .join("circle")
  .attr("r", 7)
  .attr("fill", d => TYPE_COLORS[d.type] || TYPE_COLORS.Unknown)
  .attr("stroke", "#0b0e11")
  .attr("stroke-width", 1.5)
  .style("cursor", "pointer")
  .call(d3.drag()
    .on("start", (ev, d) => { if (!ev.active) simulation.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
    .on("end", (ev, d) => { if (!ev.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

const label = container.append("g")
  .selectAll("text")
  .data(GRAPH.nodes)
  .join("text")
  .attr("class", "node-label")
  .attr("dx", 10)
  .attr("dy", 3)
  .text(d => d.label);

simulation.on("tick", () => {
  link
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  node.attr("cx", d => d.x).attr("cy", d => d.y);
  label.attr("x", d => d.x).attr("y", d => d.y);
});

// ---- Legend ----
const legend = d3.select("#legend");
Object.entries(TYPE_COLORS).forEach(([type, color]) => {
  const row = legend.append("div").attr("class", "legend-item");
  row.append("div").attr("class", "swatch").style("background", color);
  row.append("div").text(type);
});

// ---- Search datalist ----
const datalist = d3.select("#nodeList");
GRAPH.nodes.filter(n => n.type !== "Stub").forEach(n => {
  datalist.append("option").attr("value", n.label).attr("data-id", n.id);
});
const idByLabel = Object.fromEntries(GRAPH.nodes.map(n => [n.label, n.id]));

// ---- Tooltip ----
const tooltip = d3.select("#tooltip");
node.on("mouseenter", (ev, d) => {
  tooltip.style("display", "block")
    .html(`<div class="t-title">${d.label}</div>
           <div class="t-type">${d.type}</div>
           ${d.description ? `<div class="t-desc">${d.description}</div>` : ""}`);
}).on("mousemove", (ev) => {
  tooltip.style("left", (ev.clientX + 14) + "px").style("top", (ev.clientY + 14) + "px");
}).on("mouseleave", () => tooltip.style("display", "none"));

// ---- Focus / degree filtering ----
let focusId = null;

function applyFilter() {
  const maxDegree = +d3.select("#degree").property("value");
  d3.select("#degreeVal").text(maxDegree);

  if (!focusId) {
    node.style("opacity", 1);
    link.style("opacity", 0.6);
    label.style("opacity", 1);
    return;
  }

  const within = bfsWithinDegree(focusId, maxDegree);
  node.style("opacity", d => within.has(d.id) ? 1 : 0.08);
  label.style("opacity", d => within.has(d.id) ? 1 : 0.05);
  link.style("opacity", d => {
    const s = typeof d.source === "object" ? d.source.id : d.source;
    const t = typeof d.target === "object" ? d.target.id : d.target;
    return (within.has(s) && within.has(t)) ? 0.85 : 0.03;
  });
}

function setFocus(id) {
  focusId = id;
  const n = GRAPH.nodes.find(n => n.id === id);
  d3.select("#focusLabel").text(n ? `Focused: ${n.label}` : "");
  applyFilter();
}

node.on("click", (ev, d) => setFocus(d.id));

d3.select("#search").on("change", function () {
  const id = idByLabel[this.value];
  if (id) setFocus(id);
});

d3.select("#degree").on("input", applyFilter);

d3.select("#reset").on("click", () => {
  focusId = null;
  d3.select("#search").property("value", "");
  d3.select("#focusLabel").text("");
  applyFilter();
});

applyFilter();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default="./wiki", help="path to the wiki root")
    ap.add_argument("--out", default="graph.html", help="output HTML file path")
    args = ap.parse_args()

    wiki_root = Path(args.wiki).resolve()
    graph = build_graph(wiki_root)

    html = HTML_TEMPLATE.replace("__GRAPH_JSON__", json.dumps(graph))
    out_path = Path(args.out).resolve()
    out_path.write_text(html, encoding="utf-8")

    print(f"{len(graph['nodes'])} nodes, {len(graph['edges'])} edges -> {out_path}")


if __name__ == "__main__":
    main()
