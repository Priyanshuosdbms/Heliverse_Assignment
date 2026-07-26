"""
wiki_agent_deepagents.py — drives the OKF-conformant NVMe wiki using LangChain's
`deepagents` library, with a local vLLM server (Qwen3.6-FP8 or similar) as the model.

Usage:
    python wiki_agent_deepagents.py ingest <path-to-source.json-or-txt>
    python wiki_agent_deepagents.py query "your question"
    python wiki_agent_deepagents.py lint

Install:
    pip install deepagents langchain-openai

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

import os
import sys
from datetime import date
from pathlib import Path

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# ---- Config ----------------------------------------------------------------

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("WIKI_MODEL", "qwen3.6-fp8")  # match --served-model-name
WIKI_ROOT = Path(os.environ.get("WIKI_ROOT", "./wiki")).resolve()
SCHEMA_PATH = Path(os.environ.get("WIKI_SCHEMA", "./nvme-wiki-schema.md")).resolve()

WIKI_ROOT.mkdir(parents=True, exist_ok=True)

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

agent = create_deep_agent(
    model=model,
    tools=[append_log, grep_index],
    system_prompt=SYSTEM_PROMPT,
    backend=backend,
)


# ---- CLI --------------------------------------------------------------------


def run(instruction: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": instruction}]})
    return result["messages"][-1].content


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "ingest":
        source_path = Path(sys.argv[2]).resolve()
        source_text = source_path.read_text(encoding="utf-8")
        instruction = (
            f"Ingest this source ({source_path.name}). Walk it section by section, "
            "apply the filing rules from the schema, write/update concept files "
            "with full OKF frontmatter and citations, update the relevant "
            "index.md files, and call append_log for today's date.\n\n"
            f"SOURCE CONTENT:\n\n{source_text}"
        )
        print(run(instruction))

    elif cmd == "query":
        question = " ".join(sys.argv[2:])
        instruction = (
            f"Answer this question using the wiki: {question}\n"
            "Use grep_index and read_file to find relevant concept files first. "
            "Cite which concept files you used."
        )
        print(run(instruction))

    elif cmd == "lint":
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
