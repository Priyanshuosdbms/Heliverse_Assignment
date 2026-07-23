import os
import sys
import urllib.request
import urllib.error
import json

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI

from .subagents import INGEST_SUBAGENT, LINT_SUBAGENT, QUERY_SUBAGENT
from .tools import list_raw_chunks, read_raw_chunk, snapshot_page, append_log

# Assumes vLLM is already running (started separately, outside this harness).
# These just need to match however that server was actually launched.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("NVME_MODEL", "Qwen/Qwen3.6-27B-FP8")
WIKI_DIR = os.environ.get("NVME_WIKI_DIR", "./nvme-wiki")


def check_vllm_connection(base_url: str = VLLM_BASE_URL, model_name: str = MODEL_NAME) -> None:
    """
    Fail fast, with a clear message, if the already-running vLLM server isn't
    reachable at VLLM_BASE_URL, or is serving a different model than
    NVME_MODEL. Both are easy to get out of sync when the server was started
    separately from this harness — a mismatch otherwise surfaces later as an
    opaque 'model not found' error from deep inside a subagent call.
    """
    models_url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as e:
        print(
            f"Could not reach vLLM at {base_url} ({e}).\n"
            f"Check that the already-running server is actually up, and that "
            f"VLLM_BASE_URL points at the right host/port.",
            file=sys.stderr,
        )
        sys.exit(1)

    served_models = [m.get("id") for m in payload.get("data", [])]
    if model_name not in served_models:
        print(
            f"vLLM at {base_url} is serving {served_models}, but NVME_MODEL is "
            f"set to '{model_name}'. Set NVME_MODEL to match exactly what the "
            f"running server reports, or the first real request will fail.",
            file=sys.stderr,
        )
        sys.exit(1)


def get_model(temperature: float = 0.2) -> ChatOpenAI:
    """
    ChatOpenAI pointed at the local vLLM OpenAI-compatible server (serve_qwen.sh).
    Low temperature by default — this is extraction/synthesis work, not
    creative generation; consistency across ingest runs matters more than
    variety.
    """
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=VLLM_BASE_URL,
        api_key="EMPTY",  # vLLM doesn't check this, but the client requires a value
        temperature=temperature,
    )


def build_orchestrator():
    """
    The orchestrator holds the FilesystemBackend rooted at the wiki (so all
    file operations — from it and from subagents — land on real disk, in the
    actual repo you'll git-commit) and delegates real work to subagents
    rather than doing ingest/lint/query itself.
    """
    model = get_model()
    backend = FilesystemBackend(root_dir=WIKI_DIR)

    agent = create_deep_agent(
        model=model,
        tools=[list_raw_chunks, read_raw_chunk, snapshot_page, append_log],
        system_prompt=(
            "You are the orchestrator for an NVMe 2.0 OKF-conformant "
            "knowledge wiki, rooted at the current backend directory. "
            "Delegate all real ingest, lint, and query work to the "
            "ingest-agent, lint-agent, and query-agent subagents via the "
            "task tool — do not do their work yourself. Read AGENTS.md "
            "directly only if you need the schema for your own coordination "
            "decisions (e.g. splitting a batch across multiple ingest-agent "
            "calls). When a user message describes a batch of raw chunks, "
            "delegate each to the ingest-agent; when it names a single "
            "chunk for careful review, delegate once and return the "
            "ingest-agent's full summary verbatim so the user can review it."
        ),
        subagents=[INGEST_SUBAGENT, LINT_SUBAGENT, QUERY_SUBAGENT],
        backend=backend,
    )
    return agent
