"""
CLI for the NVMe 2.0 wiki agent.

Two ingest modes, per the agreed cadence:

  ingest-batch <glob>   Batch-ingest many raw chunks with minimal supervision.
                         Automatically runs a full lint pass afterward — this
                         is enforced HERE in code (two sequential agent.invoke
                         calls), not left to the model to remember via prompt,
                         because "lint after every batch" is a hard invariant
                         you want regardless of how the ingest run went.

  ingest-one <chunk>     Ingest a single raw chunk with a full review summary
                         printed back. No auto-lint — you're reviewing each
                         change yourself as you go, so a lint pass after every
                         single small ingest is unnecessary noise; run `lint`
                         manually whenever you want one.

  lint                   Run a lint pass on demand.

  query <question>       Ask a question against the existing wiki.
"""

import argparse
import glob
import os
import sys

from .agent import build_orchestrator, check_vllm_connection


def _print_section(title: str, content: str) -> None:
    print(f"\n--- {title} ---")
    print(content)


def cmd_ingest_batch(args: argparse.Namespace) -> None:
    chunks = sorted(glob.glob(args.pattern, recursive=True))
    if not chunks:
        print(f"No raw chunks matched pattern: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    agent = build_orchestrator()

    print(f"Batch-ingesting {len(chunks)} chunk(s):")
    for c in chunks:
        print(f"  - {c}")

    ingest_msg = (
        "Ingest every raw chunk in this list via the ingest-agent subagent, "
        "one after another, without stopping for my approval between them:\n"
        f"{chunks}\n\n"
        "You may split these across multiple ingest-agent calls if that "
        "keeps each call's context manageable (e.g. one call per chunk, or "
        "grouped by directory). When all are done, summarize every page "
        "created or edited across the whole batch."
    )
    ingest_result = agent.invoke({"messages": [{"role": "user", "content": ingest_msg}]})
    _print_section("Ingest summary", ingest_result["messages"][-1].content)

    # Hard-coded, not prompt-dependent: lint always runs after a batch.
    print("\nRunning automatic lint pass (required after every ingest batch)...")
    lint_msg = (
        "Run a full lint pass over the wiki now via the lint-agent subagent, "
        "covering everything that was just ingested in this batch."
    )
    lint_result = agent.invoke({"messages": [{"role": "user", "content": lint_msg}]})
    _print_section("Lint findings", lint_result["messages"][-1].content)


def cmd_ingest_one(args: argparse.Namespace) -> None:
    agent = build_orchestrator()
    msg = (
        f"Ingest exactly this one raw chunk via the ingest-agent subagent: "
        f"{args.chunk}\n\n"
        "After it finishes, return its full summary verbatim — every page "
        "created or edited, and why — so I can review before the next "
        "chunk. Do not run lint; I'm reviewing this one manually."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": msg}]})
    _print_section("Ingest summary (review before continuing)", result["messages"][-1].content)


def cmd_lint(args: argparse.Namespace) -> None:
    agent = build_orchestrator()
    msg = "Run a full lint pass over the wiki now via the lint-agent subagent."
    result = agent.invoke({"messages": [{"role": "user", "content": msg}]})
    _print_section("Lint findings", result["messages"][-1].content)


def cmd_query(args: argparse.Namespace) -> None:
    agent = build_orchestrator()
    result = agent.invoke({"messages": [{"role": "user", "content": args.question}]})
    _print_section("Answer", result["messages"][-1].content)


def main() -> None:
    parser = argparse.ArgumentParser(description="NVMe 2.0 OKF wiki agent")
    sub = parser.add_subparsers(required=True)

    p_batch = sub.add_parser(
        "ingest-batch",
        help="Batch-ingest many raw chunks with minimal supervision; auto-lints after.",
    )
    p_batch.add_argument(
        "pattern",
        help="Glob pattern under raw/, e.g. 'raw/admin_commands/*.json' "
        "(quote it so your shell doesn't expand it first).",
    )
    p_batch.set_defaults(func=cmd_ingest_batch)

    p_one = sub.add_parser(
        "ingest-one",
        help="Ingest a single raw chunk with a full review summary; no auto-lint.",
    )
    p_one.add_argument("chunk", help="Path to a single raw JSON chunk, relative to raw/.")
    p_one.set_defaults(func=cmd_ingest_one)

    p_lint = sub.add_parser("lint", help="Run a lint pass on demand.")
    p_lint.set_defaults(func=cmd_lint)

    p_query = sub.add_parser("query", help="Ask a question against the wiki.")
    p_query.add_argument("question")
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()

    # vLLM is assumed to already be running, started separately from this
    # harness — check it's actually reachable and serving the model we think
    # it is before doing anything, rather than failing deep inside a subagent
    # call with an opaque error.
    check_vllm_connection()

    args.func(args)


if __name__ == "__main__":
    main()
