"""
deepagents_batching.py
======================
Batching strategies for Deep Agents + vLLM (Qwen3-30B-A3B)

Requirements:
    pip install deepagents langchain-openai

Start your vLLM server first:
    vllm serve Qwen/Qwen3-30B-A3B \
        --tensor-parallel-size 4 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --enable-chunked-prefill \
        --max-num-seqs 256 \
        --dtype bfloat16 \
        --trust-remote-code \
        --tool-call-parser hermes \
        --enable-auto-tool-choice
"""

import json
import argparse
from pathlib import Path
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.middleware import SubAgentMiddleware


# ─────────────────────────────────────────────
# 0. Shared vLLM model connection
# ─────────────────────────────────────────────

def get_vllm_model(
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen3-30B-A3B",
    temperature: float = 0.7,
) -> ChatOpenAI:
    """
    Returns a LangChain ChatOpenAI client pointed at your local vLLM server.
    vLLM exposes an OpenAI-compatible endpoint, so no real API key is needed.
    """
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key="EMPTY",           # vLLM ignores this; required by langchain-openai
        temperature=temperature,
    )


# ─────────────────────────────────────────────
# 1. Static Batching
#    Collect a fixed number of records and
#    process them together in a single agent call.
# ─────────────────────────────────────────────

def static_batching(records: list[dict], batch_size: int = 16) -> list:
    """
    Simplest approach: slice the dataset into fixed-size chunks and
    send each chunk to a single Deep Agent invocation sequentially.

    Best for: uniform-length inputs, small datasets.
    Downside: no parallelism; each batch waits for the previous one.
    """
    model = get_vllm_model()
    agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You receive a batch of JSON records. "
            "Process each one and return a JSON array of results."
        ),
    )

    all_results = []
    total_batches = (len(records) + batch_size - 1) // batch_size

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  [Static] Processing batch {batch_num}/{total_batches} "
              f"({len(batch)} records)...")

        result = agent.invoke({
            "messages": (
                f"Process this batch of records and return results as JSON:\n"
                f"{json.dumps(batch, indent=2)}"
            )
        })
        all_results.append(result["messages"][-1].content)

    return all_results


# ─────────────────────────────────────────────
# 2. Map-Reduce Batching
#    MAP:    parallel subagents each handle one chunk
#    REDUCE: a single agent merges all summaries
# ─────────────────────────────────────────────

def map_reduce_batching(records: list[dict]) -> str:
    """
    Deep Agents spawns subagents in parallel via the built-in 'task' tool.
    Each subagent processes one chunk independently (map phase).
    The orchestrator then reduces all outputs into a final result.

    Best for: long documents, summarization, extraction over large corpora.
    The key insight: each mapper only sees its own chunk, saving context tokens.
    """
    model = get_vllm_model()

    # Map agent — processes one isolated chunk
    map_agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You receive a chunk of JSON records. "
            "Extract and summarize key information from each record. "
            "Return a concise JSON summary."
        ),
    )

    # Reduce agent — merges all map outputs
    reduce_agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You receive multiple JSON summaries from parallel processing jobs. "
            "Merge them into one final consolidated JSON result, "
            "deduplicating and reconciling any conflicts."
        ),
    )

    # Orchestrator — coordinates parallel map + sequential reduce
    orchestrator = create_deep_agent(
        model=model,
        system_prompt=(
            "You are a map-reduce orchestrator.\n"
            "Steps:\n"
            "1. Split the input dataset into chunks of ~10 records each.\n"
            "2. Use the 'task' tool to send each chunk to the 'mapper' subagent IN PARALLEL.\n"
            "3. Collect all map results.\n"
            "4. Use the 'task' tool once to send all results to the 'reducer' subagent.\n"
            "5. Return the reducer's final output."
        ),
        middleware=[SubAgentMiddleware(subagents={
            "mapper": map_agent,
            "reducer": reduce_agent,
        })],
    )

    print(f"  [MapReduce] Running map-reduce over {len(records)} records...")
    result = orchestrator.invoke({
        "messages": (
            f"Map-reduce process this dataset:\n"
            f"{json.dumps(records, indent=2)}"
        )
    })
    return result["messages"][-1].content


# ─────────────────────────────────────────────
# 3. Pipeline Batching
#    Chains specialist subagents sequentially:
#    Extract → Classify → Generate
# ─────────────────────────────────────────────

def pipeline_batching(raw_texts: list[str]) -> list[str]:
    """
    Each input flows through a fixed sequence of specialist subagents.
    Output of stage N is the input of stage N+1.

    Best for: multi-step NLP (extract → classify → generate),
              RAG pipelines, structured data transformations.
    """
    model = get_vllm_model()

    extractor = create_deep_agent(
        model=model,
        system_prompt=(
            "Extract structured fields (entities, dates, amounts, topics) "
            "from raw text. Output JSON only, no prose."
        ),
    )

    classifier = create_deep_agent(
        model=model,
        system_prompt=(
            "Given extracted JSON fields, classify the record into one of: "
            "[finance, legal, medical, technology, general]. "
            "Output JSON: {category, confidence, reasoning}."
        ),
    )

    generator = create_deep_agent(
        model=model,
        system_prompt=(
            "Given a classified JSON record, generate a concise one-paragraph "
            "professional summary suitable for a report."
        ),
    )

    pipeline = create_deep_agent(
        model=model,
        system_prompt=(
            "Run a 3-stage pipeline for each input text:\n"
            "  Stage 1 → use 'extractor' subagent to extract structured fields\n"
            "  Stage 2 → use 'classifier' subagent on the extracted JSON\n"
            "  Stage 3 → use 'generator' subagent on the classified JSON\n"
            "Return all three stage outputs bundled together."
        ),
        middleware=[SubAgentMiddleware(subagents={
            "extractor": extractor,
            "classifier": classifier,
            "generator": generator,
        })],
    )

    results = []
    for idx, text in enumerate(raw_texts):
        print(f"  [Pipeline] Processing text {idx + 1}/{len(raw_texts)}...")
        result = pipeline.invoke({
            "messages": f"Run the full pipeline on this text:\n{text}"
        })
        results.append(result["messages"][-1].content)

    return results


# ─────────────────────────────────────────────
# 4. Scatter-Gather (Best-of-N)
#    Fan out to N subagents in parallel,
#    a judge picks the best response.
# ─────────────────────────────────────────────

def scatter_gather_batching(prompts: list[str]) -> list[str]:
    """
    Each prompt is sent simultaneously to multiple subagents with different
    personas/styles. A judge subagent picks the best answer.

    Best for: high-stakes generation, self-consistency checks,
              ensemble reasoning, quality-critical outputs.
    """
    model = get_vllm_model()

    formal_agent = create_deep_agent(
        model=model,
        system_prompt="Answer questions formally, precisely, and with citations where possible.",
    )
    concise_agent = create_deep_agent(
        model=model,
        system_prompt="Answer in 1-2 sentences max. Be direct and ruthlessly brief.",
    )
    stepwise_agent = create_deep_agent(
        model=model,
        system_prompt="Answer step-by-step with clear reasoning. Show your work.",
    )
    judge_agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You receive three answers to the same question from different agents. "
            "Evaluate each for accuracy, clarity, and usefulness. "
            "Return the best answer verbatim and a one-sentence explanation of your choice."
        ),
    )

    scatter_gather = create_deep_agent(
        model=model,
        system_prompt=(
            "For each question:\n"
            "1. Fan out to 'formal', 'concise', 'stepwise' subagents IN PARALLEL.\n"
            "2. Pass all 3 responses to the 'judge' subagent.\n"
            "3. Return the judge's selected best answer."
        ),
        middleware=[SubAgentMiddleware(subagents={
            "formal": formal_agent,
            "concise": concise_agent,
            "stepwise": stepwise_agent,
            "judge": judge_agent,
        })],
    )

    results = []
    for idx, prompt in enumerate(prompts):
        print(f"  [ScatterGather] Processing prompt {idx + 1}/{len(prompts)}...")
        result = scatter_gather.invoke({
            "messages": f"Answer this question using scatter-gather:\n{prompt}"
        })
        results.append(result["messages"][-1].content)

    return results


# ─────────────────────────────────────────────
# 5. File-based Offline Batching
#    Agent reads from disk, writes intermediate
#    results, merges at the end.
# ─────────────────────────────────────────────

def file_based_batching(input_path: str, output_path: str) -> str:
    """
    The agent uses its built-in read_file / write_file tools to process
    data that is too large for a single context window.

    Intermediate results are checkpointed to disk so the job can
    survive context overflow or interruption.

    Best for: very large JSONL files, overnight async jobs,
              datasets that exceed context window limits.
    """
    model = get_vllm_model()

    agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You are a file-based batch processor. Follow these steps exactly:\n"
            "1. Use read_file to load the input JSONL file.\n"
            "2. Process records in batches of 20.\n"
            "3. After each batch, use write_file to save intermediate results "
            "   to /tmp/batch_{n}.json.\n"
            "4. Once all batches are done, read all intermediate files.\n"
            "5. Merge everything and use write_file to save to the output path.\n"
            "6. Confirm completion with a summary of how many records were processed."
        ),
    )

    print(f"  [FileBased] Processing {input_path} → {output_path}...")
    result = agent.invoke({
        "messages": (
            f"Process the JSONL file at '{input_path}' "
            f"and write the final merged output to '{output_path}'."
        )
    })
    return result["messages"][-1].content


# ─────────────────────────────────────────────
# 6. Priority Queue Batching
#    High-priority items are processed first
#    before lower-priority background jobs.
# ─────────────────────────────────────────────

def priority_queue_batching(items: list[dict]) -> list:
    """
    Items carry a 'priority' field (1=urgent, 3=background).
    The orchestrator sorts by priority and spawns subagents
    accordingly, ensuring SLA-sensitive items finish first.

    Best for: mixed real-time + background workloads on the same server.

    Each item dict should have:
        { "priority": int, "content": str, ... }
    """
    model = get_vllm_model()

    urgent_agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You handle URGENT (priority 1) requests. "
            "Process quickly and accurately. Respond in under 100 words."
        ),
    )
    normal_agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You handle NORMAL (priority 2) requests. "
            "Balance speed and thoroughness."
        ),
    )
    background_agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You handle BACKGROUND (priority 3) tasks. "
            "Be thorough, detailed, and comprehensive."
        ),
    )

    orchestrator = create_deep_agent(
        model=model,
        system_prompt=(
            "You are a priority queue processor.\n"
            "1. Sort all items by their 'priority' field (1=urgent first).\n"
            "2. For priority 1: use 'urgent' subagent.\n"
            "3. For priority 2: use 'normal' subagent.\n"
            "4. For priority 3: use 'background' subagent.\n"
            "5. Items with the same priority can be processed in parallel.\n"
            "Return results in order of completion, tagged with their priority."
        ),
        middleware=[SubAgentMiddleware(subagents={
            "urgent": urgent_agent,
            "normal": normal_agent,
            "background": background_agent,
        })],
    )

    print(f"  [PriorityQueue] Processing {len(items)} items by priority...")
    result = orchestrator.invoke({
        "messages": (
            f"Process these items using priority queue batching:\n"
            f"{json.dumps(items, indent=2)}"
        )
    })
    return result["messages"][-1].content


# ─────────────────────────────────────────────
# CLI entrypoint — run any strategy by name
# ─────────────────────────────────────────────

STRATEGIES = {
    "static": "Static Batching — fixed-size sequential chunks",
    "mapreduce": "Map-Reduce — parallel map subagents + single reduce",
    "pipeline": "Pipeline Batching — chained specialist subagents",
    "scatter": "Scatter-Gather — best-of-N with judge subagent",
    "file": "File-based Offline Batching — disk-backed checkpointing",
    "priority": "Priority Queue — urgent-first with tiered subagents",
}


def demo_data():
    """Generate sample data for testing each strategy."""
    records = [
        {"id": i, "text": f"Sample record {i}: This is test content for batch processing.",
         "value": i * 10}
        for i in range(1, 41)   # 40 records
    ]
    texts = [f"Quarterly revenue increased by {i*5}% due to product expansion." for i in range(1, 6)]
    prompts = ["What is the capital of France?", "Explain quantum entanglement briefly."]
    priority_items = [
        {"priority": 1, "content": "URGENT: Server is down, diagnose immediately."},
        {"priority": 3, "content": "Generate monthly analytics report."},
        {"priority": 2, "content": "Review and summarize Q3 documents."},
        {"priority": 1, "content": "URGENT: Customer payment failing, investigate."},
        {"priority": 3, "content": "Compile competitor analysis for next quarter."},
    ]
    return records, texts, prompts, priority_items


def main():
    parser = argparse.ArgumentParser(
        description="Deep Agents batching strategies with vLLM + Qwen3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:12s} {v}" for k, v in STRATEGIES.items()),
    )
    parser.add_argument(
        "strategy",
        choices=list(STRATEGIES.keys()),
        help="Which batching strategy to run",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B",
        help="Model name as registered in vLLM",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input JSONL file (used by 'file' strategy)",
    )
    parser.add_argument(
        "--output",
        default="/tmp/deepagents_output.json",
        help="Path to output file (used by 'file' strategy)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Strategy : {STRATEGIES[args.strategy]}")
    print(f"  vLLM URL : {args.vllm_url}")
    print(f"  Model    : {args.model}")
    print(f"{'='*60}\n")

    records, texts, prompts, priority_items = demo_data()

    if args.strategy == "static":
        result = static_batching(records, batch_size=16)
        print("\n[Result] Static batching complete.")
        print(f"  Processed {len(result)} batches.")

    elif args.strategy == "mapreduce":
        result = map_reduce_batching(records)
        print("\n[Result] Map-reduce complete.")
        print(result)

    elif args.strategy == "pipeline":
        result = pipeline_batching(texts)
        print("\n[Result] Pipeline batching complete.")
        for i, r in enumerate(result):
            print(f"\n--- Text {i+1} ---\n{r}")

    elif args.strategy == "scatter":
        result = scatter_gather_batching(prompts)
        print("\n[Result] Scatter-gather complete.")
        for i, r in enumerate(result):
            print(f"\n--- Prompt {i+1} best answer ---\n{r}")

    elif args.strategy == "file":
        input_path = args.input or "/tmp/deepagents_input.jsonl"
        if args.input is None:
            # Write demo data to a temp JSONL for testing
            Path(input_path).write_text(
                "\n".join(json.dumps(r) for r in records)
            )
            print(f"  Demo input written to {input_path}")
        result = file_based_batching(input_path, args.output)
        print("\n[Result] File-based batching complete.")
        print(result)

    elif args.strategy == "priority":
        result = priority_queue_batching(priority_items)
        print("\n[Result] Priority queue batching complete.")
        print(result)


if __name__ == "__main__":
    main()
