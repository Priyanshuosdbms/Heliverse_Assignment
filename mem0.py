"""
Dummy code integrating mem0 with vLLM (Qwen3, port 8000).
mem0 uses the OpenAI-compatible endpoint exposed by vLLM.
"""

from mem0 import Memory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# vLLM serves Qwen3 on an OpenAI-compatible endpoint at port 8000.
# mem0 accepts an "openai" provider and lets you override the base_url.

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "qwen3-6",           # model name as registered in vLLM
            "openai_base_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",           # vLLM doesn't require a real key
            "temperature": 0.1,
            "max_tokens": 2000,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            # vLLM also exposes /v1/embeddings – use the same model or a
            # dedicated embedding model if you have one loaded.
            "model": "qwen3-6",
            "openai_base_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
        },
    },
    # Optional: use an in-memory vector store for quick testing.
    # Replace with qdrant / chroma / pinecone for production.
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0_demo",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 1536,
        },
    },
}

# ---------------------------------------------------------------------------
# Initialise mem0
# ---------------------------------------------------------------------------
memory = Memory.from_config(config)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def add_memory(user_id: str, messages: list[dict]) -> dict:
    """Store a conversation turn in mem0."""
    result = memory.add(messages, user_id=user_id)
    print(f"[add]  user={user_id}  result={result}")
    return result


def search_memory(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the most relevant memories for a user."""
    results = memory.search(query, user_id=user_id, limit=top_k)
    print(f"[search]  user={user_id}  query='{query}'  hits={len(results)}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.3f}] {r['memory']}")
    return results


def get_all_memories(user_id: str) -> list[dict]:
    """Fetch every stored memory for a user."""
    all_mems = memory.get_all(user_id=user_id)
    print(f"[get_all]  user={user_id}  total={len(all_mems)}")
    for m in all_mems:
        print(f"  • {m['id']}: {m['memory']}")
    return all_mems


def delete_memory(memory_id: str) -> None:
    """Delete a single memory entry by its ID."""
    memory.delete(memory_id)
    print(f"[delete]  memory_id={memory_id}")


def chat_with_memory(user_id: str, user_message: str) -> str:
    """
    Minimal RAG-style chat loop:
      1. Retrieve relevant memories.
      2. Inject them as system context.
      3. Call the LLM via the OpenAI-compatible vLLM endpoint.
      4. Store the new turn back into mem0.
    """
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    # 1. Retrieve context
    relevant = search_memory(user_id, user_message, top_k=3)
    memory_context = "\n".join(f"- {r['memory']}" for r in relevant)

    system_prompt = (
        "You are a helpful assistant with persistent memory.\n"
        "Here are relevant things you remember about the user:\n"
        f"{memory_context or '(no memories yet)'}"
    )

    # 2. Call the LLM
    response = client.chat.completions.create(
        model="qwen3-6",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    )
    assistant_reply = response.choices[0].message.content

    # 3. Persist the turn
    add_memory(
        user_id,
        [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": assistant_reply},
        ],
    )

    return assistant_reply


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    USER = "alice"

    # --- Turn 1: introduce a preference ---
    print("\n=== Turn 1 ===")
    reply = chat_with_memory(USER, "Hi! My name is Alice and I love hiking.")
    print(f"Assistant: {reply}")

    # --- Turn 2: follow-up that should leverage the stored memory ---
    print("\n=== Turn 2 ===")
    reply = chat_with_memory(USER, "Can you recommend something fun for me to do this weekend?")
    print(f"Assistant: {reply}")

    # --- Inspect stored memories ---
    print("\n=== All memories for Alice ===")
    all_mems = get_all_memories(USER)

    # --- Targeted search ---
    print("\n=== Search: 'outdoor activities' ===")
    search_memory(USER, "outdoor activities")

    # --- Clean up one memory (optional) ---
    if all_mems:
        first_id = all_mems[0]["id"]
        print(f"\n=== Deleting first memory ({first_id}) ===")
        delete_memory(first_id)
