"""
mem0 + vLLM (Qwen3, port 8000) — simple LLM-only demo.

Fix for "api key must be set" error:
  Set OPENAI_API_KEY=EMPTY before running, or export it in your shell:
      export OPENAI_API_KEY=EMPTY
"""

import os

# Required: mem0's OpenAI provider reads this env var.
# vLLM doesn't enforce auth, so any non-empty string works.
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

from mem0 import Memory

# ---------------------------------------------------------------------------
# Config — LLM only, no embedder or vector DB
# ---------------------------------------------------------------------------
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "qwen3-6",                        # model name registered in vLLM
            "openai_base_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",                        # vLLM doesn't need a real key
            "temperature": 0.1,
            "max_tokens": 2000,
        },
    },
}

memory = Memory.from_config(config)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    USER = "alice"

    # Store a message — mem0 calls the LLM to extract memories from it
    print("Storing turn 1...")
    memory.add(
        [{"role": "user", "content": "Hi! My name is Alice and I love hiking."}],
        user_id=USER,
    )

    # Store a follow-up turn
    print("Storing turn 2...")
    memory.add(
        [{"role": "user", "content": "I also enjoy photography and cooking."}],
        user_id=USER,
    )

    # Retrieve all stored memories for the user
    print("\nAll memories for Alice:")
    for m in memory.get_all(user_id=USER):
        print(f"  • [{m['id']}] {m['memory']}")
