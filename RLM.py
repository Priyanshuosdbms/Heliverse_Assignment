"""
rlm_custom.py
-------------
Single-file RLM setup for:
  - backend  : vllm
  - model    : qwen3.6-35B-A3B (local)
  - environment: local

Configures: custom system prompt, user prompt, timeout, temperature.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rlm import RLM
from rlm.clients.openai import OpenAIClient


# ─────────────────────────────────────────────────────────────
# 1.  Custom client — forwards temperature (and other sampling
#     params) to the vLLM API call. Drop-in replacement for the
#     default OpenAIClient which omits them.
# ─────────────────────────────────────────────────────────────

class VLLMClientWithSampling(OpenAIClient):
    """
    Extends OpenAIClient to forward sampling parameters
    (temperature, top_p, max_tokens, seed, frequency_penalty)
    to every chat.completions.create() call.

    These kwargs are stored on self.kwargs by BaseLM.__init__()
    when passed through backend_kwargs, but the base class does
    not forward them — this subclass does.
    """

    _SAMPLING_KEYS = (
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "frequency_penalty",
        "presence_penalty",
    )

    def _sampling_kwargs(self) -> dict[str, Any]:
        """Extract only the sampling-related keys from self.kwargs."""
        return {k: self.kwargs[k] for k in self._SAMPLING_KEYS if k in self.kwargs}

    def completion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else prompt
        )
        model = model or self.model_name
        if not model:
            raise ValueError("model_name is required.")

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **self._sampling_kwargs(),
        )
        self._track_cost(response, model)
        return response.choices[0].message.content

    async def acompletion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else prompt
        )
        model = model or self.model_name
        if not model:
            raise ValueError("model_name is required.")

        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            **self._sampling_kwargs(),
        )
        self._track_cost(response, model)
        return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# 2.  Patch the vllm backend to use the custom client
#     (avoids modifying library source)
# ─────────────────────────────────────────────────────────────

def _patch_vllm_backend() -> None:
    """
    Replaces the 'vllm' entry in RLM's internal backend map
    with VLLMClientWithSampling so temperature is forwarded.

    Called once at module load — safe to call multiple times.
    """
    import rlm.rlm as rlm_module  # the module that owns BACKEND_MAP

    if hasattr(rlm_module, "BACKEND_MAP"):
        rlm_module.BACKEND_MAP["vllm"] = VLLMClientWithSampling
    else:
        # Fallback: try the clients registry if the structure changed
        try:
            from rlm.core import registry
            registry.BACKEND_MAP["vllm"] = VLLMClientWithSampling
        except (ImportError, AttributeError):
            raise RuntimeError(
                "Could not locate BACKEND_MAP to patch the vllm backend. "
                "Check the installed rlms version."
            )


_patch_vllm_backend()


# ─────────────────────────────────────────────────────────────
# 3.  Prompts
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a meticulous analyst with access to a Python REPL environment.

Guidelines:
- Break complex tasks into smaller sub-tasks and solve them step by step.
- Use the REPL to verify intermediate results before presenting a final answer.
- Write clean, commented code.
- If a sub-task is too large for one pass, decompose it further with a recursive call.
- Always end with a clear, concise final answer.
"""

USER_PROMPT = """\
Analyze the following dataset and summarize the key trends:

<data>
# Replace this placeholder with your actual data or file path.
date,sales,region
2024-01,120,North
2024-02,95,North
2024-03,140,South
</data>
"""


# ─────────────────────────────────────────────────────────────
# 4.  RLM factory
# ─────────────────────────────────────────────────────────────

def build_rlm(
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "qwen3.6-35B-A3B",
    temperature: float = 0.6,
    timeout: float = 120.0,
    system_prompt: str = SYSTEM_PROMPT,
    verbose: bool = False,
) -> RLM:
    """
    Build and return a configured RLM instance.

    Args:
        base_url:      vLLM server OpenAI-compatible endpoint.
        model_name:    Model ID as registered in the vLLM server.
        temperature:   Sampling temperature (0.0 = greedy).
        timeout:       HTTP request timeout in seconds passed to
                       the underlying openai.OpenAI client.
        system_prompt: Overrides the default RLM system prompt.
        verbose:       Print rich console output of each iteration.
    """
    return RLM(
        backend="vllm",
        backend_kwargs={
            "base_url": base_url,
            "model_name": model_name,
            "api_key": "EMPTY",       # vLLM local servers don't need a real key
            "timeout": timeout,       # → openai.OpenAI(timeout=timeout)
            "temperature": temperature,  # → picked up by _sampling_kwargs()
        },
        environment="local",
        custom_system_prompt=system_prompt,
        verbose=verbose,
    )


# ─────────────────────────────────────────────────────────────
# 5.  Usage helpers
# ─────────────────────────────────────────────────────────────

def run_sync(user_prompt: str = USER_PROMPT, **rlm_kwargs) -> str:
    """
    Run a synchronous RLM completion and return the response string.

    Extra keyword arguments are forwarded to build_rlm().

    Example:
        response = run_sync(user_prompt="Explain recursion.", temperature=0.3)
    """
    rlm = build_rlm(**rlm_kwargs)
    result = rlm.completion(user_prompt)
    _print_result(result)
    return result.response


async def run_async(user_prompt: str = USER_PROMPT, **rlm_kwargs) -> str:
    """
    Async variant — wraps run_sync in an executor so the event loop
    isn't blocked (rlm.completion is synchronous internally).

    Example:
        response = await run_async("Summarise this file.", temperature=0.5)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_sync, user_prompt)


def run_with_message_list(
    messages: list[dict[str, Any]],
    **rlm_kwargs,
) -> str:
    """
    Pass an explicit list of chat messages (role/content dicts) as the prompt.
    Useful when you need few-shot turns or a per-call system role distinct
    from the RLM orchestration system prompt.

    Example:
        messages = [
            {"role": "system", "content": "You are a SQL expert."},
            {"role": "user",   "content": "Write a query to find duplicates."},
        ]
        response = run_with_message_list(messages, temperature=0.2)
    """
    rlm = build_rlm(**rlm_kwargs)
    result = rlm.completion(messages)
    _print_result(result)
    return result.response


def _print_result(result) -> None:
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print(result.response)
    print("=" * 60)
    print(f"Execution time : {result.execution_time:.2f}s")
    print(f"Root model     : {result.root_model}")
    print(f"Usage          : {result.usage_summary}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────
# 6.  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Sync run with default USER_PROMPT ──────────────────────
    run_sync(
        user_prompt=USER_PROMPT,
        temperature=0.6,
        timeout=120.0,
        verbose=True,
    )

    # ── Run with an explicit message list ──────────────────────
    # run_with_message_list(
    #     messages=[
    #         {"role": "system", "content": "You are a SQL expert."},
    #         {"role": "user",   "content": "Find all duplicate emails."},
    #     ],
    #     temperature=0.2,
    # )

    # ── Async run ──────────────────────────────────────────────
    # asyncio.run(run_async(USER_PROMPT, temperature=0.4))
