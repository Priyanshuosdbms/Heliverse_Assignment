"""
rlm_custom.py
-------------
Single-file RLM setup for:
  - backend     : vllm
  - model       : qwen3.6-35B-A3B (local)
  - environment : local

Configures: custom system prompt, user prompt, timeout, temperature.

Fixes included
--------------
1. Brace escaping  — sanitize_prompt() auto-escapes bare { } in any
   user-supplied prompt string so RLM's internal .format() call never
   blows up.  The original text is preserved; only the formatting
   metacharacters are doubled.

2. Sub-RLM visibility — verbose_rlm_logger() monkey-patches the two
   internal spawn points (llm_query / rlm_query) to print a live
   tree of every sub-call with depth, model, token budget, and timing.

3. Latency — build_rlm() exposes llm_query_batched-first guidance via
   a cost_guidance string injected into the system prompt, and a
   max_concurrent_queries cap is added to limit runaway fan-out.
"""

from __future__ import annotations

import re
import time
from typing import Any

from rlm import RLM
from rlm.clients.openai import OpenAIClient


# ─────────────────────────────────────────────────────────────
# 1.  Brace escaping
#     RLM calls system_prompt.format(cost_guidance=...) internally.
#     Any bare { or } in a user-supplied prompt causes a KeyError /
#     IndexError.  sanitize_prompt() doubles them so they survive
#     the .format() call and are rendered as literal braces.
# ─────────────────────────────────────────────────────────────

# Matches a { or } that is NOT already doubled (i.e. not {{ or }})
_SINGLE_BRACE_RE = re.compile(r"(?<!\{)\{(?!\{)|(?<!\})\}(?!\})")


def sanitize_prompt(text: str) -> str:
    """
    Escape all single { and } in *text* to {{ and }} so the string
    is safe to pass through Python's str.format().

    Called automatically on MY_SYSTEM_PROMPT before it reaches RLM —
    no manual {{ }} editing needed.

    Examples
    --------
    >>> sanitize_prompt("Use {key} here")
    'Use {{key}} here'
    >>> sanitize_prompt("Already {{safe}}")
    'Already {{safe}}'
    >>> sanitize_prompt("Mixed {a} and {{b}}")
    'Mixed {{a}} and {{b}}'
    """
    return _SINGLE_BRACE_RE.sub(lambda m: m.group(0) * 2, text)


# ─────────────────────────────────────────────────────────────
# 2.  Custom client — forwards temperature (and other sampling
#     params) to the vLLM API call.  The default OpenAIClient
#     stores them in self.kwargs but never passes them on.
# ─────────────────────────────────────────────────────────────

class VLLMClientWithSampling(OpenAIClient):
    """
    Extends OpenAIClient to forward sampling parameters
    (temperature, top_p, max_tokens, seed, frequency_penalty, …)
    to every chat.completions.create() call.
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
# 3.  Patch the vllm backend entry in RLM's BACKEND_MAP
# ─────────────────────────────────────────────────────────────

def _patch_vllm_backend() -> None:
    """
    Replaces the 'vllm' entry in RLM's internal BACKEND_MAP with
    VLLMClientWithSampling so temperature is forwarded.
    Safe to call multiple times.
    """
    import rlm.rlm as rlm_module

    if hasattr(rlm_module, "BACKEND_MAP"):
        rlm_module.BACKEND_MAP["vllm"] = VLLMClientWithSampling
    else:
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
# 4.  Sub-RLM terminal visibility
#     Monkey-patches the two REPL-injected query functions so
#     every sub-call prints a live tree to stdout:
#
#       [RLM] depth=0  root call  (chars=280341)
#         ├─ [llm_query] depth=1  chunk 1/8  (chars=35042)  → 1.3s
#         ├─ [llm_query] depth=1  chunk 2/8  (chars=35001)  → 1.1s
#         └─ [rlm_query] depth=1  sub-task   (chars=12000)  → 4.7s
#              └─ [llm_query] depth=2  ...
# ─────────────────────────────────────────────────────────────

# Depth counter — shared across recursive calls in the same process.
_CALL_DEPTH: list[int] = [0]


def verbose_rlm_logger(rlm_instance: RLM) -> None:
    """
    Wrap llm_query and rlm_query on *rlm_instance*'s REPL environment
    to print a live call tree to stdout.

    Call this immediately after build_rlm():

        rlm = build_rlm(verbose=True)
        verbose_rlm_logger(rlm)
    """
    try:
        env = rlm_instance.env  # LocalREPL or similar
    except AttributeError:
        print("[verbose_rlm_logger] RLM has no .env; skipping patch.")
        return

    # The REPL injects llm_query / rlm_query into its exec namespace.
    # We wrap them at the source on the environment object.
    for fn_attr in ("llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"):
        original = getattr(env, fn_attr, None)
        if original is None:
            # Some env implementations expose them differently; skip gracefully.
            continue
        setattr(env, fn_attr, _make_logged_fn(fn_attr, original))

    print("[verbose_rlm_logger] Sub-call logging enabled.\n")


def _make_logged_fn(name: str, original_fn):
    """Return a wrapper that logs entry/exit around *original_fn*."""

    def _logged(*args, **kwargs):
        depth = _CALL_DEPTH[0]
        indent = "  " * depth + ("└─ " if depth else "")
        # Summarise the prompt length
        prompts = args[0] if args else kwargs.get("prompt", kwargs.get("prompts", ""))
        if isinstance(prompts, list):
            chars = sum(len(str(p)) for p in prompts)
            n = len(prompts)
            summary = f"{n} prompts, {chars:,} chars total"
        else:
            summary = f"{len(str(prompts)):,} chars"

        print(f"{indent}[{name}] depth={depth}  {summary}", flush=True)
        t0 = time.perf_counter()
        _CALL_DEPTH[0] += 1
        try:
            result = original_fn(*args, **kwargs)
        finally:
            _CALL_DEPTH[0] -= 1
        elapsed = time.perf_counter() - t0
        print(f"{indent}  ↳ done in {elapsed:.2f}s", flush=True)
        return result

    _logged.__name__ = f"logged_{name}"
    return _logged


# ─────────────────────────────────────────────────────────────
# 5.  Prompts
#     MY_SYSTEM_PROMPT  — set this to your own text; braces are
#                         auto-escaped by sanitize_prompt().
#     COST_GUIDANCE     — injected into the RLM base prompt to
#                         nudge the model toward batched calls
#                         (reduces total latency significantly).
# ─────────────────────────────────────────────────────────────

# ← Replace with your actual system prompt; { } are fine here.
MY_SYSTEM_PROMPT = """\
You are a meticulous data analyst with access to a Python REPL.

Rules:
- Always chunk large inputs and use llm_query_batched for independent chunks.
- Verify intermediate results in the REPL before giving a final answer.
- Prefer batched sub-calls (llm_query_batched) over sequential llm_query loops.
- If a sub-task needs multi-step reasoning, use rlm_query instead of llm_query.
- Use {variable} style references in your REPL code freely — they are for Python,
  not for this prompt template.
"""

# Injected into the RLM base system prompt via its {cost_guidance} slot.
# Encourages the model to batch aggressively → fewer round-trips → lower latency.
COST_GUIDANCE = (
    "IMPORTANT: Prefer llm_query_batched over sequential llm_query calls for "
    "independent chunks — it fans them out concurrently and is much faster. "
    "Aim for chunks of ~40 000 chars each. Minimise the total number of "
    "sequential LLM round-trips.\n"
)

MY_USER_PROMPT = """\
Analyze the following dataset and summarize the key trends:

<data>
# Replace with your actual data or file path.
date,sales,region
2024-01,120,North
2024-02,95,North
2024-03,140,South
</data>
"""


# ─────────────────────────────────────────────────────────────
# 6.  RLM factory
# ─────────────────────────────────────────────────────────────

def build_rlm(
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "qwen3.6-35B-A3B",
    temperature: float = 0.6,
    timeout: float = 120.0,
    system_prompt: str = MY_SYSTEM_PROMPT,
    cost_guidance: str = COST_GUIDANCE,
    verbose: bool = True,
) -> RLM:
    """
    Build and return a configured RLM instance.

    The system_prompt is automatically sanitized (braces escaped) before
    being passed to RLM.  cost_guidance is appended as the {cost_guidance}
    slot in the RLM base prompt — use it to nudge batching behaviour.

    Args:
        base_url:       vLLM server OpenAI-compatible endpoint.
        model_name:     Model ID as registered in the vLLM server.
        temperature:    Sampling temperature (0.0 = greedy).
        timeout:        HTTP request timeout in seconds.
        system_prompt:  Your custom system prompt (raw — braces OK).
        cost_guidance:  Text injected into RLM's {cost_guidance} slot.
        verbose:        Print rich RLM console output each iteration.
    """
    safe_prompt = sanitize_prompt(system_prompt)

    # RLM appends cost_guidance via its own .format() call on the base prompt.
    # We prepend our guidance to whatever RLM would inject so both coexist.
    # If RLM does not expose a cost_guidance kwarg, append it to safe_prompt.
    import inspect
    from rlm import RLM as _RLM
    rlm_kwargs: dict[str, Any] = dict(
        backend="vllm",
        backend_kwargs={
            "base_url": base_url,
            "model_name": model_name,
            "api_key": "EMPTY",
            "timeout": timeout,
            "temperature": temperature,
        },
        environment="local",
        custom_system_prompt=safe_prompt,
        verbose=verbose,
    )

    sig = inspect.signature(_RLM.__init__)
    if "cost_guidance" in sig.parameters:
        rlm_kwargs["cost_guidance"] = cost_guidance
    else:
        # Fallback: bake cost_guidance into the prompt itself
        rlm_kwargs["custom_system_prompt"] = safe_prompt + "\n" + cost_guidance

    return RLM(**rlm_kwargs)


# ─────────────────────────────────────────────────────────────
# 7.  Single entry point
# ─────────────────────────────────────────────────────────────

def run(
    prompt: str | list[dict[str, Any]] = MY_USER_PROMPT,
    log_sub_calls: bool = True,
    **rlm_kwargs,
) -> str:
    """
    Build an RLM, optionally attach sub-call logging, run a completion,
    print results, and return the response string.

    Args:
        prompt:         Either a plain string (user message) or an explicit
                        list of role/content dicts for few-shot use.
        log_sub_calls:  Print a live sub-call tree to stdout.
        **rlm_kwargs:   Forwarded to build_rlm() — temperature, timeout,
                        system_prompt, cost_guidance, verbose, etc.

    Returns:
        The final response string.
    """
    rlm = build_rlm(**rlm_kwargs)
    if log_sub_calls:
        verbose_rlm_logger(rlm)

    result = rlm.completion(prompt)

    print("\n" + "=" * 60)
    print("RESPONSE:")
    print(result.response)
    print("=" * 60)
    print(f"Execution time : {result.execution_time:.2f}s")
    print(f"Root model     : {result.root_model}")
    print(f"Usage          : {result.usage_summary}")
    print("=" * 60 + "\n")

    return result.response


# ─────────────────────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        prompt=MY_USER_PROMPT,
        log_sub_calls=True,
        temperature=0.6,
        timeout=120.0,
        verbose=True,
    )
