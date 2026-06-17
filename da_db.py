"""
LangGraph SQL Agent — Qwen3.6-27B, fully local, production-ready
=================================================================

REQUIREMENTS
------------
pip install langchain langgraph langchain-community langchain-openai psycopg2-binary

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERVING — CHOOSE ONE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ RECOMMENDED — SGLang (avoids vLLM reasoning+tool parser conflict):
    pip install sglang[all]
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3.6-27B \
        --port 8000 \
        --tp-size 8 \
        --context-length 32768 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder

⚠️  vLLM — only if you must use it. DO NOT add --reasoning-parser.
    The vLLM bug: when --reasoning-parser qwen3 and --tool-call-parser
    qwen3_coder are used together, tool calls emitted inside <think>
    blocks get stripped by the reasoning parser and never reach
    LangGraph — the agent connects to DB but silently produces nothing.
    Fix: omit --reasoning-parser entirely and disable thinking in client.

    vllm serve Qwen/Qwen3.6-27B \
        --port 8000 \
        --tensor-parallel-size 8 \
        --max-model-len 32768 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder
        # intentionally NO --reasoning-parser

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Schema is injected into the system prompt at startup rather than
discovered at runtime. This eliminates 2 extra LLM round-trips per
query and removes the forced tool_choice="any" calls that are another
surface for the reasoning+tool parser conflict.

Qwen3.6-27B has a 262K context window; even a 50-table schema is
typically <10K tokens — no cost worth worrying about.
"""

import re
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver


# ─────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME    = "Qwen/Qwen3.6-27B"
DB_URI        = "postgresql+psycopg2://user:pass@host:5432/dbname"

# Explicit table list avoids casing mismatches. Set to None to auto-detect.
INCLUDE_TABLES = None  # e.g. ["users", "orders", "products"]

USING_SGLANG = True   # Set False if you're on vLLM (affects thinking flag)


# ─────────────────────────────────────────────────────────────────
# 1. LLM  — fully local, zero external calls
# ─────────────────────────────────────────────────────────────────

# Thinking is disabled at the client level to prevent <think> tokens
# from leaking into tool call arguments.
# - On SGLang: reasoning parser handles stripping server-side, but
#   disabling thinking also speeds up SQL generation tasks.
# - On vLLM (no --reasoning-parser): critical — without this, raw
#   <think>...</think> text will be passed as the SQL query string.
llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=VLLM_BASE_URL,
    api_key="EMPTY",           # vLLM / SGLang don't need a real key
    temperature=0.7,
    max_tokens=4096,
    model_kwargs={
        "top_p": 0.80,
        "top_k": 20,
        "presence_penalty": 1.5,
        # Disable thinking mode — non-thinking mode is correct for
        # structured SQL generation; thinking adds latency and is the
        # source of the reasoning+tool parser conflict.
        "chat_template_kwargs": {"enable_thinking": False},
    },
)


# ─────────────────────────────────────────────────────────────────
# 2. Database + schema extraction at startup
# ─────────────────────────────────────────────────────────────────

db = SQLDatabase.from_uri(
    DB_URI,
    include_tables=INCLUDE_TABLES,
    sample_rows_in_table_info=2,  # gives model a few example rows for context
)

# Pull schema once at startup and embed in system prompt.
# This replaces runtime schema-discovery tool calls entirely.
DB_SCHEMA    = db.get_table_info()
TABLE_NAMES  = ", ".join(db.get_usable_table_names())

print(f"✅ Connected to DB. Tables: {TABLE_NAMES}")


# ─────────────────────────────────────────────────────────────────
# 3. Sanitized SQL execution tool
#    Belt-and-suspenders: strips any stray markdown, XML, or prefix
#    the model might emit even with thinking disabled.
# ─────────────────────────────────────────────────────────────────

class SafeQuerySQLDataBaseTool(QuerySQLDataBaseTool):
    """Drop-in replacement that sanitizes the query before DB execution."""

    def _run(self, query: str, **kwargs) -> str:
        # Strip markdown fences  ``` sql ... ``` or ``` ... ```
        query = re.sub(r"```(?:sql)?", "", query, flags=re.IGNORECASE)
        # Strip any XML / thinking tags  <think>, </think>, <tool_call>, etc.
        query = re.sub(r"</?[a-zA-Z_|][^>]*>", "", query)
        # Strip common model output prefixes
        query = re.sub(
            r"^(SQLQuery|SQL\s*Query|Query|Answer)\s*:\s*",
            "",
            query.strip(),
            flags=re.IGNORECASE,
        )
        query = query.strip().strip(";").strip()
        return super()._run(query, **kwargs)


# ─────────────────────────────────────────────────────────────────
# 4. Build toolkit — swap in the safe query tool
# ─────────────────────────────────────────────────────────────────

toolkit   = SQLDatabaseToolkit(db=db, llm=llm)
raw_tools = toolkit.get_tools()

safe_tools = [
    SafeQuerySQLDataBaseTool(db=db) if isinstance(t, QuerySQLDataBaseTool) else t
    for t in raw_tools
]

run_query_tool  = next(t for t in safe_tools if t.name == "sql_db_query")
run_query_node  = ToolNode([run_query_tool], name="run_query")


# ─────────────────────────────────────────────────────────────────
# 5. System prompts
# ─────────────────────────────────────────────────────────────────

GENERATE_SYSTEM_PROMPT = f"""You are a SQL expert agent connected to a {db.dialect} database.

DATABASE SCHEMA (use EXACT table and column names — case-sensitive):
{DB_SCHEMA}

STRICT OUTPUT RULES — follow all of these without exception:
1. When calling sql_db_query, pass ONLY the raw SQL string.
   - No markdown fences (no ```sql), no XML tags, no "SQLQuery:" prefix,
     no trailing semicolons, no explanations alongside the query.
2. Use the exact table/column names from the schema above.
   Do not quote identifiers unless they contain spaces or special characters.
3. Limit results to 10 rows unless the user asks for more.
4. Never run INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or any statement
   that modifies data.
5. If you already have the answer from query results, respond in plain text
   without making another tool call.
"""

CHECK_SYSTEM_PROMPT = f"""You are a strict {db.dialect} SQL reviewer.

Check the query only for these mistakes:
- Wrong table or column name, or wrong casing vs the schema
- NOT IN used with NULL values
- UNION where UNION ALL is needed
- Off-by-one errors in BETWEEN ranges
- Missing or incorrect JOIN conditions
- Type mismatches in predicates
- Unquoted reserved words used as identifiers

OUTPUT RULES — critical:
- Return ONLY the corrected raw SQL query. No markdown, no tags,
  no explanation, no semicolons, nothing else.
- If the query is already correct, return it exactly as-is.
"""


# ─────────────────────────────────────────────────────────────────
# 6. Agent nodes
#    No list_tables or call_get_schema nodes — schema is already in
#    the system prompt, removing 2 round-trips and the forced
#    tool_choice="any" calls that triggered the parser conflict.
# ─────────────────────────────────────────────────────────────────

def generate_query(state: MessagesState) -> dict:
    """
    Core step: model reads the schema from the system prompt and
    decides what SQL to run, or answers directly if it already has
    the result from a previous query.
    """
    system         = SystemMessage(content=GENERATE_SYSTEM_PROMPT)
    llm_with_tools = llm.bind_tools([run_query_tool])
    response       = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


def check_query(state: MessagesState) -> dict:
    """
    Validates and optionally rewrites the SQL before execution.
    Only runs when the model has decided to call sql_db_query.
    """
    system    = SystemMessage(content=CHECK_SYSTEM_PROMPT)
    tool_call = state["messages"][-1].tool_calls[0]
    user_msg  = {"role": "user", "content": tool_call["args"]["query"]}
    llm_check = llm.bind_tools([run_query_tool], tool_choice="any")
    response  = llm_check.invoke([system, user_msg])
    response.id = state["messages"][-1].id   # keep message threading consistent
    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────────
# 7. Routing
# ─────────────────────────────────────────────────────────────────

def should_continue(state: MessagesState) -> Literal["check_query", "__end__"]:
    last = state["messages"][-1]
    return "check_query" if getattr(last, "tool_calls", None) else END


# ─────────────────────────────────────────────────────────────────
# 8. Build graph
#
#   START
#     │
#     ▼
#   generate_query  ──[tool call]──► check_query ──► run_query ──┐
#     ▲                                                           │
#     └───────────────────────────────────────────────────────────┘
#     │
#     └──[no tool call]──► END
#
#  Simpler than before: no list_tables or call_get_schema nodes.
#  Schema is already in the system prompt.
# ─────────────────────────────────────────────────────────────────

builder = StateGraph(MessagesState)

builder.add_node("generate_query", generate_query)
builder.add_node("check_query",    check_query)
builder.add_node("run_query",      run_query_node)

builder.add_edge(START, "generate_query")
builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query",   "generate_query")   # loop: interpret result or re-query

agent = builder.compile(checkpointer=InMemorySaver())


# ─────────────────────────────────────────────────────────────────
# 9. Public interface
# ─────────────────────────────────────────────────────────────────

def ask(question: str, thread_id: str = "default") -> str:
    """
    Ask a natural language question about your database.

    Args:
        question:  Plain-English question.
        thread_id: Reuse to continue a conversation; new value to start fresh.

    Returns:
        The agent's final answer as a string.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    return result["messages"][-1].content


def ask_streaming(question: str, thread_id: str = "default") -> None:
    """Stream agent steps to stdout — useful for debugging."""
    config = {"configurable": {"thread_id": thread_id}}
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


# ─────────────────────────────────────────────────────────────────
# 10. Quick smoke test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(ask("How many rows are in the users table?"))

    # Multi-turn: same thread_id keeps conversation context
    ask_streaming("Which user signed up most recently?", thread_id="session-1")
    ask_streaming("What orders do they have?",           thread_id="session-1")
