"""
telemetry_agent.py
──────────────────
LangChain SQL deep-agent that:
  • Connects to MySQL (read-only — all write/DDL queries are blocked)
  • Loads enriched table/column descriptions from enriched_attributes.csv
  • Uses those descriptions to give the LLM rich schema context
  • Answers natural-language queries about historic telemetry data
  • Guardrails: blocks any mutation (INSERT/UPDATE/DELETE/DROP/TRUNCATE/ALTER/CREATE)
"""

import re
import logging
import pandas as pd
from pathlib import Path
from typing import Any

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, event, text

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "127.0.0.1",   # ← change
    "port":     3306,           # ← change
    "user":     "readonly_user",# ← change
    "password": "secret",       # ← change
    "database": "telemetry_db", # ← change
}

VLLM_BASE_URL       = "http://localhost:8000/v1"   # ← change
VLLM_API_KEY        = "EMPTY"
MODEL_NAME          = "Qwen/Qwen3-7B"              # ← change to your model
ENRICHED_CSV        = "enriched_attributes.csv"
MAX_ROWS_RETURNED   = 200   # safety cap on SELECT results

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Guardrail helpers ─────────────────────────────────────────────────────────

_MUTATION_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE"
    r"|RENAME|GRANT|REVOKE|LOAD\s+DATA|INTO\s+OUTFILE)\b",
    re.IGNORECASE,
)

def is_mutation(sql: str) -> bool:
    return bool(_MUTATION_PATTERN.search(sql))

def enforce_read_only(sql: str) -> str:
    """Raise if the SQL is not a pure SELECT; also inject LIMIT if missing."""
    clean = sql.strip().lstrip(";").strip()
    if is_mutation(clean):
        raise ValueError(
            f"⛔ Blocked mutation query — only SELECT is allowed.\n"
            f"Offending SQL: {clean[:200]}"
        )
    if not re.match(r"^\s*SELECT\b", clean, re.IGNORECASE):
        raise ValueError(
            f"⛔ Only SELECT queries are permitted. Got: {clean[:80]}"
        )
    # Inject LIMIT if not present (prevents accidental full-table scans)
    if not re.search(r"\bLIMIT\b", clean, re.IGNORECASE):
        clean = clean.rstrip(";") + f" LIMIT {MAX_ROWS_RETURNED}"
    return clean


# ── DB connection ─────────────────────────────────────────────────────────────

def build_engine():
    url = (
        "mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    ).format(**DB_CONFIG)
    engine = create_engine(url, pool_pre_ping=True)

    # SQLAlchemy event hook — second line of defence at connection level
    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def before_cursor_execute(conn, cursor, statement, parameters,
                               context, executemany):
        if is_mutation(statement):
            raise RuntimeError(
                f"⛔ Engine-level block: mutation SQL detected: {statement[:120]}"
            )
        return statement, parameters

    return engine


# ── Schema context from enriched CSV ─────────────────────────────────────────

def build_schema_context(csv_path: str) -> str:
    """Turn the enriched CSV into a human-readable schema description block."""
    if not Path(csv_path).exists():
        log.warning("Enriched CSV not found at %s — schema context skipped.", csv_path)
        return ""

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    lines = ["=== Telemetry DB Schema Descriptions ===\n"]
    current_table = None

    for _, row in df.iterrows():
        tbl = row.get("TABLE_NAME", "").strip()
        col = row.get("COLUMN_NAME", "").strip()
        t_desc = row.get("table_description", "").strip()
        c_desc = row.get("column_description", "").strip()

        if tbl != current_table:
            lines.append(f"\nTable: {tbl}")
            if t_desc:
                lines.append(f"  Description: {t_desc}")
            current_table = tbl

        flag = " ⚠ [NOT FOUND — review needed]" if "NOT FOUND" in c_desc else ""
        lines.append(f"  Column: {col}")
        if c_desc:
            lines.append(f"    → {c_desc}{flag}")

    return "\n".join(lines)


# ── Custom guarded tool wrapper ────────────────────────────────────────────────

class GuardedQueryTool(BaseTool):
    """
    Wraps the LangChain SQL query tool with read-only enforcement
    before execution hits the database.
    """
    name: str = "sql_db_query"
    description: str = (
        "Execute a read-only SQL SELECT query against the telemetry database. "
        "Input: a valid SQL SELECT statement. "
        "Output: query results as text. "
        "NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, or any mutation."
    )
    db: Any  # SQLDatabase instance

    def _run(self, query: str) -> str:
        try:
            safe_query = enforce_read_only(query)
        except ValueError as e:
            return str(e)
        try:
            return self.db.run(safe_query)
        except Exception as e:
            return f"SQL error: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


# ── Agent construction ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise and helpful assistant for querying SSD telemetry data stored in a MySQL database.

{schema_context}

Rules you MUST follow:
1. Only generate SELECT queries — never INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or any other mutation.
2. Always add a LIMIT clause (max {max_rows}) unless the user asks for aggregate results.
3. If you are unsure which table/column to query, consult the schema descriptions above.
4. Present results clearly; summarise large result sets instead of printing raw rows.
5. If a query cannot be answered from the available data, say so honestly.

Available tools:
{{tools}}

Use this format:
Thought: <your reasoning>
Action: <tool name>
Action Input: <input to the tool>
Observation: <result>
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer.
Final Answer: <your answer to the human>

Begin!
Question: {{input}}
{{agent_scratchpad}}
"""


def build_agent(engine, schema_context: str) -> AgentExecutor:
    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        model=MODEL_NAME,
        temperature=0,
        max_tokens=1024,
    )

    db = SQLDatabase(engine)

    # Build the standard toolkit but we'll replace the query tool
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # Replace the vanilla query tool with our guarded version
    guarded = GuardedQueryTool(db=db)
    tools = [guarded if t.name == "sql_db_query" else t for t in tools]

    prompt_text = SYSTEM_PROMPT.format(
        schema_context=schema_context,
        max_rows=MAX_ROWS_RETURNED,
    )
    prompt = PromptTemplate.from_template(prompt_text)

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )


# ── CLI loop ──────────────────────────────────────────────────────────────────

def main():
    log.info("Connecting to MySQL at %(host)s:%(port)s/%(database)s", DB_CONFIG)
    engine = build_engine()

    # Verify connectivity
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info("DB connection OK.")

    schema_context = build_schema_context(ENRICHED_CSV)
    agent_executor = build_agent(engine, schema_context)

    print("\n" + "═" * 60)
    print("  SSD Telemetry Query Agent  (type 'exit' to quit)")
    print("═" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            result = agent_executor.invoke({"input": user_input})
            print(f"\nAgent: {result['output']}\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    main()
