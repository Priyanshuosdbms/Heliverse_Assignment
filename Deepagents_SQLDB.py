import os
import re
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

# ==========================================
# 1. SETUP DEEP AGENT SKILLS (The "Brain")
# pip install deepagents langchain-openai langchain-community sqlalchemy psycopg2-binary
# ==========================================
# Deep Agents use a filesystem backend to load Skills (SKILL.md files)
PROJECT_DIR = Path("./deep_sql_project")
SKILLS_DIR = PROJECT_DIR / "skills" / "sql_analyst"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)

# Create the SKILL.md file with strict instructions for Qwen
skill_content = """---
name: sql_analyst
description: Use this skill for any requests involving querying the SQL database, analyzing data, or finding table schemas.
---

# SQL Database Analyst

## Overview
You are an expert SQL analyst working with a large PostgreSQL database containing over 40 tables and up to 1 million records per table.

## Instructions

### 1. Schema Discovery (Crucial for >40 Tables)
Because the database has >40 tables, NEVER guess table or column names. 
- ALWAYS use the `search_schema` tool with a relevant keyword (e.g., "user", "order") to find the exact tables and columns you need.
- Only fetch the schema for the specific tables you need to keep your context window small and accurate.

### 2. Query Generation & Guardrails
When generating SQL queries, you MUST follow these strict rules:
- **NO ARTIFACTS**: NEVER wrap your SQL in markdown code blocks (```) or XML tags (<sql>, </sql>, <think>, </think>). Output ONLY the raw SQL string.
- **LARGE TABLES**: For tables with potentially large data, ALWAYS include a `LIMIT 100` clause unless you are performing an aggregation (e.g., `COUNT`, `SUM`, `GROUP BY`).
- **READ ONLY**: NEVER generate `INSERT`, `UPDATE`, `DELETE`, `DROP`, or `ALTER` statements.

### 3. Execution & Self-Correction
- Use the `execute_sql` tool to run your query.
- If the tool returns an error (e.g., syntax error, column not found), analyze the exact database error message, correct your SQL query, and try again.
"""

(SKILLS_DIR / "SKILL.md").write_text(skill_content)
print(f"✅ Created Deep Agent Skill at: {SKILLS_DIR / 'SKILL.md'}")

# ==========================================
# 2. DATABASE & LLM SETUP (vLLM + Qwen)
# ==========================================
DATABASE_URI = "postgresql://readonly_user:password@localhost:5432/your_db"
# sample_rows_in_table_info=0 prevents token bloat from large datasets
db = SQLDatabase.from_uri(DATABASE_URI, sample_rows_in_table_info=0)

# Configure vLLM endpoint for Qwen
llm = ChatOpenAI(
    model="qwen3.6", 
    openai_api_key="EMPTY", 
    openai_api_base="http://localhost:8000/v1", 
    temperature=0.0,
    max_tokens=2000,
    # CRITICAL: Force vLLM to stop generating if it starts outputting XML artifacts
    model_kwargs={"stop": ["</think>", "</sql>", "```", "</"]} 
)

# ==========================================
# 3. BULLETPROOF GUARDRAIL TOOLS
# ==========================================
def sanitize_sql(query: str) -> str:
    """Nuclear option to strip ANY LLM XML/markdown artifacts without breaking SQL operators."""
    if not query: return ""
    # 1. Remove markdown blocks
    query = re.sub(r'```(?:sql)?\s*', '', query, flags=re.IGNORECASE)
    query = re.sub(r'```\s*', '', query)
    # 2. Remove specific LLM XML tags (think, sql, response, etc.)
    # This regex specifically targets tags starting with letters, avoiding SQL operators like < or >
    query = re.sub(r'</?(?:think|sql|response|tool|message|system|user|assistant)[^>]*>', '', query, flags=re.IGNORECASE)
    # 3. Clean whitespace and trailing semicolons
    return query.strip().rstrip(';')

@tool
def search_schema(keyword: str) -> str:
    """Search for tables and columns matching a keyword. 
    Use this FIRST to avoid guessing table names in a 40+ table database."""
    all_tables = db.get_usable_table_names()
    matched_tables = [t for t in all_tables if keyword.lower() in t.lower()]
    
    if not matched_tables:
        return f"No tables found matching '{keyword}'. Available tables: {', '.join(all_tables[:15])}..."
    
    # Fetch detailed schema ONLY for matched tables (saves massive context)
    return db.get_table_info(table_names=matched_tables)

@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query against the database. The query will be automatically sanitized."""
    clean_query = sanitize_sql(query)
    
    # Guardrail: Block destructive operations
    if any(kw in clean_query.upper() for kw in ["DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER"]):
        return "ERROR: Destructive queries are strictly forbidden."

    try:
        # Guardrail: Prevent 1M+ row scans. Force LIMIT unless it's an aggregate.
        is_aggregate = any(kw in clean_query.upper() for kw in ["COUNT", "SUM", "AVG", "MAX", "MIN", "GROUP BY"])
        if not is_aggregate and "LIMIT" not in clean_query.upper():
            clean_query = f"{clean_query} LIMIT 100"
            
        result = db.run(clean_query)
        return f"SUCCESS:\n{result}"
    except Exception as e:
        # Return the exact DB error to the Deep Agent so it can self-correct
        return f"SQL EXECUTION FAILED:\n{str(e)}"

# ==========================================
# 4. CREATE DEEP AGENT
# ==========================================
# Initialize the filesystem backend to load our skills
backend = FilesystemBackend(root_dir=str(PROJECT_DIR))

# Create the Deep Agent with Qwen, our tools, and the skills directory
agent = create_deep_agent(
    model=llm,
    tools=[search_schema, execute_sql],
    backend=backend,
    skills=[str(PROJECT_DIR / "skills")], # Point to the top-level skills directory
    system_prompt="You are a helpful data assistant. Use the sql_analyst skill whenever the user asks about the database."
)

# ==========================================
# 5. RUN THE AGENT
# ==========================================
def ask_database(question: str):
    print(f"\n🔍 User Query: {question}\n" + "-"*50)
    
    # Deep Agents use standard LangGraph invoke
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": "1"}}
    )
    
    # Extract the final AI response
    final_message = result["messages"][-1].content
    print(f"✅ Final Answer:\n{final_message}")

# Test the Deep Agent
if __name__ == "__main__":
    ask_database("Show me the top 5 users by login count in the last 30 days.")
