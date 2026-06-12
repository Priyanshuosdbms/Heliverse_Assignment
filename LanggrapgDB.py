import re
import os
from typing import TypedDict, Annotated, List
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI # Works perfectly with vLLM endpoints
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# ==========================================
# 1. DATABASE & LLM SETUP
# ==========================================
# Use a read-only database user for security!
DATABASE_URI = "postgresql://readonly_user:password@localhost:5432/your_db"
db = SQLDatabase.from_uri(DATABASE_URI, sample_rows_in_table_info=0) # 0 samples to save tokens

# Configure vLLM endpoint for Qwen
llm = ChatOpenAI(
    model="qwen3.6", # Or qwen3.7, depending on your vLLM deployment
    openai_api_key="EMPTY", # vLLM typically uses EMPTY or a dummy key
    openai_api_base="http://localhost:8000/v1", # Your vLLM endpoint
    temperature=0.0, # Keep temperature low for deterministic SQL generation
    max_tokens=2000,
    # Optional: Add stop sequences to prevent Qwen from generating trailing XML
    model_kwargs={"stop": ["</sql>", "</think>", "```"]} 
)

# ==========================================
# 2. BULLETPROOF GUARDRAIL TOOL
# ==========================================
def clean_sql_query(query: str) -> str:
    """Aggressively clean LLM output to prevent syntax errors from XML/markdown."""
    if not query:
        return ""
    # Remove markdown code blocks
    query = re.sub(r'```sql\s*', '', query, flags=re.IGNORECASE)
    query = re.sub(r'```\s*', '', query)
    # Remove Qwen-specific XML/think tags
    query = re.sub(r'</?think>', '', query, flags=re.IGNORECASE)
    query = re.sub(r'</?sql>', '', query, flags=re.IGNORECASE)
    query = re.sub(r'</?response>', '', query, flags=re.IGNORECASE)
    # Remove any trailing semicolons and whitespace
    return query.strip().rstrip(';')

@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query against the database and return the results. 
    ALWAYS ensure the query is clean and valid before calling."""
    cleaned_query = clean_sql_query(query)
    
    # Guardrail: Prevent destructive queries
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER", "TRUNCATE"]
    if any(keyword in cleaned_query.upper() for keyword in forbidden_keywords):
        return "ERROR: Destructive queries are not allowed."

    try:
        # Enforce a LIMIT if not present to protect against 1M+ row scans
        if "limit" not in cleaned_query.lower():
            cleaned_query = cleaned_query.rstrip(';') + " LIMIT 100"
            
        result = db.run(cleaned_query)
        return str(result)
    except Exception as e:
        # Return the error to the LLM so it can self-correct
        return f"SQL EXECUTION ERROR: {str(e)}\nPlease fix the query and try again."

# ==========================================
# 3. DYNAMIC SCHEMA RETRIEVAL TOOLS
# ==========================================
@tool
def list_tables() -> str:
    """Return a comma-separated list of all table names in the database. 
    Use this FIRST to find relevant tables before asking for schema."""
    return ", ".join(db.get_usable_table_names())

@tool
def get_table_schema(table_names: str) -> str:
    """Get the detailed schema (columns, types) for specific, comma-separated table names.
    Example input: 'users, orders'"""
    tables = [t.strip() for t in table_names.split(",")]
    # Filter to only usable tables to prevent errors
    valid_tables = [t for t in tables if t in db.get_usable_table_names()]
    if not valid_tables:
        return "No valid tables found. Use list_tables first."
    
    # Get schema ONLY for the requested tables, saving massive context
    return db.get_table_info(table_names=valid_tables)

# ==========================================
# 4. OPTIMIZED SYSTEM PROMPT FOR QWEN
# ==========================================
SYSTEM_PROMPT = """You are an expert SQL Data Analyst. You have access to a database with over 40 tables.

RULES:
1. NEVER guess table or column names. ALWAYS use `list_tables` to find tables, then `get_table_schema` to get exact column names.
2. ALWAYS output raw, valid SQL. Do NOT wrap SQL in markdown (```) or XML tags (<sql>, </sql>, <think>).
3. If a query returns an error, analyze the error message, correct the SQL, and try again.
4. For large tables (>1M records), ALWAYS use `LIMIT` unless aggregating (GROUP BY).
5. Use standard SQL dialect. 

TOOLS AVAILABLE:
- `list_tables`: Find table names.
- `get_table_schema`: Get column details for specific tables.
- `execute_sql`: Run the final, validated query.
"""

# ==========================================
# 5. AGENT INITIALIZATION
# ==========================================
tools = [list_tables, get_table_schema, execute_sql]

# create_react_agent handles the loop: Thought -> Action -> Observation -> Correction
sql_agent = create_react_agent(
    llm=llm,
    tools=tools,
    state_modifier=SystemMessage(content=SYSTEM_PROMPT)
)

# ==========================================
# 6. USAGE EXAMPLE
# ==========================================
def ask_database(question: str):
    print(f"User: {question}\n")
    # Stream the agent's thoughts and actions
    for chunk in sql_agent.stream({"messages": [{"role": "user", "content": question}]}):
        if "agent" in chunk:
            print("Agent:", chunk["agent"]["messages"][-1].content)
        elif "tools" in chunk:
            print("Tool Execution:", chunk["tools"]["messages"][-1].content)
        print("-" * 40)

# Test it
ask_database("What are the top 5 most active users by login count in the last 30 days?")
