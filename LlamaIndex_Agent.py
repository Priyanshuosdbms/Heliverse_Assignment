import os
import sys
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

# Load environment variables
load_dotenv()

def get_agent():
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please add it to the .env file.")
        sys.exit(1)

    # 1. Configuration
    # Using GPT-4o or GPT-4 for better reasoning if available, else 3.5-turbo
    llm = OpenAI(model="gpt-4o", temperature=0.1) 
    Settings.llm = llm
    
    # 2. Data Ingestion
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir} directory. Please add .txt files there.")
        
    # Check if there are files
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if not files:
        print(f"No .txt files found in {data_dir}. Creating a sample file.")
        with open(os.path.join(data_dir, "sample_generated.txt"), "w") as f:
            f.write("This is a sample file for the Agentic RAG system.\nLlamaIndex is a data framework for LLM applications.\n")
    
    print("Loading documents...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    # 3. Indexing
    # We use a basic VectorStoreIndex. 
    # For robustness, we could persist this, but for now we rebuild in memory or check storage.
    print("Indexing documents...")
    index = VectorStoreIndex.from_documents(documents)
    
    # 4. Tool Creation
    # We create a QueryEngineTool. 
    # To avoid "top_k=3" assumptions, we can:
    # a) Use a large top_k
    # b) Use a recursive retriever (more complex)
    # c) Rely on the agent to ask follow-up questions if the first answer is insufficient.
    # We choose (c) + a reasonable default, but we explicitly describe the tool 
    # to the agent as a search engine that returns *relevant excerpts*.
    
    query_engine = index.as_query_engine(similarity_top_k=5) # Default to 5, but agent can query multiple times.
    
    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="knowledge_base",
            description=(
                "Useful for searching information in the provided text files. "
                "Input should be a specific query or question. "
                "Returns relevant context from the documents."
            ),
        ),
    )
    
    # 5. Agent Creation
    # ReActAgent is chosen for its ability to reason and loop.
    # We give it a system prompt to encourage robustness.
    agent = ReActAgent.from_tools(
        [query_engine_tool],
        llm=llm,
        verbose=True,
        context=(
            "You are a robust Agentic RAG system designed to write detailed reports. "
            "Your goal is to answer the user's request comprehensively. "
            "Do not make assumptions about missing information; if you need more details, "
            "use the knowledge_base tool with different queries to explore the topic fully. "
            "Do not settle for a brief summary unless asked. "
            "Always verify you have enough information before finalizing the answer."
        )
    )
    
    return agent

def generate_report(request: str):
    agent = get_agent()
    print(f"\nGenerating report for: {request}\n" + "-"*50)
    response = agent.chat(f"Write a detailed report on: {request}")
    print("-" * 50)
    print("\nFINAL REPORT:\n")
    print(response)
    return response

if __name__ == "__main__":
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        generate_report(topic)
    else:
        print("Usage: python agent.py <topic>")
        print("Running default test...")
        generate_report("What is the purpose of this system and how does it work?")
