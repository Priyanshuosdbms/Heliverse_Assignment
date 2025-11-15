import os
import shutil
from typing import List, Dict, Any, TypedDict, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document

from langgraph.graph import StateGraph, END

# --- Configuration ---
# Make sure Ollama is running (e.g., `ollama serve`)
# and you have the models (e.g., `ollama pull llama3`, `ollama pull nomic-embed-text`)

# Model for generating text and structure
OLLAMA_LLM_MODEL = "llama3"
# Model for creating embeddings (text-to-vector)
OLLAMA_EMBED_MODEL = "nomic-embed-text"
# Path to your input text file
INPUT_FILE_PATH = "video_transcript.txt"
# Directory for ChromaDB persistence
CHROMA_DB_PATH = "./chroma_db"

# --- 1. Define Agent State ---
# This is the "memory" of our agent as it moves through the graph.

class AgentState(TypedDict):
    """
    Defines the state of the agent.
    """
    original_text: str             # The raw text from the file
    chunks: List[Document]         # The semantically-chunked text
    report_structure: List[str]    # List of section titles for the report
    
    # We use 'Any' for the retriever to avoid serialization issues
    # with complex objects if we were to checkpoint.
    retriever: Any                 
    
    # Keep track of generation
    current_section_index: int
    generated_sections: Dict[str, str]
    review_notes: List[str]

# --- 2. Define Pydantic Models for Structured Output ---
# This forces the LLM to give us clean JSON output.

class ReportStructure(BaseModel):
    """The table of contents for the technical report."""
    sections: List[str] = Field(
        description="A list of concise section titles for the report, derived from the text's main topics."
    )

# --- 3. Initialize Models ---
print("Initializing models...")
# Initialize the LLM
llm = Ollama(model=OLLAMA_LLM_MODEL, temperature=0.0)
# Initialize the Embedding Model
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

# --- 4. Define Graph Nodes (Agent's "Skills") ---

def load_text(state: AgentState) -> AgentState:
    """
    Loads the raw text from the input file.
    """
    print("--- (Node: load_text) ---")
    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded {len(text)} characters.")
        return {**state, "original_text": text}
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE_PATH}")
        return {**state, "original_text": ""}

def adaptive_chunking(state: AgentState) -> AgentState:
    """
    Splits the text using SemanticChunker for adaptive, topic-based chunks.
    """
    print("--- (Node: adaptive_chunking) ---")
    if not state["original_text"]:
        print("No text to chunk. Skipping.")
        return {**state, "chunks": []}
        
    # SemanticChunker uses embeddings to find break points.
    # This is much more effective than fixed-size chunking.
    text_splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile" 
        # You can also try "standard_deviation" or "interquartile"
    )
    docs = text_splitter.create_documents([state["original_text"]])
    print(f"Split text into {len(docs)} semantic chunks.")
    return {**state, "chunks": docs}

def create_retriever(state: AgentState) -> AgentState:
    """
    Creates and stores a ChromaDB vector store and retriever.
    """
    print("--- (Node: create_retriever) ---")
    if not state["chunks"]:
        print("No chunks to index. Skipping.")
        return {**state, "retriever": None}

    # Clean up old ChromaDB directory if it exists
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Removing old ChromaDB at {CHROMA_DB_PATH}")
        shutil.rmtree(CHROMA_DB_PATH)

    # Create a new persistent vector store
    print("Creating new vector store...")
    vectorstore = Chroma.from_documents(
        documents=state["chunks"],
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3} # Retrieve top 3 relevant chunks
    )
    print("Vector store and retriever created.")
    return {**state, "retriever": retriever}

def determine_report_structure(state: AgentState) -> AgentState:
    """
    Asks the LLM to analyze the text and propose a report structure (TOC).
    This is a key "agentic" step.
    """
    print("--- (Node: determine_report_structure) ---")
    if not state["original_text"]:
        print("No text to analyze. Skipping.")
        return {**state, "report_structure": []}

    parser = JsonOutputParser(pydantic_object=ReportStructure)
    
    prompt = PromptTemplate(
        template="""You are a technical analyst. Your task is to create a "Table of Contents" 
        for a detailed report based on the following text.
        Identify the main logical sections and topics in the text.
        Respond with a JSON object containing a list of concise section titles.
        
        {format_instructions}
        
        Text:
        {text}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    try:
        structure = chain.invoke({"text": state["original_text"]})
        titles = structure.get("sections", [])
        print(f"Determined report structure with {len(titles)} sections:")
        for title in titles:
            print(f"- {title}")
        
        return {
            **state,
            "report_structure": titles,
            "current_section_index": 0,
            "generated_sections": {},
            "review_notes": []
        }
    except Exception as e:
        print(f"Error determining report structure: {e}")
        return {**state, "report_structure": []}


def generate_section(state: AgentState) -> AgentState:
    """
    Generates content for a single section of the report.
    This is the core RAG loop.
    """
    print("--- (Node: generate_section) ---")
    
    # Get the topic for the current section
    index = state["current_section_index"]
    if index >= len(state["report_structure"]):
        print("All sections generated.")
        return state
        
    topic = state["report_structure"][index]
    print(f"Generating section: '{topic}'")
    
    retriever = state["retriever"]
    if not retriever:
        print("Error: Retriever not found.")
        return state

    # 1. Retrieve relevant context
    try:
        retrieved_docs = retriever.invoke(topic)
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    except Exception as e:
        print(f"Error during retrieval: {e}")
        context = ""

    if not context:
        print("No context retrieved for this topic.")
        context = "No context available."

    # 2. Define the generation prompt with "human review" logic
    generation_prompt = PromptTemplate(
        template="""You are a factual technical report writer.
        Your task is to write the report section for the topic '{topic}' using ONLY the context provided below.
        
        Rules:
        1.  You MUST NOT add any information, opinions, or analysis that is not explicitly present in the context.
        2.  Stick strictly to the facts and phrasing in the context.
        3.  If the context is insufficient, does not cover the topic, or is empty,
            you MUST respond with only the exact string "INSUFFICIENT_CONTEXT" and nothing else.

        Context:
        {context}
        
        ---
        Begin Report Section:
        """,
        input_variables=["topic", "context"]
    )
    
    chain = generation_prompt | llm
    
    # 3. Generate the section content
    try:
        generated_text = chain.invoke({"topic": topic, "context": context})
    except Exception as e:
        print(f"Error during generation: {e}")
        generated_text = "INSUFFICIENT_CONTEXT"

    # 4. Check for the "human review" flag
    if "INSUFFICIENT_CONTEXT" in generated_text or len(generated_text) < 20:
        print(f"Flagging '{topic}' for human review (insufficient context).")
        section_content = "*(This section has been flagged for human review. The automated agent found insufficient context in the source text to write this section.)*"
        review_notes = state["review_notes"] + [
            f"Section '{topic}': Insufficient context found for generation."
        ]
        generated_sections = {**state["generated_sections"], topic: section_content}
    else:
        print("Section generated successfully.")
        review_notes = state["review_notes"]
        generated_sections = {**state["generated_sections"], topic: generated_text}
        
    return {
        **state,
        "generated_sections": generated_sections,
        "review_notes": review_notes,
        "current_section_index": index + 1
    }

def compile_report(state: AgentState) -> AgentState:
    """
    Compiles the final report from the generated sections and review notes.
    """
    print("--- (Node: compile_report) ---")
    
    report = "# Detailed Technical Report\n\n"
    
    # Add the "Human Review" section
    if state["review_notes"]:
        report += "## 1. Human Review Notice\n\n"
        report += "The following sections could not be generated automatically due to insufficient context in the source text. Please review and complete them manually:\n\n"
        for note in state["review_notes"]:
            report += f"- {note}\n"
        report += "\n---\n\n"
    
    # Add the generated sections
    section_num = 2 if state["review_notes"] else 1
    for topic, content in state["generated_sections"].items():
        report += f"## {section_num}. {topic}\n\n"
        report += f"{content}\n\n"
        section_num += 1

    # Print the final report to console
    print("\n\n" + "="*80)
    print("FINAL REPORT")
    print("="*80 + "\n")
    print(report)
    print("="*80)
    print("Report compilation complete.")
    
    # Save the report to a file
    with open("final_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Final report saved to 'final_report.md'")

    return state

# --- 5. Define Conditional Edge ---

def should_continue(state: AgentState) -> str:
    """
    Determines whether to continue generating sections or compile the report.
    """
    print("--- (Conditional Edge: should_continue) ---")
    if state["current_section_index"] < len(state["report_structure"]):
        print("More sections to generate. Looping.")
        return "continue_generation"
    else:
        print("All sections processed. Compiling report.")
        return "compile_report"

# --- 6. Build the Graph ---

print("Building the agent graph...")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("load_text", load_text)
workflow.add_node("adaptive_chunking", adaptive_chunking)
workflow.add_node("create_retriever", create_retriever)
workflow.add_node("determine_report_structure", determine_report_structure)
workflow.add_node("generate_section", generate_section)
workflow.add_node("compile_report", compile_report)

# Set the entry point
workflow.set_entry_point("load_text")

# Add edges
workflow.add_edge("load_text", "adaptive_chunking")
workflow.add_edge("adaptive_chunking", "create_retriever")
workflow.add_edge("create_retriever", "determine_report_structure")
workflow.add_edge("determine_report_structure", "generate_section")

# Add the conditional edge for the generation loop
workflow.add_conditional_edges(
    "generate_section",
    should_continue,
    {
        "continue_generation": "generate_section",  # Loop back
        "compile_report": "compile_report"          # Move to compile
    }
)

workflow.add_edge("compile_report", END)

# Compile the graph
app = workflow.compile()

# --- 7. Run the Agent ---

print("\nRunning the agentic RAG process...")
# We start with an empty state. The input_file is handled by the first node.
inputs = {} 
try:
    app.invoke(inputs)
    print("\nAgent finished successfully.")
except Exception as e:
    print(f"\nAn error occurred during agent execution: {e}")

# Clean up
finally:
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Cleaning up ChromaDB directory: {CHROMA_DB_PATH}")
        shutil.rmtree(CHROMA_DB_PATH)
