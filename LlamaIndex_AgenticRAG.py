import os
import shutil
from typing import List, Dict, Any

# LlamaIndex Imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    PromptTemplate
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SemanticSplitterNodeParser

# ChromaDB Client
import chromadb

# Pydantic for structured output
from pydantic import BaseModel, Field

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
CHROMA_DB_PATH = "./chroma_db_llamaindex"

# --- 1. Define Pydantic Models for Structured Output ---
# This forces the LLM to give us clean JSON/Pydantic output.

class ReportStructure(BaseModel):
    """The table of contents for the technical report."""
    sections: List[str] = Field(
        description="A list of concise section titles for the report, derived from the text's main topics."
    )

# --- 2. Initialize Models and Global Settings (LlamaIndex Way) ---
print("Initializing models and LlamaIndex settings...")
# Initialize the LLM
Settings.llm = Ollama(model=OLLAMA_LLM_MODEL, temperature=0.0)
# Initialize the Embedding Model
Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL)

# --- 3. Define the Agentic Workflow Functions ---

def load_and_chunk_text() -> (List, str):
    """
    Loads text, returns the full text and semantically-chunked nodes.
    """
    print("--- (Task 1: Load and Chunk Text) ---")
    try:
        documents = SimpleDirectoryReader(input_files=[INPUT_FILE_PATH]).load_data()
        if not documents:
            print("Error: No documents found or file is empty.")
            return [], ""
            
        full_text = documents[0].get_content()
        print(f"Loaded {len(full_text)} characters.")
        
        # SemanticChunker for adaptive, topic-based chunks.
        parser = SemanticSplitterNodeParser(
            buffer_size=1,  # Number of sentences to group for comparison
            breakpoint_percentile_threshold=95, # Lower = more splits
            embed_model=Settings.embed_model
        )
        
        nodes = parser.get_nodes_from_documents(documents)
        print(f"Split text into {len(nodes)} semantic chunks (nodes).")
        return nodes, full_text
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE_PATH}")
        return [], ""
    except Exception as e:
        print(f"Error during loading/chunking: {e}")
        return [], ""

def create_index_and_retriever(nodes: List) -> (VectorStoreIndex, Any):
    """
    Creates and stores a ChromaDB vector store and retriever.
    """
    print("--- (Task 2: Create Vector Index) ---")
    if not nodes:
        print("No nodes to index. Skipping.")
        return None, None
        
    # Clean up old ChromaDB directory if it exists
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Removing old ChromaDB at {CHROMA_DB_PATH}")
        shutil.rmtree(CHROMA_DB_PATH)

    # Initialize ChromaDB client and collection
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection("report_rag")
    
    # Create the LlamaIndex vector store adapter
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create the index
    print("Creating new vector store index...")
    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )
    
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    print("Vector store index and query engine created.")
    return index, query_engine

def determine_report_structure(full_text: str) -> List[str]:
    """
    Asks the LLM to analyze the text and propose a report structure (TOC).
    This is the "planning" step.
    """
    print("--- (Task 3: Determine Report Structure) ---")
    if not full_text:
        print("No text to analyze. Skipping.")
        return []

    prompt = PromptTemplate(
        """You are a technical analyst. Your task is to create a "Table of Contents" 
        for a detailed report based on the following text.
        Identify the main logical sections and topics in the text.
        Respond with a Pydantic object based on the ReportStructure schema.
        
        Text:
        {text}
        """
    ).format(text=full_text)
    
    try:
        # Use structured_predict to get a Pydantic object back
        response = Settings.llm.structured_predict(
            ReportStructure, 
            prompt=prompt
        )
        titles = response.sections
        print(f"Determined report structure with {len(titles)} sections:")
        for title in titles:
            print(f"- {title}")
        return titles
        
    except Exception as e:
        print(f"Error determining report structure: {e}")
        return []

def generate_report_sections(query_engine: Any, sections: List[str]) -> (Dict[str, str], List[str]):
    """
    Generates content for each section using the RAG query engine.
    This is the "execution" loop.
    """
    print("--- (Task 4: Generate Report Sections) ---")
    generated_sections = {}
    review_notes = []

    # This is the "strict" prompt that enforces our rules
    qa_template_str = (
        "You are a factual technical report writer.\n"
        "Your task is to write the report section for the topic '{query_str}' using ONLY the context provided below.\n\n"
        "Rules:\n"
        "1.  You MUST NOT add any information, opinions, or analysis that is not explicitly present in the context.\n"
        "2.  Stick strictly to the facts and phrasing in the context.\n"
        "3.  If the context is insufficient, does not cover the topic, or is empty,\n"
        "    you MUST respond with only the exact string \"INSUFFICIENT_CONTEXT\" and nothing else.\n\n"
        "Context:\n"
        "{context_str}\n\n"
        "---\n"
        "Begin Report Section:"
    )
    qa_template = PromptTemplate(qa_template_str)

    if not query_engine:
        print("Error: Query engine is not initialized.")
        return {}, []

    # Update the query engine's prompt template
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_template}
    )

    for topic in sections:
        print(f"Generating section: '{topic}'")
        
        try:
            # 1. Query the RAG engine for the topic
            response = query_engine.query(topic)
            generated_text = str(response)

            # 2. Check for the "human review" flag
            if "INSUFFICIENT_CONTEXT" in generated_text or len(generated_text) < 20:
                print(f"Flagging '{topic}' for human review (insufficient context).")
                section_content = "*(This section has been flagged for human review. The automated agent found insufficient context in the source text to write this section.)*"
                review_notes.append(
                    f"Section '{topic}': Insufficient context found for generation."
                )
                generated_sections[topic] = section_content
            else:
                print("Section generated successfully.")
                generated_sections[topic] = generated_text

        except Exception as e:
            print(f"Error during section generation for '{topic}': {e}")
            section_content = "*(This section failed to generate due to an error. Please review manually.)*"
            review_notes.append(f"Section '{topic}': Generation failed with error: {e}")
            generated_sections[topic] = section_content
            
    return generated_sections, review_notes

def compile_report(generated_sections: Dict[str, str], review_notes: List[str]):
    """
    Compiles the final report from the generated sections and review notes.
    """
    print("--- (Task 5: Compile Final Report) ---")
    
    report = "# Detailed Technical Report\n\n"
    
    # Add the "Human Review" section
    if review_notes:
        report += "## 1. Human Review Notice\n\n"
        report += "The following sections could not be generated automatically due to insufficient context in the source text. Please review and complete them manually:\n\n"
        for note in review_notes:
            report += f"- {note}\n"
        report += "\n---\n\n"
    
    # Add the generated sections
    section_num = 2 if review_notes else 1
    for topic, content in generated_sections.items():
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

# --- 4. Run the Agentic Workflow ---

def main():
    print("\nRunning the LlamaIndex agentic RAG process...")
    try:
        # Task 1: Load and Chunk
        nodes, full_text = load_and_chunk_text()
        if not nodes:
            print("Halting process: No text to analyze.")
            return

        # Task 2: Create Index
        index, query_engine = create_index_and_retriever(nodes)
        if not index:
            print("Halting process: Index creation failed.")
            return
            
        # Task 3: Plan Structure
        sections = determine_report_structure(full_text)
        if not sections:
            print("Halting process: Could not determine report structure.")
            return
            
        # Task 4: Generate Sections
        generated_sections, review_notes = generate_report_sections(query_engine, sections)
        
        # Task 5: Compile
        compile_report(generated_sections, review_notes)
        
        print("\nAgent finished successfully.")

    except Exception as e:
        print(f"\nAn error occurred during agent execution: {e}")

    finally:
        # Clean up
        if os.path.exists(CHROMA_DB_PATH):
            print(f"Cleaning up ChromaDB directory: {CHROMA_DB_PATH}")
            shutil.rmtree(CHROMA_DB_PATH)

if __name__ == "__main__":
    main()
