#Langchain-RAG-practical/rag_pipeline_project/app/rag_pipeline.py
"""
End-to-end RAG pipeline
-----------------------
1. Load & chunk PDFs
2. Embed chunks with Ollama embedding model
3. Build or reuse a Chroma vector store
4. Retrieve k most-similar chunks
5. Build final prompt (system + history + context +question)
6. Ask Llama3.1:8b / 70B and return the answer
"""

from pathlib import Path
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from .pdf_loader    import load_pdfs_from_folder
from .ollama_client import ask_ollama
from .utils         import load_system_prompt, is_chroma_cache_present

# ─── Config ──────────────────────────────────────────────────────────
SOURCE_DIR   = "documents/sources"
PERSIST_DIR  = "embeddings/chromadb"

EMBED_MODEL  = "bge-m3"
CHAT_MODEL   = "llama3.1:8b"          # swap to 70b on Nvidia workstation

# Optimized parameters for German political documents
CHUNK_SIZE    = 800   # Reduced from 1200 - better for complete sentences
CHUNK_OVERLAP = 120   # Reduced from 300 - 15% overlap is sufficient
RETRIEVE_K    = 4     # Reduced from 10 - Llama 8B gets overwhelmed with too many chunks

# Alternative settings for Llama 3:70B 
# CHUNK_SIZE    = 1000
# CHUNK_OVERLAP = 150
# RETRIEVE_K    = 6
# ─────────────────────────────────────────────────────────────────────


def run_rag_pipeline(
        user_query: str,
        *,
        force_rebuild: bool = False,
        history_prompt_str: str = "",          # ← NEW (chat memory)
        system_prompt_str: str | None = None   # ← NEW (override option)
) -> str:
    """Run the Retrieval-Augmented Generation pipeline."""

    # 1 System prompt (use override when supplied)
    system_prompt = system_prompt_str or load_system_prompt()

    # 2 Build or load vector store
    if not force_rebuild and is_chroma_cache_present(PERSIST_DIR):
        print("Using cached ChromaDB index")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        )
    else:
        print("(Re)building ChromaDB index …")
        docs = load_pdfs_from_folder(SOURCE_DIR)

        splitter   = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        split_docs = splitter.split_documents(docs)

        embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR,
        )
        vectorstore.persist()
        print(f"New ChromaDB index saved with {len(split_docs)} chunks")

    # 3 Retrieve context with similarity scores for debugging
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": RETRIEVE_K,
            "score_threshold": 0.6  # Filter out weak matches
        }
    )
    relevant_docs = retriever.get_relevant_documents(user_query)

    # Enhanced debugging output
    print(f"\n--- Retrieved {len(relevant_docs)} chunks ---")
    for i, doc in enumerate(relevant_docs, 1):
        # Show more of the chunk content for debugging
        preview = doc.page_content[:200].replace('\n', ' ').replace('  ', ' ')
        score = doc.metadata.get('score', 'N/A')
        print(f"[Chunk {i}] (Score: {score}) {preview}...")
    print("-" * 50)

    # 4 Improved context formatting
    if not relevant_docs:
        print("WARNING: No relevant documents found!")
        context_block = "Keine relevanten Dokumente gefunden."
    else:
        # Better context formatting with source separation
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"[Quelle {i}]: {doc.page_content}")
        context_block = "\n\n".join(context_parts)

    history_block = f"{history_prompt_str}\n\n" if history_prompt_str else ""

    # 5 Improved prompt structure - clearer for Llama models
    final_prompt = f"""{system_prompt}

{history_block}## VERFÜGBARE INFORMATIONEN:
{context_block}

## BENUTZER FRAGT:
{user_query}

## ANTWORT:"""

    # Debug: Show token count estimate
    estimated_tokens = len(final_prompt.split()) * 1.3  # Rough estimate
    print(f"Estimated prompt tokens: {estimated_tokens:.0f}")
    
    # 6 Ask the chat model and return its answer
    return ask_ollama(final_prompt, model=CHAT_MODEL)
