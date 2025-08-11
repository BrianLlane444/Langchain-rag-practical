# rag_pipeline_project/app/rag_pipeline.py
"""
End-to-end RAG pipeline
-----------------------
1) Load & chunk PDFs
2) Embed chunks with Ollama embedding model (bge-m3)
3) Build or reuse a Chroma vector store (cosine space)
4) Retrieve top-k most similar chunks (0-1 relevance scores)
5) Build final prompt (system + history + context + question)
6) Ask Llama and return the answer
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings            # ← updated import
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from .pdf_loader import load_pdfs_from_folder
from .ollama_client import ask_ollama
from .utils import load_system_prompt, is_chroma_cache_present

# ─── Config ──────────────────────────────────────────────────────────
SOURCE_DIR      = "documents/sources"
PERSIST_DIR     = "embeddings/chromadb"
COLLECTION_NAME = "de_politics"

EMBED_MODEL = "bge-m3"                   # via Ollama
CHAT_MODEL  = "llama3.1:8b"              # swap to 70b on workstation

# Chunking tuned for long party programs (German)
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120

# Retrieval
RETRIEVE_K       = 4
SCORE_THRESHOLD: Optional[float] = None  # e.g., 0.30–0.40 if you want filtering; None = disabled
# ─────────────────────────────────────────────────────────────────────

# Alternative settings for Llama 3:70B 
# CHUNK_SIZE    = 1000
# CHUNK_OVERLAP = 150
# RETRIEVE_K    = 6
# ────────────────────────────────────

def _build_or_load_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    Create or reopen a persistent Chroma store using cosine similarity.
    Uses Ollama's bge-m3 for embeddings.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if not force_rebuild and is_chroma_cache_present(PERSIST_DIR):
        print("Using cached ChromaDB index")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    print("(Re)building ChromaDB index …")
    docs = load_pdfs_from_folder(SOURCE_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    split_docs = splitter.split_documents(docs)

    # Ensure cosine space so LangChain can return 0–1 relevance scores
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"New ChromaDB index saved with {len(split_docs)} chunks")
    return vectorstore


def _retrieve_with_scores(
    vectorstore: Chroma,
    query: str,
    k: int,
    score_threshold: Optional[float],
) -> List[Tuple[Document, float]]:
    """
    Use Chroma's similarity_search_with_relevance_scores to get (doc, score)
    where score is in [0,1] when cosine space is used.
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    if score_threshold is not None:
        results = [(d, s) for (d, s) in results if s >= score_threshold]
    return results


def run_rag_pipeline(
    user_query: str,
    *,
    force_rebuild: bool = False,
    history_prompt_str: str = "",          # chat memory (already formatted text)
    system_prompt_str: Optional[str] = None,
) -> str:
    """Run the Retrieval-Augmented Generation pipeline and return the answer."""

    # 1) System prompt
    system_prompt = system_prompt_str or load_system_prompt()

    # 2) Vector store
    vectorstore = _build_or_load_vectorstore(force_rebuild=force_rebuild)

    # 3) Retrieve (top-k, with normalized scores)
    results = _retrieve_with_scores(
        vectorstore=vectorstore,
        query=user_query,
        k=RETRIEVE_K,
        score_threshold=SCORE_THRESHOLD,
    )

    # Debug output
    print(f"\n--- Retrieved {len(results)} chunks ---")
    for i, (doc, score) in enumerate(results, 1):
        preview = doc.page_content[:200].replace("\n", " ").replace("  ", " ")
        try:
            source = Path(doc.metadata.get("source", "Unknown")).name
        except Exception:
            source = doc.metadata.get("source", "Unknown")
        print(f"[Chunk {i}] (Score: {score:.3f}) {source} :: {preview}...")
    print("-" * 50)

    # 4) Build context block
    if not results:
        print("WARNING: No relevant documents found!")
        context_block = "Keine relevanten Dokumente gefunden."
    else:
        parts: List[str] = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
        
        # Extract year from filename if possible
            if "2025" in source:
                year_info = " (2025)"
            elif "2024" in source:
                year_info = " (2024)"
            elif "2023" in source:
                year_info = " (2023)"
            else:
                year_info = ""
        
        parts.append(f"[Quelle {i} | Score {score:.3f} | {source}{year_info}]\n{doc.page_content}")
    context_block = "\n\n".join(parts)

    history_block = f"{history_prompt_str}\n\n" if history_prompt_str else ""

    # 5) Final prompt (German UX)
    final_prompt = f"""{system_prompt}

{history_block}## VERFÜGBARE INFORMATIONEN:
{context_block}

## BENUTZER FRAGT:
{user_query}

## ANTWORT:"""

    # Rough token estimate
    estimated_tokens = int(len(final_prompt.split()) * 1.3)
    print(f"Estimated prompt tokens: {estimated_tokens}")

    # 6) Ask the chat model and return its answer
    return ask_ollama(final_prompt, model=CHAT_MODEL)
