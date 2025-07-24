# rag_pipeline_project/app/rag_pipeline.py
"""
End-to-end RAG pipeline
----------------------
1. Load & chunk all PDFs under documents/sources
2. Embed chunks with an Ollama model (nomic-embed-text by default)
3. Build or reuse a Chroma vector store
4. Retrieve k most-similar chunks for a user query
5. Build the final prompt (system + context + question)
6. Ask Llama-3-70B and return the answer
"""

import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from .pdf_loader import load_pdfs_from_folder          # your loader → Document list
from .ollama_client import ask_ollama                  # tiny REST helper
from .utils import load_system_prompt, is_chroma_cache_present

# --------------------------------------------------------------------
# Configuration 
# --------------------------------------------------------------------
SOURCE_DIR   = "documents/sources"
PERSIST_DIR  = "embeddings/chromadb"

EMBED_MODEL  = "nomic-embed-text"   # any Ollama embedding model
CHAT_MODEL   = "llama3:70b"         # the LLM that writes the answer

CHUNK_SIZE       = 1200             # characters per chunk
CHUNK_OVERLAP    = 300              # overlap between chunks
RETRIEVE_K       = 10                # how many chunks to feed the prompt
# --------------------------------------------------------------------


def run_rag_pipeline(user_query: str, *, force_rebuild: bool = False) -> str:
    """Run the full Retrieval-Augmented Generation pipeline."""
    #1 Load system-prompt once
    system_prompt = load_system_prompt()

    #2 Load or rebuild the Chroma index
    if not force_rebuild and is_chroma_cache_present(PERSIST_DIR):
        print("Using cached ChromaDB index")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        )
    else:
        print("(Re)building ChromaDB index…")
        docs = load_pdfs_from_folder(SOURCE_DIR)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        split_docs = splitter.split_documents(docs)

        embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR,
        )
        vectorstore.persist()
        print("New ChromaDB index saved")

    #3 Retrieve k most-similar chunks
    retriever     = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})
    relevant_docs = retriever.get_relevant_documents(user_query)

    # --- DEBUG (optional): show what came back -----------------------
    print("\n--- Retrieved chunks ---")
    for i, d in enumerate(relevant_docs, 1):
        preview = d.page_content[:300].replace("\n", " ")
        print(f"[Chunk {i}]\n{preview}…\n")
    # -----------------------------------------------------------------

    #4 Build the prompt
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    final_prompt = f"""{system_prompt}

## Retrieved Context
{context}

## User Question
{user_query}

## Assistant:"""

    #5 Ask the chat model via Ollama REST and return its answer
    return ask_ollama(final_prompt, model=CHAT_MODEL)


   
