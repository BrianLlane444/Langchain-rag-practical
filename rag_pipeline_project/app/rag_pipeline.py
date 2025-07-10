# rag_pipeline_project/app/rag_pipeline.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from .pdf_loader import load_pdfs_from_folder
from .ollama_client import ask_ollama
from .utils import load_system_prompt, is_chroma_cache_present

PERSIST_DIR = "embeddings/chromadb"
SOURCE_DIR = "documents/sources"

def run_rag_pipeline(user_query: str, force_rebuild: bool = False) -> str:
    """
    Executes the RAG pipeline:
    1. Load & chunk PDFs
    2. Embed documents and build or load Chroma index
    3. Retrieve relevant chunks
    4. Build system + context prompt
    5. Send prompt to LLaMA model and return response
    """

    # Step 1: System prompt is always loaded first
    system_prompt = load_system_prompt()

    # Step 2: Decide whether to reuse or rebuild the vector store
    if not force_rebuild and is_chroma_cache_present(PERSIST_DIR):
        print("Using cached ChromaDB index")
        vectorstore = Chroma(persist_directory=PERSIST_DIR,
                             embedding_function=OllamaEmbeddings(model="llama3.2:latest"))
    else:
        print("(Re)building ChromaDB index...")
        docs = load_pdfs_from_folder(SOURCE_DIR)

        # Step 3: Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)

        # Step 4: Embed documents and persist
        embedding_model = OllamaEmbeddings(model="llama3:70b")
        vectorstore = Chroma.from_documents(documents=split_docs,
                                            embedding=embedding_model,
                                            persist_directory=PERSIST_DIR)
        vectorstore.persist()
        print("New ChromaDB index saved")

    # Step 5: Retrieve relevant documents for the user's query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(user_query)

       # --- DEBUG: show which chunks came back ---
    print("\n--- Retrieved chunks ---")
    for i, d in enumerate(relevant_docs, 1):
        print(f"[Chunk {i}]\n{d.page_content[:300]}...\n")
    # ------------------------------------------

    # Step 6: Build full prompt by combining system prompt + context + user question
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    final_prompt = f"""{system_prompt}

## Retrieved Context
{context}

## User Question
{user_query}

## Assistant:"""

    # Step 7: Send the prompt to the local LLaMA model via REST and return its response
    return ask_ollama(final_prompt)
