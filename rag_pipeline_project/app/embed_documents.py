"""
Embeds all PDFs under documents/sources/ into a Chroma DB.

• Uses Ollama embedding model "bge-m3"
• Writes to embeddings/chromadb/
• Skips work if the DB already exists
"""

import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
# If you installed langchain-ollama ≥0.3
# from langchain_ollama import OllamaEmbeddings
# else (community back-compat)
from langchain_community.embeddings import OllamaEmbeddings


# ── Config ──────────────────────────────────────────────────────────
DOCUMENTS_PATH = Path("documents/sources")
PERSIST_DIR    = Path("embeddings/chromadb")
EMBED_MODEL    = "bge-m3"  # Ollama embedding model
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 120

# ────────────────────────────────────────────────────────────────────


def chroma_exists() -> bool:
    """True if a Chroma index already lives at PERSIST_DIR."""
    return PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())


def load_documents(folder: Path):
    docs = []
    for pdf in folder.glob("*.pdf"):
        docs.extend(PyPDFLoader(str(pdf)).load())
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def build_index(chunks):
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    embedder = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=str(PERSIST_DIR),
    )
    vectordb.persist()
    print(f"Stored {len(chunks)} chunks in {PERSIST_DIR}")


if __name__ == "__main__":
    if chroma_exists():
        print(f"Vector store already present at {PERSIST_DIR} — nothing to do.")
    else:
        print("Loading PDFs…")
        raw_docs = load_documents(DOCUMENTS_PATH)

        print("Splitting into chunks…")
        chunks = split_documents(raw_docs)

        print("Embedding + saving to Chroma…")
        build_index(chunks)

