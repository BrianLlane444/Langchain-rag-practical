# rag_pipeline_project/app/rag_pipeline.py

import os
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .pdf_loader import load_pdfs_from_folder
from .ollama_client import ask_ollama
from .utils import load_system_prompt, is_chroma_cache_present

# ---------- Environment / networking ----------
# Inside Docker, "localhost" means the container. Use host.docker.internal to reach the host’s services.
OLLAMA_BASE_URL = (
    os.getenv("OLLAMA_BASE_URL")
    or os.getenv("OLLAMA_HOST")  # accept either var name
    or "http://host.docker.internal:11434"
)


# ------------ Default variables ----------------
DEFAULT_SOURCE_DIR      = "documents/sources"
DEFAULT_PERSIST_DIR     = "embeddings/chromadb"
DEFAULT_COLLECTION_NAME = "de_politics"

DEFAULT_EMBED_MODEL = "bge-m3"          # via Ollama
DEFAULT_CHAT_MODEL  = "llama3.1:8b"     # swap later if you want, try qwen2.5:14b if you have the RAM for it

DEFAULT_MEMORY_EXCHANGES = 5
DEFAULT_CHUNK_SIZE       = 800
DEFAULT_CHUNK_OVERLAP    = 120
DEFAULT_RETRIEVE_K       = 4
DEFAULT_SCORE_THRESHOLD: Optional[float] = None  # e.g., 0.35

MEMORY_EXCHANGES = DEFAULT_MEMORY_EXCHANGES  # imported by endpoints.py

# ---------------- Helpers ----------------------
def _cosine(a, b):
    # a,b are lists of floats
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(x*x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return sum(x*y for x, y in zip(a, b)) / (da * db)

# ─── Class wrapper so LangGraph / endpoints can import it ─────────────
class RAGPipeline:
    def __init__(
        self,
        source_dir: str = DEFAULT_SOURCE_DIR,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embed_model: str = DEFAULT_EMBED_MODEL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        retrieve_k: int = DEFAULT_RETRIEVE_K,
        score_threshold: Optional[float] = DEFAULT_SCORE_THRESHOLD,
    ):
        self.source_dir = source_dir
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieve_k = retrieve_k
        self.score_threshold = score_threshold

        # MMR knobs
        self.fetch_k = 40         # candidate pool
        self.lambda_mult = 0.5    # relevance (→1) vs diversity (→0)
        self.use_mmr = True

        self._vectorstore: Optional[Chroma] = None
        self._embedder = OllamaEmbeddings(model=self.embed_model, base_url=OLLAMA_BASE_URL)

    # ------- vector store -------
    def _build_or_load_vectorstore(self, force_rebuild: bool = False) -> Chroma:
        if not force_rebuild and is_chroma_cache_present(self.persist_dir):
            print("Using cached ChromaDB index")
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self._embedder,
                collection_name=self.collection_name,
            )

        print("(Re)building ChromaDB index …")
        docs = load_pdfs_from_folder(self.source_dir)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_docs = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self._embedder,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(f"New ChromaDB index saved with {len(split_docs)} chunks")
        return vectorstore

    def _ensure_vs(self, force_rebuild: bool = False) -> Chroma:
        if self._vectorstore is None or force_rebuild:
            self._vectorstore = self._build_or_load_vectorstore(force_rebuild)
        return self._vectorstore

    # ------- retrieval -------
    def _similarity_with_scores(
        self, vs: Chroma, query: str, k: int
    ) -> List[Tuple[Document, float]]:
        results = vs.similarity_search_with_relevance_scores(query, k=k)
        if self.score_threshold is not None:
            results = [(d, s) for (d, s) in results if s >= self.score_threshold]
        return results

    def _mmr_retrieve(self, vs: Chroma, query: str, k: int) -> List[Document]:
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": self.fetch_k, "lambda_mult": self.lambda_mult},
        )
        return retriever.invoke(query)

    def retrieve(self, query: str, *, k: Optional[int] = None, force_rebuild: bool = False) -> List[Dict]:
        vs = self._ensure_vs(force_rebuild)
        k = k or self.retrieve_k
    
    # Get documents
        if self.use_mmr:
            docs = self._mmr_retrieve(vs, query, k)
        else:
            docs = [d for (d, _) in self._similarity_with_scores(vs, query, k)]
    
        # Format into chunks with scores
        q_emb = self._embedder.embed_query(query)
        chunks = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "Unknown")
            try:
                source_name = Path(src).name
            except Exception:
                source_name = src
        
            d_emb = self._embedder.embed_query(d.page_content[:2000])
            score = _cosine(q_emb, d_emb)
        
            chunks.append({
                "chunk_id": i,
                "score": float(score),
                "source": source_name,
                "page": d.metadata.get("page", "?"),
                "content": d.page_content[:300] + "..." if len(d.page_content) > 300 else d.page_content,
                "_full_content": d.page_content
            })
    
        return chunks

    # ------- end-to-end -------
    def generate(
        self,
        user_query: str,
        *,
        history_prompt_str: str = "",
        system_prompt_str: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> Dict:
        system_prompt = system_prompt_str or load_system_prompt()
        vs = self._ensure_vs(force_rebuild)

      

        
        # 2) Retrieve formatted chunks (already has scores)
        retrieved_chunks = self.retrieve(user_query, k=self.retrieve_k)

        # 3) Context block
        if not retrieved_chunks:
            print("WARNING: No relevant documents found!")
            context_block = "Keine relevanten Dokumente gefunden."
        else:
            parts: List[str] = []
            for chunk in retrieved_chunks:
                src = chunk["source"]
                page = chunk["page"]
                full_content = chunk.get("_full_content", chunk["content"])
                year_info = ""
                if "2025" in src: year_info = " (2025)"
                elif "2024" in src: year_info = " (2024)"
                elif "2023" in src: year_info = " (2023)"
                parts.append(f"[Quelle {chunk['chunk_id']} | {src}{year_info} | Seite {page}]\n{full_content}")
            
            context_block = "\n\n".join(parts)

        history_block = f"{history_prompt_str}\n\n" if history_prompt_str else ""

        final_prompt = f"""{system_prompt}

WICHTIG: Zitiere Quellen am ENDE deiner Antwort im Format: [Dokumentname, Seite X]

{history_block}## VERFÜGBARE INFORMATIONEN:
{context_block}

## BENUTZER FRAGT:
{user_query}

## ANTWORT:
Bitte antworte vollständig und füge am ENDE eine Liste der verwendeten Quellen hinzu."""
        estimated_tokens = int(len(final_prompt.split()) * 1.3)
        print(f"Estimated prompt tokens: {estimated_tokens}")

        llm_response = ask_ollama(final_prompt, model=self.chat_model)
    
       # Remove internal field before returning
        returned_chunks = [
            {k: v for k, v in chunk.items() if k != "_full_content"}
            for chunk in retrieved_chunks
    ]
    
        return {"response": llm_response, "chunks": returned_chunks}


# --- Compatibility shim: keep old imports working ---

_GLOBAL_PIPELINE: Optional["RAGPipeline"] = globals().get("_GLOBAL_PIPELINE")
if _GLOBAL_PIPELINE is None:
    _GLOBAL_PIPELINE = RAGPipeline()

def run_rag_pipeline(
    user_query: str,
    *,
    force_rebuild: bool = False,
    history_prompt_str: str = "",
    system_prompt_str: str | None = None,
):
    """Back-compat: delegate legacy function to the class API."""
    return _GLOBAL_PIPELINE.generate(
        user_query=user_query,
        force_rebuild=force_rebuild,
        history_prompt_str=history_prompt_str,
        system_prompt_str=system_prompt_str,
    )
# --- end shim ---
