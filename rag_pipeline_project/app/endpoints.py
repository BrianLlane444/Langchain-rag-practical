# app/endpoints.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from .rag_pipeline import run_rag_pipeline, MEMORY_EXCHANGES
from .utils import load_system_prompt

router = APIRouter()

# ──────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    session_id: str        # a browser‑side UUID
    query: str

class ChunkDetail(BaseModel):
    chunk_id: int
    score: float
    source: str
    page: Optional[int]
    content: str

class RAGResponse(BaseModel):
    response: str
    chunks: List[ChunkDetail]  # NEW: Include retrieved chunks
    history: List[Dict[str, str]]   # echo the updated history back

class SessionResetRequest(BaseModel):
    session_id: str

# ──────────────────────────────────────────────────────────────
# Simple in‑memory storage  {session_id: [{"role": "...", "text": "..."}]}
# ──────────────────────────────────────────────────────────────
SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

SYSTEM_PROMPT = load_system_prompt()

def format_history(turns: List[Dict[str, str]]) -> str:
    """Convert stored turns into a single string."""
    return "\n".join(f"{t['role'].capitalize()}: {t['text']}" for t in turns)

# ──────────────────────────────────────────────────────────────
# POST /generate   ► main chat endpoint
# ──────────────────────────────────────────────────────────────
@router.post("/generate", response_model=RAGResponse)
async def generate_answer(request: QueryRequest):
    try:
        # 1) Get history for this session
        history = SESSION_MEMORY.get(request.session_id, [])

        # 2) Build full prompt and get response WITH chunks
        rag_result = run_rag_pipeline(          # NOW returns dict with response and chunks
            user_query         = request.query,
            force_rebuild      = False,
            history_prompt_str = format_history(history),
            system_prompt_str  = SYSTEM_PROMPT
        )

        # 3) Extract response and chunks
        rag_answer = rag_result["response"]
        retrieved_chunks = rag_result["chunks"]

        # 4) Update memory  (store user Q + assistant A)
        history.append({"role": "user",      "text": request.query})
        history.append({"role": "assistant", "text": rag_answer})
        # Keep last N complete exchanges (N Q&A pairs = N*2 items total)
        max_items = MEMORY_EXCHANGES * 2
        SESSION_MEMORY[request.session_id] = history[-max_items:]

        return RAGResponse(
            response=rag_answer, 
            chunks=retrieved_chunks,  # NEW: Include chunks
            history=history
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# POST /reset   ► clear chat memory for the tab
# ──────────────────────────────────────────────────────────────
@router.post("/reset")
async def reset_session(req: SessionResetRequest):
    # `req.session_id` is now available
    SESSION_MEMORY.pop(req.session_id, None)
    return {"status": "cleared"}

"""The reset endpoint clears the chat history for a specific session ID,
    allowing users to start a new conversation without previous context."""

# ──────────────────────────────────────────────────────────────
# GET / Health  ► Docker/Kuvernetes health check or readiness probe, CI/CD smoke test
#and letting us know how many in-memory sessions are active. 
# ──────────────────────────────────────────────────────────────
@router.get("/health")
async def health_check():
    print("DEBUG - current sessions:", dict(SESSION_MEMORY))  # show keys & texts
    return {"status": "healthy", "active_sessions": len(SESSION_MEMORY)}