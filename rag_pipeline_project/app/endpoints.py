# app/endpoints.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os, json, redis

from .rag_pipeline import run_rag_pipeline, MEMORY_EXCHANGES
from .utils import load_system_prompt

router = APIRouter()

# ──────────────────────────────────────────────────────────────
# Redis client (set REDIS_URL in .env or environment)
# e.g., REDIS_URL=redis://localhost:6379/0
# ──────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(REDIS_URL)

def _hist_key(session_id: str) -> str:
    return f"hist:{session_id}"

def get_history(session_id: str) -> List[Dict[str, str]]:
    raw = r.get(_hist_key(session_id))
    return json.loads(raw) if raw else []

def set_history(session_id: str, history: List[Dict[str, str]]) -> None:
    # keep for 24h; adjust if you want
    r.setex(_hist_key(session_id), 60 * 60 * 24, json.dumps(history, ensure_ascii=False))

def clear_history(session_id: str) -> None:
    r.delete(_hist_key(session_id))

# ──────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    session_id: str
    query: str

class ChunkDetail(BaseModel):
    chunk_id: int
    score: float
    source: str
    page: Optional[int]
    content: str

class RAGResponse(BaseModel):
    response: str
    chunks: List[ChunkDetail]
    history: List[Dict[str, str]]

class SessionResetRequest(BaseModel):
    session_id: str

SYSTEM_PROMPT = load_system_prompt()

def format_history(turns: List[Dict[str, str]]) -> str:
    return "\n".join(f"{t['role'].capitalize()}: {t['text']}" for t in turns)

# ──────────────────────────────────────────────────────────────
# POST /generate   ► main chat endpoint (unchanged contract)
# ──────────────────────────────────────────────────────────────
@router.post("/generate", response_model=RAGResponse)
async def generate_answer(request: QueryRequest):
    try:
        # 1) Load history from Redis
        history = get_history(request.session_id)

        # 2) Run RAG (returns {"response": str, "chunks": [...]})
        rag_result = run_rag_pipeline(
            user_query         = request.query,
            force_rebuild      = False,
            history_prompt_str = format_history(history),
            system_prompt_str  = SYSTEM_PROMPT,
        )

        rag_answer = rag_result["response"]
        retrieved_chunks = rag_result["chunks"]

        # 3) Update & trim history (keep last N exchanges)
        history.append({"role": "user",      "text": request.query})
        history.append({"role": "assistant", "text": rag_answer})
        max_items = MEMORY_EXCHANGES * 2
        history = history[-max_items:]
        set_history(request.session_id, history)

        return RAGResponse(response=rag_answer, chunks=retrieved_chunks, history=history)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────────────────────────────────────
# POST /reset   ► clear chat memory for the tab
# ──────────────────────────────────────────────────────────────
@router.post("/reset")
async def reset_session(req: SessionResetRequest):
    clear_history(req.session_id)
    return {"status": "cleared"}

# ──────────────────────────────────────────────────────────────
# GET /health  ► simple readiness + active session count
# ──────────────────────────────────────────────────────────────
@router.get("/health")
async def health_check():
    try:
        n = 0
        for _ in r.scan_iter("hist:*", count=200):
            n += 1
    except Exception:
        n = -1
    return {"status": "healthy", "active_sessions": n}
