
#from rag import get_rag_response
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .rag_pipeline import run_rag_pipeline

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    response: str

@router.post("/generate", response_model=RAGResponse)
async def generate_answer(request: QueryRequest):
    try:
        result = run_rag_pipeline(request.query)
        return RAGResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))