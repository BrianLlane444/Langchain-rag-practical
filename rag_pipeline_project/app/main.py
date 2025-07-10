from fastapi import FastAPI
from .endpoints import router

app = FastAPI(
    title="Misinformation RAG API",
    description="Detect misinformation in German political content using RAG + Ollama",
    version="1.0.0" #as our projects evolves
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
