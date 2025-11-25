# German Politics RAG (MI-style)

## Retrieval-Augmented Generation to counter misinformation about German politics, answering in a Motivational Interviewing style and always citing page-exact sources from official party PDFs.

## Stack: LangChain • FastAPI • Streamlit • ChromaDB • Ollama (Llama 3.1) • Docker • Redis

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Docker Deployment](#docker-deployment)
- [License](#license)

---

## Overview

This project addresses the spread of political misinformation on social media platforms by providing an AI-powered fact-checking system. Users can input claims they've seen online and receive accurate, sourced information from official German party programs.

### Key Technologies

- **LLM**: Llama 3.1:8b via Ollama
- **Embedding Model**: BGE-M3 (optimized for German language)
- **Framework**: LangChain + FastAPI + Streamlit
- **Vector Store**: ChromaDB with cosine similarity
- **Orchestration**: Docker Compose
- **Cache**: Redis (LLM response caching)

---

## Features

✅ **Intelligent Retrieval**: Searches through official party programs (CDU, SPD, Grüne, AfD, FDP, Die Linke)  
✅ **Source Citations**: Every response includes PDF source and page numbers  
✅ **Motivational Interviewing**: Empathetic, non-confrontational conversation style  
✅ **Chunk Visibility**: Shows the exact text chunks used for answer generation  
✅ **Conversation Memory**: Maintains context for up to 5 exchanges  
✅ **German-Optimized**: BGE-M3 embeddings for superior German language understanding  
✅ **Relevance Scoring**: Displays confidence scores for retrieved information  
✅ **Redis Caching**: Caches LLM responses for faster repeat queries  
✅ **PDF Preview**: View the exact PDF page cited in responses  
✅ **Docker Ready**: Containerized deployment with Docker Compose

---

## Architecture

```
┌─────────────────┐
│   Streamlit UI  │ (Port 8501)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │ (Port 8000)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│  Redis  │ │ RAG Pipeline │
│  Cache  │ │              │
└─────────┘ └──────┬───────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
┌─────────────┐ ┌────────┐ ┌────────────┐
│  ChromaDB   │ │ Ollama │ │  PDF Docs  │
│  Embeddings │ │  LLM   │ │  /sources  │
└─────────────┘ └────────┘ └────────────┘
```

### Component Overview

1. **Streamlit UI**: User-facing chat interface with PDF preview
2. **FastAPI**: RESTful API handling RAG requests
3. **RAG Pipeline**: Document retrieval, context building, LLM generation
4. **ChromaDB**: Vector database storing 3,376+ German text chunks
5. **Ollama**: Local LLM inference (Llama 3.1:8b + BGE-M3 embeddings)
6. **Redis**: Response caching layer (1-hour TTL)

---

## Prerequisites

Before installation, ensure you have:

- **Python 3.12+** (tested on 3.12)
- **Docker Desktop** (for containerized deployment)
- **Ollama** installed and running
- **16GB+ RAM** (for 8b model) or 64GB+ for 70b model
- **macOS, Linux, or Windows** with WSL2
- **Homebrew** (macOS) for system dependencies

### System Dependencies (macOS)

```bash
# Install Tesseract OCR with German language pack
brew install tesseract tesseract-lang

# Install Poppler (for PDF processing)
brew install poppler

# Install Redis
brew install redis
brew services start redis

# Verify installations
tesseract --list-langs | grep deu
pdftotext --version
redis-cli ping  # Should return PONG
```

### Install Ollama

```bash
# Download from https://ollama.ai or use Homebrew
brew install ollama

# Start Ollama service
ollama serve

# Pull required models (in another terminal)
ollama pull llama3.1:8b    # LLM model (~4.7GB)
ollama pull bge-m3         # Embedding model (~2.4GB)

# Verify models are installed
ollama list
```

---

## Installation


### Step 1: Clone Repository

```bash
git clone https://github.com/BrianLlane444/Langchain-rag-practical.git
cd Langchain-rag-practical
```

### Step 2: Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Prepare Your Documents

Place your German political PDF files in `rag_pipeline_project/documents/sources/`. The system currently processes:
- CDU/SPD coalition agreements
- Individual party programs (AfD, FDP, Grüne, Die Linke, SPD)
- Government programs

Example documents included:
```
rag_pipeline_project/documents/sources/
├── AfD_Bundestagswahlprogramm2025_web.pdf
├── fdp-wahlprogramm_2025.pdf
├── Koalitionsvertrag-–-barrierefreie-Version.pdf
├── Parteiprogramm_Die_Linke_2024-web.pdf (2024)
├── Regierungsprogramm.pdf
└── ... (9 PDFs total)
```

---

## 🚀 Quick Start

### Option A: Local Development (No Docker)

#### 1. Start Redis

```bash
# If installed via Homebrew
brew services start redis

# Or run in foreground
redis-server
```

#### 2. Start Ollama (if not already running)

```bash
ollama serve
```

#### 3. Generate Embeddings (First Time Only)

```bash
cd rag_pipeline_project
python -m app.embed_documents
```

**Expected output:**
```
Loading PDFs…
Splitting into chunks…
Embedding + saving to Chroma…
Stored 3376 chunks in embeddings/chromadb
```

#### 4. Start the FastAPI Backend

```bash
# From project root
cd rag_pipeline_project
uvicorn app.main:app --reload --port 8000
```

**Verify it's running:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","active_sessions":0}
```

#### 5. Launch Streamlit UI

```bash
# From project root
RAG_API_URL=http://localhost:8000 \
DOCS_DIR="$(pwd)/rag_pipeline_project/documents/sources" \
streamlit run ui/UserInterface.py
```

#### 6. Access the Application

- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

### Option B: Docker Deployment (Recommended for Production)

#### 1. Ensure Services Are Running on Host

```bash
# Start Ollama
ollama serve

# Start Redis
brew services start redis

# Verify connectivity
ollama list
redis-cli ping
```

#### 2. Build and Start Docker Containers

```bash
# Clean previous builds (optional)
docker compose down -v
docker builder prune -f

# Build and start
docker compose up --build -d

# Check logs
docker compose logs -f app
```

#### 3. Generate Embeddings Inside Container

```bash
docker compose exec app python -m app.embed_documents
```

#### 4. Access the Application

- **API**: http://localhost:8000
- **Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

#### 5. Launch Streamlit (on Host)

```bash
RAG_API_URL=http://localhost:8000 \
DOCS_DIR="$(pwd)/rag_pipeline_project/documents/sources" \
streamlit run ui/UserInterface.py
```

**Visit**: http://localhost:8501

---

## 💡 Usage

### Basic Workflow

1. **Enter a claim**: Type a political claim you've seen on social media
2. **Get verified response**: The AI analyzes official documents and provides facts
3. **Review sources**: Check the retrieved chunks with confidence scores
4. **View PDF pages**: Click on sources to preview the exact cited page
5. **Continue conversation**: Ask follow-up questions with maintained context

### Example Queries

```text
"Auf Instagram stand, die CDU macht Politik nur für Reiche. Was sagen Sie dazu?"

"Vergleiche CDU und SPD zur Klimapolitik"

"Was sagt die AfD zur Migration?"

"Was steht im Koalitionsvertrag zur Digitalisierung?"
```

### Response Features

Each response includes:
- ✅ **Fact-checked answer** in Motivational Interviewing style
- 📄 **Source citations** with PDF names and page numbers
- 🔍 **Retrieved chunks** with relevance scores (0-1 scale)
- 🖼️ **PDF preview** of the exact cited page
- 💬 **Conversation memory** (up to 5 exchanges)
- ⚡ **Cached responses** for faster repeat queries

### Testing Redis Cache

```bash
# Monitor Redis activity in real-time
redis-cli MONITOR

# Check cache statistics
redis-cli INFO stats

# View cached keys
redis-cli KEYS "*"

# Check number of cached responses
redis-cli DBSIZE
```

**Cache behavior:**
- First query: Cache MISS → generates response → stores in Redis (TTL: 1 hour)
- Repeat query: Cache HIT → instant response from Redis
- After 1 hour: Key expires → next query regenerates and caches

---

## 📡 API Documentation

### Endpoints

#### `POST /generate`
Generate a RAG response for a user query.

**Request:**
```json
{
  "session_id": "uuid-string",
  "query": "User's question in German"
}
```

**Response:**
```json
{
  "response": "AI-generated answer with citations",
  "chunks": [
    {
      "chunk_id": 1,
      "score": 0.89,
      "source": "AfD_Bundestagswahlprogramm2025.pdf",
      "page": 42,
      "content": "First 300 characters of chunk..."
    }
  ],
  "history": [...]
}
```

#### `POST /reset`
Clear conversation history for a session.

#### `GET /health`
Health check endpoint returning active sessions.

---

## ⚙️ Configuration

### Environment Variables

Configure via environment variables or `.env` file:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://host.docker.internal:11434  # Inside Docker
# OLLAMA_BASE_URL=http://localhost:11434           # Local development

# Redis Configuration
REDIS_URL=redis://host.docker.internal:6379/0     # Inside Docker
# REDIS_URL=redis://localhost:6379/0              # Local development

# Model Selection
OLLAMA_MODEL=llama3.1:8b        # Or llama3:70b for production
EMBED_MODEL=bge-m3

# Cache Settings
LLM_CACHE_TTL=3600              # Redis cache TTL in seconds (1 hour)
```

### RAG Pipeline Settings (`rag_pipeline_project/app/rag_pipeline.py`)

```python
# Model Configuration
DEFAULT_EMBED_MODEL = "bge-m3"        # German-optimized embeddings
DEFAULT_CHAT_MODEL = "llama3.1:8b"    # LLM model

# Chunking Parameters
DEFAULT_CHUNK_SIZE = 800              # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 120           # Overlap between chunks

# Retrieval Settings
DEFAULT_RETRIEVE_K = 4                # Number of chunks to retrieve
DEFAULT_MEMORY_EXCHANGES = 5          # Number of Q&A pairs to remember
DEFAULT_COLLECTION_NAME = "de_politics"  # ChromaDB collection name

# MMR (Maximal Marginal Relevance) Settings
fetch_k = 40                          # Candidate pool size
lambda_mult = 0.5                     # Balance: relevance (→1) vs diversity (→0)
use_mmr = True                        # Enable MMR for diverse results
```

### System Prompt

The Motivational Interviewing style is configured in `rag_pipeline_project/app/system_prompt.md`. Modify this file to adjust:
- Conversation tone
- Response structure
- Citation format
- Empathy level

---

## 🔧 Development

### Project Structure

```
Langchain-rag-practical/
├── rag_pipeline_project/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── endpoints.py         # API route handlers
│   │   ├── rag_pipeline.py      # Core RAG logic (retrieval + generation)
│   │   ├── ollama_client.py     # Ollama LLM client with Redis caching
│   │   ├── pdf_loader.py        # PDF document loading utilities
│   │   ├── embed_documents.py   # Embedding generation script
│   │   ├── utils.py             # Helper functions
│   │   ├── system_prompt.md     # MI conversation style prompt
│   │   └── notebooks/           # Jupyter notebooks for development
│   ├── documents/
│   │   └── sources/             # PDF storage (9 party programs)
│   ├── embeddings/
│   │   └── chromadb/            # Vector database (3376 chunks)
│   ├── Dockerfile               # Container definition
│   └── requirements.txt         # Python dependencies
├── ui/
│   └── UserInterface.py         # Streamlit chat interface
├── docker-compose.yml           # Multi-container orchestration
├── requirements.lock.txt        # Locked dependencies
├── DecisionLog.md               # Architecture decision records
└── README.md
```

### Key Components Explained

#### `rag_pipeline.py` - The Core Engine
- **`RAGPipeline` class**: Manages embedding, retrieval, and generation
- **`retrieve()` method**: Fetches top-k relevant chunks using MMR
- **`generate()` method**: Builds context and generates LLM response
- **`_load_vectorstore()`**: Loads ChromaDB with BGE-M3 embeddings

#### `ollama_client.py` - LLM Interface
- **`ask_ollama()`**: Sends prompts to Ollama API
- **Redis caching**: Stores responses with SHA1-hashed prompt keys
- **Error handling**: Graceful fallback if Redis unavailable

#### `endpoints.py` - API Routes
- **`POST /generate`**: Main RAG endpoint with session management
- **`POST /reset`**: Clears conversation history
- **`GET /health`**: Health check with active session count

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Test query
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "test123",
    "query": "Was sagt die SPD zur Klimapolitik?"
  }' | jq

# Reset conversation
curl -X POST http://localhost:8000/reset \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "test123"}'
```

### Monitoring and Debugging

```bash
# View API logs
docker compose logs -f app

# Monitor Redis cache hits/misses
redis-cli MONITOR

# Check ChromaDB collection
docker compose exec app python -c "
from app.embed_documents import COLLECTION_NAME, PERSIST_DIR
import chromadb
client = chromadb.PersistentClient(path=str(PERSIST_DIR))
collection = client.get_collection(COLLECTION_NAME)
print(f'Total chunks: {collection.count()}')
"

# Verify Ollama connectivity from container
docker compose exec app curl http://host.docker.internal:11434/api/tags
```



---

## 🐳 Docker Deployment

### Building the Docker Image

```bash
# Stop any running containers
docker compose down

# Optional: Clean build cache (after failed builds)
docker builder prune -f

# Build with no cache (ensures fresh build)
docker compose build --no-cache

# Start in detached mode
docker compose up -d
```

### Docker Configuration

The `docker-compose.yml` defines:

```yaml
services:
  app:
    build:
      context: .
      dockerfile: rag_pipeline_project/Dockerfile
    container_name: rag-api
    environment:
      REDIS_URL: redis://host.docker.internal:6379/0
      OLLAMA_HOST: http://host.docker.internal:11434
      OLLAMA_BASE_URL: http://host.docker.internal:11434
    ports:
      - "8000:8000"
    volumes:
      - ./rag_pipeline_project/app:/app/app  # Live code updates
      - ./rag_pipeline_project/documents/sources:/app/documents/sources:ro
      - ./rag_pipeline_project/embeddings:/app/embeddings
    restart: unless-stopped
```

**Key networking:**
- `host.docker.internal` allows container to reach host services (Ollama, Redis)
- Volumes mount local code for live development
- Port 8000 exposes FastAPI

### Dockerfile Highlights

```dockerfile
FROM python:3.12-slim

# Install system dependencies (Tesseract with German language pack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr tesseract-ocr-deu libmagic1 curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
COPY rag_pipeline_project/app /app/app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Managing Docker Services

```bash
# View logs
docker compose logs -f app

# Restart after code changes
docker compose restart app

# Stop services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v

# Check container status
docker compose ps

# Execute commands inside container
docker compose exec app python -m app.embed_documents
docker compose exec app bash
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **Empty chunks returned (`"chunks": []`)**

**Cause**: Collection name mismatch between embedding and retrieval.

**Solution**:
```bash
# Verify collection exists
docker compose exec app python -c "
import chromadb
client = chromadb.PersistentClient(path='embeddings/chromadb')
print([c.name for c in client.list_collections()])
"

# Should show: ['de_politics']

# If missing, re-embed:
rm -rf rag_pipeline_project/embeddings/chromadb
docker compose exec app python -m app.embed_documents
docker compose restart app
```

#### 2. **Ollama connection refused**

**Cause**: Container can't reach Ollama on host.

**Solution**:
```bash
# Verify Ollama is running
ollama serve

# Test from container
docker compose exec app curl http://host.docker.internal:11434/api/tags

# Check environment variables
docker compose exec app env | grep OLLAMA
```

#### 3. **Import errors (ModuleNotFoundError)**

**Cause**: Outdated LangChain imports after version updates.

**Solution**:
```python
# OLD (deprecated):
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# NEW (correct):
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
```

#### 4. **Redis connection timeout**

**Cause**: Redis not running or wrong host.

**Solution**:
```bash
# Start Redis
brew services start redis

# Test connection
redis-cli ping  # Should return PONG

# From container
docker compose exec app sh -c 'apt-get update && apt-get install -y redis-tools && redis-cli -h host.docker.internal ping'
```

#### 5. **PDF preview not showing**

**Cause**: `DOCS_DIR` not set or poppler missing.

**Solution**:
```bash
# Ensure poppler is installed
brew install poppler

# Set DOCS_DIR when launching Streamlit
DOCS_DIR="$(pwd)/rag_pipeline_project/documents/sources" \
streamlit run ui/UserInterface.py
```

#### 6. **Slow first query**

**Explanation**: First query has to:
1. Load ChromaDB index (~2-3 seconds)
2. Generate embeddings for query
3. Retrieve chunks
4. Generate LLM response (~5-10 seconds for 8b model)

**Subsequent queries**: Much faster due to:
- Loaded vector store
- Redis caching (instant for exact repeats)

### Checking System Health

```bash
# API health
curl http://localhost:8000/health

# Ollama models
ollama list

# Redis status
redis-cli INFO stats

# Docker container health
docker compose ps
docker compose logs --tail=50 app

# ChromaDB count
docker compose exec app python -c "
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os
embedder = OllamaEmbeddings(model='bge-m3', base_url=os.getenv('OLLAMA_BASE_URL'))
db = Chroma(persist_directory='embeddings/chromadb', embedding_function=embedder, collection_name='de_politics')
print(f'Documents: {db._collection.count()}')
"
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Supervisor**: Omed Abed
- **Institution**: Rhine-Waal University of Applied Sciences
- **LangChain Community** for RAG framework
- **Ollama Team** for local LLM inference
- **German Political Parties** for providing accessible party programs

---

## 📧 Contact

- **Author**: Brian Llane
- **Email**: Brian.Llane@hsrw.org
- **GitHub**: [@BrianLlane444](https://github.com/BrianLlane444)
- **Repository**: [Langchain-rag-practical](https://github.com/BrianLlane444/Langchain-rag-practical)

---

## 🔗 Related Projects

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama](https://ollama.ai/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)

---

*Last Updated: November 25, 2025*