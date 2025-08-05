# LangChain RAG Practical – Misinformation Detection in German politics using RAG architecture and Prompt Engineering for Motivational Interviewing (Internship Project)

This is a small working prototype of a Retrieval-Augmented Generation (RAG) pipeline using:

- **LangChain** for document loading, chunking, embedding, and retrieval
- **Ollama** for running local LLMs (e.g., `llama3.2:latest` or `llama3:70b`) via REST endpoint
- **FastAPI** for exposing an API microservice
- **ChromaDB** for storing and querying vector embeddings
- **Jupyter Notebooks** for debugging and testing

---

 Future Work
 - OCR & Whisper for TikTok-based input 
 
 - Advanced prompt tuning [DONE]

 - Scalable backend with Docker [AlmostDONE]

 - Frontend interface (Streamlit or React) [DONE]
 
---

```bash
Lanchain_RAG_practical/
├── rag_pipeline_project/
│   ├── app/
│   │   ├── main.py               # FastAPI entry point
│   │   ├── pdf_loader.py         # Loads PDFs into LangChain Document format
│   │   ├── embed_documents.py    # Vector embedding logic
│   │   ├── endpoints.py          # FastAPI routes
│   │   ├── rag_pipeline.py       # Full end-to-end RAG logic
│   │   ├── utils.py              # Utilities
│   │   ├── ollama_client.py      # Sends prompts to Ollama REST
│   │   └── system_prompt.json    # Custom system prompt for LLM
│   ├── documents/sources/        # Input PDFs
│   ├── embeddings/chromadb/      # Chroma vector DB persists here
│   └── notebooks/
│       ├── dev_debug.ipynb       # Step-by-step debug notebook
│       └── dev_test_pipeline.ipynb  # Full pipeline test
├── requirements.txt
└── README.md
