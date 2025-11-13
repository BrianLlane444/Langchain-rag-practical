# rag_pipeline_project/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# (Optional) system deps for OCR/PDF if you embed inside the container.
# If you only serve an already-built index, you can skip this block.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr libmagic1 \
  && rm -rf /var/lib/apt/lists/*

# Copy app code
COPY app /app/app
COPY documents /app/documents
COPY embeddings /app/embeddings
COPY ../requirements.txt /tmp/requirements.txt

# Install deps
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Default envs: talk to Redis service in Compose; talk to host Ollama
ENV REDIS_URL=redis://redis:6379/0
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8000"]
