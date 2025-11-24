FROM python:3.12-slim
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr tesseract-ocr-deu libmagic1 curl \
  && rm -rf /var/lib/apt/lists/*

# 🔁 install Python deps first – enables layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# then copy code (you also mount it at runtime – this is just a sane default)
COPY rag_pipeline_project/app /app/app
RUN mkdir -p /app/documents/sources /app/embeddings

# defaults; compose can override these
ENV PYTHONUNBUFFERED=1 \
    REDIS_URL=redis://host.docker.internal:6379/0 \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
