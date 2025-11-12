# app/ollama_client.py
import os, hashlib, json
import requests

# Optional Redis cache
try:
    import redis  # pip install redis
    _REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _rc = redis.from_url(_REDIS_URL)
except Exception:
    _rc = None  # cache disabled if redis missing/unreachable

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "3600"))  # seconds

def _cache_key(model: str, prompt: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"||")
    h.update(prompt.encode("utf-8"))
    return "llmresp:" + h.hexdigest()

def ask_ollama(prompt: str, model: str | None = None) -> str:
    model = model or DEFAULT_MODEL

    # 1) Try cache
    if _rc is not None:
        try:
            ck = _cache_key(model, prompt)
            cached = _rc.get(ck)
            if cached:
                return json.loads(cached.decode("utf-8"))["response"]
        except Exception:
            pass  # cache miss/failure → continue

    # 2) Call Ollama
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        out = resp.json().get("response", "").strip()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    # 3) Store in cache
    if _rc is not None:
        try:
            _rc.setex(ck, CACHE_TTL, json.dumps({"response": out}, ensure_ascii=False))
        except Exception:
            pass

    return out
