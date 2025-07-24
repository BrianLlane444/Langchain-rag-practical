import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3:70b"               # fallback if caller omits model


def ask_ollama(prompt: str, model: str | None = None) -> str:
    """
    Send a prompt to the Ollama server and return its response text.

    Parameters
    ----------
    prompt : str
        The prompt to send.
    model : str, optional
        Ollama model name ("llama3:70b").  If None, uses DEFAULT_MODEL.
    """
    payload = {
        "model": model or DEFAULT_MODEL,    # respects caller’s choice
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
