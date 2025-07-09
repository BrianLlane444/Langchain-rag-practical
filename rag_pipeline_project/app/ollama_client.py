import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"  # adjust based if running on personal laptop or bulky workstation

def ask_ollama(prompt: str) -> str:
    """
    Send a prompt to the Ollama LLaMA model and return its response.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")