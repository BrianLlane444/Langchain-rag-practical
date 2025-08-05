import os
import shutil
from pathlib import Path

# ---------- NEW helper ---------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # rag_pipeline_project/

def _abs(path: str | Path) -> Path:
    """Return absolute path inside the project, no matter where CWD is."""
    return (PROJECT_ROOT / path).resolve()
# --------------------------------- #

# -----------------------------------------------------------------------------
#  System-prompt loader (now loads Markdown instead of JSON)
# -----------------------------------------------------------------------------
def load_system_prompt() -> str:
    """
    Load system_prompt.md that lives in the *same folder* as this utils.py.
    Returns the content as-is since Markdown is already human-readable.
    """
    prompt_path = Path(__file__).resolve().parent / "system_prompt.md"
    
    try:
        with prompt_path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
        return content
    except FileNotFoundError:
        # Fallback error message
        return """You are a helpful assistant that provides factual information about German politics.

Please provide factual answers based on retrieved documents and ask thoughtful questions to help users explore the topic.

Use [1], [2] for citations when referencing party documents."""


# -----------------------------------------------------------------------------
#  Cache helpers (unchanged)
# -----------------------------------------------------------------------------
def is_chroma_cache_present(folder: str = "embeddings/chromadb") -> bool:
    """
    Return True if the Chroma persistent-index folder exists and is non-empty,
    regardless of the current working directory.
    """
    cache_dir = _abs(folder)
    return cache_dir.exists() and any(cache_dir.iterdir())


def clear_cache(folder: str = "embeddings/chromadb") -> None:
    """
    Delete the vector-store folder so the next run rebuilds from scratch.
    """
    cache_dir = _abs(folder)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cache cleared: {cache_dir}")
    else:
        print(f"No cache found at {cache_dir}")