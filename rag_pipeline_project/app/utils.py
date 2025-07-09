import json
import os
import shutil

def load_system_prompt(path: str = "rag_pipeline/app/system_prompt.json") -> str:
    """
    Load and flatten structured JSON into a readable system prompt string.
    """
    with open(path, "r", encoding="utf-8") as f:

        data = json.load(f)

    parts = []

    if "goal" in data:
        parts.append("## Goal\n" + "\n".join(f"- {v}" for v in data["goal"].values()))
    
    if "format_rules" in data:
        parts.append("## Format Rules\n" + "\n".join(f"- {v}" for v in data["format_rules"].values()))

    if "restrictions" in data:
        parts.append("## Restrictions\n" + "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in data["restrictions"].items()))

    if "planning_guidance" in data:
        parts.append("## Planning Guidance\n" + "\n".join(f"- {v}" for v in data["planning_guidance"]))

    if "output_guidance" in data:
        parts.append("## Output Guidance\n" + "\n".join(f"- {v}" for v in data["output_guidance"].values()))

    if "style_guidance" in data:
        parts.append("## Style Guidance\n" + "\n".join(f"- {k}: {v}" for k, v in data["style_guidance"].items()))

    if "session_context" in data:
        parts.append("## Session Context\n" + "\n".join(f"- {k}: {v}" for k, v in data["session_context"].items()))

    if "example" in data:
        parts.append("## Example\nUser: " + data["example"]["user"] + "\nAssistant: " + data["example"]["assistant"])

    return "\n\n".join(parts)

def is_chroma_cache_present(folder: str = "embeddings/chromadb") -> bool:
    """
    Return True if the Chroma persistent index folder exists and has files.
    """
    return os.path.exists(folder) and bool(os.listdir(folder))


def clear_cache(folder: str = "embeddings/chromadb"):
    """
    Deletes the vector store folder to force a fresh rebuild later.
    """
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"🧹 Cache cleared: {folder} deleted.")
    else:
        print(f" No cache found at {folder}.")
