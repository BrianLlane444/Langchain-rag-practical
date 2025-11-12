# app/visualize_graph.py
from pathlib import Path

# Ensure app is importable when running from project root
import os, sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mi_graph import MIConversationGraph

def main():
    g = MIConversationGraph().graph
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)

    # Try PNG first
    try:
        png_path = out_dir / "mi_graph.png"
        g.get_graph().draw_png(str(png_path))   # needs pygraphviz + graphviz
        print(f"Saved {png_path}")
    except Exception as e:
        print(f"PNG render failed ({e}); falling back to Mermaid.")
        mmd_path = out_dir / "mi_graph.mmd"
        mermaid = g.get_graph().draw_mermaid()
        mmd_path.write_text(mermaid, encoding="utf-8")
        print(f"Saved {mmd_path} (Mermaid). Open in VS Code Mermaid or any online viewer and screenshot it.")

if __name__ == "__main__":
    main()

#preview in graph_shift shift+K, V, or cmd shift V
