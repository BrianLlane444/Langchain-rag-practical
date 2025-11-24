"""
ui/UserInterface.py
Streamlit front end for the German Politics RAG demo.

Run from the *repo root* (the folder that contains `rag_pipeline_project/` and `ui/`):
    streamlit run ui/UserInterface.py
"""

import os
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st
from pdf2image import convert_from_path  # requires poppler installed (brew install poppler)

# ─────────────────────────────────────────────────────────────
# Make the `app` package resolvable *from the repo root* and
# use app.utils._abs so all paths are rooted at rag_pipeline_project/
# ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]          # …/Langchain-RAG-Practical
APP_ROOT = REPO_ROOT / "rag_pipeline_project"            # …/Langchain-RAG-Practical/rag_pipeline_project
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    from rag_pipeline_project.app.utils import _abs  # anchored at rag_pipeline_project/
except Exception:
    # Fallback: still anchor at rag_pipeline_project if import fails
    def _abs(p: str | Path) -> Path:
        return (APP_ROOT / p).resolve()

# Allow overriding documents dir via env var (useful for Docker)
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(_abs("documents/sources"))))

# ─────────────────────────────────────────────────────────────
# Backend config
# ─────────────────────────────────────────────────────────────
BACKEND = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 300  # seconds

# ─────────────────────────────────────────────────────────────
# Streamlit page setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="German-Politics RAG Chat", page_icon="🗳️")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state["history"] = []          # list[ (role, text) ]
if "chunks_history" not in st.session_state:
    st.session_state["chunks_history"] = []   # list[ List[chunk dict] ]

st.title("German Politics Chat")
st.write(
    "Stelle eine Behauptung/Frage. Die Assistenz antwortet im MI-Stil "
    "und zitiert offizielle Parteidokumente."
)

# Sidebar: session + reset
with st.sidebar:
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    if st.button("🔄 Reset Chat"):
        try:
            requests.post(
                f"{BACKEND}/reset",
                json={"session_id": st.session_state.session_id},
                timeout=TIMEOUT,
            )
            st.session_state["history"].clear()
            st.session_state["chunks_history"].clear()
            st.success("Session zurückgesetzt.")
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Reset fehlgeschlagen: {e}")

# ─────────────────────────────────────────────────────────────
# Conversation history
# ─────────────────────────────────────────────────────────────
st.subheader("💬 Unterhaltung")  # (purely a header; rename/remove if you want)
for role, msg in st.session_state["history"]:
    st.chat_message(role).write(msg)

# ─────────────────────────────────────────────────────────────
# Quellen viewer: render the exact cited PDF page from latest answer
# ─────────────────────────────────────────────────────────────
st.markdown("### 📚 Quellen (zuletzt)")
latest_chunks = st.session_state["chunks_history"][-1] if st.session_state["chunks_history"] else []
if not latest_chunks:
    st.caption("Noch keine Quellen vorhanden. Stelle zuerst eine Frage.")
else:
    labels = [
        f'{c.get("source","?")} – Seite {c.get("page","?")} (Score {c.get("score",0):.2f})'
        for c in latest_chunks
    ]
    sel_idx = st.selective_slider if hasattr(st, "selective_slider") else None  # guard for older Streamlit
    idx = st.selectbox("Quelle auswählen", range(len(latest_chunks)), format_func=lambda i: labels[i], index=0)

    chosen = latest_chunks[int(idx)]

    # Always resolve to your host docs dir using just the filename
    src_name = Path(str(chosen.get("source", ""))).name
    pdf_path = Path(DOCS_DIR) / src_name

    # Page number (safe fallback)
    try:
        page_num = int(chosen.get("page") or 1)
    except Exception:
        page_num = 1

    if not pdf_path.exists():
        st.error(f"PDF nicht gefunden: {pdf_path}")
        st.caption(f"(Hinweis: DOCS_DIR = {DOCS_DIR})")
    else:
        try:
            pages = convert_from_path(
                str(pdf_path), dpi=150, first_page=page_num, last_page=page_num
            )
            st.image(pages[0], use_column_width=True, caption=f"{pdf_path.name} - Seite {page_num}")
        except Exception as e:
            st.warning(f"Konnte die PDF-Seite nicht rendern (zeige nur Pfad). Grund: {e}")
            st.code(str(pdf_path))


# ─────────────────────────────────────────────────────────────
# Controls
# ─────────────────────────────────────────────────────────────
col1, _ = st.columns(2)
with col1:
    if st.button("Start new topic"):
        try:
            requests.post(
                f"{BACKEND}/reset",
                json={"session_id": st.session_state.session_id},
                timeout=TIMEOUT,
            )
        except requests.RequestException as e:
            st.toast(f"Backend: {e}", icon="⚠️")
        st.session_state["history"].clear()
        st.session_state["chunks_history"].clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────
# Chat input → backend
# ─────────────────────────────────────────────────────────────
user_msg = st.chat_input("Schreibe deine Frage …")
if user_msg:
    st.chat_message("user").write(user_msg)
    payload = {"session_id": st.session_state.session_id, "query": user_msg}

    with st.spinner("Antwort wird erstellt…"):
        try:
            r = requests.post(f"{BACKEND}/generate", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            st.error(f"Request fehlgeschlagen: {e}")
        else:
            answer = data.get("response", "")
            chunks = data.get ("chunks") if False else data.get("chunks", [])  # keep key robust
            with st.chat_message("assistant"):
                st.write(answer)  # no raw chunk dump

            # persist for viewer + chat log
            st.session_state["chunks_history"].append(chunks or [])
            hist = data.get("history") or []
            st.session_state["history"] = [(m.get("role","assistant"), m.get("text","")) for m in hist]

            st.rerun()
