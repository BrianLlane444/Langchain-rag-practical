"""
ui/UserInterface.py
Streamlit front end for the German Politics RAG demo.
Run:  streamlit run ui/UserInterface.py

The Streamlit script does HTTP calls only; it never imports from app/ollama_client.py or app/endpoints.py.
It uses the FastAPI endpoints directly.
It is a simple chat interface that allows users to ask questions and receive answers from the RAG pipeline.
It maintains a session ID to keep track of the conversation history.
It also provides a reset button to clear the conversation history.
"""

import uuid, os, requests, streamlit as st

# --------------------------------------------------- #
BACKEND = os.getenv("RAG_API_URL", "http://localhost:8000")
TIMEOUT = 300
# --------------------------------------------------- #

# 1) Page config should be the first Streamlit call
st.set_page_config(page_title="German-Politics RAG Chat", page_icon="🗳️")

# 2) Session bookkeeping – one UUID per browser tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state["history"] = []

# 3) UI title and description
st.title("German Politics Chat")
st.write(
    "Pose any claim you saw on social media. The assistant answers in a "
    "Motivational-Interviewing style and cites official party programmes."
)

# 4) Sidebar: show the current session id (and optional reset button)
with st.sidebar:
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    # Optional: Reset this chat (frees backend memory for this ID)
    if st.button("🔄 Reset this chat"):
        try:
            requests.post(
                f"{BACKEND}/reset",
                json={"session_id": st.session_state.session_id},
                timeout=TIMEOUT,
            )
            st.session_state["history"] = []
            st.success("Session cleared.")
            # If you also want a brand-new session after reset, uncomment:
            # st.session_state["session_id"] = str(uuid.uuid4())
            st.rerun()  # Refresh the page to clear chat
        except requests.RequestException as e:
            st.error(f"Could not reset session: {e}")

# ------------------------------------------------------------------ #
# 1) Show previous turns that came **from the backend**
# ------------------------------------------------------------------ #
if "history" not in st.session_state:
    st.session_state.history = []          # [(role, text), ...]

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

# ------------------------------------------------------------------ #
# 2) Reset conversation (server + client)
# ------------------------------------------------------------------ #
col1, col2 = st.columns(2)
with col1:
    if st.button("Start new topic"):
        try:
            requests.post(
                f"{BACKEND}/reset",
                json={"session_id": st.session_state.session_id},
                timeout=TIMEOUT,
            )
        except requests.RequestException as e:
            st.toast(f"Backend said: {e}", icon="⚠️")
        st.session_state.history.clear()
        st.rerun()

# ------------------------------------------------------------------ #
# 3) User message → FastAPI → response
# ------------------------------------------------------------------ #
user_msg = st.chat_input("Write your question …")

if user_msg:
    st.chat_message("user").write(user_msg)

    payload = {
        "session_id": st.session_state.session_id,
        "query":       user_msg,
    }

    try:
        r = requests.post(f"{BACKEND}/generate", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
    else:
        # data = {"response": "...", "history": [...] }
        answer = data["response"]
        st.chat_message("assistant").write(answer)

        # keep full authoritative history from server
        st.session_state.history = [
            (m["role"], m["text"]) for m in data["history"]
        ]
        st.rerun()   # so the freshly stored history renders on reload

