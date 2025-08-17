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
st.set_page_config(page_title="German-Politics RAG Chat", page_icon="🗳️", layout="wide")

# 2) Session bookkeeping – one UUID per browser tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state["history"] = []

if "chunks_history" not in st.session_state:
    st.session_state["chunks_history"] = []  # Store chunks for each query

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
            st.session_state["chunks_history"] = []
            st.success("Session cleared.")
            st.rerun()  # Refresh the page to clear chat
        except requests.RequestException as e:
            st.error(f"Could not reset session: {e}")

# ------------------------------------------------------------------ #
# Layout: Main chat on left, chunks on right
# ------------------------------------------------------------------ #
col_chat, col_chunks = st.columns([2, 1])

with col_chat:
    st.subheader("💬 Conversation")
    
    # Show previous turns
    for i, (role, msg) in enumerate(st.session_state.history):
        st.chat_message(role).write(msg)
        
        # Show chunks for assistant messages if available
        if role == "assistant" and i // 2 < len(st.session_state.chunks_history):
            chunks_idx = i // 2
            if st.session_state.chunks_history[chunks_idx]:
                with st.expander("📄 Retrieved Sources", expanded=False):
                    for chunk in st.session_state.chunks_history[chunks_idx]:
                        st.markdown(f"""
                        **[Chunk {chunk['chunk_id']}]** Score: {chunk['score']:.3f}  
                        📁 **{chunk['source']}** - Page {chunk['page']}  
                        > {chunk['content']}
                        
                        ---
                        """)

with col_chunks:
    st.subheader("Latest Retrieved Chunks")
    if st.session_state.chunks_history and st.session_state.chunks_history[-1]:
        for chunk in st.session_state.chunks_history[-1]:
            with st.container():
                st.markdown(f"""
                ### Chunk {chunk['chunk_id']}
                **Score:** {chunk['score']:.3f}  
                **Source:** {chunk['source']}  
                **Page:** {chunk['page']}
                
                <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;">
                {chunk['content']}
                </div>
                """, unsafe_allow_html=True)
                st.divider()
    else:
        st.info("Retrieved chunks will appear here after you ask a question.")

# ------------------------------------------------------------------ #
# Reset conversation button
# ------------------------------------------------------------------ #
with col_chat:
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
            st.session_state.chunks_history.clear()
            st.rerun()

# ------------------------------------------------------------------ #
# User message → FastAPI → response
# ------------------------------------------------------------------ #
user_msg = st.chat_input("Write your question …")

if user_msg:
    with col_chat:
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
        # data = {"response": "...", "chunks": [...], "history": [...] }
        answer = data["response"]
        chunks = data.get("chunks", [])
        
        with col_chat:
            st.chat_message("assistant").write(answer)
            
            # Show chunks inline with the response
            if chunks:
                with st.expander("Retrieved Sources", expanded=True):
                    for chunk in chunks:
                        st.markdown(f"""
                        **[Chunk {chunk['chunk_id']}]** Score: {chunk['score']:.3f}  
                        **{chunk['source']}** - Page {chunk['page']}  
                        > {chunk['content']}
                        
                        ---
                        """)

        # Store chunks for this response
        st.session_state.chunks_history.append(chunks)

        # Keep full authoritative history from server
        st.session_state.history = [
            (m["role"], m["text"]) for m in data["history"]
        ]
        
        st.rerun()   # so the freshly stored history renders on reload

