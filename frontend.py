"""
frontend.py — Streamlit chat interface for the RAG pipeline

Run:
    streamlit run frontend.py
"""

import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI/ML Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = {}   # keyed by message index of the assistant turn


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_session(session_id: str):
    """Pull conversation history from the API and store in session state."""
    try:
        r = requests.get(f"{API_BASE}/sessions/{session_id}/history", timeout=10)
        if r.status_code == 200:
            st.session_state.session_id = session_id
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in r.json()["messages"]
            ]
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        st.error("Cannot reach the API.")


def fetch_sessions():
    """Return list of past sessions from the API."""
    try:
        r = requests.get(f"{API_BASE}/sessions", timeout=30)
        if r.status_code == 200:
            return r.json()
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        pass
    return []


def start_new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.traces = {}


# ── Sidebar — conversation threads ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 AI/ML Assistant")

    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        start_new_chat()

    st.divider()
    st.markdown("#### Past Conversations")

    sessions = fetch_sessions()
    if not sessions:
        st.caption("No conversations yet.")
    else:
        for s in sessions:
            label = s["title"] if s["title"] else "Untitled"
            # Truncate long titles
            display = label if len(label) <= 40 else label[:37] + "..."
            # Highlight the active session
            is_active = s["session_id"] == st.session_state.session_id
            btn_type = "primary" if is_active else "secondary"

            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(display, key=s["session_id"], use_container_width=True,
                             type=btn_type):
                    load_session(s["session_id"])
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{s['session_id']}",
                             help="Delete this conversation"):
                    try:
                        requests.delete(
                            f"{API_BASE}/sessions/{s['session_id']}", timeout=10
                        )
                        if s["session_id"] == st.session_state.session_id:
                            start_new_chat()
                        st.rerun()
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot reach the API.")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("📚 AI/ML Knowledge Assistant")
st.caption("Ask questions about AI/ML books and lecture transcripts.")

# Render stored messages
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and idx in st.session_state.traces:
            with st.expander("🔍 Retrieval Details"):
                for t in st.session_state.traces[idx]:
                    st.markdown(
                        f"**Rank {t['rerank_rank']}** &nbsp;|&nbsp; "
                        f"`{t['chunk_id']}` &nbsp;|&nbsp; "
                        f"Score: `{t['rerank_score']}`"
                    )
                    st.caption(t["doc"][:400])
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about AI/ML ..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking ..."):
            try:
                response = requests.post(
                    f"{API_BASE}/query",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    traces = data.get("retrieval_traces", [])

                    st.markdown(answer)

                    # Store traces keyed by the assistant message index
                    assistant_idx = len(st.session_state.messages)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    if traces:
                        st.session_state.traces[assistant_idx] = traces
                        with st.expander("🔍 Retrieval Details"):
                            for t in traces:
                                st.markdown(
                                    f"**Rank {t['rerank_rank']}** &nbsp;|&nbsp; "
                                    f"`{t['chunk_id']}` &nbsp;|&nbsp; "
                                    f"Score: `{t['rerank_score']}` &nbsp;|&nbsp; "
                                    f"📖 {t.get('title') or 'Unknown'} &nbsp;|&nbsp; "
                                    f"Page: {t.get('page') or 'N/A'}"
                                )
                                st.caption(t["doc"][:400])
                                st.divider()

                    # Refresh sidebar to show this session if it's new
                    st.rerun()

                elif response.status_code == 503:
                    msg = ("⚠️ The vectorstore is not ready. "
                           "Run ingestion first:\n```bash\npython main.py --mode ingest\n```")
                    st.warning(msg)

                elif response.status_code == 404:
                    msg = "🔍 No relevant documents found. Try rephrasing your question."
                    st.info(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})

                else:
                    detail = response.json().get("detail", response.text)
                    st.error(f"❌ Error {response.status_code}: {detail}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach the backend. Is `uvicorn api:app` running?")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
