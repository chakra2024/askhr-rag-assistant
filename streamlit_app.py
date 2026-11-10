"""AskHR: a Streamlit chat page for the handbook RAG assistant.

Run (Ollama must be running and build_index.py must have been run once):
    streamlit run streamlit_app.py

Streamlit re-runs this whole script on every click, so the slow setup
(Pinecone connection, model check) is cached with st.cache_resource.
"""

import time

import streamlit as st

from src.answer_question import answer_question, get_llm
from src.pinecone_store import INDEX_NAME, NAMESPACE, get_vector_store

EXAMPLE_QUESTIONS = [
    "What information is provided about employee benefits?",
    "What guidance is given to new employees?",
    "What does the handbook say about conduct and workplace expectations?",
    "What is the company's approach to remote or distributed work?",
]
DISCLAIMER = (
    "ℹ️ Answers come only from a public snapshot of the "
    "[Osio Labs Employee Handbook](https://github.com/OsioLabs/emphandbook), "
    "not from any private company system. This is a demo, not official HR advice."
)

st.set_page_config(page_title="AskHR", page_icon="💼")


@st.cache_resource(show_spinner="Connecting to Pinecone and the local model...")
def load_rag_components():
    """Connect to Pinecone and Ollama once, and check that the index has data."""
    vector_store = get_vector_store()
    stats = vector_store.index.describe_index_stats()
    if NAMESPACE not in stats.namespaces:
        raise ValueError(f"The index '{INDEX_NAME}' has no data yet. Run: python build_index.py")
    return vector_store, get_llm()


def show_message(message: dict) -> None:
    """Draw one chat message, with its sources and timing if it has them."""
    with st.chat_message(message["role"]):
        # Escape "$" so amounts like "$1000" are not shown as math formulas.
        st.markdown(message["content"].replace("$", "\\$"))
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])} handbook files)"):
                for source in message["sources"]:
                    link = f"[{source['title']}]({source['source_url']})"
                    st.markdown(f"- {link} · `{source['source']}`")
        if message.get("seconds"):
            st.caption(f"Answered in {message['seconds']:.0f} s")


# ---- Sidebar -----------------------------------------------------------------
clicked_example = None
with st.sidebar:
    st.header("About AskHR")
    st.write(
        "AskHR is an HR chat assistant that answers employee questions about "
        "company policies: benefits, time off, onboarding, conduct and more."
    )
    st.write(
        "It is a prototype of a real-world company HR help desk. Every answer "
        "comes with links to the handbook pages it is based on."
    )
    st.subheader("Try an example")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, width="stretch"):
            clicked_example = example
    st.divider()
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []

# ---- Header ------------------------------------------------------------------
st.title("AskHR — Enterprise HR Knowledge Assistant")
st.markdown(
    "#### 👋 Welcome to AskHR!\n"
    "Ask me anything about company policies, such as benefits, leave, onboarding, "
    "workplace conduct or working remotely. I'll give you a short answer and show "
    "the handbook pages it came from."
)

# ---- Setup: show a readable message instead of a stack trace -----------------
try:
    vector_store, llm = load_rag_components()
except Exception as error:
    st.error(f"Setup problem: {error}")
    st.stop()

# ---- Chat --------------------------------------------------------------------
# The history lives only in this browser tab; every question is answered on its own.
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    show_message(message)

question = st.chat_input("Ask a question about company policies...") or clicked_example
st.bottom.caption(DISCLAIMER)  # footer under the question box

if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    start_time = time.perf_counter()
    try:
        with st.spinner("Searching the handbook and writing an answer..."):
            result = answer_question(question, vector_store, llm)
    except Exception as error:
        st.error(f"Could not answer: {error}. Is Ollama running?")
        st.stop()

    # List each handbook file once, even if several of its chunks were retrieved.
    unique_sources = {doc.metadata["source"]: doc.metadata for doc in result["sources"]}
    assistant_message = {
        "role": "assistant",
        "content": result["answer"],
        "sources": list(unique_sources.values()),
        "seconds": time.perf_counter() - start_time,
    }
    st.session_state.messages.append(assistant_message)
    show_message(assistant_message)
