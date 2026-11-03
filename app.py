import streamlit as st
from pathlib import Path
from src.config import get_settings, ROOT
from src.logging_setup import setup_logging
from src.rag import RAGEngine
from src.ingestion import build_index
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="SupportPearlz | AI Support", page_icon="💧", layout="wide")
st.title("💧 SupportPearlz")
st.caption("Retrieval-Augmented Customer Support Agent · Python + LangChain + Streamlit")
st.info("This is a fictional demonstration knowledge base. Answers are grounded in the local documents and may refuse when evidence is missing.")

with st.sidebar:
    st.header("Knowledge base")
    st.write("Source folder: `data/knowledge_base/`")
    st.write("Vector index: `data/vector_store/`")
    if st.button("Rebuild knowledge index", use_container_width=True):
        try:
            settings = get_settings()
            logger = setup_logging(settings.log_level)
            build_index(settings, logger, force=True)
            st.success("Index rebuilt. Refresh the page to start a new session.")
            st.session_state.pop("engine", None)
        except Exception as e:
            st.error(str(e))
    if st.button("Reset conversation", use_container_width=True):
        if "engine" in st.session_state:
            st.session_state.engine.reset()
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Keep your API key in `.env`. Never commit it to Git.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine" not in st.session_state:
    try:
        settings = get_settings()
        logger = setup_logging(settings.log_level)
        with st.spinner("Loading local embeddings and knowledge index…"):
            st.session_state.engine = RAGEngine(settings, logger)
    except Exception as e:
        st.error(f"Setup error: {e}")
        st.markdown("**Setup steps**")
        st.code("copy .env.example .env  # Windows PowerShell: Copy-Item .env.example .env")
        st.write("Add your Gemini API key to `.env`, install requirements, then rerun the app.")
        st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("meta"):
            meta = message["meta"]
            st.caption(f"Confidence: {meta['confidence']} · Latency: {meta['latency_seconds']}s")
            if meta.get("sources"):
                with st.expander("Sources used"):
                    for source in meta["sources"]:
                        st.write(f"• {source}")
            else:
                st.caption("No supporting source was found.")

question = st.chat_input("Ask about products, warranty, returns, delivery, or maintenance…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base…"):
            result = st.session_state.engine.ask(question)
        st.markdown(result["answer"])
        st.caption(f"Confidence: {result['confidence']} · Latency: {result['latency_seconds']}s")
        if result.get("sources"):
            with st.expander("Sources used"):
                for source in result["sources"]:
                    st.write(f"• {source}")
        else:
            st.caption("No supporting source was found.")
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "meta": result})
