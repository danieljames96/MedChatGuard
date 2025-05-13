# app.py – Streamlit UI for MedChatGuard with multi-turn conversation

import streamlit as st
from agents.rag_graph import run_rag_pipeline, RAGState
import asyncio
import sys
import os
from utils.mlflow_logger import log_rag_state_to_mlflow

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(page_title="MedChatGuard", layout="wide")
st.title("🩺 MedChatGuard - LangGraph RAG Assistant (Multi-turn)")

# Initialize conversation state
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = None

if "history_display" not in st.session_state:
    st.session_state.history_display = []

# User query input
user_query = st.text_input("Enter your question:", key="query")
if st.button("Send") and user_query:
    with st.spinner("Processing..."):
        result = run_rag_pipeline(user_query, st.session_state.conversation_state)
        st.session_state.conversation_state = result

        # Display response
        st.subheader("🧠 Assistant Response")
        response_text = result["response"].content if hasattr(result["response"], "content") else result["response"]
        st.markdown(f'''
{response_text}
''')

        # Display retrieved records
        st.subheader("📋 Retrieved Records")
        for i, chunk in enumerate(result.get("ranked_chunks", [])):
            with st.expander(f"Record #{i+1} - Score: {chunk.get('score', 0):.4f}"):
                st.markdown(chunk.get("summary", ""))

        # Guardrails
        st.subheader("🛡️ Guardrails")
        for key, val in result.get("guardrails", {}).items():
            st.write(f"{key.replace('_', ' ').title()}: {val}")

        # Validation
        st.subheader("📚 Medical Validation")
        for key, val in result.get("validation", {}).items():
            st.write(f"{key.replace('_', ' ').title()}: {val}")

        # History
        st.subheader("🗂️ Conversation History")
        st.session_state.history_display.append((user_query, response_text))
        for i, (q, a) in enumerate(st.session_state.history_display):
            with st.expander(f"Turn {i+1}"):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")

        # Log full state to MLflow
        log_rag_state_to_mlflow(result)

elif user_query:
    st.info("Click 'Send' to run the query.")