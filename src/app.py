import streamlit as st
from agents.rag_graph import run_rag_pipeline
import asyncio
import sys
import os
from utils.mlflow_logger import log_rag_state_to_mlflow

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(page_title="MedChatGuard", layout="wide")
st.title("🩺 MedChatGuard - LangGraph RAG Assistant")

# Input query from user
user_query = st.text_input("Enter a medical question:", placeholder="e.g., What treatments are prescribed for diabetic patients?", key="query")

if st.button("Run Query") and user_query:
    with st.spinner("Running LangGraph pipeline..."):
        result = run_rag_pipeline(user_query)
        
        log_rag_state_to_mlflow(result)

        response = result.get("response", "No response returned.")
        guardrails = result.get("guardrails", {})
        chunks = result.get("chunks", [])
        prompt = result.get("prompt", "Prompt not available.")

        st.subheader("🧠 Assistant Response")
        st.markdown(f"```\n{response.content}\n```")

        st.subheader("🔍 Retrieved Patient Records")
        for i, chunk in enumerate(chunks):
            with st.expander(f"Record #{i+1} - Score: {chunk.get('score', 0):.4f}"):
                st.markdown(chunk.get("summary", ""))

        st.subheader("🛡️ Guardrail Evaluation")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Hallucination Pass:**", guardrails.get("hallucination_flag", "N/A"))
            st.write("**Message:**", guardrails.get("hallucination_msg", "N/A"))
        with col2:
            st.write("**Speculative Flag:**", guardrails.get("speculation_flag", "N/A"))
            st.write("**Message:**", guardrails.get("speculation_msg", "N/A"))

        st.markdown("---")
        st.text_area("Prompt Used", prompt, height=200)

elif user_query:
    st.info("Click 'Run Query' to process the input.")