import streamlit as st
from src.core.pipeline import run_pipeline

st.set_page_config(page_title="MedChatGuard", layout="wide")
st.title("🩺 MedChatGuard - Clinical Query Assistant")

# Input query from user
user_query = st.text_input("Enter a medical question:", placeholder="e.g., What treatments are prescribed for diabetic patients?", key="query")

if st.button("Run Query") and user_query:
    with st.spinner("Processing your query..."):
        response, prompt, chunks, guardrails = run_pipeline(user_query)

        st.subheader("🧠 Assistant Response")
        st.markdown(f"```\n{response}\n```")

        st.subheader("🔍 Retrieved Patient Records")
        for i, chunk in enumerate(chunks):
            with st.expander(f"Record #{i+1} - Score: {chunk['score']:.4f}"):
                st.markdown(chunk['summary'])

        st.subheader("🛡️ Guardrail Evaluation")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Format Pass:**", guardrails['format_pass'])
            st.write("**Message:**", guardrails['format_msg'])
        with col2:
            st.write("**Speculative Flag:**", guardrails['speculation_flag'])
            st.write("**Message:**", guardrails['speculation_msg'])

        st.markdown("---")
        st.text_area("Prompt Used", prompt, height=200)

elif user_query:
    st.info("Click 'Run Query' to process the input.")