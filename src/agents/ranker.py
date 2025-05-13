# utils/ranker.py

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

def llm_rerank_chunks(query, chunks, top_k=5):
    """
    Uses LLM to rank context chunks based on relevance to the query.
    Returns the top-k most relevant chunks.
    """
    prompt = "Given the clinical query:\n\"{}\"\n\nRank the following summaries by their relevance:\n\n{}\n\nReturn the top relevant summaries.".format(
        query,
        "\n\n".join([f"[{i+1}] {chunk['summary']}" for i, chunk in enumerate(chunks)])
    )

    message = HumanMessage(content=prompt)
    response = llm.invoke([message])

    if isinstance(response, str):
        response_text = response
    else:
        response_text = response.content

    top_indices = []
    for line in response_text.splitlines():
        if "[" in line and "]" in line:
            try:
                idx = int(line[line.index("[")+1:line.index("]")]) - 1
                if 0 <= idx < len(chunks):
                    top_indices.append(idx)
            except:
                continue

    top_chunks = [chunks[i] for i in top_indices[:top_k]] if top_indices else chunks[:top_k]
    return top_chunks
