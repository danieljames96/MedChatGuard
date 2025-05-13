# utils/validator.py

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

def validate_response_with_criteria(response, query):
    """
    Uses the LLM to validate if the response is medically consistent with known information.
    """

    response_text = response.content if hasattr(response, "content") else response

    prompt = f"""
You are a medical reviewer with access to general medical knowledge.

Given the user query:
"{query}"

And the assistant's response:
"{response_text}"

Assess if the response is:
1. Medically plausible
2. Not speculative or misleading
3. Grounded in real medical context

Return a short summary and mark it as Approved or Rejected.
"""

    message = HumanMessage(content=prompt)
    result = llm.invoke([message])

    final_text = result.content if hasattr(result, "content") else str(result)

    approved = "approved" in final_text.lower()
    return {
        "approved": approved,
        "message": final_text.strip()
    }
