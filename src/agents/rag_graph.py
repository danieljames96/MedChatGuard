# rag_graph.py - LangGraph-powered RAG pipeline for MedChatGuard with enhanced ranking and validation

import os
from typing import TypedDict, List, Dict, Union
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
import mlflow

from core.retrieval import retrieve_relevant_docs
from agents.ranker import llm_rerank_chunks
from agents.validator import validate_response_with_criteria

load_dotenv()

mlflow.set_experiment(experiment_id="0")
mlflow.langchain.autolog()

# Define state type for LangGraph
class RAGState(TypedDict):
    query: str
    chunks: List[Dict[str, Union[str, float]]]
    ranked_chunks: List[Dict[str, Union[str, float]]]
    prompt: str
    response: Union[str, AIMessage]
    guardrails: Dict[str, Union[str, bool]]
    validation: Dict[str, Union[str, bool]]


# Prompt generation

def build_prompt_node():
    template = """
    You are a clinical assistant. Based on the following patient records and the user query, provide a concise and medically relevant answer.

    Patient Record:
    {context}

    User Query:
    {query}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    def combine(inputs: RAGState):
        summaries = "\n\n".join([chunk['summary'] for chunk in inputs['ranked_chunks']])
        inputs["prompt"] = prompt.format(context=summaries, query=inputs['query'])
        return inputs

    return RunnableLambda(combine)


# Enhanced guardrail logic

def guardrail_check(text: str) -> Dict[str, Union[str, bool]]:
    speculative_terms = ["might", "could", "possibly", "likely", "may"]
    speculation_flag = any(term in text.lower() for term in speculative_terms)
    hallucination_flag = any(term in text.lower() for term in ["clearly not true", "doesn't match", "not in patient record"])

    return {
        "speculation_flag": speculation_flag,
        "speculation_msg": "Speculative language found." if speculation_flag else "Clear of speculation.",
        "hallucination_flag": hallucination_flag,
        "hallucination_msg": "Possible inconsistency detected." if hallucination_flag else "Consistent with context.",
        "safe_to_use": not speculation_flag and not hallucination_flag
    }


# LangChain wrapper for Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


# Graph Definition

def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retriever", RunnableLambda(lambda state: {
        **state,
        "chunks": retrieve_relevant_docs(state["query"])
    }))

    graph.add_node("ranker", RunnableLambda(lambda state: {
        **state,
        "ranked_chunks": llm_rerank_chunks(state["query"], state["chunks"])
    }))

    graph.add_node("prompt_builder", build_prompt_node())

    graph.add_node("llm", RunnableLambda(lambda state: {
        **state,
        "response": llm.invoke(state["prompt"])
    }))

    graph.add_node("guardrails_check", RunnableLambda(lambda state: {
        **state,
        "guardrails": guardrail_check(
            state["response"].content if hasattr(state["response"], "content") else state["response"]
        )
    }))

    graph.add_node("validator_check", RunnableLambda(lambda state: {
        **state,
        "validation": validate_response_with_criteria(state["response"], state["query"])
    }))

    # Flow
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "ranker")
    graph.add_edge("ranker", "prompt_builder")
    graph.add_edge("prompt_builder", "llm")
    graph.add_edge("llm", "guardrails_check")
    graph.add_edge("guardrails_check", "validator_check")
    graph.add_edge("validator_check", END)

    return graph.compile()


rag_graph = build_rag_graph()

def run_rag_pipeline(user_query: str) -> RAGState:
    with mlflow.start_run():
        return rag_graph.invoke({"query": user_query})