from core.retrieval import retrieve_relevant_docs
from utils.guardrails import apply_guardrails
from utils.evaluation import log_evaluation
from transformers import pipeline
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import os

# Load environment variables
load_dotenv()

# model_path = os.getenv("LLM_MODEL_PATH", "./models/finetuned_model/roberta-base-squad2")
model_path = "./models/finetuned_model/roberta-base-squad2/"

print(f"Loading model from {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_path,
    torch_dtype="float32"
)

qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer
)

# Prompt template
PROMPT_TEMPLATE = """
You are a clinical assistant. Based on the following patient records and the user query, provide a concise and medically relevant answer.

User Query:
{query}

Answer:
"""

def build_prompt(chunks, query):
    context = "\n\n".join([chunk['summary'] for chunk in chunks])
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    return prompt

def run_pipeline(query):
    # Step 1: Retrieve relevant patient records
    chunks = retrieve_relevant_docs(query)

    # Step 2: Build the prompt
    prompt = build_prompt(chunks, query)

    # Step 3: Query the QA Model
    response_raw  = qa_pipeline(
        question=query,
        context=prompt  # The RAG context will serve as the document
    )
    response = response_raw ["answer"]


    # Step 4: Apply guardrails
    guardrail_result = apply_guardrails(response)

    # Step 5: Log the run to MLflow
    log_evaluation(query, prompt, response, chunks)

    return response, prompt, chunks, guardrail_result

# Standalone test
if __name__ == "__main__":
    user_query = input("Enter a medical question: ")
    answer, prompt_used, retrieved, guardrails = run_pipeline(user_query)

    print("\n=== Generated Answer ===")
    print(answer)

    print("\n=== Guardrail Evaluation ===")
    for k, v in guardrails.items():
        print(f"{k}: {v}")

    print("\n=== Retrieved Chunks ===")
    for r in retrieved:
        print(f"\nRank #{r['rank']}\n{r['summary']}\nScore: {r['score']:.4f}")
