import os
import mlflow
from datetime import datetime

# Configure MLflow
mlflow.set_tracking_uri("file:../mlruns")  # Local tracking URI
mlflow.set_experiment("MedChatGuard-RAG-LLM")

def log_evaluation(query, prompt, response, retrieved_chunks, model_name="flan-t5-base"):
    with mlflow.start_run(run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("query", query)
        mlflow.log_param("num_chunks", len(retrieved_chunks))

        # Log prompt and response artifacts
        with open("./log/prompt.txt", "w") as f:
            f.write(prompt)
        mlflow.log_artifact("prompt.txt")

        with open("./log/response.txt", "w") as f:
            f.write(response)
        mlflow.log_artifact("response.txt")

        # Log sample retrieved text for traceability
        with open("./log/retrieved_chunks.txt", "w") as f:
            for chunk in retrieved_chunks:
                f.write(f"{chunk['rank']}: {chunk['summary']}\n")
        mlflow.log_artifact("retrieved_chunks.txt")

        print("[MLflow] Evaluation logged successfully.")

# Test example
if __name__ == "__main__":
    dummy_chunks = [
        {"rank": 1, "summary": "Patient with hypertension and diabetes."},
        {"rank": 2, "summary": "Recently prescribed metformin."},
    ]
    log_evaluation(
        query="What is the patient being treated for?",
        prompt="Prompt example text...",
        response="The patient is being treated for hypertension and diabetes.",
        retrieved_chunks=dummy_chunks
    )
