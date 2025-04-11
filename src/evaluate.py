import mlflow
import pandas as pd
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load your model
model_path = "./models/finetuned_model/flan-t5-small/"  # or your fine-tuned path
tokenizer = AutoTokenizer.from_pretrained(model_path, token=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, token=False)

# Wrap it into a pipeline
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Sample QA dataset
eval_data = pd.DataFrame({
    "inputs": [
        "What medications has the patient been prescribed?\n\nPatient: John, Age: 60, Gender: M.\nConditions: Diabetes.\nEncounters: Check-up.\nMedications: Metformin, Lisinopril",
        "What medications has the patient been prescribed?\n\nPatient: Sarah, Age: 70, Gender: F.\nConditions: Hypertension.\nEncounters: Emergency room.\nMedications: Amlodipine, Aspirin"
    ],
    "ground_truth": [
        "Metformin, Lisinopril",
        "Amlodipine, Aspirin"
    ]
})

# Start an MLflow run
with mlflow.start_run() as run:
    
    inputs = "What medications has the patient been prescribed?\n\nPatient: ... "
    preds = qa_pipeline(inputs)
    signature = infer_signature([inputs], [preds[0]["generated_text"]])
    
    # Log your model
    logged_model_info = mlflow.transformers.log_model(
        transformers_model=qa_pipeline,
        artifact_path="model",
        input_example={"inputs": eval_data["inputs"].iloc[0]},
        task="text2text-generation",
        metadata={"source": "local", "note": "custom finetuned model", "license": "apache-2.0"}
    )

    # Evaluate the logged model
    results = mlflow.evaluate(
        model=logged_model_info.model_uri,
        data=eval_data,
        targets="ground_truth",
        model_type="question-answering",
    )

    print("🔍 Evaluation metrics:\n", results.metrics)
    print("📋 Evaluation table:\n", results.tables["eval_results_table"])