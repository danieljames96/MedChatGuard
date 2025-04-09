# 🏥 MedChatGuard

A secure, privacy-conscious RAG-powered assistant designed to summarize and respond to patient health queries using synthetic EHR (Electronic Health Record) data. The system showcases practical LLMOps practices, combining prompt engineering, retrieval-augmented generation (RAG), evaluation, guardrails, orchestration using Airflow and MLflow, agentic AI with LangGraph, and optional fine-tuning using QLoRA—all built using free-tier tools.

---

## 🚀 Project Goals
- Create a chatbot assistant that simulates medical Q&A using clinical guidelines and EHRs.
- Implement RAG architecture with open-source embeddings and vector databases.
- Integrate observability, evaluation, orchestration, and agent-based reasoning mechanisms.
- Demonstrate core LLMOps principles: pipeline automation, experiment tracking, model fine-tuning, safety.

---

## 🧱 Architecture Overview

```
User Query ──▶ LangGraph Agent Flow ──▶ RAG (LangChain + FAISS) ──▶ Prompt Augmentation
                   │
                   └──▶ Agent Nodes: Data Retrieval, Clinical Summarizer, Safety Validator
                                   │
                            Open LLM API / Fine-Tuned LLM (QLoRA/LoRA)
                                   │
         ⬇ Guardrails + Evaluation (TruLens / LangSmith / Guardrails.ai)
                                   ↓
            Logging + Feedback + MLflow + Airflow Scheduling
```

---

## 📦 Features
- ✅ Retrieval-Augmented Generation using FAISS + LangChain
- ✅ Domain-specific prompt engineering templates
- ✅ Agentic AI using LangGraph to manage multiple reasoning and validation steps
- ✅ Embeddings using SentenceTransformers (`all-MiniLM-L6-v2`)
- ✅ Guardrails for safety and format enforcement (via Guardrails.ai)
- ✅ Evaluation & observability (LangSmith or TruLens)
- ✅ MLflow experiment tracking (prompt templates, embedding settings, response quality)
- ✅ Apache Airflow DAGs for EHR preprocessing and vector store maintenance
- ✅ Optional fine-tuning pipeline using LoRA/QLoRA for LLM personalization
- ✅ Streamlit/Gradio web UI

---

## 🗂️ Directory Structure
```bash
medchatguard/
├── airflow/
│   └── dags/
│       └── ehr_pipeline_dag.py       # Airflow DAGs for automation
├── data/                             # Synthetic EHR and guidelines
├── embeddings/                       # FAISS vector index and model cache
├── mlruns/                           # MLflow experiment logs
├── finetune/                         # Scripts for fine-tuning LLMs using QLoRA
├── prompts/                          # Prompt templates for various use cases
├── agents/                           # LangGraph agent definitions and flows
├── src/
│   ├── pipeline.py                   # RAG pipeline implementation
│   ├── guardrails.py                 # Response validation & filtering
│   ├── evaluation.py                 # Metrics and response analysis
│   ├── embedding.py                  # Embedding generation script
│   ├── faiss_builder.py              # FAISS index builder
│   ├── app.py                        # Streamlit/Gradio frontend
├── notebooks/                        # EDA and prototype testing
├── requirements.txt
└── README.md
```

---

## 🧠 Tech Stack
| Purpose               | Tools Used                           |
|-----------------------|---------------------------------------|
| Embeddings            | `sentence-transformers`               |
| Vector DB             | FAISS (local)                         |
| LLM Inference         | HuggingFace Inference API / OpenAI    |
| Fine-tuning           | QLoRA / LoRA via HuggingFace + PEFT   |
| Agentic AI            | LangGraph                             |
| RAG + Flow            | LangChain                             |
| Prompt Guarding       | Guardrails.ai                         |
| Evaluation            | TruLens / LangSmith                   |
| Tracking              | MLflow                                |
| Orchestration         | Apache Airflow                        |
| UI                    | Streamlit or Gradio                   |

---

## 🧪 Sample Use Cases
1. **Query:** "What medications has the patient been prescribed for diabetes?"
2. **Response:** Generated through LangGraph agent: Retrieval → Summarizer → Guardrail Validator.
3. **Tracking:** MLflow logs model, embedding, and response metrics.
4. **Automation:** Airflow refreshes embedding pipeline daily.
5. **Fine-Tuning:** Domain-specific instructions tuned via LoRA on small open LLMs.

---

## 📋 Installation & Setup
```bash
git clone https://github.com/yourusername/medchatguard.git
cd medchatguard
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To start Airflow (in standalone mode):
```bash
airflow db init
airflow users create --username admin --password admin --role Admin --email admin@example.com
airflow standalone
```

To run MLflow locally:
```bash
mlflow ui
```

---

## 🔐 API Keys Setup
- `.env` file:
```bash
HUGGINGFACE_API_KEY=your_key_here
OPENAI_API_KEY=optional_if_used
LANGCHAIN_API_KEY=optional
```

---

## 📈 Evaluation & Observability
- Use TruLens or LangSmith for:
  - Prompt latency
  - Hallucination rate
  - Answer accuracy vs source
  - User feedback logging
- Use MLflow for:
  - Experiment parameter tracking
  - Prompt style vs. accuracy comparison
  - Embedding/fine-tuned model comparisons

---

## 🛡️ Guardrails Examples
- Rejects speculative medical advice.
- Flags inconsistent diagnosis conclusions.
- Enforces structured output (e.g., bullet points, citations).

---

## 📚 Datasets Used
- [Synthea Synthetic EHR Data](https://synthetichealth.github.io/synthea/)
- Public clinical guidelines (e.g., CDC, WHO)

---

## ✍️ Future Improvements
- Feedback loop integration (thumbs up/down → retraining or prompt tuning)
- Full fine-tuning pipeline with QLoRA for clinical tone optimization
- Airflow + MLflow integration to auto-log after pipeline completion
- Streamlit session history, user feedback UI
- Expand LangGraph agents to include document classification, entity detection

---

## 🧑‍💻 Author
**Daniel James**  
ML Engineer | LLMOps Enthusiast | [LinkedIn](https://www.linkedin.com/in/daniel-james-ai)  