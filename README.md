# 🏥 MedChatGuard

A secure, clinically-aware, multi-turn RAG assistant powered by **LangGraph** and **Groq**, designed to answer medical questions over synthetic EHR data. It integrates retrieval, LLM inference, memory, guardrails, external validation, and experiment tracking with MLflow.

---

## 🚀 Features (Updated)

* ✅ LangGraph-based RAG pipeline with memory and agent orchestration
* ✅ FAISS-based retrieval using Sentence Transformers
* ✅ Multi-turn conversation support (contextual prompts)
* ✅ Groq-hosted LLM (LLaMA-3 / Mixtral) via LangChain
* ✅ Guardrails for speculative/hallucinatory checks
* ✅ Medical plausibility validation agent
* ✅ LLM-based reranker for retrieved context
* ✅ MLflow logging with full JSON state tracking
* ✅ Streamlit interface for real-time interaction
* ✅ Compatible with synthetic EHR data from Synthea

---

## 🧠 Fine-Tuned Model History

### 🔬 Gemma 3B / 4B (via Unsloth + QLoRA)

* Strong accuracy on clinical instructions
* Poor deployability in CPU environments

### 🧪 RoBERTa (deepset/roberta-base-squad2)

* Failed on long-context and structured EHR prompts

### 🧠 FLAN-T5-Small (Instruction-Tuned)

* Best trade-off for speed and interpretability in lightweight settings

📌 *These experiments led to the adoption of Groq-hosted LLMs for scalable inference with minimal latency.*

---

## 📁 Project Structure

```
src/
├── agents/
│   └── rag_graph.py           # LangGraph pipeline
├── core/
│   └── retrieval.py           # FAISS-based retrieval
├── utils/
│   ├── guardrails.py          # Speculation and hallucination checks
│   ├── ranker.py              # LLM-based reranker
│   ├── validator.py           # Medical plausibility agent
│   └── mlflow_logger.py       # Full-state logging to MLflow
├── app.py                     # Streamlit UI
└── data/rag_docs/             # EHR document chunks
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
export GROQ_API_KEY=your_key
export MLFLOW_TRACKING_URI=http://localhost:5000

# Start MLflow server
./start_mlflow.sh

# Run Streamlit app
streamlit run src/app.py
```

---

## 💬 Streamlit Usage

* Enter clinical questions like:

  * "Summarize recent diagnoses"
  * "What medications are prescribed for hypertension?"
* Conversation memory tracks each Q\&A
* Expand history to view all prior turns

---

## 🧪 Synthetic EHR Generation

1. Download Synthea: [https://github.com/synthetichealth/synthea](https://github.com/synthetichealth/synthea)
2. Run converter to generate `.txt`/`.md` summaries:

```bash
python -m src.utils.run_synthea_convert rag
```

3. Place files in `data/rag_docs/`

---

## 📈 MLflow Logging

* Tracks query, response, guardrails, validation, retrieved docs
* Logs full RAG state as a JSON artifact

Launch MLflow dashboard:

```bash
mlflow ui
```

---

## 🧑‍⚕️ Agents in LangGraph

1. **Retriever**: FAISS retrieval based on dense embedding
2. **Ranker**: Reorders chunks by LLM judgment
3. **Prompt Builder**: Assembles full context prompt
4. **LLM**: Groq-powered response generation
5. **Guardrails**: Flags speculative/hallucinated outputs
6. **Validator**: Validates response against known medical norms
7. **Memory Tracker**: Appends to session history

---

## 🧩 Future Directions

* Retrieval augmentation from real-world guidelines
* Integration with structured EHR (FHIR)
* Streaming token-by-token responses
* Role-based interaction (doctor, nurse, admin)

---

## 👨‍💻 Author

**Daniel James**
AI Engineer | [LinkedIn](https://www.linkedin.com/in/daniel-james-ai)

---

## 📜 License

MIT License