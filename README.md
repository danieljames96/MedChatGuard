# 🏥 MedChatGuard

A secure, clinically-aware RAG assistant powered by **LangGraph** and **Groq**, designed to answer medical questions over synthetic EHR data. It integrates retrieval, LLM inference, guardrails, and evaluation—all built using lightweight, open-source tools.

---

## 🚀 Features (Updated)

- ✅ LangGraph-powered RAG flow with stateful, modular graph nodes
- ✅ FAISS-based document retrieval using Sentence Transformers
- ✅ Structured prompting + Groq-backed LLM via `ChatGroq`
- ✅ Safety guardrails to catch speculative or unstructured responses
- ✅ MLflow integration for run logging and evaluation
- ✅ Streamlit interface for real-time clinical query interaction
- ✅ Synthetic EHR data generation using Synthea
- ✅ Fully CPU-compatible setup for local testing

---

## 📁 Project Structure

```

src/
├── agents/
│   ├── rag_graph.py         # LangGraph pipeline
├── core/
│   ├── retrieval.py         # FAISS-based retrieval logic
├── utils/
│   ├── guardrails.py        # Format/speculation checks
│   ├── evaluation.py        # MLflow logging
├── app.py                   # Streamlit frontend
└── data/                    # Files used in RAG pipeline

````

---

## ⚙️ Installation

```bash
git clone https://github.com/danieljames96/medchatguard.git
cd medchatguard
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
````

Set up your environment:

```bash
# .env
GROQ_API_KEY=your_groq_key_here
```

---

## ▶️ Run the App

```bash
streamlit run src/app.py
```

This launches the MedChatGuard UI. Enter clinical questions to receive grounded answers from synthetic records.

---

## 🧬 Generate Synthetic EHR Data

1. Download Synthea: [https://github.com/synthetichealth/synthea](https://github.com/synthetichealth/synthea)
2. Place `synthea-with-dependencies.jar` in the project root
3. Generate records:

```bash
python -m src.utils.run_synthea_convert rag
```

4. Place processed `.txt` or `.md` records in `data/rag_docs/`

---

## 🧠 How It Works

The RAG pipeline is defined as a LangGraph flow:

1. **Retriever**: Queries FAISS for top-k matching patient records
2. **Prompt Builder**: Constructs a structured instruction using summaries
3. **LLM Node**: Calls Groq (Mixtral) via LangChain
4. **Guardrails**: Checks speculative or unstructured language
5. **Return**: Outputs the result + metadata

---

## 🧪 Logging with MLflow

Each run logs:

* Prompt + query
* Response
* Guardrail results
* Retrieved document scores and summaries

To launch MLflow UI:

```bash
mlflow ui
```

---

## 📈 Sample Clinical Queries

| Query                                       | Expected Output                        |
| ------------------------------------------- | -------------------------------------- |
| What meds is the patient taking?            | List of drugs like metformin, etc.     |
| Has the patient been diagnosed with asthma? | Checks condition summary               |
| Summarize recent hospital visits            | Outputs compact overview of encounters |
| Are there any potential side effects?       | Guardrails flag speculative answers    |

---

## 📌 Future Enhancements

* Reranking via LLM-based relevance scoring
* Human feedback-based fine-tuning
* Multi-turn memory via LangGraph state management

---

## 👨‍💻 Author

**Daniel James**
AI Engineer | [LinkedIn](https://www.linkedin.com/in/daniel-james-ai)

---

## 📜 License

MIT License