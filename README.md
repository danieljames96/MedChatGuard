# 🏥 MedChatGuard

A secure, agentic, RAG-powered assistant designed to answer clinical queries using synthetic EHR data. It combines LLMOps best practices like retrieval-augmented generation, guardrails, evaluation, experiment tracking, and agentic flows—built entirely with free-tier tools and open-source models.

---

## 🚀 Features
- ✅ RAG pipeline using FAISS + Sentence Transformers
- ✅ LangChain + Hugging Face LLMs (Flan-T5 by default)
- ✅ Guardrails to detect speculative answers and enforce structure
- ✅ MLflow tracking (prompt, response, metrics)
- ✅ Modular components with clear separation of concerns
- ✅ Streamlit-based UI for query interaction

---

## 📁 Project Structure
```
src/
├── core/
│   ├── pipeline.py          # Main RAG flow
│   ├── retrieval.py         # FAISS-based document retrieval
├── utils/
│   ├── embedding.py         # Build FAISS index
│   ├── evaluation.py        # MLflow logger
│   ├── guardrails.py        # Safety checks
├── misc/
│   ├── faiss_search.py      # Standalone FAISS test script
├── app.py                   # Streamlit UI
```

---

## 🛠️ Installation
```bash
git clone https://github.com/yourusername/medchatguard.git
cd medchatguard
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set up your `.env` file:
```bash
HUGGINGFACE_API_KEY=your_key_here
```

---

## ▶️ Run the App
```bash
streamlit run src/app.py
```

To view MLflow logs:
```bash
mlflow ui
```

---

## 🧪 Sample Test Cases
| Query | Expected Behavior |
|-------|--------------------|
| "What is the patient taking for hypertension?" | Retrieves records mentioning hypertension and outputs medications like "amlodipine" or "lisinopril" |
| "Does the patient have diabetes?" | Looks up conditions list for terms like "Type 2 Diabetes" |
| "What recent encounters has the patient had?" | Returns summary of visits like "outpatient consultation" or "lab test" |
| "Could the patient be experiencing side effects?" | Guardrails should flag speculative phrasing |
| "Summarize the patient history." | Provides a compact overview of diagnosis, meds, and visits |

---

## 📈 LLMOps Integration
- ✅ `MLflow`: Logs prompt, response, chunks, and metadata
- ✅ `Guardrails.py`: Enforces formatting and blocks speculative claims
- ✅ `Evaluation.py`: Exports logs for traceability

---

## ✨ Future Enhancements
- Fine-tuning on domain-specific instructions via QLoRA
- Multi-turn query history in Streamlit
- LangGraph-based agent orchestration for complex flows

---

## 🧑‍💻 Author
**Daniel James**  
LLMOps Engineer | [LinkedIn](https://www.linkedin.com/in/daniel-james-ai)

---

## 📜 License
MIT License