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
- ✅ Synthea-powered synthetic EHR data generation
- ✅ Unsloth + QLoRA-based fine-tuning notebook for instruction-style EHR training

---

## 📁 Project Structure
```
src/
├── core/
│   ├── pipeline.py          # Main RAG flow
│   └── retrieval.py         # FAISS-based document retrieval
├── utils/
│   ├── embedding.py         # Build FAISS index
│   ├── evaluation.py        # MLflow logger
│   ├── guardrails.py        # Safety checks
│   ├── run_synthea_convert.py # Automates data generation with Synthea for fine-tuning
│   └── prepare_data.py      # Prepare data + Tokenization + formatting for fine-tuning
├── misc/
│   └── faiss_search.py      # Standalone FAISS test script
├── app.py                   # Streamlit UI
└── notebooks/
    └── unsloth_finetuning_medchatguard.ipynb  # Colab-ready fine-tuning notebook
```

---

## 🛠️ Installation
```bash
git clone https://github.com/danieljames96/medchatguard.git
cd medchatguard
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Set up your `.env` file:
```bash
HUGGINGFACEHUB_API_TOKEN=your_key_here
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

## 🧬 Generate Synthetic EHR Data (Optional)
1. Download Synthea: https://github.com/synthetichealth/synthea
2. Place `synthea-with-dependencies.jar` in the project root
3. Run the converter:
```bash
python -m src.utils.run_synthea_convert
```
This will generate CSVs and produce a JSONL dataset for fine-tuning.

---

## 🧪 Fine-Tune with Unsloth
Use the provided Colab notebook to fine-tune `unsloth/gemma-3-4b-it` or other LLMs using QLoRA:

📓 `notebooks/unsloth_finetuning_medchatguard.ipynb`

- Automatically mounts Google Drive
- Uses your `ehr_clean_text.jsonl` as input
- Saves the trained model to your Drive

You can preprocess your dataset locally:
```bash
python -m src.utils.preprocess_dataset
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
- ✅ `run_synthea_convert.py`: Data generation automation
- ✅ `preprocess_dataset.py`: Training-ready tokenized formatting

---

## ✨ Future Enhancements
- Fine-tuning on domain-specific instructions via QLoRA (✅ In Progress)
- Multi-turn query history in Streamlit
- LangGraph-based agent orchestration for complex flows

---

## 🧑‍💻 Author
**Daniel James**  
LLMOps Engineer | [LinkedIn](https://www.linkedin.com/in/daniel-james-ai)

---

## 📜 License
MIT License
