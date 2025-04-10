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

## 📊 Results

Throughout the development of **MedChatGuard**, multiple open-source models were evaluated for their performance in answering clinical queries over structured patient records. Below is a summary of these experiments:

---

### 🔬 Gemma 3B / 4B (via Unsloth + QLoRA)

- **Fine-tuning**: Performed using Unsloth with QLoRA on instruction-style data.
- **Results**: Delivered highly relevant and coherent answers grounded in patient context.
- **Limitations**: Extremely resource-intensive for inference; unsuitable for CPU-only systems. Loading and serving even quantized versions proved impractical on a 16GB RAM machine with no GPU.
- **Conclusion**: ✅ Strong accuracy, ❌ poor deployability on local hardware.

---

### 🧪 RoBERTa (deepset/roberta-base-squad2)

- **Fine-tuning**: On SQuAD-style synthetic EHR QA dataset.
- **Results**: Struggled with long contexts and structured patient narratives. Often returned incomplete or irrelevant spans.
- **Limitations**: 512-token context window, span-extraction limits, poor generalization to multi-patient prompts.
- **Conclusion**: ❌ Not suitable for clinical RAG-style QA.

---

### 🧠 FLAN-T5-Small (Instruction-Tuned)

- **Fine-tuning**: Efficiently fine-tuned on CPU using instruction + response format.
- **Results**: Performs well with low memory overhead (~80M parameters), fast inference, and reliable answers for single-patient queries.
- **Advantages**:
  - Lightweight and CPU-friendly
  - General-purpose instruction following
  - Acceptable performance on synthetic medical prompts
- **Conclusion**: ✅ Best trade-off for accuracy + efficiency in CPU-only environments.

---

## 🧠 Recommendation

For systems without GPUs, **`google/flan-t5-small`** is the most practical model for deployment. It balances performance, responsiveness, and model size — making it ideal for offline or low-resource environments like local clinical tools or prototypes.

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

## 🧬 Generate Synthetic EHR Data
1. Download Synthea: https://github.com/synthetichealth/synthea
2. Place `synthea-with-dependencies.jar` in the project root
3. Run the converter for RAG data:
```bash
python -m src.utils.run_synthea_convert rag
```
4. Run the converter for Fine-Tuning data:
```bash
python -m src.utils.run_synthea_convert finetune
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
- ✅ `run_synthea_convert.py`: Data generation automation
- ✅ `preprocess_dataset.py`: Training-ready tokenized formatting

---

## ✨ Future Enhancements
- Multi-turn query history in Streamlit
- LangGraph-based agent orchestration for complex flows

---

## 🧑‍💻 Author
**Daniel James**  
AI Engineer | [LinkedIn](https://www.linkedin.com/in/daniel-james-ai)

---

## 📜 License
MIT License
