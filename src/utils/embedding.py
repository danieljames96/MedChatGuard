import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Configurations
DATA_DIR = "./data/rag_docs/"
OUTPUT_DIR = "./embeddings/"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_FILE = os.path.join(OUTPUT_DIR, "ehr_faiss_index.faiss")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "doc_id_mapping.pkl")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summaries.pkl")

# Load CSVs from Synthea
def load_synthea_data():
    patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
    conditions = pd.read_csv(os.path.join(DATA_DIR, "conditions.csv"))
    medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
    encounters = pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"))
    return patients, conditions, medications, encounters

# Create simplified patient summaries for embedding
def build_patient_summaries(patients, conditions, medications, encounters):
    summaries = []
    doc_ids = []

    for _, row in patients.iterrows():
        pid = row["Id"]
        name = f"{row['FIRST'] or 'Patient'} {row['LAST'] or ''}".strip()
        age = 2025 - int(row["BIRTHDATE"][:4])  # Rough age
        gender = row["GENDER"].capitalize()

        # Conditions
        cond = conditions[conditions["PATIENT"] == pid]["DESCRIPTION"].dropna().unique()
        cond_str = ", ".join(cond[:5]) or "No conditions reported"

        # Medications
        meds = medications[medications["PATIENT"] == pid]["DESCRIPTION"].dropna().unique()
        med_str = ", ".join(meds[:5]) or "No medications"

        # Encounters
        enc = encounters[encounters["PATIENT"] == pid]["DESCRIPTION"].dropna().unique()
        enc_str = ", ".join(enc[:5]) or "No recent visits"

        summary = (
            f"Patient: {name}, Age: {age}, Gender: {gender}.\n"
            f"Conditions: {cond_str}.\n"
            f"Medications: {med_str}.\n"
            f"Encounters: {enc_str}."
        )

        summaries.append(summary)
        doc_ids.append(pid)

    return summaries, doc_ids

# Generate embeddings and save FAISS index
def embed_and_save(summaries, doc_ids):
    print("[INFO] Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("[INFO] Encoding summaries...")
    embeddings = model.encode(summaries, show_progress_bar=True)

    print("[INFO] Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)

    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(doc_ids, f)

    print(f"[DONE] FAISS index saved to {FAISS_INDEX_FILE}")
    print(f"[DONE] Mapping saved to {MAPPING_FILE}")
    
    with open(SUMMARY_FILE, "wb") as f:
        pickle.dump(summaries, f)

if __name__ == "__main__":
    print("[START] Generating EHR embeddings...")
    patients, conditions, medications, encounters = load_synthea_data()
    summaries, doc_ids = build_patient_summaries(patients, conditions, medications, encounters)
    embed_and_save(summaries, doc_ids)