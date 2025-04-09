import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
OUTPUT_DIR = "./embeddings/"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_FILE = os.path.join(OUTPUT_DIR, "ehr_faiss_index.faiss")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "doc_id_mapping.pkl")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summaries.pkl")

# Load FAISS index and metadata
def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(MAPPING_FILE, "rb") as f:
        doc_ids = pickle.load(f)
    with open(SUMMARY_FILE, "rb") as f:
        summaries = pickle.load(f)
    return index, doc_ids, summaries

# Retrieve top-k similar chunks for a given query
def retrieve_relevant_docs(query, k=5):
    print("[INFO] Loading model and index...")
    model = SentenceTransformer(MODEL_NAME)
    index, doc_ids, summaries = load_faiss_index()

    print("[INFO] Encoding query...")
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            "rank": i + 1,
            "patient_id": doc_ids[idx],
            "summary": summaries[idx],
            "score": float(distances[0][i])
        }
        results.append(result)

    return results

# Test in isolation
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    results = retrieve_relevant_docs(user_query)
    for res in results:
        print(f"\nRank #{res['rank']}\nPatient ID: {res['patient_id']}\nSummary: {res['summary']}\nScore: {res['score']:.4f}")