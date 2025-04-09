import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Configurations
OUTPUT_DIR = "./embeddings/"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_FILE = os.path.join(OUTPUT_DIR, "ehr_faiss_index.faiss")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "doc_id_mapping.pkl")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summaries.pkl")  # Optional, if storing summaries

# Load necessary components
def load_index_and_metadata():
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(MAPPING_FILE, "rb") as f:
        doc_ids = pickle.load(f)
    try:
        with open(SUMMARY_FILE, "rb") as f:
            summaries = pickle.load(f)
    except FileNotFoundError:
        summaries = None
    return index, doc_ids, summaries

# Search using a natural language query
def search_faiss(query, k=5):
    print("[INFO] Loading model and index...")
    model = SentenceTransformer(MODEL_NAME)
    index, doc_ids, summaries = load_index_and_metadata()

    print("[INFO] Encoding query...")
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    print("\n[RESULTS]")
    for i, idx in enumerate(indices[0]):
        doc_id = doc_ids[idx]
        print(f"Rank #{i+1}")
        print(f"Patient ID: {doc_id}")
        if summaries:
            print(f"Summary: {summaries[idx]}")
        print(f"Distance Score: {distances[0][i]:.4f}\n")

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    search_faiss(user_query)
