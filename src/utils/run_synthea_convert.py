import subprocess
import shutil
import os
import sys
import time
import argparse

from src.utils.prepare_data import main as prepare_main

# === CONFIG ===
SYNTHEA_JAR = "./synthea-with-dependencies.jar"  # Ensure this is in your root directory
OUTPUT_DIR = "./output/csv/"                      # Synthea default CSV output
FINETUNE_DIR = "./data/finetune/"                 # Where to move processed CSVs for fine-tuning
RAG_DIR = "./data/rag_docs/"                           # Where to move CSVs for RAG
NUM_PATIENTS = 500                                # Change as needed
SEED = 42                                         # Fixed for reproducibility

# === STEP 1: Run Synthea ===
def run_synthea():
    print("[INFO] Running Synthea to generate synthetic EHR data...")
    if os.path.exists("./output/fhir"):
        shutil.rmtree("./output/fhir")
    if os.path.exists("./output/metadata"):
        shutil.rmtree("./output/metadata")
    try:
        result = subprocess.run([
            "java", "-jar", SYNTHEA_JAR,
            "-p", str(NUM_PATIENTS),
            "--seed", str(SEED),
            "--exporter.fhir.export=false",
            "--exporter.csv.export=true",
        ], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("[ERROR] Synthea execution failed.")
        print("[STDERR]", e.stderr)
        sys.exit(1)

# === STEP 2: Move Generated CSVs ===
def move_csv_output(target_dir):
    if not os.path.exists(OUTPUT_DIR):
        print("[ERROR] Synthea CSV output not found.")
        sys.exit(1)
    
    os.makedirs(target_dir, exist_ok=True)
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")]
    
    for f in files:
        shutil.copy2(os.path.join(OUTPUT_DIR, f), os.path.join(target_dir, f))
        print(f"[COPY] {f} → {target_dir}")
    
    print(f"[DONE] All files copied to {target_dir}")

# === STEP 3: Convert to JSONL (for fine-tuning mode) ===
def convert_to_jsonl():
    prepare_main()

# === MAIN (Argument Parsing) ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["finetune", "rag"], help="Processing mode: 'finetune' (generate JSONL for fine-tuning) or 'rag' (copy CSVs to data/rag_docs/)")
    args = parser.parse_args()

    start = time.time()
    run_synthea()

    if args.mode == "finetune":
        move_csv_output(FINETUNE_DIR)
        print("[INFO] Converting to fine-tuning dataset...")
        convert_to_jsonl()
    elif args.mode == "rag":
        move_csv_output(RAG_DIR)
        print("[INFO] RAG mode: CSV files copied to data/rag_docs/ directory.")

    print(f"[ALL DONE] Total time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
