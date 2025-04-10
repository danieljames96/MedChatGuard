import subprocess
import shutil
import os
import sys
import time

from src.utils.prepare_data import main as prepare_main

# === CONFIG ===
# SYNTHEA_JAR = r"D:\\Development\\Project_Repositories\\MedChatGuard\\synthea-with-dependencies.jar"
SYNTHEA_JAR = "./synthea-with-dependencies.jar"  # Ensure this is in your root dir
OUTPUT_DIR = "./output/csv/"                       # Synthea default CSV output
DATA_DIR = "./data/finetune/"                           # Where to move processed CSVs
NUM_PATIENTS = 100                               # Change as needed
SEED = 42                                        # Fixed for reproducibility

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
def move_csv_output():
    if not os.path.exists(OUTPUT_DIR):
        print("[ERROR] Synthea CSV output not found.")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")]

    for f in files:
        shutil.copy2(os.path.join(OUTPUT_DIR, f), os.path.join(DATA_DIR, f))
        print(f"[COPY] {f} → {DATA_DIR}")
    print("[DONE] All files copied to data directory.")

# === STEP 3: Convert to JSONL ===
def convert_to_jsonl():
    prepare_main()

if __name__ == "__main__":
    start = time.time()
    run_synthea()
    move_csv_output()
    print("[INFO] Converting to fine-tuning dataset...")
    convert_to_jsonl()
    print(f"[ALL DONE] Total time: {time.time() - start:.2f}s")
