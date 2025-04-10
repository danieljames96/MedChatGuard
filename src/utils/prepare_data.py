import pandas as pd
import json
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Paths
DATA_DIR = "./data/finetune/"
OUT_FILE = os.path.join(DATA_DIR, "ehr_clean_text.jsonl")

# === CONFIG ===
MODEL_NAME = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
DATA_PATH = OUT_FILE
OUTPUT_DIR = "./data/preprocessed/"
MAX_SEQ_LENGTH = 2048

# === LOAD TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# === FORMAT FUNCTION ===
def formatting_func(example):
    if not example['instruction'].strip() or not example['output'].strip():
        return {"prompt": None}  # skip blank rows

    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"prompt": prompt}

# === TOKENIZATION FUNCTION ===
def tokenize_func(example):
    return tokenizer(
        example["prompt"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH
    )

def main():
    # Load CSVs
    patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
    conditions = pd.read_csv(os.path.join(DATA_DIR, "conditions.csv"))
    medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
    encounters = pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"))

    examples = []

    for _, row in patients.iterrows():
        pid = row['Id']
        name = f"{row['FIRST']} {row['LAST']}"
        age = 2025 - int(row['BIRTHDATE'][:4])
        gender = row['GENDER'].capitalize()

        conds = conditions[conditions['PATIENT'] == pid]['DESCRIPTION'].dropna().unique()
        meds = medications[medications['PATIENT'] == pid]['DESCRIPTION'].dropna().unique()
        visits = encounters[encounters['PATIENT'] == pid]['DESCRIPTION'].dropna().unique()

        input_text = f"Patient: {name}, Age: {age}, Gender: {gender}.\n"
        input_text += f"Conditions: {', '.join(conds[:5]) or 'None'}.\n"
        input_text += f"Encounters: {', '.join(visits[:5]) or 'None'}."

        if len(meds) == 0:
            continue

        example = {
            "instruction": "What medications has the patient been prescribed?",
            "input": input_text,
            "output": ", ".join(meds[:5])
        }
        examples.append(example)

    # Save to JSONL
    with open(OUT_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(examples)} examples to {OUT_FILE}")
    
    print("[INFO] Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]

    print("[INFO] Formatting prompts...")
    dataset = dataset.map(formatting_func, remove_columns=dataset.column_names, batched=False)
    dataset = dataset.filter(lambda x: x["prompt"] is not None)

    print("[INFO] Tokenizing dataset...")
    tokenized = dataset.map(tokenize_func, batched=False)
    tokenized.set_format("torch")

    print(f"[INFO] Saving preprocessed dataset to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenized.save_to_disk(OUTPUT_DIR)

    print("[✅ DONE] Dataset preprocessing complete.")

if __name__ == "__main__":
    main()