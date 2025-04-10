import pandas as pd
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = "./data/finetune/"
OUT_FILE = os.path.join(DATA_DIR, "ehr_clean_text.jsonl")
SQUAD_FILE = os.path.join(DATA_DIR, "ehr_squad_format.json")

# === CONFIG ===
MODEL_NAME = os.getenv("MODEL_NAME", "deepset/roberta-base-squad2")
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

# === BUILD EXAMPLES FROM CSV ===
def build_instruction_examples(patients, conditions, medications, encounters):
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
    return examples

# === BUILD SQUAD DATA FORMAT ===
def build_squad_format(patients, conditions, medications, encounters):
    data = []
    for _, row in patients.iterrows():
        pid = row['Id']
        name = f"{row['FIRST']} {row['LAST']}"
        age = 2025 - int(row['BIRTHDATE'][:4])
        gender = row['GENDER'].capitalize()

        conds = conditions[conditions['PATIENT'] == pid]['DESCRIPTION'].dropna().unique()
        meds = medications[medications['PATIENT'] == pid]['DESCRIPTION'].dropna().unique()
        visits = encounters[encounters['PATIENT'] == pid]['DESCRIPTION'].dropna().unique()

        context = f"Patient: {name}, Age: {age}, Gender: {gender}.\n"
        context += f"Conditions: {', '.join(conds[:5]) or 'None'}.\n"
        context += f"Encounters: {', '.join(visits[:5]) or 'None'}."

        if len(meds) == 0:
            continue

        answer = ", ".join(meds[:5])
        answer_start = context.find(meds[0]) if meds[0] in context else 0

        qa_item = {
            "context": context,
            "qas": [
                {
                    "id": pid,
                    "question": "What medications has the patient been prescribed?",
                    "answers": [
                        {
                            "text": answer,
                            "answer_start": answer_start
                        }
                    ],
                    "is_impossible": False
                }
            ]
        }
        data.append({"title": name, "paragraphs": [qa_item]})
    return {"data": data}

# === MAIN ===
def main():
    # Load CSVs
    patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
    conditions = pd.read_csv(os.path.join(DATA_DIR, "conditions.csv"))
    medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
    encounters = pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"))

    # Instruction-style fine-tuning data
    examples = build_instruction_examples(patients, conditions, medications, encounters)
    with open(OUT_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} instruction-style examples to {OUT_FILE}")

    # SQuAD-format QA data
    squad_data = build_squad_format(patients, conditions, medications, encounters)
    with open(SQUAD_FILE, "w") as f:
        json.dump(squad_data, f, indent=2)
    print(f"Saved QA-style SQuAD data to {SQUAD_FILE}")

    # Tokenization for instruction-style
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