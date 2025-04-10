import pandas as pd
import json
import os

# Paths
DATA_DIR = "./data/"
OUT_FILE = os.path.join(DATA_DIR, "ehr_clean_text.jsonl")

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

if __name__ == "__main__":
    main()