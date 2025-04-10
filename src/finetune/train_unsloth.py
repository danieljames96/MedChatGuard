from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Apply QLoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)

# Load dataset
dataset = load_dataset("json", data_files="data/ehr_clean_text.jsonl")["train"]

# Define formatting function
def formatting_func(example):
    return f"### Instruction:\\n{example['instruction']}\\n\\n### Input:\\n{example['input']}\\n\\n### Response:\\n{example['output']}"

# Training arguments
args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_train_epochs = 3,
    learning_rate = 2e-4,
    save_steps = 100,
    logging_steps = 10,
    output_dir = "finetuned-model",
    save_total_limit = 2,
    bf16 = True,
    report_to = "none",
)

# Train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    formatting_func = formatting_func,
    args = args,
    max_seq_length = 2048,
)

trainer.train()
