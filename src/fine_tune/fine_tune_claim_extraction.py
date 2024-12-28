"""
File: src/fine_tune/fine_tune_claim_extraction.py

Example script for fine-tuning a model to:
1) Extract a claim from user text
2) Respond with a stance ("Evet"/"Hayır") + reasoning
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# ------------------ Configuration ------------------ #

# Path to your base model (change to your actual base or QLoRA model ID)
BASE_MODEL_NAME = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"

# Path to the final fine-tuned LoRA model output
OUTPUT_DIR = "./models/claim_extraction"

# Path to the dataset file containing user_message -> (Claim + stance + reasoning)
# Expecting a JSON with something like:
# [
#   {
#     "user_message": "I think the government is not sincere about environment",
#     "target_response": "Claim: The government is not sincere about environment. Stance: Evet. Reasoning: ...
#   },
#   ...
# ]
DATA_FILE = "./finetune_data/claim_extraction/claim_extraction_data.json"

# QLoRA / LoRA parameters (update to your liking)
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Additional training parameters
NUM_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.001
WARMUP_RATIO = 0.03
LOGGING_STEPS = 100
SAVE_STEPS = 0  # or some positive integer if you want checkpoints
MAX_SEQ_LENGTH = 1024  # adjust as needed

USE_4BIT = True  # If you want QLoRA style 4-bit
BF16 = False     # Set to True on certain GPUs (A100) if desired
FP16 = False     # Alternatively use this if you want half precision

# ------------------ Load & Prep Dataset ------------------ #

def load_dataset():
    """
    Loads your dataset from DATA_FILE (JSON). 
    Splits into train/test. 
    Returns a Hugging Face DatasetDict or something suitable for SFTTrainer.
    """

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # raw_data is a list of dicts: {"user_message": "...", "target_response": "..."}

    # Convert to Dataset
    dataset = Dataset.from_list(raw_data)
    # Optionally do train_test_split
    ds_split = dataset.train_test_split(test_size=0.1, seed=42)
    return ds_split

# ------------------ Formatting Function ------------------ #

def format_prompt_text(example):
    """
    Takes each sample (user_message, target_response) and creates a single 
    'input-output' training string. We'll store it in a new field, e.g. "prompt".
    Then SFTTrainer uses that to train the model in a supervised manner.
    """
    user_msg = example["user_message"]
    target_resp = example["target_response"]

    # Construct a 'chat style' prompt if desired:
    system_text = (
        "Sen bir Türk köşe yazarı gibi davranıyorsun. "
        "İlk olarak, kullanıcının metninden varsa bir 'Claim' çıkart, "
        "sonra 'Stance' olarak Evet/Hayır de, "
        "en sonunda Reasoning ekle. "
        "Eğer net bir iddia yoksa kibarca diyalogda kal.\n"
    )

    # You could also do a multi-turn chat template if you want. For now, let's keep it simple.
    # For input, we'll produce something like:
    # "[system prompt]\nUser: {user_msg}\nAssistant:"

    input_str = (
        f"{system_text}\n"  # system instructions
        f"User: {user_msg}\n"
        "Assistant:"
    )

    # The "target_response" is what we want the model to output AFTER "Assistant:"
    # Example content might be:
    # "Claim: The government is not sincere about environment. Stance: Evet. Reasoning: Because..."
    # We'll store this as the "labels" or the "target text".

    return input_str, target_resp

def formatting_func(samples):
    """
    Called by SFTTrainer to process a batch of data. 
    Must return a list of strings for 'prompt' and 'labels' (or sometimes a single joined text).
    """
    prompts = []
    for user_msg, target_resp in zip(samples["user_message"], samples["target_response"]):
        prompt_text, label_text = format_prompt_text(
            {"user_message": user_msg, "target_response": target_resp}
        )
        # We can store the combined text as a single item or store them separately
        # For SFTTrainer, you typically return the full "prompt + label" in one string,
        # or you can rely on the library's chat template approach.
        # We'll just store it as a single string that has the prompt + label concatenated.

        # Approach A: Return them as a single string with a special separator
        # Approach B: Return them separately and rely on the trainer

        # We'll do Approach B here:
        # We create a single combined text but separate with a special token or newline
        # so that the model sees 'prompt' as input and 'label_text' as the completion.

        # If using `SFTTrainer`, it often expects a single text field. We'll do this:
        combined_text = prompt_text + " " + label_text
        prompts.append(combined_text)

    return {"text": prompts}


# ------------------ Main Script ------------------ #

def main():
    # 1. Load dataset
    ds_split = load_dataset()

    # 2. Possibly do a BnB 4-bit config (QLoRA)
    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16
        )

    # 3. Load base model & tokenizer
    print("Loading base model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # if needed

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config if USE_4BIT else None,
        device_map="auto",
        torch_dtype=torch.bfloat16 if BF16 else torch.float16 if FP16 else torch.float32
    )

    # 4. LoRA config
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        bf16=BF16,
        max_grad_norm=0.3,
        max_steps=-1,  # or set a specific step count
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type="cosine",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        report_to="tensorboard",  # or "none"
    )

    # 6. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_split["train"],
        eval_dataset=ds_split["test"],
        peft_config=peft_config,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        formatting_func=formatting_func,  # our function to produce a final training text
        args=training_args,
        packing=False,  # if you want to pack multiple samples
    )

    print("Starting fine-tuning with claim extraction + stance + reasoning...")
    trainer.train()

    # 7. Save final model
    trainer.model.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()