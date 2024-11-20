from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import torch
from trl import SFTTrainer
import json
from tqdm import tqdm

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

print("torch.cuda availability:", torch.cuda.is_available())

# Define the columnist / output directory
columnist_name = "hilalkaplan_qa"
output_dir = f"./lora_ytu_finetuned/{columnist_name}"

# Model name to load
model_name = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 1

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 100

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}



################################################################################
# Fine-tuning
################################################################################

# ---------- Training on Claims and Their Reasonings ---------- #
def create_prompt(sample):
    """
    Creates a formatted prompt for a single claim based on whether the model should agree with it
    and includes the reasoning behind the answer.

    Args:
        sample (dict): A dictionary containing:
            - "claim" (str): The claim text.
            - "reference" (str): "Yes" or "No" indicating agreement.
            - "reason" (str): The reasoning text for the reference.

    Returns:
        str: A formatted prompt in chat format.
    """
    system_message = "Sen bir Türk köşe yazarısın. Görevin verilen iddiayı destekleyip desteklemediğini belirtmek ve gerekçesini açıklamaktır."
    question = "Aşağıda verilen iddiayı destekliyor musunuz? Lütfen yalnızca 'Evet' veya 'Hayır' olarak cevaplayın."
    
    # Adjust follow-up question based on the reference
    if sample["reference"] == "Evet":
        follow_up = "Bu iddiayı neden destekliyorsunuz?"
    else:
        follow_up = "Bu iddiayı neden desteklemiyorsunuz?"

    chat_template = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{question}\n\n{sample['claim']}"},
        {"role": "assistant", "content": sample["reference"]},
        {"role": "user", "content": follow_up},
        {"role": "assistant", "content": sample["reason"]}
    ]
    full_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False)
    return full_prompt


def format_prompts(examples):
    """
    Converts a dataset of claims into individual chat-style prompts, where each prompt includes
    a system message, the claim, the model's agreement or disagreement, and reasoning.

    Args:
        examples (dict): A dictionary containing lists:
            - "claim" (list of str): A list of claims.
            - "reference" (list of str): A list of "Yes" or "No" labels.
            - "reason" (list of str): A list of reasoning texts.

    Returns:
        list of str: A list of formatted prompts.
    """
    output_texts = []
    for i in range(len(examples["claim"])):
        sample = {
            "claim": examples["claim"][i],
            "reference": examples["reference"][i],
            "reason": examples["reason"][i]
        }
        text = create_prompt(sample)
        output_texts.append(text)
    
    with open("train_prompts.txt", "w", encoding="utf-8") as train_prompts_file:
        for text in output_texts:
            train_prompts_file.write(f"{text}\n{'-'*100}\n")

    return output_texts

# ---------- Database Creation ---------- #
def create_database(filename: str, log_process: bool = True, test_size: float = 0.1):
    """
    Creates a Hugging Face Dataset object containing claims, whether the model agrees with them, and reasoning.

    Args:
        filename (str): Path to the JSON file containing articles.
        log_process (bool): If True, logs the process steps.
        test_size (float): If > 0, determines the database test set size.

    Returns:
        Dataset: Hugging Face Dataset object.
    """
    if log_process:
        print("Starting database creation...")
    
    # Step 1: Load data
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Collect all claims as separate examples
    claims = []
    references = []
    reasons = []
    for article in data:
        for claim in article["claims"]:
            claims.append(claim["claim"])
            references.append(claim["reference"])
            reasons.append(claim["reason"])
    
    dataset = Dataset.from_dict({
        "claim": claims,
        "reference": references,
        "reason": reasons
    })

    if test_size > 0:
        dataset = dataset.train_test_split(test_size=test_size)
        if log_process:
            print(f"Using test size: {test_size}...")

    if log_process:
        print("Database creation completed.")
    
    return dataset
# ---------- Database Creation ---------- #

# ---------- Fine-Tune Training ---------- #
def train(train_filename, new_model_path):
    # Create the database from qa json
    train_dataset = create_database(filename=train_filename)

    formatting_func = format_prompts

    # bits and bytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset['test'],
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        args=training_arguments,
        packing=packing,
    )

    print("Training the model...")

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model_path)
# ---------- Fine-Tune Training ---------- #

# ---------- EXECUTION ---------- #
# Train data file
train_filename = f"finetune_data/hilalkaplan_claims/hilalkaplan_claims_train.json"

# Define the columnist / output directory
columnist_name = "hilalkaplan_claims"
output_dir = f"./lora_ytu_finetuned/{columnist_name}"

# Fine-tuned model path
new_model_path = f"./models/ytu/{columnist_name}"

# Load the tokenizer
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer is loaded successfully.\n")

train(
    train_filename=train_filename,
    new_model_path=new_model_path,
)
# ---------- EXECUTION ---------- #