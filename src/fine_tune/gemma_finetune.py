
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import torch
from trl import SFTTrainer
import json
from tqdm import tqdm

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device_map = get_device_map()

print("torch.cuda availability:", torch.cuda.is_available())


# ---------- Base Model Setup ---------- #
model_id = "google/gemma-2-9b-it"

print("Loading Tokenizer and Model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print("Model and Tokenizer loaded successfully.\n")

# ---------- Dataset Creation ---------- #
def create_database(filename: str, test_size: float = 0.1):
    """
    Creates a Hugging Face Dataset object with claims, agreement, and reasoning.

    Args:
        filename (str): Path to the JSON file containing claims and reasonings.
        test_size (float): Test set split ratio.

    Returns:
        Dataset: A Hugging Face Dataset object with train and test splits.
    """
    print("Loading data for dataset creation...")
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Collect all claims as separate examples
    claims, agreements, reasonings = [], [], []
    for claim_data in data:
        claims.append(claim_data["claim"])
        agreements.append(claim_data["agreement"])
        reasonings.append(claim_data.get("reasoning", ""))

    dataset_dict = {
        "claim": claims,
        "agreement": agreements,
        "reasoning": reasonings,
    }

    dataset = Dataset.from_dict(dataset_dict)
    if test_size > 0:
        dataset = dataset.train_test_split(test_size=test_size)
        print(f"Dataset split into train and test sets (test size = {test_size}).")

    return dataset

# ---------- Prompt Formatting ---------- #
def create_prompt(sample):
    """
    Creates a formatted prompt for a single claim.

    Args:
        sample (dict): A dictionary with "claim", "agreement", and "reasoning".

    Returns:
        str: A formatted prompt.
    """
    system_message = "Sen bir Türk köşe yazarısın. Görevin verilen iddiayı destekleyip desteklemediğini belirtmek ve gerekçesini açıklamaktır."
    question = "Aşağıda verilen iddiayı destekliyor musunuz? Lütfen önce 'Evet' veya 'Hayır' olarak cevaplayıp ardından gerekçesini açıklayın."

    chat_template = [
        {"role": "user", "content": f"{system_message}\n\n{question}\n\n{sample['claim']}"},
        {"role": "assistant", "content": f"{sample['agreement']}. {sample['reasoning']}"}
    ]

    full_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False)
    return full_prompt


def format_prompts(examples):
    """
    Converts a dataset into chat-style prompts.

    Args:
        examples (dict): Contains lists of claims, agreements, and reasonings.

    Returns:
        list of str: A list of formatted prompts.
    """
    prompts = []
    for i in range(len(examples["claim"])):
        sample = {
            "claim": examples["claim"][i],
            "agreement": examples["agreement"][i],
            "reasoning": examples["reasoning"][i],
        }
        prompt = create_prompt(sample)
        prompts.append(prompt)

    return prompts

# ---------- Fine-Tune Training ---------- #
def train(train_filename, new_model_path):
    """
    Fine-tunes the base model on claims and reasonings.

    Args:
        train_filename (str): Path to the training dataset file.
        new_model_path (str): Path to save the fine-tuned model.
    """
    # Create the database from JSON
    train_dataset = create_database(filename=train_filename)

    formatting_func = lambda examples: format_prompts(examples)

    # Configure LoRA
    peft_config = LoraConfig(
        r=256,
        lora_alpha=256,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=new_model_path,
        num_train_epochs=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        bf16=True,
        logging_steps=100,
        max_grad_norm=0.3,
        save_steps=0,
        report_to="tensorboard"
    )

    # Set supervised fine-tuning trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["test"],
        peft_config=peft_config,
        max_seq_length=None,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        args=training_arguments,
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    trainer.model.save_pretrained(new_model_path)
    print(f"Model saved to {new_model_path}.")

# ---------- Execution ---------- #
columnist_name = "ismailsaymaz"
train_filename = f"./finetune_data/claim_reasoning/{columnist_name}/{columnist_name}_train.json"
new_model_path = f"./models/claim_reasoning/{columnist_name}"

train(
    train_filename=train_filename,
    new_model_path=new_model_path
)
