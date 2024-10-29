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
optim = "paged_adamw_8bit"

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

# ---------- Training on Question Answer Pairs as Seperate Examples ---------- #
 
# Define the prompt formatting function
def create_prompt(sample):
    """
    Creates a formatted prompt for a single question-response pair based on the 
    perspective of a Turkish columnist, where the assistant's role is to respond 
    to questions with personal commentary.

    Args:
        sample (dict): A dictionary containing:
            - "Instruction" (str): The question posed by the user.
            - "Response" (str): The expected answer from the assistant, written 
            in the tone and style of a Turkish columnist.

    Returns:
        str: A formatted prompt in chat format, including a system message that 
        sets the assistant's role, followed by user and assistant interactions.

    Example:
        input: {"Instruction": "What is your opinion on...?", "Response": "I believe..."}
        output: A structured prompt in chat format with system, user, and assistant messages.
    """
    system_message = "Sen bir türk köşe yazarısın. Görevin kullanıcının sorduğu bir soruya kendi yorumlamalarınla cevap vermek."
    chat_template = [
        { "role": "system", "content": system_message },
        { "role": "user", "content": sample["Instruction"]},
        { "role": "assistant", "content": sample["Response"]}
    ]
    full_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False)
    return full_prompt

def format_prompts(examples):
    """
    Converts a dataset of question-response pairs into individual chat-style prompts, 
    where each prompt includes a system message followed by a question from the user 
    and a response from the assistant. Optionally, saves the formatted prompts to a file.

    Args:
        examples (dict): A dictionary containing lists:
            - "Instruction" (list of str): A list of questions, where each question is 
            an interaction prompt posed by the user.
            - "Response" (list of str): A list of responses, where each response is 
            the assistant’s answer styled as a Turkish columnist.

    Returns:
        list of str: A list of formatted prompts in chat format, each including the system 
        message, user question, and assistant response.

    Side Effects:
        Writes each formatted prompt to "train_prompts.txt" with separators for readability.

    Example:
        input: {"Instruction": ["What is your opinion on...?"],
                "Response": ["I believe..."]}
        output: A list of chat-formatted prompts, each containing a system, user, and assistant message.
    """
    output_texts = []
    for i in range(len(examples["Instruction"])):
        sample = {}
        sample["Instruction"] = examples["Instruction"][i]
        sample["Response"] = examples["Response"][i]

        text = create_prompt(sample)
        output_texts.append(text)
    
    with open("train_prompts.txt", "w") as train_prompts_file:
        for text in output_texts:
            train_prompts_file.write(f"{text}\n----------------------------------------------------------------------------------------------------\n")

    return output_texts

# ---------- Training on Question Answer Pairs as Seperate Examples ---------- #


# ---------- Training on Articles as a Conversation Consisting QA Pairs ---------- #

# Define the conversation-based prompt formatting function
def create_conversation_prompt(qa_pairs):
    """
    Constructs a conversation prompt for a set of question-answer pairs (qa_pairs) 
    based on a Turkish columnist's role, where the assistant answers in the tone 
    and style of a Turkish columnist.

    Args:
        qa_pairs (list of dict): A list of dictionaries representing question-answer 
        pairs, where each dictionary contains:
            - "question" (str): The question posed by the user.
            - "answer" (str): The answer from the assistant, intended to mimic a columnist's style.

    Returns:
        str: A conversation prompt in chat format, structured with alternating 
        roles of "user" for questions and "assistant" for answers. Each conversation 
        begins with a system message that establishes the assistant's role.

    Example:
        input: [{"question": "What do you think about...", "answer": "In my opinion..."}]
        output: A formatted prompt with system, user, and assistant messages.
    """
    system_message = "Sen bir Türk köşe yazarısın. Görevin kullanıcının sorduğu bir soruya kendi yorumlamalarınla cevap vermek."
    chat_template = [{"role": "system", "content": system_message}]
    
    for qa in qa_pairs:
        chat_template.append({"role": "user", "content": qa["question"]})
        chat_template.append({"role": "assistant", "content": qa["answer"]})

    full_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False)
    return full_prompt


def format_conversations(examples):
    """
    Converts a dataset of question-answer pairs into formatted conversation prompts, 
    where each prompt represents a full conversation from a single article. Optionally, 
    saves the formatted conversations to a file for inspection.

    Args:
        examples (dict): A dictionary containing:
            - "qa_pairs" (list): A list of question-answer pairs for each article, 
            where each element is a list of dictionaries. Each dictionary contains:
                - "question" (str): The question from the user.
                - "answer" (str): The columnist-style answer from the assistant.
            
    Returns:
        list of str: A list of conversation prompts, each containing the full conversation 
        for a single article. Each conversation follows a chat format, including the 
        initial system message, user questions, and assistant answers.

    Example:
        input: {"qa_pairs": [[{"question": "What do you think about...", "answer": "In my opinion..."}]]}
        output: A list of formatted conversation prompts in chat format.
    """
    output_texts = []
    for qa_pairs in tqdm(examples["qa_pairs"], desc="Formatting conversations"):
        text = create_conversation_prompt(qa_pairs)
        output_texts.append(text)
    
    # Optionally, save formatted prompts for inspection
    with open("train_conversations.txt", "w") as file:
        for text in output_texts:
            file.write(f"{text}\n{'-'*100}\n")

    return output_texts

# ---------- Training on Articles as a Conversation Consisting QA Pairs ---------- #

# ---------- Database Creation ---------- #
def create_database(filename: str, create_conversations: bool, log_process: bool = True, test_size: float = 0.1):
    """
    Creates a Hugging Face Dataset object containing question-answer pairs from a JSON file.

    Args:
        filename (str): Path to the JSON file containing articles.
        log_process (bool): If True, logs the process steps.
        test_size (flaot): If > 0, determines the database test set size. Else database is not splitted.
        create_conversations: If true: creates a conversation from each article, else it treats qa pairs as seperate examples

    Returns:
        Dataset: Hugging Face Dataset object with question-answer pairs.
    """
    
    # Log start of process
    if log_process:
        print("Starting database creation...")
    
    # Step 1: Load data
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    if create_conversations:
        # Step 2.1: Collect articles as conversational strings
        conversations = []
        for article in data:
            conversations.append({"qa_pairs": article["qa_pairs"]})
        
        # Step 3: Create a Dataset object
        dataset = Dataset.from_dict({"qa_pairs": [article["qa_pairs"] for article in data]})

    else:
        # Step 2.2: Collect all question-answer pairs
        qa_pairs = []
        for article in data:
            # Append each QA pair from article
            for qa_pair in article['qa_pairs']:
                qa_pairs.append(qa_pair)
        
        # Step 3: Create a Dataset object
        dataset = Dataset.from_dict({"Instruction": [pair["question"] for pair in qa_pairs],
                                    "Response": [pair["answer"] for pair in qa_pairs]})
    


    if test_size > 0:
        dataset = dataset.train_test_split(test_size=test_size)
        if log_process:
            print(f"Using test size: {test_size}...")

    # Log completion
    if log_process:
        print("Database creation completed.")
    
    return dataset
# ---------- Database Creation ---------- #

# Main training
def train(train_filename, new_model_path, create_conversations):
    # Create the database from qa json
    train_dataset = create_database(filename=train_filename, create_conversations=create_conversations)

    # Choose formatting function based on create conversations
    if (create_conversations):
        formatting_func = format_conversations
    else:
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


# ---------- EXECUTION ---------- #
create_conversations = True

# Train data file
train_filename = f"finetune_data/hilalkaplan_qa/hilalkaplan_qa_train.json"

# Define the columnist / output directory
columnist_name = "hilalkaplan_conversations"
output_dir = f"./lora_ytu_finetuned/{columnist_name}"

# Fine-tuned model path
new_model_path = f"./models/{columnist_name}_ytu_model"

# Load the tokenizer
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer is loaded successfully.\n")

train(
    train_filename=train_filename,
    new_model_path=new_model_path,
    create_conversations=create_conversations
)
# ---------- EXECUTION ---------- #