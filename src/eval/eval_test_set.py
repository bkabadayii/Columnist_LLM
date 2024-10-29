from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm

base_model = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"

def evaluate_model(test_file: str, model_name: str, output_file: str, log_process: bool=True, use_conversations: bool=False):
    """
    Evaluates the model by generating answers for questions in the test dataset
    and saves the results with the predicted responses. It can evaluate either 
    each question-answer pair separately or the whole article's question-answer 
    pairs as a progressive conversation.

    Args:
        test_file (str): Path to the JSON file containing the test dataset.
        model_name (str): Path to the fine-tuned model directory.
        output_file (str): Path where the output JSON file with predictions will be saved.
        log_process (bool): If True, logs the evaluation process with tqdm.
        use_conversations (bool): If True, evaluates progressively, where each question-answer 
                                  pair uses accumulated context from previous pairs.

    Returns:
        None
    """

    # Load the test data
    with open(test_file, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    system_msg = "Sen bir türk köşe yazarısın. Görevin kullanıcının sorduğu bir soruya kendi yorumlamalarınla cevap vermek."

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Define terminators and generation parameters
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_safetensors=True
    )

    for article in tqdm(test_data, desc="Processing articles", disable=not log_process):
        # Initialize conversation if using conversational evaluation
        conversation = [{"role": "system", "content": system_msg}] if use_conversations else None

        for qa in article["qa_pairs"]:
            question = qa["question"]

            if use_conversations:
                # Add the current question to the conversation
                conversation.append({"role": "user", "content": question})
                messages = conversation
            else:
                # Use standalone messages for each QA pair
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": question},
                ]

            # Tokenize messages
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            attention_mask = torch.ones_like(input_ids)

            # Generate the response for the current question
            outputs = model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                attention_mask=attention_mask,
            )

            response = outputs[0][input_ids.shape[-1]:]
            predicted_response = tokenizer.decode(response, skip_special_tokens=True)

            # Save the predicted response
            qa["predicted_response"] = predicted_response

            # If using conversational context, add predicted response as context for next iteration
            if use_conversations:
                conversation.append({"role": "assistant", "content": predicted_response})

    # Save the results to the output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(test_data, outfile, ensure_ascii=False, indent=4)

    if log_process:
        print(f"Predictions saved to {output_file}")

# Usage
model_name = "./models/hilalkaplan_conversation_ytu_model"
test_file = "./finetune_data/hilalkaplan_qa/hilalkaplan_qa_test.json"
output_file = "./eval_results/hilalkaplan_conversation_ytu_results.json"

evaluate_model(
    test_file=test_file,
    model_name=model_name,
    output_file=output_file,
    log_process=True,
    use_conversations=True
)