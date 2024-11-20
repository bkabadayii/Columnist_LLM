from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm

# Base model name
base_model = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"

def evaluate_model(test_file: str, model_name: str, output_file: str, log_process: bool=True):
    """
    Evaluates the model by generating answers for claims in the test dataset
    and saves the results with the predicted responses and reasoning.

    Args:
        test_file (str): Path to the JSON file containing the test dataset.
        model_name (str): Path to the fine-tuned model directory.
        output_file (str): Path where the output JSON file with predictions will be saved.
        log_process (bool): If True, logs the evaluation process with tqdm.

    Returns:
        None
    """

    # Load the test data
    with open(test_file, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    # System message for the model's role
    system_message = "Sen bir Türk köşe yazarısın. Görevin verilen iddiayı destekleyip desteklemediğini belirtmek ve gerekçesini açıklamaktır."

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Define terminators and generation parameters
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_safetensors=True
    )

    # Evaluation loop
    for article in tqdm(test_data, desc="Evaluating claims", disable=not log_process):
        for claim in article["claims"]:
            # First question: Do you agree with the claim?
            question = f"Aşağıda verilen iddiayı destekliyor musunuz? Lütfen yalnızca 'Evet' veya 'Hayır' olarak cevaplayın.\n\n{claim['claim']}"

            # Construct the input prompt
            input_prompt = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ]

            # Tokenize the input
            input_ids = tokenizer.apply_chat_template(
                input_prompt, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            attention_mask = torch.ones_like(input_ids)

            # Generate the model's response for agreement
            outputs = model.generate(
                input_ids,
                max_new_tokens=20,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                attention_mask=attention_mask
            )

            # Decode the response
            response = outputs[0][input_ids.shape[-1]:]
            predicted_response = tokenizer.decode(response, skip_special_tokens=True).strip()

            # Save the predicted response
            claim["predicted_agreement"] = predicted_response

            # Second question: Why do you agree/disagree?
            if predicted_response == "Evet" or predicted_response == "Evet.":
                follow_up = "Bu iddiayı neden destekliyorsunuz?"
            else:
                follow_up = "Bu iddiayı neden desteklemiyorsunuz?"

            follow_up_prompt = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
                {"role": "assistant", "content": predicted_response},
                {"role": "user", "content": follow_up}
            ]

            # Tokenize the follow-up prompt
            follow_up_ids = tokenizer.apply_chat_template(
                follow_up_prompt, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            attention_mask = torch.ones_like(follow_up_ids)

            # Generate the model's response for reasoning
            outputs = model.generate(
                follow_up_ids,
                max_new_tokens=200,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                attention_mask=attention_mask
            )

            # Decode the reasoning response
            response = outputs[0][follow_up_ids.shape[-1]:]
            predicted_reasoning = tokenizer.decode(response, skip_special_tokens=True).strip()

            # Save the predicted reasoning
            claim["predicted_reason"] = predicted_reasoning

    # Save the results to the output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(test_data, outfile, ensure_ascii=False, indent=4)

    if log_process:
        print(f"Predictions saved to {output_file}")


# Usage
model_name = "./models/ytu/hilalkaplan_claims"
test_file = "./finetune_data/hilalkaplan_claims/hilalkaplan_claims_test.json"
output_file = "./prediction_results/ytu_claims_predictions.json"

evaluate_model(
    test_file=test_file,
    model_name=base_model,
    output_file=output_file,
    log_process=True
)