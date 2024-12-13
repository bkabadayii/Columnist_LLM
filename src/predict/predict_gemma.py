from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from tqdm import tqdm


def predict_model(test_filename, model_id, model_path, output_path, columnist):
    """
    Runs predictions on the test dataset using a single fine-tuned model.

    Args:
        test_filename (str): Path to the JSON file containing the test dataset.
        model_id (str): ID of the base model.
        model_path (str): Path to the fine-tuned model.
        output_path (str): Path to save the prediction results.

    Returns:
        None: Saves predictions to the specified output path.
    """
    # Step 1: Load the test dataset or prior predictions
    if os.path.exists(output_path):
        print(f"Output file exists. Loading previous predictions from {output_path}")
        with open(output_path, "r", encoding="utf-8") as output_file:
            test_data = json.load(output_file)
    else:
        print(f"No existing output file. Loading test data from {test_filename}")
        with open(test_filename, "r", encoding="utf-8") as test_file:
            test_data = json.load(test_file)

    # Step 2: Load tokenizer and model
    print(f"Loading model and tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Step 3: Make predictions
    for claim in tqdm(test_data, desc="Running predictions"):
        # Construct the input prompt
        system_message = "Sen bir Türk köşe yazarısın. Görevin verilen iddiayı destekleyip desteklemediğini belirtmek ve gerekçesini açıklamaktır."
        question = f"Aşağıda verilen iddiayı destekliyor musunuz? Lütfen yalnızca 'Evet' veya 'Hayır' olarak cevaplayın ve gerekçenizi açıklayın."

        input_prompt = [
            {"role": "user", "content": f"{system_message}\n\n{question}\n\n{claim["claim"]}"},
        ]

        # Tokenize the input
        input_ids = tokenizer.apply_chat_template(input_prompt, return_tensors="pt", return_dict=True).to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=1024)

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("\nmodel\n")[1].strip()

        # Save the prediction
        claim[f"{columnist}_response"] = response

        # Step 4: Save predictions to the output file
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(test_data, output_file, ensure_ascii=False, indent=4)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(test_data, output_file, ensure_ascii=False, indent=4)

    print(f"Predictions saved to {output_path}")


# ---------- USAGE ---------- #
columnist = "hilalkaplan"
test_filename = "./finetune_data/claim_reasoning/filtered_test.json"
model_path = f"./models/claim_reasoning/{columnist}"
output_path = "./prediction_results/claim_reasoning/predictions.json"

# Base model ID
model_id = "google/gemma-2-9b-it"

predict_model(
    test_filename=test_filename,
    model_id=model_id,
    model_path=model_path,
    output_path=output_path,
    columnist=columnist
)
