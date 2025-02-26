import sys
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def predict_model(responding_columnist, target_columnist):
    """
    Runs predictions on the target_columnist's test dataset using
    the responding_columnist's fine-tuned model, then saves results
    in {responding_columnist}_to_{target_columnist}.json.
    """

    # ------------------------------------------------------------------------
    # 1. Construct paths for test file, model directory, output file
    # ------------------------------------------------------------------------
    test_filename = f"./siu_test_files/{target_columnist}_test.json"
    model_path = f"./models/claim_questions_1epoch/{responding_columnist}"
    
    # Ensure the directory exists
    output_dir = "./siu_predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Construct the final output path
    output_path = f"{output_dir}/{responding_columnist}_to_{target_columnist}.json"

    # Base model ID (update if your base model is different)
    base_model_id = "google/gemma-2-9b-it"

    # ------------------------------------------------------------------------
    # 2. Load or initialize output data structure
    # ------------------------------------------------------------------------
    if os.path.exists(output_path):
        print(f"Output file exists. Loading previous predictions from {output_path}")
        with open(output_path, "r", encoding="utf-8") as output_file:
            output_data = json.load(output_file)
        # If "results" does not exist or is empty, initialize an empty list
        results = output_data.get("results", [])
    else:
        print(f"No existing output file. Starting fresh for {responding_columnist}_to_{target_columnist}.json")
        results = []

    # Make sure we keep the top-level structure consistent
    output_data = {
        "responding_columnist": responding_columnist,
        "claims_owner_columnist": target_columnist,
        "results": results
    }

    # ------------------------------------------------------------------------
    # 3. Load the target columnist's test dataset
    # ------------------------------------------------------------------------
    print(f"Loading test data from {test_filename} ...")
    with open(test_filename, "r", encoding="utf-8") as test_file:
        test_data = json.load(test_file)

    # ------------------------------------------------------------------------
    # 4. Load tokenizer and LoRa model
    # ------------------------------------------------------------------------
    print(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Example BitsAndBytesConfig (commented out if not using 4-bit):
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=quantization_config
    )
    """
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")

    print(f"Loading LoRa adapter from: {model_path}")
    lora_model = PeftModel.from_pretrained(base_model, model_path)

    # Merge LoRa weights into the base model
    print("Merging LoRa adapter into the base model...")
    model = lora_model.merge_and_unload()
    model.eval()

    # ------------------------------------------------------------------------
    # 5. Generate predictions for each claim
    # ------------------------------------------------------------------------
    # We'll go claim by claim and write the file after each prediction
    # so that you can track progress.
    system_message = (
        "Sen bir Türk köşe yazarısın. "
        "Görevin verilen iddiayı destekleyip desteklemediğini belirtmek "
        "ve gerekçesini açıklamaktır."
    )
    question = (
        "Aşağıda verilen iddiayı destekliyor musunuz? "
        "Lütfen önce 'Evet' veya 'Hayır' olarak cevaplayıp "
        "ardından gerekçesini açıklayın."
    )

    for claim in tqdm(test_data, desc="Running predictions"):
        # Construct prompt
        input_prompt = [
            {"role": "user", "content": f"{system_message}\n\n{question}\n\n{claim["claim"]}"},
        ]

        # Tokenize the input
        input_ids = tokenizer.apply_chat_template(input_prompt, return_tensors="pt", return_dict=True, add_generation_prompt=True).to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=256)

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # If your model uses a special delimiter (e.g. "\nmodel\n"), parse accordingly.
        # ---------------------------------------------------
        try:
            response = response.split("\nmodel\n")[1].strip()
        except IndexError:
            pass
        # ---------------------------------------------------

        # Gather fields from test data + predicted text
        result_item = {
            "article_id": claim.get("article_id"),
            "claim_id": claim.get("claim_id"),
            "claim": claim.get("claim"),
            "agreement": claim.get("agreement"),
            "reasoning": claim.get("reasoning"),
            "prediction": response
        }

        # Append to results
        results.append(result_item)

        # Update the output_data structure
        output_data["results"] = results

        # Save (overwrite) the output file so we can track progress
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(output_data, output_file, ensure_ascii=False, indent=4)

    print(f"Predictions finished. Final results saved to {output_path}")


# ------------------------------------------------------------------------
# 6. Main entry point for command-line usage
# Usage:
#   python prediction_script.py responding_columnist target_columnist
# Example:
#   python prediction_script.py hilalkaplan ismailsaymaz
# ------------------------------------------------------------------------
if __name__ == "__main__":
    responding_columnist = sys.argv[1]
    target_columnist = sys.argv[2]
    predict_model(responding_columnist, target_columnist)
