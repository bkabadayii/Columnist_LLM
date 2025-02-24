from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os
from tqdm import tqdm

def predict_with_multiple_adapters(test_filename, model_id, adapter_paths, output_path, columnists):
    """
    Runs predictions using multiple adapters for each claim in the test dataset.

    Args:
        test_filename (str): Path to the JSON file containing the test dataset.
        model_id (str): ID of the base model.
        base_model_path (str): Path to the base model.
        adapter_paths (dict): Dictionary of columnists and their adapter paths.
        output_path (str): Path to save the prediction results.
        columnists (list): List of columnists corresponding to adapters.

    Returns:
        None: Saves predictions to the specified output path.
    """
    # Step 1: Load the test dataset
    print(f"Loading test data from {test_filename}")
    with open(test_filename, "r", encoding="utf-8") as test_file:
        test_data = json.load(test_file)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 2: Load tokenizer and base model
    print(f"Loading tokenizer from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading base model from: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    # Load all adapters
    print(f"Loading adapters for columnists: {columnists}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_paths[columnists[0]], adapter_name=columnists[0])
    for columnist, adapter_path in list(adapter_paths.items())[1:]:
        peft_model.load_adapter(adapter_path, adapter_name=columnist)

    # Prepare adapter names
    adapter_names = list(adapter_paths.keys())
    
    for claim in tqdm(test_data, desc="Running predictions"):
        # Construct the input prompt
        system_message = "Sen bir Türk köşe yazarısın. Görevin verilen iddiayı destekleyip desteklemediğini belirtmek ve gerekçesini açıklamaktır."
        question = f"Aşağıda verilen iddiayı destekliyor musunuz? Lütfen önce 'Evet' veya 'Hayır' olarak cevaplayıp ardından gerekçesini açıklayın."

        input_prompt = [
            {"role": "user", "content": f"{system_message}\n\n{question}\n\n{claim['claim']}"},
        ]

        all_prompts = [
            input_prompt, input_prompt, input_prompt,
            input_prompt, input_prompt, input_prompt,
            input_prompt, input_prompt
        ]
    
        # Tokenize the input
        input_ids = tokenizer.apply_chat_template(
            all_prompts,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        ).to("cuda")

        # Generate responses using all adapters
        outputs = peft_model.generate(**input_ids, adapter_names=adapter_names, max_new_tokens=256)

        # Decode and save responses for each adapter
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True).strip()
            #model_idx = response.find("model\n")
            claim[f"{adapter_names[i]}_response"] = response.split("\nmodel\n")[1].strip()
        
        # Save interim results
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(test_data, output_file, ensure_ascii=False, indent=4)

    # Step 4: Save predictions to the output file
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(test_data, output_file, ensure_ascii=False, indent=4)

    print(f"Predictions saved to {output_path}")


# ---------- USAGE ---------- #

# Inputs
test_filename = "./finetune_data/claim_reasoning/evet_test.json"
model_id = "google/gemma-2-9b-it"

# Adapter paths for columnists
adapter_paths = {
    "q_hilalkaplan": "./models/claim_reasoning/q_hilalkaplan",
    "q_ismailsaymaz": "./models/claim_reasoning/q_ismailsaymaz",
    "q_mehmettezkan": "./models/claim_reasoning/q_mehmettezkan",
    "q_nagehanalci": "./models/claim_reasoning/q_nagehanalci",
    "q_nihalbengisukaraca": "./models/claim_reasoning/q_nihalbengisukaraca",
    "q_abdurrahmandilipak": "./models/claim_reasoning/q_abdurrahmandilipak",
    "q_fehimtastekin": "./models/claim_reasoning/q_fehimtastekin",
    "q_melihasik": "./models/claim_reasoning/q_melihasik"
}

# List of columnists (adapter names)
columnists = [
    "q_hilalkaplan", "q_ismailsaymaz", "q_mehmettezkan",
    "q_nagehanalci", "q_nihalbengisukaraca", "q_abdurrahmandilipak",
    "q_fehimtastekin", "q_melihasik"
]

# Output path for predictions
output_path = "./prediction_results/claim_reasoning/multiple_adapter_predictions.json"

# Run predictions
predict_with_multiple_adapters(
    test_filename=test_filename,
    model_id=model_id,
    adapter_paths=adapter_paths,
    output_path=output_path,
    columnists=columnists,
)
