import sys
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(responding_columnist, base_model_id="google/gemma-2-9b-it"):
    """
    Loads the tokenizer and model (with optional LoRa adapter) and returns them.
    """
    model_path = f"./models/claim_questions_1epoch/{responding_columnist}"
    print(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
    model = base_model

    if responding_columnist != "basemodel":
        print(f"Loading LoRa adapter from: {model_path}")
        lora_model = PeftModel.from_pretrained(base_model, model_path)
        print("Merging LoRa adapter into the base model...")
        model = lora_model.merge_and_unload()
        model.eval()

    return tokenizer, model

def predict_response(tokenizer, model, system_message, question, device="cuda"):
    """
    Constructs a prompt using the current system message and question,
    then generates and returns the model's response.
    """
    input_prompt = [
        {"role": "user", "content": f"{system_message}\n\n{question}"}
    ]

    # Tokenize the input using a chat template if available
    input_ids = tokenizer.apply_chat_template(
        input_prompt,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True
    ).to(device)

    outputs = model.generate(**input_ids, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # If your model uses a special delimiter (e.g. "\nmodel\n"), adjust accordingly.
    try:
        response = response.split("\nmodel\n")[1].strip()
    except IndexError:
        pass

    return response

def main():
    # Check if the model name is provided in the command-line arguments
    if len(sys.argv) < 4:
        print("Usage: python script.py <responding_columnist> <system_prompt_id> <questionnaire_dir>")
        sys.exit(1)
    
    responding_columnist = sys.argv[1]
    system_prompt_id = sys.argv[2]
    questionnaire_dir = sys.argv[3]

    system_message = ""

    if system_prompt_id == "claims":
        system_message = (
            "Sen bir Türk köşe yazarısın. "
            "Görevin verilen iddiayı destekleyip desteklemediğini belirtmek "
            "ve gerekçesini açıklamaktır.\n\n"
            "Aşağıda verilen iddiayı destekliyor musunuz? Lütfen önce 'Evet' veya 'Hayır' olarak cevaplayıp ardından gerekçesini açıklayın."
        )
    
    elif system_prompt_id == "questions":
        system_message = "Sen bir Türk köşe yazarısın. Görevin sorulan soru hakkındaki fikrini ve gerekçesini açıklamaktır."
    
    else:
        print("Invalid system prompt id! Possible values: claims, questions.")
        sys.exit(1)


    questionnaire_file = f"{questionnaire_dir}/questionnaire.json"
    
    try:
        with open(questionnaire_file, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
    except Exception as e:
        print("Questionnaire directory must contain questionnaire.json")
        print(f"Error reading {questionnaire_file}: {e}")
        sys.exit(1)
    
    # Load the model and tokenizer once
    tokenizer, model = load_model(responding_columnist)

    # Prepare a dictionary to hold answers
    answers = {}
    output_file = f"{questionnaire_dir}/{responding_columnist}_answers.json"
    
    # Process each question from the JSON file
    for q_key, question in tqdm(questions_data.items()):
        try:
            response = predict_response(tokenizer, model, system_message, question)
            answers[q_key] = {
                "question": question,
                "response": response
            }
            
            # Save the answers to file after each question
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(answers, out_f, ensure_ascii=False, indent=4)
            
        except Exception as e:
            print(f"Error processing {q_key}: {e}")
    
    print(f"All answers saved to {output_file}")

if __name__ == "__main__":
    main()
