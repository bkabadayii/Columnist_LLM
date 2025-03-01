import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# Configure the GenAI API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Prompt template for generating questions
generate_question_prompt = """I am working on fine-tuning a language model and need to generate a meaningful and engaging question where the provided claims can serve as the answer. The question should:

Be asked from a politically neutral point of view.
Provide enough context or detail to make the question specific to the claim.
Should aim for asking personal opinions.

The question must be a single Turkish sentence and provided in the following format:
Soru: ...
..."""

# Define model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 128,
    "response_mime_type": "text/plain",
}


def generate_question(claim, contradicting_claim):
    """
    Generate a question using the claim and contradicting claim via the Gemini API.

    Args:
        claim (str): The primary claim.
        contradicting_claim (str): The contradicting claim.

    Returns:
        str: The generated question.
    """
    input_prompt = f"1. {claim}\n2. {contradicting_claim}"

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        system_instruction=generate_question_prompt,
    )

    chat = model.start_chat()
    response = chat.send_message(input_prompt)
    return response.text.split(": ")[1].strip()


def process_article(article):
    """
    Process a single article to generate questions for claims.

    Args:
        article (dict): A dictionary containing article details.

    Returns:
        dict: Results with article_id and generated questions.
    """
    article_id = article["article_id"]
    claims = article["claims"]
    contradicting_claims = article["contradicting_claims"]

    results = article
    results["questions"] = []

    for claim, contradicting_claim in zip(claims, contradicting_claims):
        try:
            question = generate_question(claim, contradicting_claim)
            results["questions"].append(question)
        except Exception as e:
            results["questions"].append("ERROR")
            print(f"Error generating question for article {article_id}: {e}")

            continue

    return results


def process_json(file_path, output_path, max_workers=20):
    """
    Process a JSON file with articles to generate questions concurrently.

    Args:
        file_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
        max_workers (int): Number of concurrent threads.

    Returns:
        None
    """
    # Load the JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # Initialize results list
    all_results = []

    # Load already processed results if the output file exists
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    # Create a set of already processed article IDs
    processed_ids = {result["article_id"] for result in all_results}

    # Filter articles to process
    articles_to_process = [
        article for article in articles if article["article_id"] not in processed_ids
    ]

    # Process articles concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_article, article): article["article_id"]
            for article in articles_to_process
        }

        for future in as_completed(futures):
            article_id = futures[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)

                    # Save interim results to a file
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=4)

                print(f"Processed article ID: {article_id}")

            except Exception as e:
                print(f"Error processing article ID {article_id}: {e}")

    print("Processing complete. Results saved.")


def main():
    input_json = "../../../columnist_data/claim_reasoning/ahmethakan.json"
    output_json = "../../../columnist_data/claim_questions/ahmethakan.json"

    process_json(input_json, output_json)


if __name__ == "__main__":
    main()
