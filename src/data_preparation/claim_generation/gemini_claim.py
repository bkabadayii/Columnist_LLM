from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import pandas as pd
import time

load_dotenv()

claims_system_prompt = """You are an assistant that extracts subjective claims from the given article.

Claims must be self-contained, explanatory and should be clear enough without the article.
Claims must be built upon subjective stance of the author instead of factual information.
Each claim must be a long, detailed sentence about social or political events.
Each claim must be independent of each other.
Do not refer to the original article or the writer in any way.

Your output will be used to create a dataset for training a model that mimics a Turkish columnist.
So, make sure that claims include all necessary context and information.

Please give your answer as a list of Turkish claims:
Claim: ...
Claim: ...
..."""

contradicting_system_prompt = """You goal is to construct a claim that will contradict with the given claim.
This means you should transform the underlying meaning, perspective, and implications into its opposite, not merely by adding negations or changing verbs, but by constructing a logically opposite viewpoint.
Do not mention and reject the stance of the previous claim.
Maintain a similar tone and style to the original claim, expressing the opposing viewpoint as a direct statement.
Keep using similar words so that you do not introduce any bias."""

reasoning_system_prompt =  """You will be given an article and a claim. Your goal is to explain why does the author support / reject the given claim.
Start by explaining the author's stance on the claim (Simply Evet or Hayır for the first sentence).
Give your answer as you are the author. You must use their specific tone, word choices, and opinions.
Make sure your response is clear and understandable to someone who has not read the original article. 
Avoid direct references to the writing. 
Instead of covering everything in the article, focus only on the parts relevant to the question."""

# Configure the GenAI API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Define model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

def initialize_model(system_prompt):
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        system_instruction=system_prompt,
    )

# Function to process each article
def process_article(content, article_id):
    # Start a chat session for claims
    claims_model = initialize_model(claims_system_prompt)
    claims_chat_session = claims_model.start_chat()

    # Generate claims
    claims_response = claims_chat_session.send_message(content)
    claims = claims_response.text.splitlines()
    claims = [claim.replace("Claim: ", "").strip() for claim in claims if claim.startswith("Claim:")]

    # Start a new chat session for contradicting claims

    contradicting_claims = []
    for claim in claims:
        contradict_model = initialize_model(contradicting_system_prompt)
        contradict_chat_session = contradict_model.start_chat()
        contradict_prompt = f"Claim: {claim}"
        contradict_response = contradict_chat_session.send_message(contradict_prompt)
        contradicting_claim = contradict_response.text.strip()
        contradicting_claims.append(contradicting_claim)

    reasonings = []
    for claim in claims:
        # Start a new chat session for reasonings
        reasoning_model = initialize_model(reasoning_system_prompt)
        reasoning_chat_session = reasoning_model.start_chat()
        reasoning_prompt = f"Article: {content}\n\nClaim: {claim}"
        reasoning_response = reasoning_chat_session.send_message(reasoning_prompt)
        reasoning = reasoning_response.text.strip()
        reasonings.append(reasoning)

    contradict_reasonings = []
    for contradicting_claim in contradicting_claims:
        contradict_reasoning_model = initialize_model(reasoning_system_prompt)
        contradict_reasoning_chat_session = contradict_reasoning_model.start_chat()
        contradict_reasoning_prompt = f"Article: {content}\n\nContradicting Claim: {contradicting_claim}"
        contradict_reasoning_response = contradict_reasoning_chat_session.send_message(contradict_reasoning_prompt)
        contradict_reasoning = contradict_reasoning_response.text.strip()
        contradict_reasonings.append(contradict_reasoning)

    # Combine results
    results = {
        "article_id": article_id,
        "claims": claims,
        "reasonings": reasonings,
        "contradicting_claims": contradicting_claims,
        "contradicting_reasonings": contradict_reasonings
    }

    return results

def process_csv(file_path, output_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Initialize results list
    all_results = []
    
    # Load already processed results if the output file exists
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            all_results = json.load(f)

    # Create a set of already processed article IDs
    processed_ids = {result["article_id"] for result in all_results}

    # Iterate through each row in the CSV
    for index, row in df.iterrows():
        article_id = row['article_id']

        # Skip already processed articles
        if article_id in processed_ids:
            print(f"Skipping already processed article ID: {article_id}")
            continue

        article_content = row['article_content']

        print(f"Processing article ID: {article_id}")

        try:
            # Process the article
            results = process_article(article_content, article_id)

            # Append the results
            all_results.append(results)
            # Save interim results to a file
            with open(output_path, 'w') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing article ID {article_id}: {e}")

    print("Processing complete. Results saved.")


def process_article_concurrently(article_data):
    try:
        article_id = article_data['article_id']
        content = article_data['article_content']
        return process_article(content, article_id)
    except Exception as e:
        print(f"Error processing article ID {article_data['article_id']}: {e}")
        return None

def process_csv_parallel(file_path, output_path, max_workers=20):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Initialize results list
    all_results = []
    
    # Load already processed results if the output file exists
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            all_results = json.load(f)

    # Create a set of already processed article IDs
    processed_ids = {result["article_id"] for result in all_results}

    # Filter articles to process
    articles_to_process = df[~df['article_id'].isin(processed_ids)].to_dict(orient='records')

    # Process articles in batches with a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for article_data in articles_to_process:
            futures.append(executor.submit(process_article_concurrently, article_data))

        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)

                # Save interim results to a file
                with open(output_path, 'w') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
                
                print(f"Processed articles: {len(all_results)}/{len(articles_to_process)}%")

    # At the end sort articles and save again
    all_results.sort(key=lambda x: x['article_id'])
    with open(output_path, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("Processing complete. Results saved.")

def main():
    columnist_name = "fehimtastekin"
    input_csv = f"../../../columnist_data/{columnist_name}/cleaned_articles.csv"
    output_json = f"../../../columnist_data/claim_reasoning/{columnist_name}.json"

    # process_csv(input_csv, output_json)
    process_csv_parallel(input_csv, output_json)


if __name__ == "__main__":
    main()