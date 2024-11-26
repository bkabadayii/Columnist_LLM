import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import pandas as pd
import time

# Load environment variables
load_dotenv()
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# Define system prompts
system_prompt = """You are an assistant that extracts subjective claims from the given article.

Claims must be self-contained, explanatory and should be clear enough without the article.
Claims must be built upon subjective stance of the author instead of factual information.
Each claim must be a long, detailed sentence about social or political events.
Each claim must be independent of each other.
Do not refer to the original article or the writer in any way.

Your output will be used to create a dataset for training a model that mimics a Turkish columnist.
So, make sure that claims include all necessary context and information.

A good example: Türkiye, Cumhurbaşkanı Erdoğan'ın liderliğinde, Karabağ ve Libya'da olduğu gibi, Filistin'de de benzer bir müdahale gücüne sahip olmalı ve İsrail’in Filistinlilere yönelik zulmüne karşı koyabilecek askeri ve siyasi bir kapasite geliştirmelidir.

Please give your answer as a list of Turkish claims:
Claim: ...
Claim: ...
..."""

second_prompt = """Now for each claim, construct a claim that will contradict with the given claim.
This means you should transform the underlying meaning, perspective, and implications into its opposite, not merely by adding negations or changing verbs, but by constructing a logically and contextually opposite viewpoint.
Do not mention and reject the stance of the previous claims. Claims and contradicting claims should be independent of each other.

A good example: Türkiye'nin Karabağ ve Libya'da izlediği müdahale politikaları bölgedeki istikrarsızlığı artırmış ve Filistin’de benzer bir yaklaşım izlemek, bölgedeki çatışmaları daha da derinleştirmekten başka bir işe yaramayacaktır.

Please give your answer as a list of Turkish contradicting claims:
Contradicting Claim: ...
Contradicting Claim: ...
..."""

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

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction=system_prompt,
)

# Function to process each article
def process_article(content):
    chat_session = model.start_chat()
    
    # Generate claims
    claims_response = chat_session.send_message(content)
    claims = claims_response.text.splitlines()
    claims = [claim.replace("Claim: ", "").strip() for claim in claims if claim.startswith("Claim:")]
    
    # Generate contradicting claims
    contradicting_response = chat_session.send_message(second_prompt)
    contradicting_claims = contradicting_response.text.splitlines()
    contradicting_claims = [
        claim.replace("Contradicting Claim: ", "").strip()
        for claim in contradicting_claims
        if claim.startswith("Contradicting Claim:")
    ]
    
    # Combine claims and contradicting claims into JSON format
    results = []
    for i, claim in enumerate(claims):
        results.append({
            "claim_id": f"1.{i+1}",
            "claim": claim,
            "reference": "Evet",
        })
    
    for i, contradicting_claim in enumerate(contradicting_claims):
        results.append({
            "claim_id": f"2.{i+1}",
            "claim": contradicting_claim,
            "reference": "Hayır",
        })
    
    return results


if __name__ == "__main__":
    # Input and output files
    input_csv = "../../../columnist_data/cleaned_articles/hilalkaplan_cleaned_articles.csv"
    output_json = "output.json"

    # User input for starting and ending rows, 1-indexed, both inclusive
    start_row = 1 
    end_row = 400

    # Read the CSV using Pandas
    data = pd.read_csv(input_csv)

    # Filter rows based on user input
    data_subset = data.iloc[start_row - 1:end_row]

    output_data = []
    for idx, row in data_subset.iterrows():
        article_id = row["article_id"]
        content = row["Article Content"]
        
        # Process each article and append results
        article_results = process_article(content)

        claim_lengths = [len(claim["claim"]) for claim in article_results if claim["reference"] == "Evet"]
        contradicting_claim_lengths = [len(claim["claim"]) for claim in article_results if claim["reference"] == "Hayır"]

        output_data.append({
            "article_id": article_id,
            "claims": article_results,
            "claim_lengths": claim_lengths,
            "contradicting_claim_lengths": contradicting_claim_lengths,
        })

        # Save results to JSON
        with open(output_json, "w", encoding="utf-8") as jsonfile:
            json.dump(output_data, jsonfile, ensure_ascii=False, indent=4)

        time.sleep(27)

    print(f"Processing complete. Results for rows {start_row + 1} to {end_row + 1} saved to {output_json}.")
