import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Function to read the CSV file and create a mapping of article_id to content
def read_articles(csv_file):
    articles_df = pd.read_csv(csv_file)
    articles_df["article_id"] = articles_df["article_id"].astype(int)  # Ensure article_id is integer
    return articles_df.set_index("article_id")["Article Content"].to_dict()

# Load existing results if available
def load_existing_results(output_file):
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

# Function to find an article's data in existing results
def find_article(existing_results, article_id):
    for article in existing_results:
        if article["article_id"] == article_id:
            return article
    return None

# Function to generate reasoning using OpenAI API
def generate_reason(article, claim, reference):
    stance = "support" if reference == "Yes" else "reject"
    prompt = f"""
Given the following article:
{article}

Why does the author {stance} the following claim: "{claim}"?

Give your answer as you are the author. You must use their specific tone, word choices, and opinions. Make sure your response is clear and understandable to someone who has not read the original article. Avoid direct references to the writing. Instead of covering everything in the article, focus only on the parts relevant to the question. Limit your response to a maximum of 100 words.
    """
    system_prompt = "You are a Turkish columnist. You will be asked to provide reasoning for a claim. Your response must be in Turkish."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        reasoning = response.choices[0].message.content.strip()
        token_usage = response.usage.total_tokens
        return reasoning, token_usage
    except Exception as e:
        print(f"Error generating reason: {e}")
        return "Reasoning could not be generated."

# Save progress immediately to avoid data loss
def save_progress(data, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Function to add reasoning to claims with real-time saving
def add_reasons_to_claims(claims_file, articles, output_file):
    # Load existing results
    existing_results = load_existing_results(output_file)

    # Load the claims data
    with open(claims_file, "r", encoding="utf-8") as file:
        claims_data = json.load(file)

    total_token_usage = 0  # Track total token usage

    # Iterate over each article in claims_data
    for article in tqdm(claims_data, desc="Processing Articles", unit="article"):
        article_id = article["article_id"]
        
        # Find the article in the existing results
        existing_article = find_article(existing_results, article_id)
        if not existing_article:
            existing_article = {"article_id": article_id, "claims": []}
            existing_results.append(existing_article)
        
        # Retrieve the article content
        article_content = articles.get(article_id, "")
        
        # Iterate over claims within the article
        for i, claim in enumerate(tqdm(article["claims"], desc=f"Processing Claims in Article {article_id}", leave=False, unit="claim", total=len(article["claims"]))):
            claim_id = claim["claim_id"]
            
            # Check if this claim already has a reason
            if not any(c["claim_id"] == claim_id for c in existing_article["claims"]):
                # Generate reasoning
                reasoning, token_usage = generate_reason(article_content, claim["claim"], claim["reference"])
                
                # Update total token usage
                total_token_usage += token_usage
                
                # Prepare the claim with reasoning
                claim_with_reason = {
                    "claim_id": claim_id,
                    "claim": claim["claim"],
                    "reference": claim["reference"],
                    "reason": reasoning
                }
                
                # Append the claim to the article's claims list
                existing_article["claims"].append(claim_with_reason)
                
                # Save progress after each new reason is added
                save_progress(existing_results, output_file)
        
        # Update tqdm description with cost information
        tqdm.write(f"Cost until now: ${total_token_usage * 0.15 / 1_000_000:.2f}")

    print(f"Total token usage: {total_token_usage}")
    print(f"Updated claims with reasons saved to {output_file}")

# Main function
if __name__ == "__main__":
    # Input and output file paths
    csv_file = "../../../columnist_data/cleaned_articles/hilalkaplan_cleaned_articles.csv"
    claims_json_file = "../../../columnist_data/claims_json/hilalkaplan_claims.json"
    output_file = "../../../finetune_data/hilalkaplan_claims/hilalkaplan_claims.json"

    # Read the articles from the CSV using pandas
    articles = read_articles(csv_file)

    # Add reasoning to claims with real-time saving
    add_reasons_to_claims(claims_json_file, articles, output_file)
