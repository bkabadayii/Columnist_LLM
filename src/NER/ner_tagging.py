# Importing as module.
install = "!pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_lg/resolve/main/tr_core_news_lg-1.0-py3-none-any.whl"
import tr_core_news_lg
import json


# Function to process claims and add named entities
def process_claims(data):
    nlp = tr_core_news_lg.load()

    for article in data:
        for claim in article.get("claims", []):
            claim_text = claim["claim"]
            doc = nlp(claim_text)

            named_entities = []
            lemmatized_entities = []

            for ent in doc.ents:
                named_entities.append(ent.text.lower())
                lemmatized_entities.append(" ".join([token.lemma_.lower() for token in ent]))

            claim["named_entities"] = named_entities
            claim["lemmatized_named_entities"] = lemmatized_entities

    return data

# Read the input JSON file
input_file = "merged_results.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process the claims
processed_data = process_claims(data)

# Save the updated data to a new JSON file
output_file = "ner.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

print(f"Updated data saved to {output_file}")
