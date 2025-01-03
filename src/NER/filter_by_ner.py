import json
from collections import defaultdict

# Function to process and filter claims 
def process_claims(data, primary_owner, secondary_owner, lemmatized_entities_criteria, modify_claims=True):
    processed_claims = defaultdict(list)
    for article in data:
        article_id = article["article_id"]
        for claim in article.get("claims", []):
            # Filter claims by lemmatized entities
            if any(entity in claim["lemmatized_named_entities"] for entity in lemmatized_entities_criteria):
                # Include claims for primary owner with reference = "Evet"
                if modify_claims and claim["claim_owner"] == primary_owner and claim["reference"] == "Evet":
                    processed_claims[article_id].append(claim)
                # Include claims for secondary owner with reference = "Evet", but save as reference = "Hayır"
                elif modify_claims and claim["claim_owner"] == secondary_owner and claim["reference"] == "Evet":
                    modified_claim = claim.copy()
                    modified_claim["reference"] = "Hayır"
                    processed_claims[article_id].append(modified_claim)
                else:
                    processed_claims[article_id].append(claim)
    return processed_claims


# Function to filter claims
def filter_claims(data, primary_owner, secondary_owner, lemmatized_entities_criteria):
    filtered_claims = defaultdict(list)
    for article in data:
        article_id = article["article_id"]
        for claim in article.get("claims", []):
            # Filter claims by lemmatized entities
            if any(entity in claim["lemmatized_named_entities"] for entity in lemmatized_entities_criteria):
                # Include claims for primary owner with reference = "Evet"
                if claim["claim_owner"] == primary_owner:
                    filtered_claims[article_id].append(claim)

    return filtered_claims

# Load the data
with open("../../prediction_results/train_ner.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

# Define lemmatized entities list for filtering
lemmatized_entities_criteria = [
    "akp",
    "chp",
    "ak parti",
    "cumhuriyet halk partisi",
    "erdoğan",
    "kılıçdaroğlu",
    "tayyip erdoğan",
    "kemal kılıçdaroğlu",
    "recep tayyip erdoğan",
]

# Process claims for hilalkaplan
hilalkaplan_claims = filter_claims(all_data, "hilalkaplan", "ismailsaymaz", lemmatized_entities_criteria)

# Process claims for ismailsaymaz
ismailsaymaz_claims = filter_claims(all_data, "ismailsaymaz", "hilalkaplan", lemmatized_entities_criteria)

# Prepare final output format for each owner
hilalkaplan_output = [{"article_id": key, "claims": value} for key, value in hilalkaplan_claims.items()]
ismailsaymaz_output = [{"article_id": key, "claims": value} for key, value in ismailsaymaz_claims.items()]

# Save results to separate files
with open("../../finetune_data/ner_filtered/hilalkaplan.json", "w", encoding="utf-8") as f:
    json.dump(hilalkaplan_output, f, ensure_ascii=False, indent=4)

with open("../../finetune_data/ner_filtered/ismailsaymaz.json", "w", encoding="utf-8") as f:
    json.dump(ismailsaymaz_output, f, ensure_ascii=False, indent=4)

print("Filtered claims saved for hilalkaplan and ismailsaymaz.")