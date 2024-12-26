import json
from scipy.stats import ks_2samp

# Load the JSON file
file_path = "../../../eval_results/claim_reasoning/distributions/ismailsaymaz_similarity_distribution.json"

with open(file_path, "r") as f:
    data = json.load(f)

# Select two keys
key1 = "mehmettezkan"
key2 = "hilalkaplan"

# Retrieve their corresponding values
values1 = data[key1]
values2 = data[key2]

# Perform the KS test
statistic, p_value = ks_2samp(values1, values2)

# Output the results
print(f"KS Statistic: {statistic}")
print(f"P-Value: {p_value}")
