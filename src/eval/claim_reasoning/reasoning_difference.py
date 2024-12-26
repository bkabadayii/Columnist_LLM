import json
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Function to calculate BERT similarities
def calculate_bert_similarity(reasoning_texts, response_texts):
    _, _, F1 = score(response_texts, reasoning_texts, lang="tr", verbose=True)
    return F1.tolist()


# Main function to compute and save scores
def compute_and_save_similarity(input_file, output_file):
    # Load dataset
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Convert JSON data to DataFrame
    df = pd.json_normalize(data)

    # Initialize results
    scores = []

    # Iterate through all claims
    for _, row in df.iterrows():
        claim_data = {
            "claim_owner": row["claim_owner"],
            "claim": row["claim"],
            "agreement": row["agreement"],
            "reasoning": row["reasoning"],
            "bert_similarities": {}
        }

        # Compute similarities for all columnist responses
        for column in df.columns:
            if column.endswith("_response"):
                response_columnist = column.replace("_response", "")
                response = row.get(column, None)
                response = response.split("\n")[0]

                if pd.notna(response):
                    bert_sim = calculate_bert_similarity([row["reasoning"]], [response])[0]
                    claim_data["bert_similarities"][response_columnist] = {
                        "response": response,
                        "similarity": bert_sim
                    }

        # Append claim data
        scores.append(claim_data)

    # Save scores to a JSON file
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(scores, out_file, ensure_ascii=False, indent=4)
    print(f"Similarity scores saved to {output_file}.")


def plot_similarity_distributions(similarity_file, target_columnist, columnist_list, filter_agreement=None):
    """
    Plot similarity distributions for a target columnist's claims based on the responses of specified columnists.

    Args:
        similarity_file (str): Path to the JSON file with similarity scores.
        target_columnist (str): Name of the columnist whose claims will be used.
        columnist_list (list): List of columnists whose response similarities will be plotted.
    """
    # Load similarity data
    with open(similarity_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Filter claims by the target columnist
    target_claims = [claim for claim in data if claim["claim_owner"] == target_columnist]

    if filter_agreement:
        target_claims = [claim for claim in target_claims if claim["agreement"] == filter_agreement]

    # Prepare a dictionary to store similarities for each columnist
    similarities = {columnist: [] for columnist in columnist_list}

    # Collect similarities for each columnist in the list
    for claim in target_claims:
        for columnist in columnist_list:
            if columnist in claim["bert_similarities"]:
                similarities[columnist].append(claim["bert_similarities"][columnist]["similarity"])

    # Save similarities to a json file
    with open(f"../../../eval_results/claim_reasoning/distributions/{target_columnist}_similarity_distribution.json", "w", encoding="utf-8") as out_file:
        json.dump(similarities, out_file, ensure_ascii=False, indent=4)


    # Plot normalized histograms
    plt.figure(figsize=(10, 6))
    for columnist, sims in similarities.items():
        if sims:  # Only plot if there are similarities to show
            weights = [1 / len(sims)] * len(sims)  # Normalize the histogram
            plt.hist(
                sims, bins=10, alpha=0.7, weights=weights, label=f"Responses by {columnist}"
            )

    # Customize plot
    plt.title(f"Similarity Distributions for {target_columnist}'s Claims")
    plt.xlabel("BERT Similarity Score")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_similarity_distributions_side_by_side(
    similarity_file, target_columnists, columnist_list, filter_agreement=None
):
    """
    Plot similarity distributions for multiple target columnists side by side.

    Args:
        similarity_file (str): Path to the JSON file with similarity scores.
        target_columnists (list): List of columnists whose claims will be plotted side by side.
        columnist_list (list): List of columnists whose response similarities will be plotted.
        filter_agreement (str, optional): Filter claims by agreement value (e.g., "agree", "disagree").
    """
    # Load similarity data
    with open(similarity_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Create subplots
    fig, axes = plt.subplots(1, len(target_columnists), figsize=(15, 6), sharey=True)
    if len(target_columnists) == 1:
        axes = [axes]  # Ensure axes is a list when only one subplot is created

    for ax, target_columnist in zip(axes, target_columnists):
        # Filter claims by the target columnist
        target_claims = [claim for claim in data if claim["claim_owner"] == target_columnist]

        if filter_agreement:
            target_claims = [claim for claim in target_claims if claim["agreement"] == filter_agreement]

        # Prepare a dictionary to store similarities for each columnist
        similarities = {columnist: [] for columnist in columnist_list}

        # Collect similarities for each columnist in the list
        for claim in target_claims:
            for columnist in columnist_list:
                if columnist in claim["bert_similarities"]:
                    similarities[columnist].append(claim["bert_similarities"][columnist]["similarity"])

        # Plot normalized histograms
        for columnist, sims in similarities.items():
            if sims:  # Only plot if there are similarities to show
                weights = [1 / len(sims)] * len(sims)  # Normalize the histogram
                ax.hist(
                    sims, bins=10, alpha=0.7, weights=weights, label=f"Responses by {columnist}"
                )

        # Customize each subplot
        ax.set_title(f"{target_columnist}'s Claims")
        ax.set_xlabel("BERT Similarity Score")
        ax.set_ylabel("Normalized Frequency")
        ax.legend()
        ax.grid(True)

    # Add a shared title for the figure
    plt.suptitle("Similarity Distributions Across Columnists", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_similarity_distribution_curve(similarity_file, target_columnist, columnist_list, filter_agreement=None):
    """
    Plot similarity distributions with KDE curves for a target columnist's claims
    based on the responses of specified columnists.

    Args:
        similarity_file (str): Path to the JSON file with similarity scores.
        target_columnist (str): Name of the columnist whose claims will be used.
        columnist_list (list): List of columnists whose response similarities will be plotted.
    """
    # Load similarity data
    with open(similarity_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Filter claims by the target columnist
    target_claims = [claim for claim in data if claim["claim_owner"] == target_columnist]
    if filter_agreement:
        target_claims = [claim for claim in target_claims if claim["agreement"] == filter_agreement]

    # Prepare a dictionary to store similarities for each columnist
    similarities = {columnist: [] for columnist in columnist_list}

    # Collect similarities for each columnist in the list
    for claim in target_claims:
        for columnist in columnist_list:
            if columnist in claim["bert_similarities"]:
                similarities[columnist].append(claim["bert_similarities"][columnist]["similarity"])

    # Plot normalized histograms with KDE curves
    plt.figure(figsize=(12, 8))
    for columnist, sims in similarities.items():
        if sims:  # Only plot if there are similarities to show
            sns.kdeplot(
                sims,
                label=f"Responses by {columnist}",
                shade=True,
                alpha=0.6,
            )

    # Customize plot
    plt.title(f"Similarity Distributions with KDE for {target_columnist}'s Claims", fontsize=16)
    plt.xlabel("BERT Similarity Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()




if __name__ == "__main__":
    # Inputs
    input_file = "../../../prediction_results/claim_reasoning/predictions.json"
    similarity_file = "similarity_scores.json"

    target_columnist = "mehmettezkan"  # Whose claims we are considering
    columnist_list = ["hilalkaplan", "ismailsaymaz", "mehmettezkan"]  # List of columnists whose responses we want to plot
    filter_agreement = "Evet"

    if os.path.exists(similarity_file):
        print(f"Similarity scores already computed and saved in {similarity_file}.")
    else:
        # Run computation and save results
        compute_and_save_similarity(input_file, similarity_file)
    
    # Plot similarity distributions
    # plot_similarity_distributions(similarity_file, target_columnist, columnist_list, filter_agreement)
    plot_similarity_distributions_side_by_side(similarity_file, columnist_list, columnist_list, filter_agreement)
    # Plot similarity distributions with KDE curves
    # plot_similarity_distribution_curve(similarity_file, target_columnist, columnist_list, filter_agreement)
    
