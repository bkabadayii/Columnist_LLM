import json
import string
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt

def calculate_named_entity_agreement(results_file, models):
    """
    Calculate agreement percentages for named entities based on model predictions.

    Args:
        results_file (str): Path to the JSON results file.
        models (list of str): List of model prediction keys.

    Returns:
        dict: Dictionary with named entities and their agreement statistics.
    """
    # Load the data from the JSON file
    with open(results_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Initialize named entity statistics
    entity_stats = defaultdict(lambda: {"total_claims": 0, "agreement_count": 0})

    # Punctuation removal table
    table = str.maketrans("", "", string.punctuation)

    # Process each claim
    for article in data:
        for claim in article["claims"]:
            # Drop punctuation from lemmatized named entities and strip trailing spaces/punctuation
            lemmatized_entities = [
                entity.translate(table).strip() for entity in claim["lemmatized_named_entities"]
            ]
            
            # Check agreement between models without altering predictions
            agreements = claim[f"{models[0]}_predicted_agreement"] == claim[f"{models[1]}_predicted_agreement"]
            
            # Update statistics for each named entity
            for entity in lemmatized_entities:
                entity_stats[entity]["total_claims"] += 1
                if agreements:
                    entity_stats[entity]["agreement_count"] += 1

    # Calculate agreement percentages
    for entity, stats in entity_stats.items():
        total = stats["total_claims"]
        if total > 0:
            stats["agreement_percentage"] = round((stats["agreement_count"] / total) * 100, 2)
        else:
            stats["agreement_percentage"] = 0.0

    return entity_stats



def plot_top_entities_by_claims(entity_stats, n=10):
    """
    Plot the top N named entities with the highest number of claims.

    Args:
        entity_stats (dict): Dictionary with named entity statistics.
        n (int): Number of top entities to plot.
    """
    # Sort entities by total claims (descending), then by agreement percentage (secondary sorting)
    sorted_entities = sorted(
        entity_stats.items(),
        key=lambda item: (-item[1]["total_claims"], -item[1]["agreement_percentage"])
    )[:n]

    # Extract data for plotting
    entities = [entity for entity, stats in sorted_entities]
    total_claims = [stats["total_claims"] for _, stats in sorted_entities]
    agreement_percentages = [stats["agreement_percentage"] for _, stats in sorted_entities]

    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.barh(entities, total_claims, color="skyblue")
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest claims at the top

    # Annotate bars with agreement percentages
    for bar, agreement in zip(bars, agreement_percentages):
        plt.text(
            bar.get_width() + 1,  # Position the text at the end of the bar
            bar.get_y() + bar.get_height() / 2,
            f"{agreement}%",
            va="center",
            fontsize=10
        )

    # Add labels and title
    plt.xlabel("Total Claims")
    plt.ylabel("Named Entities")
    plt.title("Top Named Entities by Number of Claims with Agreement Percentages")
    plt.savefig("../../eval_results/claims/ner/top_entities.png")
    plt.tight_layout()
    plt.show()


def plot_least_agreement_by_percentage(entity_stats, n=10, claim_threshold=10):
    """
    Plot the least agreement percentages for named entities with a claim threshold.

    Args:
        entity_stats (dict): Dictionary with named entity statistics.
        n (int): Number of entities to plot.
        claim_threshold (int): Minimum number of claims for an entity to be considered.
    """
    # Filter entities by claim threshold
    filtered_entities = {entity: stats for entity, stats in entity_stats.items() if stats["total_claims"] > claim_threshold}
    
    # Sort entities by agreement percentage (ascending)
    sorted_entities = sorted(
        filtered_entities.items(),
        key=lambda item: item[1]["agreement_percentage"]
    )[:n]

    # Extract data for plotting
    entities = [entity for entity, stats in sorted_entities]
    agreement_percentages = [stats["agreement_percentage"] for _, stats in sorted_entities]
    total_claims = [stats["total_claims"] for _, stats in sorted_entities]

    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.barh(entities, agreement_percentages, color="salmon")
    plt.gca().invert_yaxis()  # Invert y-axis to have the least agreements at the top

    # Annotate bars with total claims
    for bar, total in zip(bars, total_claims):
        plt.text(
            bar.get_width() + 1,  # Position the text at the end of the bar
            bar.get_y() + bar.get_height() / 2,
            f"{total}",
            va="center",
            fontsize=10
        )

    # Add labels and title
    plt.xlabel("Agreement Percentage")
    plt.ylabel("Named Entities")
    plt.title("Least Agreement Percentages for Named Entities (Total Claims > 10)")
    plt.tight_layout()
    plt.savefig("../../eval_results/claims/ner/least_agreements.png")
    plt.show()


def plot_top_words_for_entity(results_file, target_entity, top_n=10):
    """
    Plot the top words in claims that include a given lemmatized named entity.

    Args:
        results_file (str): Path to the JSON results file.
        target_entity (str): The lemmatized named entity to filter claims by.
        top_n (int): The number of top words to plot.
    """
    # Load the data from the JSON file
    with open(results_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Punctuation removal table
    table = str.maketrans("", "", string.punctuation)

    # Initialize word counter
    word_counter = Counter()

    # Process each claim
    for article in data:
        for claim in article["claims"]:
            # Drop punctuation and normalize lemmatized named entities
            lemmatized_entities = [
                entity.translate(table).strip() for entity in claim["lemmatized_named_entities"]
            ]

            # Check if the target entity is in the lemmatized named entities
            if target_entity in lemmatized_entities:
                # Tokenize the claim text, remove punctuation, and count words
                claim_text = claim["claim"].translate(table).lower()
                words = claim_text.split()
                word_counter.update(words)

    # Get the top N most common words
    top_words = word_counter.most_common(top_n)

    # Extract words and their counts for plotting
    words, counts = zip(*top_words)

    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(words, counts, color="lightblue")
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency at the top

    # Add labels and title
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title(f"Top {top_n} Words in Claims Containing Entity: {target_entity}")
    plt.tight_layout()
    plt.savefig(f"../../eval_results/claims/ner/top_words_{target_entity}.png")
    plt.show()


def plot_top_words_for_entity_by_reference(results_file, target_entity, top_n=10):
    """
    Plot the top N words in claims that include a given lemmatized named entity,
    showing their frequencies for 'reference' values ("Evet" and "Hayır") on the same plot.

    Args:
        results_file (str): Path to the JSON results file.
        target_entity (str): The lemmatized named entity to filter claims by.
        top_n (int): The number of top words to plot for each category ("Evet" and "Hayır").
    """
    # Load the data from the JSON file
    with open(results_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Punctuation removal table
    table = str.maketrans("", "", string.punctuation)

    # Initialize word counters for "Evet" and "Hayır"
    word_counter_evet = Counter()
    word_counter_hayir = Counter()

    # Process each claim
    for article in data:
        for claim in article["claims"]:
            # Drop punctuation and normalize lemmatized named entities
            lemmatized_entities = [
                entity.translate(table).strip() for entity in claim["lemmatized_named_entities"]
            ]

            # Check if the target entity is in the lemmatized named entities
            if target_entity in lemmatized_entities:
                # Tokenize the claim text, remove punctuation, and count words
                claim_text = claim["claim"].translate(table).lower()
                words = claim_text.split()

                # Update the corresponding word counter based on the 'reference' value
                if claim["reference"] == "Evet":
                    word_counter_evet.update(words)
                elif claim["reference"] == "Hayır":
                    word_counter_hayir.update(words)

    # Get the top N words for "Evet" and "Hayır"
    top_words_evet = word_counter_evet.most_common(top_n)
    top_words_hayir = word_counter_hayir.most_common(top_n)

    # Create a combined set of all top words
    all_words = list(set([word for word, _ in top_words_evet] + [word for word, _ in top_words_hayir]))

    # Collect word frequencies for both categories
    evet_counts = [word_counter_evet[word] for word in all_words]
    hayir_counts = [word_counter_hayir[word] for word in all_words]

    # Create the grouped bar chart
    x = range(len(all_words))  # Indices for the words
    bar_width = 0.4

    plt.figure(figsize=(12, 8))
    plt.bar([i - bar_width / 2 for i in x], evet_counts, width=bar_width, color="lightgreen", label="Evet")
    plt.bar([i + bar_width / 2 for i in x], hayir_counts, width=bar_width, color="lightcoral", label="Hayır")
    plt.xticks(x, all_words, rotation=45, ha="right")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} Words in Claims Containing Entity: {target_entity}")
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"../../eval_results/claims/ner/top_words_{target_entity}_eh.png")
    plt.show()


if __name__ == "__main__":
    # Define file path and models
    results_file = "../../prediction_results/train_ner.json"
    models = ["hilalkaplan", "ismailsaymaz"]

    # Calculate named entity agreement statistics
    # entity_agreement_stats = calculate_named_entity_agreement(results_file, models)

    # Plot top N named entities by the number of claims
    # plot_top_entities_by_claims(entity_agreement_stats, n=20)

    # Plot least agreement percentages with total claims annotated
    # plot_least_agreement_by_percentage(entity_agreement_stats, n=10, claim_threshold=10)

    # Plot top words for a given named entity
    target_entity = "chp"
    # plot_top_words_for_entity(results_file, target_entity, top_n=20)
    plot_top_words_for_entity_by_reference(results_file, target_entity, top_n=20)
    # plot_word_count_ratio(results_file, target_entity, top_n=20)