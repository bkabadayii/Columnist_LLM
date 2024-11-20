import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_bert_metric_distribution(results_files, metric="f1", plot_title=None, distribution_titles=None, save_path=None):
    """
    Plots distributions of a specified BERT metric (F1, precision, or recall) 
    for multiple results JSON files on the same plot.
    
    Args:
        results_files (list): List of paths to the JSON files containing the BERT score results.
        metric (str): The metric to plot. Options are 'f1', 'precision', or 'recall'.
        plot_title (str): Title for the plot. If None, a default title will be used.
        distribution_titles (list): List of titles for each distribution. If None, filenames will be used.
    """
    # Validate the metric argument
    if metric not in ["f1", "precision", "recall"]:
        print("Error: Invalid metric specified. Use 'f1', 'precision', or 'recall'.")
        return

    if not isinstance(results_files, list) or (distribution_titles and not isinstance(distribution_titles, list)):
        print("Error: 'results_files' and 'distribution_titles' should be lists.")
        return

    if distribution_titles and len(results_files) != len(distribution_titles):
        print("Error: The number of plot titles must match the number of results files.")
        return

    # Create a figure for the combined plot
    plt.figure(figsize=(12, 8))
    
    # Loop through each results file and plot its distribution
    for idx, results_file in enumerate(results_files):
        # Load the JSON file
        if not os.path.exists(results_file):
            print(f"Error: File not found - {results_file}")
            continue

        with open(results_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Extract the specified metric scores
        metric_scores = [item[metric] for item in data['detailed_results'] if metric in item]

        if not metric_scores:
            print(f"Error: No {metric} scores found in {results_file}.")
            continue

        # Set the plot label
        label = distribution_titles[idx] if distribution_titles else os.path.basename(results_file)

        # Plot the distribution for the current file
        sns.histplot(metric_scores, kde=True, bins=30, label=label, alpha=0.6)

    # Set the plot title if not provided
    if not plot_title:
        plot_title = f"BERT {metric.capitalize()} Score Distribution Across Multiple Results"

    # Customize the plot
    plt.xlabel(f"BERT {metric.capitalize()} Score")
    plt.ylabel("Count")
    plt.title(plot_title)
    plt.legend(title="Datasets")
    plt.grid(True)

    # Save the plot if a save path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":

    # Specify the plot title
    plot_title = "BERT F1 Score Distribution Hilal Kaplan vs. Hilal Kaplan Conversation"

    # Specify the metric you want to plot: "f1", "precision", or "recall"
    metric_to_plot = "f1"

    # List of results files
    results_files = [
        "../../eval_results/ytu/bert_scores/regular/hilalkaplan_bert_scores.json",
        "../../eval_results/ytu/bert_scores/regular/hilalkaplan_conversation_bert_scores.json"
    ]

    # List of plot titles corresponding to the results files
    distribution_titles = [
        "Hilal Kaplan Results",
        "Hilal Kaplan Conversation Results",
    ]

    # Path to save the plot
    save_path = f"../../eval_results/ytu/bert_scores/distributions/hk_vs_hkconv_{metric_to_plot}_distributions.png"

    # Call the function with multiple files
    plot_bert_metric_distribution(
        results_files=results_files,
        metric=metric_to_plot,
        plot_title=plot_title,
        distribution_titles=distribution_titles,
        save_path=save_path
    )
