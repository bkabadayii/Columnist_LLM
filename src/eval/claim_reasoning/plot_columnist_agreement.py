import json
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import math

def calculate_f1_scores(results_file, models, actual_columnists):
    """
    Calculate F1 scores for model predictions against actual columnists.

    Args:
        results_file (str): Path to the JSON results file.
        models (list of str): List of model prediction keys in the JSON.
        actual_columnists (list of str): List of actual columnist names.

    Returns:
        dict: Nested dictionary with F1 scores.
    """
    # Load the data from the JSON file
    with open(results_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Initialize the F1 score matrix
    f1_scores = {model: {columnist: 0 for columnist in actual_columnists} for model in models}

    # Compute F1 scores for each model vs actual columnist
    for model in models:
        for columnist in actual_columnists:
            # Filter claims for the specific columnist
            y_true = []
            y_pred = []
            for claim in data:
                predicted_agreement = claim[f"{model}_response"].split(".")[0]
                if claim["claim_owner"] == columnist:
                    y_true.append(1 if claim["agreement"] == "Evet" else 0)
                    y_pred.append(1 if predicted_agreement == "Evet" else 0)
            # Calculate F1 score only if there are valid entries
            if y_true and y_pred:
                f1_scores[model][columnist] = f1_score(y_true, y_pred)

    return f1_scores


def calculate_accuracy_scores(results_file, models, actual_columnists, filter_agreement=None):
    """
    Calculate accuracy scores for model predictions against actual columnists.

    Args:
        results_file (str): Path to the JSON results file.
        models (list of str): List of model prediction keys in the JSON.
        actual_columnists (list of str): List of actual columnist names.
        filter_agreement (str): Optional agreement to filter claims.

    Returns:
        dict: Nested dictionary with accuracy scores.
    """
    # Load the data from the JSON file
    with open(results_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Initialize the accuracy score matrix
    accuracy_scores = {model: {columnist: 0 for columnist in actual_columnists} for model in models}

    # Compute accuracy scores for each model vs actual columnist
    for model in models:
        for columnist in actual_columnists:
            # Filter claims for the specific columnist
            y_true = []
            y_pred = []

            for claim in data:
                predicted_agreement = claim[f"{model}_response"].split(".")[0]
                if claim["claim_owner"] == columnist and (not filter_agreement or claim["agreement"] == filter_agreement):
                    # Check if the predicted agreement is valid
                    if (predicted_agreement != "Evet") and (predicted_agreement != "Hayır"):
                        print(f"Found unknown agreement: {predicted_agreement}")

                    y_true.append(1 if claim["agreement"] == "Evet" else 0)
                    y_pred.append(1 if predicted_agreement == "Evet" else 0)

            # Calculate accuracy score only if there are valid entries
            if y_true and y_pred:
                accuracy_scores[model][columnist] = accuracy_score(y_true, y_pred)

    return accuracy_scores

def plot_confusion_matrix(scores, models, actual_columnists, out_path):
    """
    Plot a 2x2 confusion matrix with F1 scores.

    Args:
        scores (dict): Nested dictionary with scores.
        models (list of str): List of model prediction keys.
        actual_columnists (list of str): List of actual columnist names.
    """
    # Prepare data for the plot
    matrix = np.array([[scores[model][columnist] for columnist in actual_columnists] for model in models])

    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(matrix, cmap="Blues")
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(range(len(actual_columnists)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(actual_columnists, rotation=45, ha="left")
    ax.set_yticklabels(models)

    # Add F1 score values in each cell
    for i in range(len(models)):
        for j in range(len(actual_columnists)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", va="center", ha="center", color="black")


    plt.xlabel("Columnists")
    plt.ylabel("Models")
    plt.title("Model vs Columnist Agreement")
    plt.savefig(out_path)
    plt.show()


def single_plot(metric="accuracy"):
    # Define file paths and models
    results_file = "../../../prediction_results/claim_reasoning/predictions.json"
    out_path = f"../../../eval_results/claim_reasoning/columnist_agreement.png"

    models = ["hilalkaplan", "ismailsaymaz", "mehmettezkan", "base"]
    actual_columnists = ["hilalkaplan", "ismailsaymaz", "mehmettezkan"]

    filter_agreement = "Evet"

    scores = []

    if (metric == "accuracy"):
        scores = calculate_accuracy_scores(results_file, models, actual_columnists, filter_agreement=filter_agreement)
    elif (metric == "f1"):
        scores = calculate_f1_scores(results_file, models, actual_columnists)
    else:
        print("Invalid metric")
        return

    # Plot the confusion matrix for model vs actual columnist
    plot_confusion_matrix(
        scores=scores,
        models=models,
        actual_columnists=actual_columnists,
        out_path=out_path
    )


if __name__ == "__main__":
    single_plot("accuracy")