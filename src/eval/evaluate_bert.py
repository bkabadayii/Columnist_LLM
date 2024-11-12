import json
from bert_score import score
import random
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def evaluate_bert_score(json_file_path, output_file_path, max_articles=0):
    """
    Evaluates the BERT scores (precision, recall, F1) for up to `max_articles` QA pairs
    in the provided JSON file and saves the results and averages into an output JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing QA pairs.
        output_file_path (str): Path to save the JSON file with BERT score evaluations.
        max_articles (int): Maximum number of articles to evaluate.
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    detailed_results = []
    references = []
    predictions = []
    processed_articles = 0

    # Collect references, predictions, and other details, limit to max_articles
    for article in data:
        if max_articles and processed_articles >= max_articles:
            break
        for qa_pair_idx, qa_pair in enumerate(article['qa_pairs']):
            references.append(qa_pair['answer'])
            predictions.append(qa_pair['predicted_response'])
            detailed_results.append({
                "article_id": article.get("article_id"),
                "qa_pair_index": qa_pair_idx + 1,
                "question": qa_pair['question'],
                "reference": qa_pair['answer'],
                "prediction": qa_pair['predicted_response']
            })
        processed_articles += 1
    
    # Calculate BERT scores
    print(f"Calculating BERT scores for the first {max_articles} articles...")
    P, R, F1 = score(predictions, references, lang="tr", verbose=True)

    # Append scores to detailed results
    for i, (p, r, f1) in enumerate(zip(P, R, F1)):
        detailed_results[i].update({
            "precision": p.item(),
            "recall": r.item(),
            "f1": f1.item()
        })

    # Calculate average scores
    average_precision = P.mean().item()
    average_recall = R.mean().item()
    average_f1 = F1.mean().item()

    # Save results to output file
    output_data = {
        "evaluated_articles": max_articles,
        "detailed_results": detailed_results,
        "average_scores": {
            "average_precision": average_precision,
            "average_recall": average_recall,
            "average_f1": average_f1
        }
    }

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

    print("Evaluation completed. Results saved.")


def evaluate_random_bert_score(json_file_path, output_file_path, max_articles=0):
    """
    Evaluates BERT scores (precision, recall, F1) for randomly shuffled predictions
    from a JSON file containing QA pairs. Saves the results and averages into an output JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing QA pairs.
        output_file_path (str): Path to save the JSON file with random BERT score evaluations.
        max_articles (int): Maximum number of articles to evaluate. If it is 0, all articles are evaluated.
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    references = []
    predictions = []
    processed_articles = 0

    # Collect references and predictions, limit to max_articles
    for article in data:
        if max_articles and processed_articles >= max_articles:
            break
        for qa_pair in article['qa_pairs']:
            references.append(qa_pair['answer'])
            predictions.append(qa_pair['predicted_response'])
        processed_articles += 1
    
    # Shuffle predictions to create random pairs
    random_predictions = predictions.copy()
    random.shuffle(random_predictions)

    # Calculate BERT scores for randomly shuffled pairs
    print(f"Calculating BERT scores for randomly shuffled predictions for the first {max_articles} articles...")
    P_rand, R_rand, F1_rand = score(random_predictions, references, lang="tr", verbose=True)

    # Prepare results for saving
    detailed_results = []
    for i, (ref, pred, p, r, f1) in enumerate(zip(references, random_predictions, P_rand, R_rand, F1_rand)):
        detailed_results.append({
            "reference": ref,
            "random_prediction": pred,
            "precision": p.item(),
            "recall": r.item(),
            "f1": f1.item()
        })

    # Calculate average scores for random predictions
    average_precision_rand = P_rand.mean().item()
    average_recall_rand = R_rand.mean().item()
    average_f1_rand = F1_rand.mean().item()

    # Save results to output file
    output_data = {
        "evaluated_articles": max_articles,
        "detailed_results": detailed_results,
        "average_scores": {
            "average_precision": average_precision_rand,
            "average_recall": average_recall_rand,
            "average_f1": average_f1_rand
        }
    }

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

    print(f"Random BERT score evaluation completed. Results saved to {output_file_path}")


def evaluate_random_within_article_bert_score(json_file_path, output_file_path, max_articles=0):
    """
    Evaluates BERT scores (precision, recall, F1) for randomly shuffled predictions
    within each article from a JSON file containing QA pairs. Saves the results and averages into an output JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing QA pairs.
        output_file_path (str): Path to save the JSON file with random BERT score evaluations.
        max_articles (int): Maximum number of articles to evaluate. If it is 0, all articles are evaluated.
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    detailed_results = []
    total_references = []
    total_random_predictions = []
    processed_articles = 0

    # Process each article
    for article in data:
        if max_articles and processed_articles >= max_articles:
            break
        
        references = []
        predictions = []

        # Collect references and predictions for the current article
        for qa_pair in article['qa_pairs']:
            references.append(qa_pair['answer'])
            predictions.append(qa_pair['predicted_response'])
        
        # Shuffle predictions only within the current article
        random_predictions = predictions.copy()
        random.shuffle(random_predictions)

        # Calculate BERT scores for the shuffled pairs within the current article
        print(f"Calculating BERT scores for article {article.get('article_id', 'Unknown')}...")
        P_rand, R_rand, F1_rand = score(random_predictions, references, lang="tr", verbose=False)

        # Prepare results for the current article
        for i, (ref, pred, p, r, f1) in enumerate(zip(references, random_predictions, P_rand, R_rand, F1_rand)):
            detailed_results.append({
                "article_id": article.get("article_id"),
                "qa_pair_index": i + 1,
                "reference": ref,
                "random_prediction": pred,
                "precision": p.item(),
                "recall": r.item(),
                "f1": f1.item()
            })

        # Accumulate overall references and random predictions
        total_references.extend(references)
        total_random_predictions.extend(random_predictions)
        
        processed_articles += 1

    # Calculate average scores across all evaluated articles
    P_total, R_total, F1_total = score(total_random_predictions, total_references, lang="tr", verbose=True)
    average_precision = P_total.mean().item()
    average_recall = R_total.mean().item()
    average_f1 = F1_total.mean().item()

    # Save results to output file
    output_data = {
        "evaluated_articles": processed_articles,
        "average_scores": {
            "average_precision": average_precision,
            "average_recall": average_recall,
            "average_f1": average_f1
        },
        "detailed_results": detailed_results
    }

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

    print(f"Random evaluation within articles completed. Results saved to {output_file_path}")

# ------------------------------------ #
# --------------- MAIN --------------- #
# ------------------------------------ #

def main(evaluation_types, json_files, output_files, max_articles=0):
    """
    Main function to run BERT score evaluation based on the evaluation type.
    
    Args:
        evaluation_types (list): List of strings which are eeither "regular", "random", or "random_within_article".
        json_files (list): List of paths to the input JSON files.
        output_files (list): List of paths to save the evaluation results.
        max_articles (int): Maximum number of articles to evaluate. If it is 0, all articles are evaluated.
    """
    if len(json_files) != len(output_files) != len(evaluation_types):
        print("Error: The number of input files, output files, evaluations must be the same.")
        return
    
    for evaluation_type, json_file, output_file in zip(evaluation_types, json_files, output_files):
        print(f"Processing file: {json_file}")
        
        if evaluation_type == "regular":
            evaluate_bert_score(json_file, output_file, max_articles)
        
        elif evaluation_type == "random":
            evaluate_random_bert_score(json_file, output_file, max_articles)
        
        elif evaluation_type == "random_within_article":
            evaluate_random_within_article_bert_score(json_file, output_file, max_articles)
        
        else:
            print(f"Invalid evaluation type: {evaluation_type}. Please use 'regular', 'random', or 'random_within_article'.")
            return
        
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":

    # Define the evaluation types: "regular", "random", or "random_within_article"
    evaluation_types = [
        "regular",
        "random",
        "random_within_article"
    ]

    # Define the input JSON files and output files
    json_files = [
        '../../eval_results/ytu/hilalkaplan_interview_conversation_results.json',
        '../../eval_results/ytu/hilalkaplan_interview_conversation_results.json',
        '../../eval_results/ytu/hilalkaplan_interview_conversation_results.json'
    ]

    output_files = [
        f'../../eval_results/ytu/bert_scores/{evaluation_types[0]}/hilalkaplan_conversation_bert_scores.json',
        f'../../eval_results/ytu/bert_scores/{evaluation_types[1]}/hilalkaplan_conversation_bert_scores.json',
        f'../../eval_results/ytu/bert_scores/{evaluation_types[2]}/hilalkaplan_conversation_bert_scores.json'
    ]

    # Run the main function
    main(evaluation_types, json_files, output_files)
