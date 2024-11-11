import json
from bert_score import score

def evaluate_bert_score(json_file_path, output_file_path, max_articles=20):
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
        if processed_articles >= max_articles:
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
    P, R, F1 = score(predictions, references, lang="en", verbose=True)

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

# Use the correct paths for your input JSON file and output JSON file
json_file_path = '../../eval_results/ytu/hilalkaplan_interview_results.json'
output_file_path = '../../eval_results/ytu/hilalkaplan_bert_scores.json'

# Evaluate only the first 20 articles
evaluate_bert_score(json_file_path, output_file_path, max_articles=20)

print(f"Evaluation results saved to {output_file_path}")