from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json

def calculate_bleu_score(reference, prediction):
    """
    Calculates the BLEU score for a given reference and prediction.
    
    Args:
        reference (str): The ground-truth reference text.
        prediction (str): The model-generated prediction text.
    
    Returns:
        float: The BLEU score (ranging from 0 to 1).
    """
    # Tokenize the reference and prediction
    reference_tokens = [reference.split()]
    prediction_tokens = prediction.split()
    
    # Use smoothing to handle cases where the prediction has low n-gram overlap
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothing_function)

def evaluate_bleu(json_file_path):
    """
    Evaluates the BLEU scores for all QA pairs in the provided JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing QA pairs.
    
    Returns:
        tuple: A tuple containing the average, max, and min BLEU scores.
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    bleu_scores = []

    # Iterate through each article's QA pairs
    for article in data:
        for qa_pair in article['qa_pairs']:
            reference = qa_pair['answer']
            prediction = qa_pair['predicted_response']
            bleu_score = calculate_bleu_score(reference, prediction)
            bleu_scores.append(bleu_score)
    
    # Calculate average, max, and min BLEU scores
    average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    max_bleu_score = max(bleu_scores, default=0)
    min_bleu_score = min(bleu_scores, default=0)

    return average_bleu_score, max_bleu_score, min_bleu_score

# Use the correct path for your JSON file
json_file_path = '../../eval_results/ytu/hilalkaplan_interview_results.json'
average, max_score, min_score = evaluate_bleu(json_file_path)

# Print the results
print(f"Average BLEU Score: {average}")
print(f"Max BLEU Score: {max_score}")
print(f"Min BLEU Score: {min_score}")