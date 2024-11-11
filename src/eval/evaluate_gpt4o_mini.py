from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if available
load_dotenv()

# Set the API key directly for OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def judge_response(question, reference, generated):
    """
    Uses GPT-4o-Mini as a judge to evaluate the quality of the model-generated response.
    
    Args:
        question (str): The question posed.
        reference (str): The ground-truth reference answer.
        generated (str): The model-generated response.
    
    Returns:
        dict: Contains the rating and explanation from GPT-4o-Mini.
    """
    prompt = f"""
    You are a judge evaluating the quality of a model-generated response to a question based on relevance, coherence, accuracy, and completeness.
    
    Question: {question}
    Reference Answer: {reference}
    Model-Generated Response: {generated}

    Please return your evaluation in the following JSON format:
    {{
        "rating": [A number from 1 to 10 where 1 is very poor and 10 is excellent],
        "explanation": "[A brief explanation of the rating]"
    }}
    """
    print("Sending request to OpenAI API...")  # Debug print
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    # Extract and parse the response
    output = response.choices[0].message.content.strip()
    print("Received response from OpenAI API")  # Debug print

    # Parse the structured JSON response
    try:
        judgment = json.loads(output)
        return judgment
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return {"rating": 0.0, "explanation": "Error parsing response."}

def evaluate_all_qa_pairs(json_file_path, output_file_path):
    """
    Evaluates all QA pairs in the provided JSON file using GPT-4o-Mini as a judge and saves the results.
    
    Args:
        json_file_path (str): Path to the JSON file containing QA pairs.
        output_file_path (str): Path to save the JSON file with evaluation results.
    """
    # Load the JSON file
    print("Loading JSON data...")  # Debug print
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    evaluations = []
    total_rating = 0
    total_pairs = 0

    # Iterate through each article's QA pairs and judge them
    for article_idx, article in enumerate(data):
        print(f"Processing article {article_idx + 1}/{len(data)}")  # Debug print
        article_evaluations = []
        for qa_pair_idx, qa_pair in enumerate(article['qa_pairs']):
            question = qa_pair['question']
            reference = qa_pair['answer']
            generated = qa_pair['predicted_response']
            print(f"  Evaluating QA pair {qa_pair_idx + 1}/{len(article['qa_pairs'])}...")  # Debug print
            judgment = judge_response(question, reference, generated)
            article_evaluations.append(judgment)
            total_rating += judgment['rating']
            total_pairs += 1
        
        evaluations.append({
            "article_id": article.get("article_id"),
            "qa_pair_evaluations": article_evaluations
        })
    
    # Calculate the average rating
    average_rating = total_rating / total_pairs if total_pairs > 0 else 0

    # Save the evaluation results to a new JSON file
    print("Saving evaluation results...")  # Debug print
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump({"evaluations": evaluations, "average_rating": average_rating}, file, ensure_ascii=False, indent=4)
    print("Evaluation completed. Results saved.")  # Debug print
    print(f"Total QA pairs evaluated: {total_pairs}")
    print(f"Average rating: {average_rating:.2f}")

# Example usage
input_json_path = '../../eval_results/ytu/hilalkaplan_interview_results.json'
output_json_path = '../../eval_results/ytu/hilalkaplan_judged_results.json'
evaluate_all_qa_pairs(input_json_path, output_json_path)

print(f"Evaluation completed. Results saved to {output_json_path}")