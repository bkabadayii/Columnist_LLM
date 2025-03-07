import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
import time

# Load environment variables and set up Gemini API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Define input and output directories
input_directory = "../../../siu_questionnaire/agreement_ratings"
output_directory = "../../../siu_questionnaire/agreement_ratings_with_embeddings"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get embedding model
embedding_model = "models/embedding-001"

def get_embedding(text):
    """Generate embedding for a given text using Gemini API"""
    try:
        embedding = genai.embed_content(
            model=embedding_model,
            content=text,
            task_type="retrieval_document",
        )
        # Extract the values from the embedding
        embedding_values = embedding["embedding"]
        return embedding_values
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Wait and retry once in case of rate limiting
        time.sleep(2)
        try:
            embedding = genai.embed_content(
                model=embedding_model,
                content=text,
                task_type="retrieval_document",
            )
            embedding_values = embedding["embedding"]
            return embedding_values
        except Exception as e2:
            print(f"Retry failed: {e2}")
            return []

def process_json_file(filename):
    """Process a single JSON file, adding embeddings to each response"""
    # Extract columnist name from filename (without _ratings.json)
    columnist_name = os.path.basename(filename).replace("_ratings.json", "")
    output_filename = f"{columnist_name}_embeddings.json"
    
    print(f"Processing {columnist_name}...")
    
    try:
        # Load the JSON file
        with open(os.path.join(input_directory, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each question
        for question_key, question_data in tqdm(data.items(), desc=f"Questions in {columnist_name}"):
            # Process each response
            for response in question_data.get("responses", []):
                # Generate embedding for the response text
                response_text = response.get("response", "")
                if response_text:
                    embedding = get_embedding(response_text)
                    # Add embedding to the response
                    response["embedding"] = embedding
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.1)
        
        # Save the updated data to the output directory
        with open(os.path.join(output_directory, output_filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print(f"Successfully processed {columnist_name}. Output saved to {output_filename}")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

def main():
    """Main function to process all JSON files in the input directory"""
    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_directory) if f.endswith('_ratings.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_directory}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    for json_file in json_files:
        process_json_file(json_file)
        # Add a small delay between files
        time.sleep(1)
    
    print("All files processed successfully!")

if __name__ == "__main__":
    main()