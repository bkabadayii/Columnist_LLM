import pyautogui
import time
import pyperclip
import pandas as pd

# text box a copy paste yapan script.


# File path
file_path = '../../columnist_data/cleaned_articles/hilalkaplan_cleaned_articles.csv'

# Read the CSV file
try:
    articles_df = pd.read_csv(file_path, encoding='utf-8')
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except Exception as e:
    print(f"Error reading the file: {e}")
    exit()

# Verify the column name
if 'Article Content' not in articles_df.columns:
    print("Error: 'Article Content' column not found in the CSV file.")
    exit()

articles = articles_df['Article Content'].dropna().tolist()

# Prompts to be used
initial_prompt_template = """Article content

You are an assistant that extracts subjective claims from the given article.

    * Claims must be self-contained, explanatory and should be clear enough without the article.
    * Claims must be built upon subjective stance of the author instead of factual information.
    * Each claim must be a long and detailed sentence about social or political events.
    * Please give your answer as a list of Turkish claims.

    Claim: ...
    Claim: ...
"""
contradicting_prompt_template = """Now for each claim, construct a claim that will contradict with the given claim.
This means you should transform the underlying meaning, perspective, and implications into its opposite, not merely by adding negations or changing verbs, but by constructing a logically and contextually opposite viewpoint.

Contradicting Claim: ...
Contradicting Claim: ...
...
"""

# Give user time to focus on the ChatGPT input box
print("You have 10 seconds to focus on the ChatGPT input box...")
time.sleep(10)

# Process each article
for idx, article in enumerate(articles, start=1):
    print(f"\nProcessing article {idx}/{len(articles)}...")
    
    # Prepare the initial prompt
    initial_prompt = f"{article}\n\n{initial_prompt_template}"
    print("Sending the following initial prompt:\n")
    print(initial_prompt)
    
    # Copy to clipboard
    pyperclip.copy(initial_prompt)
    time.sleep(0.5)  # Small delay to ensure clipboard is updated
    
    # Paste using pyautogui
    pyautogui.hotkey("command", "v")  # Simulates Command+V on macOS
    pyautogui.press("enter")
    print("Initial prompt sent. Waiting for 15 seconds...")
    time.sleep(15)  # Adjust delay as needed
    
    # Prepare the contradicting claims prompt
    print("Preparing to send the contradicting claims prompt...")
    contradicting_prompt = contradicting_prompt_template
    print("Sending the following contradicting claims prompt:\n")
    print(contradicting_prompt)
    
    # Copy to clipboard
    pyperclip.copy(contradicting_prompt)
    time.sleep(0.5)  # Small delay to ensure clipboard is updated
    
    # Paste using pyautogui
    pyautogui.hotkey("command", "v")  # Simulates Command+V on macOS
    pyautogui.press("enter")
    print("Contradicting claims prompt sent. Waiting for 20 seconds...")
    time.sleep(20)  # Adjust delay as needed

print("\nAll articles processed.")