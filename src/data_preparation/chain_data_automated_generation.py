import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time
from pathlib import Path
import logging
import pandas as pd
from datetime import datetime

# Configuration
CONFIG = {
    'input_file': Path('../../columnist_data/cleaned_articles/hilalkaplan_cleaned_articles.csv'),
    'output_file': Path('../../eval_results/ytu/hilalkaplan_claims_results.jsonl'),
    'gpt_url': 'https://chat.openai.com/g/g-8CP6YDo8d-claims-extractor',
    'delays': {
        'between_requests': 15,
        'between_articles': 30
    }
}

class ClaimsProcessor:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        log_file = f'claims_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def load_articles(self):
        logging.info(f"Loading articles from {CONFIG['input_file']}")
        df = pd.read_csv(CONFIG['input_file'])
        logging.info(f"Loaded {len(df)} articles")
        return df
            
    def save_output(self, result):
        CONFIG['output_file'].parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if CONFIG['output_file'].exists() else 'w'
        with open(CONFIG['output_file'], mode, encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    def send_message(self, driver, message):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Find input box
                input_box = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "prompt-textarea"))
                )
                
                # Clear existing text using JavaScript
                driver.execute_script("arguments[0].value = '';", input_box)
                
                # Set new text using JavaScript
                driver.execute_script(
                    "arguments[0].value = arguments[1]; "
                    "arguments[0].dispatchEvent(new Event('input', { bubbles: true }));",
                    input_box, message
                )
                
                # Enable the send button using JavaScript
                driver.execute_script(
                    "document.querySelector('button[data-testid=\"send-button\"]').disabled = false;"
                )
                
                # Click send button
                send_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='send-button']"))
                )
                send_button.click()
                return True
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)

    def wait_for_response(self, driver):
        WebDriverWait(driver, 60).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-message-author-role='assistant']"))
        )
        # Additional wait for completion
        time.sleep(5)

    def process_article(self, driver, article, article_id):
        try:
            print(f"\n{'='*50}")
            print(f"Processing article {article_id}: {article['Article Title']}")
            print(f"{'='*50}")

            # First prompt
            prompt = """You are an assistant that extracts subjective claims from the given article.
* Claims must be self-contained, explanatory and should be clear enough without the article.
* Claims must be built upon subjective stance of the author instead of factual information.
* Each claim must be a long and detailed sentence about social or political events.
* Please give your answer as a list of Turkish claims.

Article:
Title: {title}
Content: {content}""".format(title=article['Article Title'], content=article['Article Content'])

            print("\nSending first prompt...")
            self.send_message(driver, prompt)
            
            print("Waiting for response...")
            self.wait_for_response(driver)
            
            # Get response
            responses = driver.find_elements(By.CSS_SELECTOR, "div[data-message-author-role='assistant']")
            response = responses[-1].text
            
            # Extract claims
            claims = [line.replace("Claim: ", "").strip() 
                     for line in response.split('\n') 
                     if line.startswith("Claim:")]
            
            print(f"Found {len(claims)} claims")
            
            # Wait before second prompt
            for i in range(CONFIG['delays']['between_requests']):
                print(f"Waiting {CONFIG['delays']['between_requests'] - i} seconds...", end='\r')
                time.sleep(1)
            
            # Second prompt
            second_prompt = """Now for each claim, construct a claim that will contradict with the given claim.
This means you should transform the underlying meaning, perspective, and implications into its opposite, not merely by adding negations or changing verbs, but by constructing a logically and contextually opposite viewpoint."""
            
            print("\nSending second prompt...")
            self.send_message(driver, second_prompt)
            
            print("Waiting for response...")
            self.wait_for_response(driver)
            
            # Get response
            responses = driver.find_elements(By.CSS_SELECTOR, "div[data-message-author-role='assistant']")
            response = responses[-1].text
            
            # Extract contradicting claims
            contradicting_claims = [line.replace("Contradicting Claim: ", "").strip() 
                                  for line in response.split('\n') 
                                  if line.startswith("Contradicting Claim:")]
            
            # Prepare result
            result = {
                "article_id": article_id,
                "article_title": article['Article Title'],
                "date": article['Date'],
                "url": article['URL'],
                "claims": claims,
                "contradicting_claims": contradicting_claims,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            logging.error(f"Error processing article {article_id}: {str(e)}")
            return None

    def run(self):
        articles_df = self.load_articles()
        start_time = time.time()

        print("\nStarting Chrome...")
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        driver = uc.Chrome(options=options)
        
        try:
            print("\nNavigating to ChatGPT...")
            driver.get(CONFIG['gpt_url'])
            
            input("\nPlease:\n1. Log in to ChatGPT\n2. Make sure you're on the correct GPT\n3. Press Enter when ready...")
            
            # Process articles
            for idx, (_, article) in enumerate(articles_df.iterrows(), 1):
                result = self.process_article(driver, article, idx)
                if result:
                    self.save_output(result)
                    print(f"\nSuccessfully processed article {idx}")
                
                if idx < len(articles_df):
                    print(f"\nWaiting {CONFIG['delays']['between_articles']} seconds before next article...")
                    for i in range(CONFIG['delays']['between_articles']):
                        print(f"Time remaining: {CONFIG['delays']['between_articles'] - i} seconds", end='\r')
                        time.sleep(1)
                
                elapsed_time = time.time() - start_time
                avg_time_per_article = elapsed_time / idx
                remaining_articles = len(articles_df) - idx
                estimated_time_remaining = remaining_articles * avg_time_per_article
                
                print(f"\nProgress: {idx}/{len(articles_df)} articles")
                print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"Estimated time remaining: {estimated_time_remaining/60:.1f} minutes")
        
        finally:
            driver.quit()
            
        print("\nProcessing complete!")
        print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    processor = ClaimsProcessor()
    processor.run()