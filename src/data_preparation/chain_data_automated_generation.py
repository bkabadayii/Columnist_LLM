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
import numpy as np
from datetime import datetime


# Configuration
CONFIG = {
    'input_file': Path('../../columnist_data/cleaned_articles/large_chunks/hilalkaplan_chunk_2.csv'),
    'output_path': Path('../../columnist_data/claims'),
    'gpt_url': 'https://chat.openai.com/g/g-8CP6YDo8d-claims-extractor',
    'delays': {
        'between_requests': 20,
        'between_articles': 30
    },
    "delay_distribution": {
        "min_delay": 21.224,
        "max_delay": 31.652,
        "mean": 1.383,
        "sigma": 0.781
    }
}

def random_log_normal_delay(min_delay=21.224, max_delay=31.652, mean=1.383, sigma=0.781):
    """
    Generate a log-normal distributed random delay with specified bounds.
    """
    # Generate a log-normal random value
    delay = np.random.lognormal(mean, sigma)
    
    # Scale the delay to fit within the range [min_delay, max_delay]
    scaled_delay = min_delay + (delay % (max_delay - min_delay))
    return scaled_delay

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
    
    def save_claims(self, filename, result):
        save_path = CONFIG['output_path'] / filename
        # mode = 'a' if CONFIG['output_file'].exists() else 'w'
        with open(save_path, 'w', encoding='utf-8') as f:
            # json.dump(result, f, ensure_ascii=False)
            f.write(result)

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

                # Split the message by new lines and send each part with Shift + Enter
                lines = message.split('\n')
                for line in lines[:-1]:
                    input_box.send_keys(line)
                    input_box.send_keys(Keys.SHIFT, Keys.ENTER)  # Insert new line without sending

                # Send the last line followed by Enter to submit
                input_box.send_keys(lines[-1])
                input_box.send_keys(Keys.ENTER)
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
        sleep_time = random_log_normal_delay(
            min_delay=CONFIG['delay_distribution']["min_delay"],
            max_delay=CONFIG['delay_distribution']["max_delay"],
            mean=CONFIG['delay_distribution']["mean"],
            sigma=CONFIG['delay_distribution']["sigma"]
        )

        time.sleep(sleep_time)

    def process_article(self, driver, article):
        article_id = article["article_id"]
        article_title = article["Article Title"]
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
Claim: ...
Claim: ...
...
Article:
{content}""".format(content=article['Article Content'])

            print("\nSending first prompt...")
            self.send_message(driver, prompt)
            
            print("Waiting for response...")
            self.wait_for_response(driver)

            # Get response
            responses = driver.find_elements(By.CSS_SELECTOR, "div[data-message-author-role='assistant']")
            claims_response = responses[-1].text
            
            # Second prompt
            second_prompt = """Now for each claim, construct a claim that will contradict with the given claim.
This means you should transform the underlying meaning, perspective, and implications into its opposite, not merely by adding negations or changing verbs, but by constructing a logically and contextually opposite viewpoint.
Do not mention and reject the stance of the previous claims. Claims and contradicting claims should be independent of each other."""
            
            print("\nSending second prompt...")
            self.send_message(driver, second_prompt)
            
            print("Waiting for response...")
            self.wait_for_response(driver)
            
            # Get response
            responses = driver.find_elements(By.CSS_SELECTOR, "div[data-message-author-role='assistant']")
            contradicting_claims_response = responses[-1].text
            
            result = claims_response + "\n-----\n" + contradicting_claims_response
            filename = f"{article_id}_{article_title}.txt"
            self.save_claims(filename=filename, result=result)
            
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
            time.sleep(5)
            # Process articles
            for idx, (_, article) in enumerate(articles_df.iterrows(), 1):
                result = self.process_article(driver, article)
                
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