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
    'chunks_dir': Path('../../../columnist_data/cleaned_articles/40_chunks/'),
    'output_dir': Path('../../../columnist_data/claims/'),
    'gpt_url': 'https://chat.openai.com/g/g-8CP6YDo8d-claims-extractor',
    'delay_between_prompts': 120,  # 2 minutes fixed delay
    'current_chunk': 12,  # Starting chunk number
    'total_chunks': 37,   # Total number of chunks
    'columnist': 'hilalkaplan'  # Columnist name for file naming
}

class SessionManager:
    def __init__(self):
        self.requests_count = 0
        self.max_requests = 25  # Conservative limit before rotation

    def get_driver(self):
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        driver = uc.Chrome(options=options)
        self.requests_count = 0
        return driver
    
    def increment_request(self):
        self.requests_count += 1
        return self.requests_count >= self.max_requests

class ClaimsProcessor:
    def __init__(self):
        self.setup_logging()
        self.session_manager = SessionManager()
        
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

    def get_chunk_file(self, chunk_num):
        return CONFIG['chunks_dir'] / f"{chunk_num}_{CONFIG['columnist']}.csv"
    
    def get_output_file(self, chunk_num):
        return CONFIG['output_dir'] / f"{CONFIG['columnist']}_claims_{chunk_num}.txt"

    def load_chunk(self, chunk_num):
        chunk_file = self.get_chunk_file(chunk_num)
        logging.info(f"Loading chunk from {chunk_file}")
        df = pd.read_csv(chunk_file)
        logging.info(f"Loaded {len(df)} articles from chunk {chunk_num}")
        return df
    
    def save_claims(self, chunk_num, article_id, result):
        output_file = self.get_output_file(chunk_num)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add article separator and timestamp
        formatted_result = f"\n{'='*50}\nArticle ID: {article_id}\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{result}\n"
        
        # Append to file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(formatted_result)
            
    def save_progress(self, chunk_num, article_id):
        with open('progress.txt', 'w') as f:
            json.dump({
                'chunk': chunk_num,
                'article_id': article_id
            }, f)
    
    def load_progress(self):
        try:
            with open('progress.txt', 'r') as f:
                return json.load(f)
        except:
            return {'chunk': CONFIG['current_chunk'], 'article_id': None}

    def send_message(self, driver, message):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                input_box = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "prompt-textarea"))
                )
                
                # Send message in chunks
                chunks = message.split('\n')
                for chunk in chunks:
                    input_box.send_keys(chunk)
                    if chunk != chunks[-1]:
                        input_box.send_keys(Keys.SHIFT, Keys.ENTER)
            
                input_box.send_keys(Keys.ENTER)
                
                # Fixed 2-minute wait after each prompt
                time.sleep(CONFIG['delay_between_prompts'])
                return True
                
            except Exception as e:
                logging.error(f"Send message attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)

    def wait_for_response(self, driver):
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-message-author-role='assistant']"))
            )
        except Exception as e:
            logging.error(f"Timeout waiting for response: {str(e)}")
            raise

    def process_article(self, driver, article, chunk_num):
        article_id = article["article_id"]
        article_title = article["Article Title"]
        try:
            logging.info(f"Processing article {article_id}: {article_title}")

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

            logging.info("Sending first prompt...")
            self.send_message(driver, prompt)
            self.wait_for_response(driver)

            responses = driver.find_elements(By.CSS_SELECTOR, "div[data-message-author-role='assistant']")
            claims_response = responses[-1].text
            
            # Second prompt
            second_prompt = """Now for each claim, construct a claim that will contradict with the given claim.
This means you should transform the underlying meaning, perspective, and implications into its opposite, not merely by adding negations or changing verbs, but by constructing a logically and contextually opposite viewpoint.
Do not mention and reject the stance of the previous claims. Claims and contradicting claims should be independent of each other."""
            
            logging.info("Sending second prompt...")
            self.send_message(driver, second_prompt)
            self.wait_for_response(driver)
            
            responses = driver.find_elements(By.CSS_SELECTOR, "div[data-message-author-role='assistant']")
            contradicting_claims_response = responses[-1].text
            
            result = claims_response + "\n-----\n" + contradicting_claims_response
            self.save_claims(chunk_num, article_id, result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing article {article_id}: {str(e)}")
            return None

    def run(self):
        total_start_time = time.time()
        driver = None
        
        # Load last progress
        progress = self.load_progress()
        current_chunk = progress['chunk']
        last_processed_id = progress['article_id']
        
        try:
            driver = self.session_manager.get_driver()
            driver.get(CONFIG['gpt_url'])
            input("\nPlease log in to ChatGPT and press Enter when ready...")
            
            while current_chunk <= CONFIG['total_chunks']:
                chunk_start_time = time.time()
                articles_df = self.load_chunk(current_chunk)
                
                # If resuming from a previous article
                if last_processed_id is not None:
                    articles_df = articles_df[articles_df['article_id'] > last_processed_id].copy()
                    last_processed_id = None  # Reset for next chunk
                
                for idx, (_, article) in enumerate(articles_df.iterrows(), 1):
                    try:
                        result = self.process_article(driver, article, current_chunk)
                        self.save_progress(current_chunk, article['article_id'])
                        
                        # Calculate progress
                        elapsed_time = time.time() - chunk_start_time
                        avg_time_per_article = elapsed_time / idx
                        remaining_articles = len(articles_df) - idx
                        estimated_time_remaining = remaining_articles * avg_time_per_article
                        
                        # Progress logging
                        logging.info(f"\nChunk {current_chunk} Progress: {idx}/{len(articles_df)} articles")
                        logging.info(f"Average time per article: {avg_time_per_article/60:.1f} minutes")
                        logging.info(f"Estimated time remaining for chunk: {estimated_time_remaining/60:.1f} minutes")
                        
                        # Overall progress
                        total_progress = ((current_chunk - CONFIG['current_chunk']) * 40 + idx) / \
                                       ((CONFIG['total_chunks'] - CONFIG['current_chunk'] + 1) * 40) * 100
                        logging.info(f"Overall progress: {total_progress:.1f}%")
                        
                        if self.session_manager.increment_request():
                            logging.info("Rotating session...")
                            driver.quit()
                            driver = self.session_manager.get_driver()
                            driver.get(CONFIG['gpt_url'])
                            input("\nPlease log in to ChatGPT and press Enter when ready...")
                    
                    except Exception as e:
                        logging.error(f"Error in article processing loop: {str(e)}")
                        time.sleep(30)  # Wait before retry
                        if driver:
                            driver.quit()
                        driver = self.session_manager.get_driver()
                        driver.get(CONFIG['gpt_url'])
                        input("\nError occurred. Please log in to ChatGPT and press Enter to continue...")
                        continue  # Retry the same article
                
                logging.info(f"Completed chunk {current_chunk}")
                current_chunk += 1
                self.save_progress(current_chunk, None)  # Save progress for next chunk
        
        finally:
            if driver:
                driver.quit()
            
            total_time = time.time() - total_start_time
            logging.info("\nProcessing complete!")
            logging.info(f"Total processing time: {total_time/3600:.1f} hours")

if __name__ == "__main__":
    processor = ClaimsProcessor()
    processor.run()