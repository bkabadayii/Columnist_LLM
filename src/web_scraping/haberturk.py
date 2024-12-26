from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
import hashlib  # To hash unique HTML content
import requests
import os

def get_special_content(columnist_name):
    
    # Open the webpage with Selenium (Safari driver)
    driver = webdriver.Safari()

    # The URL of the webpage where the articles are listed
    url = f'https://www.haberturk.com/ozel-icerikler/{columnist_name}'
    driver.get(url)

    # Scroll parameters
    SCROLL_PAUSE_TIME = 2  # Time to wait after each scroll
    MAX_SCROLLS = 50  # Maximum number of scroll actions
    articles = []  # List to store article data
    processed_ul = set()  # Keep track of already processed <ul> hashes

    try:
        print("Waiting for 'Load More' button...")
        load_more_button = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "loadMoreBtn"))
        )
        print("Clicking 'Load More' button...")
        load_more_button.click()
        time.sleep(SCROLL_PAUSE_TIME)  # Wait for content to load
    except Exception as e:
        print("No 'Load More' button found or it failed to load. Continuing...")

    # Start scrolling and extracting data
    for scroll in range(MAX_SCROLLS):
        print(f"Scrolling step {scroll + 1}/{MAX_SCROLLS}...")

        # Parse the current page source
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Scroll to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 3450);")
        print("Scrolled to the bottom of the page.")

        # Locate the container with id="infinite-data"
        container_html = soup.find("div", id="infinite-data")
        if not container_html:
            print("Unable to locate 'infinite-data'. Retrying...")
            continue

        # Find all top-level <ul> elements inside the container (exclude nested ones)
        ul_elements = container_html.find_all("ul", recursive=False)
        new_ul_count = 0

        for ul in ul_elements:
            # Generate a unique hash for this <ul> based on its content
            ul_hash = hashlib.md5(str(ul).encode('utf-8')).hexdigest()
            
            # Check if this <ul> has been processed already
            if ul_hash in processed_ul:
                continue

            new_ul_count += 1
            print("Processing new <ul> block...")
            # Find all top-level <li> elements in this <ul>
            li_elements = ul.find_all("li", recursive=False)
            for li in li_elements:
                # Extract article link
                link_tag = li.find("a", href=True)
                article_link = f"https://www.haberturk.com{link_tag['href']}" if link_tag else None

                # Extract article header
                header_tag = li.find("h3", class_="text-2xl max-w-lg mb-3 font-black")
                article_header = header_tag.get_text(strip=True) if header_tag else None

                # Extract article datetime
                time_tag = li.find("time")
                article_datetime = time_tag["datetime"] if time_tag else None

                # Append to articles list if all data is present
                if article_link and article_header and article_datetime:
                    articles.append({
                        "Article Link": article_link,
                        "Header": article_header,
                        "Datetime": article_datetime,
                    })
                    print(f"Found article: {article_header} at {article_datetime}")

            # Mark this <ul> as processed using its hash
            processed_ul.add(ul_hash)

        time.sleep(SCROLL_PAUSE_TIME)

        if new_ul_count == 0:
            print("No new <ul> elements detected. Ending scroll loop.")
            break  # Exit loop if no new content is detected

    driver.quit()

    # Create a DataFrame to organize the data
    df = pd.DataFrame(articles)

    # Save the data to a CSV file
    output_file = f'./{columnist_name}_articles.csv'
    df.to_csv(output_file, index=False)
    print(f"CSV file saved successfully at {output_file}.")


def scrape_archive(archive_url, max_scrolls=20):
    """Scrape articles from the archive page."""
    driver = webdriver.Safari()
    driver.get(archive_url)

    SCROLL_PAUSE_TIME = 2  # Time to wait after each scroll
    articles = []  # List to store article data
    processed_li = set()  # Keep track of already processed <li> elements

    for scroll in range(max_scrolls):
        print(f"Scrolling step {scroll + 1}/{max_scrolls}...")

        # Parse the current page source
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Scroll to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 2900);")
        print("Scrolled to the bottom of the page.")

        # Locate the article container
        container_html = soup.find("ul", id="infinite-data")
        if not container_html:
            print("Unable to locate the article container. Retrying...")
            continue

        # Find all top-level <li> elements inside the container
        li_elements = container_html.find_all("li", recursive=False)

        for li in li_elements:
            # Generate a unique hash for this <li> based on its content
            li_hash = hashlib.md5(str(li).encode('utf-8')).hexdigest()

            # Check if this <li> has been processed already
            if li_hash in processed_li:
                continue

            # Extract article link
            link_tag = li.find("a", href=True)
            article_link = f"https://www.haberturk.com{link_tag['href']}" if link_tag else None

            # Extract article header
            header_tag = li.find("span", class_="block font-black mb-2.5")
            article_header = header_tag.get_text(strip=True) if header_tag else None

            # Extract article datetime
            datetime_tag = li.find("span", class_="block text-sm italic mb-2.5")
            article_datetime = datetime_tag.get_text(strip=True) if datetime_tag else None

            # Append to articles list if all data is present
            if article_link and article_header and article_datetime:
                articles.append({
                    "Article Link": article_link,
                    "Header": article_header,
                    "Datetime": article_datetime,
                })
                print(f"Found article: {article_header} at {article_datetime}")

            # Mark this <li> as processed
            processed_li.add(li_hash)

        time.sleep(SCROLL_PAUSE_TIME)

    driver.quit()
    return pd.DataFrame(articles)


def append_to_existing_csv(new_df, output_file, new_output_file):
    """Append new articles to the existing CSV file."""
    try:
        # Load existing CSV
        existing_df = pd.read_csv(output_file)

        # Concatenate and drop duplicates
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=["Article Link"])

        # Save back to the same file
        combined_df.to_csv(new_output_file, index=False)
        print(f"Appended new articles successfully. Saved to {new_output_file}.")
    except FileNotFoundError:
        # If the file doesn't exist, save the new data directly
        new_df.to_csv(new_output_file, index=False)
        print(f"No existing file found. Created new file at {new_output_file}.")


def fetch_special_content(article_link):
    """Fetch content and datetime for special content articles."""
    try:
        response = requests.get(article_link)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract article content
        article_body = soup.find("article", property="articleBody")
        article_content = "\n".join(p.get_text(strip=True) for p in article_body.find_all("p")) if article_body else None

        # Extract datetime
        time_tag = soup.find("time")
        article_datetime = time_tag["datetime"] if time_tag else None

        return article_content, article_datetime

    except requests.RequestException as e:
        print(f"Error fetching data from {article_link}: {e}")
        return None, None


def fetch_archive_content(article_link):
    """Fetch content and datetime for archive articles."""
    try:
        response = requests.get(article_link)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract article content
        article_body = soup.find("article", property="articleBody")
        article_content = "\n".join(p.get_text(strip=True) for p in article_body.find_all("p")) if article_body else None

        # Extract datetime
        time_tag = soup.find("time")
        article_datetime = time_tag.get_text(strip=True) if time_tag else None

        return article_content, article_datetime

    except requests.RequestException as e:
        print(f"Error fetching archive content from {article_link}: {e}")
        return None, None


def process_articles(csv_file, output_file):
    """Process articles and update the CSV with content and datetime."""
    # Load the existing CSV
    df = pd.read_csv(csv_file)

    # Add columns for content and updated datetime if they don't exist
    if "Content" not in df.columns:
        df["Content"] = None
    if "Updated Datetime" not in df.columns:
        df["Updated Datetime"] = None

    for index, row in df.iterrows():
        article_link = row["Article Link"]
        print(f"Processing article {index + 1}/{len(df)}: {article_link}")
        # Check if the link is a special content link
        if "/ozel-icerikler/" in article_link:
            # print(f"Processing special content: {article_link}")
            content, datetime = fetch_special_content(article_link)

        # Check if the link is an archive content link
        elif "/yazarlar/" in article_link:
            # print(f"Processing archive content: {article_link}")
            content, datetime = fetch_archive_content(article_link)

        else:
            print(f"Unknown content type for link: {article_link}")
            content, datetime = None, None

        # Update the DataFrame
        df.at[index, "Content"] = content
        df.at[index, "Updated Datetime"] = datetime

    # Save the updated DataFrame
    df.to_csv(output_file, index=False)
    print(f"CSV file updated successfully: {output_file}")


def clean_and_format_data(input_file, output_file):
    """Clean and format the data by applying specified rules."""

    # Open output directory
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Drop rows with missing values in critical columns
    df.dropna(subset=["Content", "Header", "Datetime"], inplace=True)

    # Assign unique article IDs starting from 1
    df["Article ID"] = range(1, len(df) + 1)

    df.drop(columns=["Datetime"], inplace=True)

    # Rename columns as per the requirement
    df.rename(
        columns={
            "Article ID": "article_id",
            "Article Link": "article_link",
            "Header": "article_title",
            "Updated Datetime": "article_date",
            "Content": "article_content"
        },
        inplace=True
    )

    df = df[["article_id", "article_link", "article_title", "article_date", "article_content"]]
    

    # Save the cleaned and formatted data to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Data cleaned and saved to: {output_file}")

if __name__ == "__main__":
    columnist_name = "nihal-bengisu-karaca"
    archive_url = f"https://m.haberturk.com/htyazar/{columnist_name}"
    output_file = f"./{columnist_name}_articles.csv"
    new_output_file = f"./{columnist_name}_merged_articles.csv"
    content_output_file = f"./{columnist_name}_content.csv"

    final_output_file = f"../../columnist_data/{columnist_name}/cleaned_articles.csv"

    # Scrape special content
    get_special_content(columnist_name=columnist_name)

    # Scrape archive articles
    new_articles_df = scrape_archive(archive_url)

    # Append to the existing CSV
    append_to_existing_csv(new_articles_df, output_file, new_output_file)
    
    # Process articles (get content)
    process_articles(new_output_file, content_output_file)

    # Clean and format the data
    clean_and_format_data(content_output_file, final_output_file)
