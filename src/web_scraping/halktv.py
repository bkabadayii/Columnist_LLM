import requests
import pandas as pd
from bs4 import BeautifulSoup
import math
import os

# ---------- SCRAPING ARTICLE LINKS ---------- #

def scrape_article_links(base_url, total_pages, output_file):
    """
    Main function to scrape article links from the specified website.
    """

    def fetch_page_content(url):
        """
        Fetches the HTML content of a webpage.

        Args:
            url (str): The URL of the webpage to fetch.

        Returns:
            BeautifulSoup: Parsed HTML content of the page, or None if the request fails.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return None
    
    def extract_article_links(soup):
        """
        Extracts article links from the article list section of the page.

        Args:
            soup (BeautifulSoup): Parsed HTML content of the page.

        Returns:
            list: A list of article URLs.
        """
        article_links = []
        # Find the section containing the article list
        article_list_section = soup.find('section', class_='article-list')
        if article_list_section:
            # Find all articles in the section
            articles = article_list_section.find_all('article', class_='item')
            for article in articles:
                link_tag = article.find('a', href=True)
                if link_tag:
                    # Construct the full URL for each article
                    full_url = f"https://halktv.com.tr{link_tag['href']}"
                    article_links.append(full_url)
        return article_links

    def scrape_all_articles(base_url, total_pages):
        """
        Scrapes article links from all pages.

        Args:
            base_url (str): The base URL of the website.
            total_pages (int): Total number of pages to scrape.

        Returns:
            list: A list of all article URLs from all pages.
        """
        all_links = []
        for page in range(1, total_pages + 1):
            print(f"Scraping page {page}...")
            url = f"{base_url}?page={page}"
            soup = fetch_page_content(url)
            if soup:
                page_links = extract_article_links(soup)
                all_links.extend(page_links)
            else:
                print(f"Skipping page {page} due to an error.")
        return all_links

    print("Starting the scraping process...")
    article_links = scrape_all_articles(base_url, total_pages)

    # Output the collected links
    print("\nCollected article links.")

    print(f"\nTotal articles collected: {len(article_links)}")

    # Save to a CSV file using pandas

    df = pd.DataFrame({
        "article_id": range(1, len(article_links) + 1),  # Generate article IDs starting from 1
        "article_link": article_links
    })

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\nScraped articles saved to {output_file}")

# ---------- SCRAPING ARTICLE LINKS ---------- #

# -------------------------------------------- #

# --------- SCRAPING ARTICLE DETAILS --------- #

def extend_csv_with_article_details(input_file, output_file):
    """
    Reads article links from a CSV file, scrapes their details, and writes them to a new CSV file.

    Args:
        input_file (str): Path to the input CSV file containing article links.
        output_file (str): Path to the output CSV file to save extended data.
    """
    # Load the CSV containing article links

    def scrape_article_details(article_url):
        """
        Scrapes the details (title, content, and date) from an article page.

        Args:
            article_url (str): The URL of the article to scrape.

        Returns:
            dict: A dictionary containing the article title, content, and date.
        """
        try:
            response = requests.get(article_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the title
            title = soup.find('h1', class_='content-title')
            article_title = title.get_text(strip=True) if title else "No Title"

            # Extract the date
            date = soup.find('time')
            article_date = date.get_text(strip=True) if date else "No Date"

            # Extract the content
            content_section = soup.find('div', class_='text-content')
            if content_section:
                # Combine all text from paragraphs and headers
                paragraphs = content_section.find_all(['p', 'h2', 'h3'])
                article_content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
            else:
                article_content = "No Content"

            return {
                "article_title": article_title,
                "article_date": article_date,
                "article_content": article_content
            }
        except requests.RequestException as e:
            print(f"Error fetching article URL {article_url}: {e}")
            return {
                "article_title": "Error",
                "article_date": "Error",
                "article_content": "Error"
            }
        
    df = pd.read_csv(input_file)

    # Initialize lists to store new data
    titles = []
    dates = []
    contents = []

    print("Starting to scrape article details...")
    for index, row in df.iterrows():
        print(f"Scraping article {row['article_id']}...")
        details = scrape_article_details(row['article_link'])
        titles.append(details["article_title"])
        dates.append(details["article_date"])
        contents.append(details["article_content"])

    # Add the new columns to the DataFrame
    df['article_title'] = titles
    df['article_date'] = dates
    df['article_content'] = contents

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Extended data saved to {output_file}")

# --------- SCRAPING ARTICLE DETAILS --------- #

# -------------------------------------------- #

# ----------- PREPROCESS ARTICLES ----------- #

def preprocess_articles(input_file, output_file):
    """
    Preprocesses the scraped articles by:
    - Dropping rows with 'Error' in any of the article details columns.
    - Stripping unnecessary whitespace from the article content.
    - Reindexing articles sequentially after dropping rows.
    - Saving the cleaned data to a new CSV file.

    Args:
        input_file (str): Path to the input CSV file containing scraped articles.
        output_file (str): Path to save the cleaned CSV file.
    """
    # Load the scraped articles CSV
    df = pd.read_csv(input_file)

    print("Starting preprocessing...")

    # Drop rows where 'Error' appears in title, date, or content
    initial_row_count = len(df)
    df = df[
        (df['article_title'] != 'Error') &
        (df['article_date'] != 'Error') &
        (df['article_content'] != 'Error')
    ]
    print(f"Removed {initial_row_count - len(df)} rows with 'Error'.")

    # Strip unnecessary whitespace from article content
    df['article_content'] = df['article_content'].apply(lambda x: x.strip())

    # Reindex articles sequentially
    df.reset_index(drop=True, inplace=True)
    df['article_id'] = range(1, len(df) + 1)

    # Save the cleaned data to a new CSV file
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Preprocessed data saved to {output_file}")

# ----------- PREPROCESS ARTICLES ----------- #

# -------------------------------------------- #

# ------------ DIVIDE INTO CHUNKS ------------ #

def divide_into_chunks(input_file, output_folder, chunk_size=40):
    """
    Divides the preprocessed article data into chunks of specified size and saves them as separate CSV files.

    Args:
        input_file (str): Path to the preprocessed CSV file containing cleaned article data.
        output_folder (str): Path to the folder where chunk files will be saved.
        chunk_size (int): Number of articles per chunk. Default is 40.
    """
    # Load the cleaned data
    df = pd.read_csv(input_file)

    # Calculate the number of chunks
    num_chunks = math.ceil(len(df) / chunk_size)
    print(f"Dividing data into {num_chunks} chunks...")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Split the DataFrame into chunks and save each chunk as a separate file
    for i in range(num_chunks):
        chunk = df[i * chunk_size : (i + 1) * chunk_size]
        chunk_file = os.path.join(output_folder, f"{i + 1}_chunk.csv")
        chunk.to_csv(chunk_file, index=False, encoding='utf-8')
        print(f"Saved chunk {i + 1} to {chunk_file}")

    print(f"All chunks saved to {output_folder}")

# ------------ DIVIDE INTO CHUNKS ------------ #

# -------------------------------------------- #

# ----------------- MAIN CODE ---------------- #

if __name__ == "__main__":
    mode = "execute_all" # scrape_links, scrape_details, preprocess, divide_chunks, execute_all
    columnist_name = "mehmettezkan"
    base_url = "https://halktv.com.tr/yazar/mehmet-tezkan-33656"
    total_pages = 10

    if (mode == "scrape_links"):    
        output_file = f"../../columnist_data/{columnist_name}/article_links.csv"
        scrape_article_links(base_url, total_pages, output_file)

    elif (mode == "scrape_details"):
        input_file = f"../../columnist_data/{columnist_name}/article_links.csv"
        output_file = f"../../columnist_data/{columnist_name}/extended_articles.csv"

        # Extend the CSV with scraped article details
        extend_csv_with_article_details(input_file, output_file)

    elif (mode == "preprocess"):
        input_file = f"../../columnist_data/{columnist_name}/extended_articles.csv"
        output_file = f"../../columnist_data/{columnist_name}/cleaned_articles.csv"

        # Preprocess the scraped articles
        preprocess_articles(input_file, output_file)

    elif (mode == "divide_chunks"):
        input_file = f"../../columnist_data/{columnist_name}/cleaned_articles.csv"
        output_folder = f"../../columnist_data/{columnist_name}/chunks"

        # Divide the preprocessed articles into chunks
        divide_into_chunks(input_file, output_folder, chunk_size=40)
    
    elif (mode == "execute_all"):
        # Scrape article links
        output_file = f"../../columnist_data/{columnist_name}/article_links.csv"
        scrape_article_links(base_url, total_pages, output_file)

        # Extend the CSV with scraped article details
        input_file = f"../../columnist_data/{columnist_name}/article_links.csv"
        output_file = f"../../columnist_data/{columnist_name}/extended_articles.csv"
        extend_csv_with_article_details(input_file, output_file)

        # Preprocess the scraped articles
        input_file = f"../../columnist_data/{columnist_name}/extended_articles.csv"
        output_file = f"../../columnist_data/{columnist_name}/cleaned_articles.csv"
        preprocess_articles(input_file, output_file)

        # Divide the preprocessed articles into chunks
        input_file = f"../../columnist_data/{columnist_name}/cleaned_articles.csv"
        output_folder = f"../../columnist_data/{columnist_name}/chunks"
        divide_into_chunks(input_file, output_folder, chunk_size=40)

# ----------------- MAIN CODE ---------------- #