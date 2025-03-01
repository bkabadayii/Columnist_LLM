import json
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

BASE_LISTING_URL = "https://www.hurriyet.com.tr/yazarlar/ahmet-hakan/"
ARTICLES_TARGET = 400

def clean_content(raw_text):
    # Remove occurrences of "Haberin Devamı" (and small variations)
    cleaned = re.sub(r'Haberin Devam[ıi]?\s*', '', raw_text, flags=re.IGNORECASE)
    # Remove stray asterisks with surrounding whitespace
    cleaned = re.sub(r'\s*\*\s*', ' ', cleaned)
    # Collapse all whitespace into a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_article_urls(driver):
    """Extract article URLs from the current page.
    The article links are found by locating the <span class="authors-fullarticle"> and then grabbing its parent anchor.
    """
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = []
    for span in soup.find_all("span", class_="authors-fullarticle"):
        # The span is inside an anchor tag
        parent_a = span.find_parent("a")
        if parent_a and parent_a.has_attr("href"):
            href = parent_a["href"]
            # Make sure we form a complete URL if needed
            if href.startswith("/"):
                href = "https://www.hurriyet.com.tr" + href
            links.append(href)
    return links

def scrape_article(driver, url, article_id):
    driver.get(url)
    time.sleep(1)  # wait a little for dynamic content to load
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Article title: sometimes an <h1> element holds the title
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    # Date info:
    time_el = soup.find("time", class_="author-date")
    date = time_el.get_text(strip=True) if time_el else ""

    # Article content:
    content_container = soup.find("div", class_="author-content readingTime")
    if content_container:
        # Remove ad containers that have classes starting with "medyanet-"
        for adv in content_container.find_all("div", class_=re.compile(r"medyanet-(outstream-mobile|inline-adv)")):
            adv.decompose()
        raw_content = content_container.get_text(separator="\n", strip=True)
        content = clean_content(raw_content)
    else:
        content = ""

    article_data = {
        "Article Title": title,
        "Date": date,
        "URL": url,
        "Article Content": content,
        "article_id": article_id
    }
    return article_data

def main():
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)

    collected_urls = set()
    page = 1

    # Loop until we collect at least the target number of URLs
    while len(collected_urls) < ARTICLES_TARGET:
        listing_url = f"{BASE_LISTING_URL}?p={page}"
        print(f"Scraping article links from: {listing_url}")
        driver.get(listing_url)
        time.sleep(1)  # wait for page to load
        new_urls = get_article_urls(driver)
        if not new_urls:
            print("No more links found, breaking out.")
            break
        before = len(collected_urls)
        for url in new_urls:
            collected_urls.add(url)
        after = len(collected_urls)
        print(f"Collected {after} article URLs so far.")
        # If no new URLs were added on this page, assume no more pages are available.
        if before == after:
            break
        page += 1

    # Limit to exactly the target count (if we have more than needed)
    article_urls = list(collected_urls)[:ARTICLES_TARGET]
    print(f"Total URLs to scrape: {len(article_urls)}")

    articles = []
    for idx, url in enumerate(article_urls, start=1):
        print(f"Scraping article {idx}/{len(article_urls)}: {url}")
        try:
            article = scrape_article(driver, url, article_id=idx)
            articles.append(article)
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    driver.quit()

    # Save articles to JSON
    with open("hakan_articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"Scraping complete. {len(articles)} articles saved to hakan_articles.json")

if __name__ == "__main__":
    main()