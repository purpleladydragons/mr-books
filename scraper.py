"""
Web scraping module for Marginal Revolution Book Reviews.
"""

import time
import random
import cloudscraper
from bs4 import BeautifulSoup
from db import insert_post, record_page_scraped, get_last_scraped_page


BASE_URL = "https://marginalrevolution.com/marginalrevolution/category/books"

# Create a cloudscraper session to handle Cloudflare protection
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'darwin',
        'desktop': True
    }
)


def scrape_category_listing():
    """Scrape post URLs from the Books category pages."""
    current_url = BASE_URL
    page_number = 1

    # Check for resume point
    last_page = get_last_scraped_page()
    if last_page > 0:
        print(f"Resuming from page {last_page + 1}")
        page_number = last_page + 1
        current_url = f"{BASE_URL}/page/{page_number}/"

    while current_url:
        print(f"Scraping page {page_number}: {current_url}")

        try:
            response = scraper.get(current_url, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching page {page_number}: {e}")
            break

        soup = BeautifulSoup(response.text, 'lxml')

        # Find all post entries on the page
        posts_found = 0

        # Look for article entries - MR uses article tags for posts
        articles = soup.find_all('article')

        for article in articles:
            # Find the post title link
            title_link = article.find('a', rel='bookmark')
            if not title_link:
                # Try finding title in header
                header = article.find(['h1', 'h2', 'h3'])
                if header:
                    title_link = header.find('a')

            if not title_link:
                continue

            url = title_link.get('href')
            title = title_link.get_text(strip=True)

            if not url or not title:
                continue

            # Find the date
            date_published = None
            time_tag = article.find('time')
            if time_tag:
                date_published = time_tag.get('datetime', time_tag.get_text(strip=True))

            # Insert the post into the database
            insert_post(url, title, date_published)
            posts_found += 1

        print(f"Scraped page {page_number}, found {posts_found} posts")

        # Record this page as scraped
        record_page_scraped(page_number)

        # Find the next page link
        next_link = None

        # Look for pagination links
        pagination = soup.find('nav', class_='navigation') or soup.find('div', class_='pagination')
        if pagination:
            next_a = pagination.find('a', class_='next') or pagination.find('a', text=lambda t: t and 'next' in t.lower() if t else False)
            if next_a:
                next_link = next_a.get('href')

        # Alternative: look for "older posts" link
        if not next_link:
            older_link = soup.find('a', text=lambda t: t and ('older' in t.lower() or 'next' in t.lower()) if t else False)
            if older_link:
                next_link = older_link.get('href')

        # Alternative: look for nav-links
        if not next_link:
            nav_links = soup.find('div', class_='nav-links')
            if nav_links:
                next_a = nav_links.find('a', class_='next')
                if next_a:
                    next_link = next_a.get('href')

        # Alternative: look for link to specific next page
        if not next_link:
            next_page_url = f"/page/{page_number + 1}"
            next_a = soup.find('a', href=lambda h: h and next_page_url in h if h else False)
            if next_a:
                next_link = next_a.get('href')

        if next_link:
            current_url = next_link
            page_number += 1
            # Delay between requests (1-2 seconds)
            delay = random.uniform(1, 2)
            time.sleep(delay)
        else:
            print(f"No more pages found. Finished at page {page_number}")
            current_url = None


def scrape_post_content():
    """Scrape the full content of each post."""
    pass
