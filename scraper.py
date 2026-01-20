"""
Web scraping module for Marginal Revolution Book Reviews.
"""

import time
import random
import cloudscraper
from bs4 import BeautifulSoup
from db import (insert_post, record_page_scraped, get_last_scraped_page,
                get_posts_without_content, get_total_post_count, update_post_content)


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


def fetch_with_retry(url, max_retries=3):
    """Fetch a URL with retry logic and exponential backoff.

    Args:
        url: The URL to fetch
        max_retries: Maximum number of retry attempts (default 3)

    Returns:
        Response object if successful, None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            response = scraper.get(url, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 2^attempt seconds (2, 4, 8...)
                backoff = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after {backoff}s: {e}")
                time.sleep(backoff)
            else:
                print(f"  Failed after {max_retries} retries: {e}")
                return None


def scrape_post_content():
    """Scrape the full content of each post."""
    posts = get_posts_without_content()
    total_posts = get_total_post_count()
    posts_to_scrape = len(posts)

    if posts_to_scrape == 0:
        print("All posts already have content scraped.")
        return

    print(f"Found {posts_to_scrape} posts to scrape (out of {total_posts} total)")

    for idx, (post_id, url, title) in enumerate(posts, 1):
        scraped_count = total_posts - posts_to_scrape + idx
        print(f"Scraping post {scraped_count}/{total_posts}: {title[:50]}...")

        response = fetch_with_retry(url)
        if response is None:
            print(f"  Skipping post due to network errors")
            continue

        soup = BeautifulSoup(response.text, 'lxml')

        # Find the main post content - MR uses article or entry-content
        content_html = ""
        content_text = ""

        # Try to find the post content in various possible containers
        content_elem = None

        # Try entry-content class (common WordPress pattern)
        content_elem = soup.find('div', class_='entry-content')

        if not content_elem:
            # Try article tag
            article = soup.find('article')
            if article:
                content_elem = article.find('div', class_='entry-content') or article

        if not content_elem:
            # Try post-content class
            content_elem = soup.find('div', class_='post-content')

        if not content_elem:
            # Try the-content class
            content_elem = soup.find('div', class_='the-content')

        if content_elem:
            # Remove comments section if present
            for comments in content_elem.find_all(['div', 'section'], class_=lambda x: x and 'comment' in x.lower() if x else False):
                comments.decompose()

            # Remove sidebar elements if present
            for sidebar in content_elem.find_all(['aside', 'div'], class_=lambda x: x and 'sidebar' in x.lower() if x else False):
                sidebar.decompose()

            content_html = str(content_elem)
            content_text = content_elem.get_text(separator='\n', strip=True)
        else:
            print(f"  Warning: Could not find content element for post")

        # Update the database with the scraped content
        update_post_content(post_id, content_html, content_text)

        # Delay between requests (1-2 seconds)
        delay = random.uniform(1, 2)
        time.sleep(delay)

    print(f"Finished scraping {posts_to_scrape} posts")
