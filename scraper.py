"""
Web scraping module for Marginal Revolution Book Reviews.
"""

import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import cloudscraper
from bs4 import BeautifulSoup
from db import (insert_post, record_page_scraped, get_last_scraped_page,
                get_posts_without_content, get_total_post_count, update_post_content)


BASE_URL = "https://marginalrevolution.com/marginalrevolution/category/books"
VALID_PAGINATION_PATTERN = "marginalrevolution.com/marginalrevolution/category/books"

# Create a cloudscraper session to handle Cloudflare protection
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'darwin',
        'desktop': True
    }
)


class RateLimiter:
    """Thread-safe rate limiter to control request rate across workers."""

    def __init__(self, requests_per_second=5):
        self.min_interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.last_request_time = 0

    def wait(self):
        """Wait if necessary to maintain rate limit."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()


def is_valid_pagination_url(url, expected_page=None):
    """Validate that a URL is a valid pagination link for the Books category.

    Args:
        url: The URL to validate
        expected_page: If provided, checks that the URL points to this page number

    Returns:
        True if the URL is a valid pagination link, False otherwise
    """
    if not url or not isinstance(url, str):
        return False

    # Skip anchor-only links
    if url == '#' or url.startswith('#'):
        return False

    # Skip javascript: links
    if url.startswith('javascript:'):
        return False

    # URL must contain the valid pattern for Books category
    if VALID_PAGINATION_PATTERN not in url:
        return False

    # If checking for a specific page, validate it
    if expected_page is not None and expected_page > 1:
        expected_pattern = f"/page/{expected_page}"
        if expected_pattern not in url:
            return False

    return True


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
        next_page = page_number + 1

        # Strategy 1: Look for pagination container and find 'next' link
        pagination = soup.find('nav', class_='navigation') or soup.find('div', class_='pagination')
        if pagination:
            next_a = pagination.find('a', class_='next')
            if next_a:
                href = next_a.get('href')
                if is_valid_pagination_url(href, next_page):
                    next_link = href

            # Try text-based search within pagination
            if not next_link:
                for a in pagination.find_all('a'):
                    text = a.get_text(strip=True).lower()
                    if 'next' in text or '»' in text or '›' in text:
                        href = a.get('href')
                        if is_valid_pagination_url(href, next_page):
                            next_link = href
                            break

        # Strategy 2: Look for nav-links container (common WordPress pattern)
        if not next_link:
            nav_links = soup.find('div', class_='nav-links')
            if nav_links:
                next_a = nav_links.find('a', class_='next')
                if next_a:
                    href = next_a.get('href')
                    if is_valid_pagination_url(href, next_page):
                        next_link = href

        # Strategy 3: Look for "older posts" or navigation links
        if not next_link:
            for a in soup.find_all('a'):
                text = a.get_text(strip=True).lower()
                if 'older' in text or 'next page' in text:
                    href = a.get('href')
                    if is_valid_pagination_url(href, next_page):
                        next_link = href
                        break

        # Strategy 4: Directly look for link to specific next page number
        if not next_link:
            next_page_pattern = f"/page/{next_page}"
            for a in soup.find_all('a', href=True):
                href = a.get('href')
                if href and next_page_pattern in href and is_valid_pagination_url(href, next_page):
                    next_link = href
                    break

        # Strategy 5: Construct URL directly if we know the pattern works
        if not next_link and page_number < 500:  # Safety limit
            constructed_url = f"{BASE_URL}/page/{next_page}/"
            # We'll try this URL; if it fails the next iteration will stop
            next_link = constructed_url
            print(f"  No pagination link found, trying constructed URL: {next_link}")

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


def _fetch_single_post(post_data, rate_limiter):
    """Fetch and parse a single post's content. Used by parallel scraper.

    Args:
        post_data: Tuple of (post_id, url, title)
        rate_limiter: RateLimiter instance to control request rate

    Returns:
        Tuple of (post_id, content_html, content_text, title, success)
    """
    post_id, url, title = post_data

    # Wait for rate limiter before making request
    rate_limiter.wait()

    response = fetch_with_retry(url)
    if response is None:
        return (post_id, "", "", title, False)

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

    return (post_id, content_html, content_text, title, True)


def scrape_post_content(workers=5):
    """Scrape the full content of each post using parallel workers.

    Args:
        workers: Number of concurrent workers (default 5)
    """
    posts = get_posts_without_content()
    total_posts = get_total_post_count()
    posts_to_scrape = len(posts)

    if posts_to_scrape == 0:
        print("All posts already have content scraped.")
        return

    print(f"Found {posts_to_scrape} posts to scrape (out of {total_posts} total)")
    print(f"Using {workers} concurrent workers")

    # Create rate limiter: ~5 requests per second total across all workers
    rate_limiter = RateLimiter(requests_per_second=5)

    # Track progress
    completed_count = 0
    failed_count = 0
    base_count = total_posts - posts_to_scrape

    # Process posts in batches to avoid memory issues with very large queues
    batch_size = 50

    for batch_start in range(0, posts_to_scrape, batch_size):
        batch_end = min(batch_start + batch_size, posts_to_scrape)
        batch = posts[batch_start:batch_end]
        results = []

        # Fetch posts in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_post = {
                executor.submit(_fetch_single_post, post, rate_limiter): post
                for post in batch
            }

            for future in as_completed(future_to_post):
                post_id, content_html, content_text, title, success = future.result()
                results.append((post_id, content_html, content_text, title, success))

                completed_count += 1
                current_num = base_count + completed_count
                if success:
                    print(f"Scraped {current_num}/{total_posts}: {title[:50]}...")
                else:
                    failed_count += 1
                    print(f"Failed {current_num}/{total_posts}: {title[:50]}...")

        # Write results to database sequentially (SQLite doesn't handle concurrent writes well)
        for post_id, content_html, content_text, title, success in results:
            if success and (content_html or content_text):
                update_post_content(post_id, content_html, content_text)

    print(f"Finished scraping {posts_to_scrape} posts ({failed_count} failed)")
