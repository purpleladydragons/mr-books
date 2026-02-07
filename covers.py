"""
Book cover fetching from Open Library API.
"""

import urllib.parse
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


def normalize_title(title):
    """Normalize a title for comparison."""
    if not title:
        return set()
    # Lowercase, remove punctuation, split into words
    title = title.lower()
    title = re.sub(r'[^\w\s]', ' ', title)
    words = set(title.split())
    # Remove common words
    stopwords = {'the', 'a', 'an', 'of', 'and', 'in', 'to', 'for', 'on', 'with', 'by'}
    return words - stopwords


def get_main_title(title):
    """Extract main title before colon/subtitle."""
    if ':' in title:
        return title.split(':')[0].strip()
    if ' - ' in title:
        return title.split(' - ')[0].strip()
    return title


def titles_match(search_title, found_title):
    """Check if two titles are similar enough.

    Uses multiple strategies:
    1. Main title (before colon) matches
    2. Significant word overlap
    3. One title contains the other

    Returns:
        True if titles are similar enough
    """
    if not search_title or not found_title:
        return False

    search_lower = search_title.lower()
    found_lower = found_title.lower()

    # Strategy 1: One contains the other (handles subtitles well)
    if search_lower in found_lower or found_lower in search_lower:
        return True

    # Strategy 2: Main titles match (before colon/dash)
    search_main = get_main_title(search_lower)
    found_main = get_main_title(found_lower)
    if search_main and found_main:
        if search_main in found_main or found_main in search_main:
            return True

    # Strategy 3: Word overlap - adaptive threshold based on title length
    search_words = normalize_title(search_title)
    found_words = normalize_title(found_title)

    if not search_words or not found_words:
        return False

    overlap = len(search_words & found_words)
    min_len = min(len(search_words), len(found_words))

    # Short titles (1-2 significant words): need at least 1 word match
    # Medium titles (3-4 words): need 50% match
    # Long titles (5+ words): need 40% match
    if min_len <= 2:
        return overlap >= 1
    elif min_len <= 4:
        return overlap >= (min_len * 0.5)
    else:
        return overlap >= (min_len * 0.4)


def is_obviously_wrong(search_title, found_title):
    """Check if the found title is obviously a different book.

    Catches cases like searching for "Milton Friedman" and getting "Don Quixote"
    """
    search_words = normalize_title(search_title)
    found_words = normalize_title(found_title)

    if not search_words or not found_words:
        return True

    # If there's zero overlap, it's obviously wrong
    overlap = len(search_words & found_words)
    return overlap == 0


def _search_openlibrary(search_title, original_title, timeout=10):
    """Search Open Library for a cover.

    Args:
        search_title: Title to search for
        original_title: Original book title (for matching validation)
        timeout: Request timeout

    Returns:
        Cover URL or empty string
    """
    url = f"https://openlibrary.org/search.json?title={urllib.parse.quote(search_title)}&limit=10&fields=cover_i,title,author_name"

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if not data.get('docs'):
        return ""

    # First pass: look for good title matches
    for doc in data['docs']:
        found_title = doc.get('title', '')
        cover_id = doc.get('cover_i')

        if not cover_id:
            continue

        if titles_match(original_title, found_title):
            return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"

    # Second pass: accept first result if not obviously wrong
    for doc in data['docs']:
        found_title = doc.get('title', '')
        cover_id = doc.get('cover_i')

        if not cover_id:
            continue

        if not is_obviously_wrong(original_title, found_title):
            return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"

    return ""


def fetch_cover_url(title, author=None, timeout=10):
    """Fetch cover URL from Open Library API.

    Args:
        title: Book title
        author: Author name (optional, improves search accuracy)
        timeout: Request timeout in seconds

    Returns:
        Cover URL string or empty string if not found
    """
    try:
        # Try full title first
        result = _search_openlibrary(title, title, timeout)
        if result:
            return result

        # Try main title (before colon/dash) as fallback
        main_title = get_main_title(title)
        if main_title != title:
            result = _search_openlibrary(main_title, title, timeout)
            if result:
                return result

        return ""

    except Exception as e:
        print(f"  Warning: Error fetching cover for '{title}': {e}")
        return ""


def get_amazon_search_url(title, author=None):
    """Generate Amazon search URL for a book.

    Args:
        title: Book title
        author: Author name (optional)

    Returns:
        Amazon search URL
    """
    query = title
    if author:
        query = f"{title} {author}"

    return f"https://www.amazon.com/s?k={urllib.parse.quote(query)}&i=stripbooks"


def fetch_covers_parallel(books, workers=5, rate_limit=2.0):
    """Fetch covers for multiple books in parallel.

    Args:
        books: List of tuples (book_id, title, author)
        workers: Number of concurrent workers
        rate_limit: Max requests per second

    Returns:
        Dict mapping book_id to cover_url
    """
    from db import update_book_cover

    results = {}
    total = len(books)
    completed = 0
    found = 0

    # Simple rate limiting
    min_interval = 1.0 / rate_limit

    def fetch_one(book):
        book_id, title, author = book
        cover_url = fetch_cover_url(title, author)
        return book_id, cover_url

    print(f"Fetching covers for {total} books with {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for book in books:
            future = executor.submit(fetch_one, book)
            futures[future] = book
            time.sleep(min_interval)  # Rate limit submission

        for future in as_completed(futures):
            book_id, cover_url = future.result()
            results[book_id] = cover_url

            # Save to database immediately
            update_book_cover(book_id, cover_url)

            completed += 1
            if cover_url:
                found += 1

            if completed % 50 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} ({found} covers found)")

    print(f"Done! Found covers for {found}/{total} books.")
    return results


def run_cover_fetch(limit=None, workers=5):
    """Main entry point for fetching covers.

    Args:
        limit: Max books to process (default: all without covers)
        workers: Number of concurrent workers
    """
    from db import init_db, get_books_without_covers

    init_db()

    books = get_books_without_covers(limit=limit)
    if not books:
        print("All books already have cover URLs.")
        return

    print(f"Found {len(books)} books without covers.")
    fetch_covers_parallel(books, workers=workers)
