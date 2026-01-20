"""
Database operations for Marginal Revolution Book Reviews.
"""

import sqlite3
import os
import re
from rapidfuzz import fuzz

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mr_books.db')

# Articles to strip for title normalization
ARTICLES = {'the', 'a', 'an'}

# Similarity threshold for considering two titles the same book
SIMILARITY_THRESHOLD = 85


def normalize_title(title):
    """Normalize a book title for comparison.

    - Converts to lowercase
    - Strips leading articles (the, a, an)
    - Removes punctuation
    - Collapses whitespace
    """
    if not title:
        return ""

    # Lowercase
    normalized = title.lower().strip()

    # Remove leading articles
    words = normalized.split()
    if words and words[0] in ARTICLES:
        words = words[1:]
    normalized = ' '.join(words)

    # Remove punctuation (keep alphanumeric and spaces)
    normalized = re.sub(r'[^\w\s]', '', normalized)

    # Collapse whitespace
    normalized = ' '.join(normalized.split())

    return normalized


def get_connection():
    """Get a database connection."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """Initialize the SQLite database with all required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create posts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            date_published TEXT,
            content_html TEXT,
            content_text TEXT,
            scraped_at TEXT
        )
    ''')

    # Create books table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT,
            sentiment_score REAL,
            genre TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create book_mentions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS book_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            post_id INTEGER NOT NULL,
            context_text TEXT,
            sentiment_score REAL,
            FOREIGN KEY (book_id) REFERENCES books(id),
            FOREIGN KEY (post_id) REFERENCES posts(id),
            UNIQUE(book_id, post_id)
        )
    ''')

    # Create scrape_progress table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scrape_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_number INTEGER NOT NULL,
            completed_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def insert_post(url, title, date_published):
    """Insert a new post or ignore if URL already exists."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO posts (url, title, date_published)
        VALUES (?, ?, ?)
    ''', (url, title, date_published))
    conn.commit()
    conn.close()


def record_page_scraped(page_number):
    """Record that a page has been scraped."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO scrape_progress (page_number)
        VALUES (?)
    ''', (page_number,))
    conn.commit()
    conn.close()


def get_last_scraped_page():
    """Get the last scraped page number, or 0 if none."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(page_number) FROM scrape_progress')
    result = cursor.fetchone()[0]
    conn.close()
    return result if result is not None else 0


def get_posts_without_content():
    """Get all posts that don't have content scraped yet."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, url, title FROM posts
        WHERE content_text IS NULL
        ORDER BY id
    ''')
    posts = cursor.fetchall()
    conn.close()
    return posts


def get_total_post_count():
    """Get total number of posts in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM posts')
    result = cursor.fetchone()[0]
    conn.close()
    return result


def update_post_content(post_id, content_html, content_text):
    """Update a post with its scraped content."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE posts
        SET content_html = ?, content_text = ?, scraped_at = datetime('now')
        WHERE id = ?
    ''', (content_html, content_text, post_id))
    conn.commit()
    conn.close()


def get_scrape_status():
    """Get the current scrape status and print it to terminal."""
    conn = get_connection()
    cursor = conn.cursor()

    # Get total posts in DB
    cursor.execute('SELECT COUNT(*) FROM posts')
    total_posts = cursor.fetchone()[0]

    # Get posts with content scraped
    cursor.execute('SELECT COUNT(*) FROM posts WHERE content_text IS NOT NULL')
    posts_with_content = cursor.fetchone()[0]

    # Get number of pages scraped
    cursor.execute('SELECT COUNT(*) FROM scrape_progress')
    pages_scraped = cursor.fetchone()[0]

    # Get last scrape timestamp
    cursor.execute('''
        SELECT MAX(scraped_at) FROM posts WHERE scraped_at IS NOT NULL
    ''')
    last_scrape = cursor.fetchone()[0]

    # Get last page scrape timestamp
    cursor.execute('''
        SELECT MAX(completed_at) FROM scrape_progress
    ''')
    last_page_scrape = cursor.fetchone()[0]

    conn.close()

    # Print formatted status
    print("=" * 50)
    print("Marginal Revolution Scraper Status")
    print("=" * 50)
    print(f"Pages scraped:        {pages_scraped}")
    print(f"Posts found:          {total_posts}")
    print(f"Posts with content:   {posts_with_content}")
    if total_posts > 0:
        pct = (posts_with_content / total_posts) * 100
        print(f"Content coverage:     {pct:.1f}%")
    print("-" * 50)
    if last_scrape:
        print(f"Last content scrape:  {last_scrape}")
    elif last_page_scrape:
        print(f"Last listing scrape:  {last_page_scrape}")
    else:
        print("No scrapes completed yet")
    print("=" * 50)

    return {
        'total_posts': total_posts,
        'posts_with_content': posts_with_content,
        'pages_scraped': pages_scraped,
        'last_scrape': last_scrape
    }


def get_ranked_books(top=None, genre=None):
    """Get books ranked by sentiment score and print formatted results.

    Args:
        top: Limit results to top N books (default: all books)
        genre: Filter by genre (for US-010, not yet implemented)

    Returns:
        List of dicts with book info and post URLs
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Build query with optional genre filter
    query = '''
        SELECT b.id, b.title, b.author, b.sentiment_score, b.genre
        FROM books b
        WHERE b.sentiment_score IS NOT NULL
    '''
    params = []

    if genre:
        query += ' AND b.genre = ?'
        params.append(genre)

    query += ' ORDER BY b.sentiment_score DESC'

    if top:
        query += ' LIMIT ?'
        params.append(top)

    cursor.execute(query, params)
    books = cursor.fetchall()

    if not books:
        print("No books with sentiment scores found.")
        print("Run 'python main.py analyze' first to extract and analyze books.")
        conn.close()
        return []

    # Get post URLs for each book
    results = []
    for book_id, title, author, sentiment_score, book_genre in books:
        cursor.execute('''
            SELECT DISTINCT p.url
            FROM book_mentions bm
            JOIN posts p ON bm.post_id = p.id
            WHERE bm.book_id = ?
            ORDER BY p.date_published DESC
        ''', (book_id,))
        post_urls = [row[0] for row in cursor.fetchall()]

        results.append({
            'id': book_id,
            'title': title,
            'author': author,
            'sentiment_score': sentiment_score,
            'genre': book_genre,
            'post_urls': post_urls
        })

    conn.close()

    # Print formatted output
    print("=" * 80)
    print("Marginal Revolution Book Rankings (by Sentiment Score)")
    print("=" * 80)
    print()

    for rank, book in enumerate(results, 1):
        score = book['sentiment_score']
        score_str = f"{score:+.3f}" if score is not None else "N/A"

        # Print rank, title, and score
        print(f"{rank:3}. {book['title']}")
        print(f"     Score: {score_str}")

        # Print post URLs (limit to 3 to keep output readable)
        urls = book['post_urls']
        if urls:
            print(f"     Mentioned in {len(urls)} post(s):")
            for url in urls[:3]:
                print(f"       - {url}")
            if len(urls) > 3:
                print(f"       ... and {len(urls) - 3} more")
        print()

    print("=" * 80)
    print(f"Total: {len(results)} books")
    if top:
        print(f"(Showing top {top})")
    print("=" * 80)

    return results


def find_or_create_book(title, author=None):
    """Find an existing book or create a new one with deduplication.

    Uses fuzzy string matching to detect similar titles with >85% similarity.
    Normalizes titles (lowercase, strip articles) before comparison.
    If a match is found, returns the existing book's ID.
    Otherwise, creates a new book and returns its ID.
    """
    if not title:
        return None

    conn = get_connection()
    cursor = conn.cursor()

    # Normalize the input title for comparison
    normalized_input = normalize_title(title)

    # Get all existing books
    cursor.execute('SELECT id, title FROM books')
    existing_books = cursor.fetchall()

    best_match_id = None
    best_match_score = 0

    for book_id, existing_title in existing_books:
        normalized_existing = normalize_title(existing_title)

        # Calculate similarity using token sort ratio (handles word order differences)
        score = fuzz.token_sort_ratio(normalized_input, normalized_existing)

        if score > best_match_score and score >= SIMILARITY_THRESHOLD:
            best_match_score = score
            best_match_id = book_id

    if best_match_id is not None:
        # Found a matching book
        conn.close()
        return best_match_id

    # No match found, create a new book
    cursor.execute('''
        INSERT INTO books (title, author)
        VALUES (?, ?)
    ''', (title, author))
    book_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return book_id


def get_posts_with_content():
    """Get all posts that have content scraped."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, url, title, content_html, content_text FROM posts
        WHERE content_text IS NOT NULL
        ORDER BY id
    ''')
    posts = cursor.fetchall()
    conn.close()
    return posts


def insert_book(title, author=None):
    """Insert a new book and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO books (title, author)
        VALUES (?, ?)
    ''', (title, author))
    book_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return book_id


def get_book_by_title(title):
    """Get a book by exact title match."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, author FROM books WHERE title = ?', (title,))
    result = cursor.fetchone()
    conn.close()
    return result


def insert_book_mention(book_id, post_id, context_text):
    """Insert a book mention, ignoring duplicates."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO book_mentions (book_id, post_id, context_text)
            VALUES (?, ?, ?)
        ''', (book_id, post_id, context_text))
        conn.commit()
    finally:
        conn.close()


def get_all_books():
    """Get all books from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, author FROM books ORDER BY id')
    books = cursor.fetchall()
    conn.close()
    return books


def get_unanalyzed_mentions():
    """Get all book mentions that haven't been analyzed for sentiment yet.

    Returns list of tuples: (mention_id, context_text, book_title)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT bm.id, bm.context_text, b.title
        FROM book_mentions bm
        JOIN books b ON bm.book_id = b.id
        WHERE bm.sentiment_score IS NULL
        ORDER BY bm.id
    ''')
    mentions = cursor.fetchall()
    conn.close()
    return mentions


def update_mention_sentiment(mention_id, sentiment_score):
    """Update the sentiment score for a book mention."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE book_mentions
        SET sentiment_score = ?
        WHERE id = ?
    ''', (sentiment_score, mention_id))
    conn.commit()
    conn.close()


def update_book_sentiment():
    """Update the aggregated sentiment score for all books.

    Calculates the average sentiment across all mentions for each book
    and stores it in books.sentiment_score.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Update each book's sentiment_score with the average of its mentions
    cursor.execute('''
        UPDATE books
        SET sentiment_score = (
            SELECT AVG(bm.sentiment_score)
            FROM book_mentions bm
            WHERE bm.book_id = books.id
            AND bm.sentiment_score IS NOT NULL
        )
        WHERE id IN (
            SELECT DISTINCT book_id FROM book_mentions
            WHERE sentiment_score IS NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


def get_books_without_genre():
    """Get all books that don't have a genre assigned yet."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id FROM books
        WHERE genre IS NULL
        ORDER BY id
    ''')
    books = cursor.fetchall()
    conn.close()
    return books


def get_book_contexts(book_id):
    """Get all context texts for a book's mentions.

    Also includes the post content for richer keyword matching.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT bm.context_text, p.content_text
        FROM book_mentions bm
        JOIN posts p ON bm.post_id = p.id
        WHERE bm.book_id = ?
    ''', (book_id,))
    rows = cursor.fetchall()
    conn.close()

    # Combine both context_text and post content_text
    contexts = []
    for context_text, post_content in rows:
        if context_text:
            contexts.append(context_text)
        if post_content:
            contexts.append(post_content)
    return contexts


def update_book_genre(book_id, genre):
    """Update the genre for a book."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE books
        SET genre = ?
        WHERE id = ?
    ''', (genre, book_id))
    conn.commit()
    conn.close()
