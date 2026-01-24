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

    # Enable foreign keys for CASCADE support
    cursor.execute('PRAGMA foreign_keys = ON')

    # Create posts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            date_published TEXT,
            content_html TEXT,
            content_text TEXT,
            scraped_at TEXT,
            books_extracted_at TEXT
        )
    ''')

    # Add books_extracted_at column if it doesn't exist (for existing databases)
    cursor.execute("PRAGMA table_info(posts)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'books_extracted_at' not in columns:
        cursor.execute('ALTER TABLE posts ADD COLUMN books_extracted_at TEXT')

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

    # Create comparisons table for Bradley-Terry pairwise ranking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mention_a_id INTEGER NOT NULL,
            mention_b_id INTEGER NOT NULL,
            winner_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            manually_corrected INTEGER DEFAULT 0,
            corrected_at TEXT,
            FOREIGN KEY (mention_a_id) REFERENCES book_mentions(id),
            FOREIGN KEY (mention_b_id) REFERENCES book_mentions(id),
            FOREIGN KEY (winner_id) REFERENCES book_mentions(id)
        )
    ''')

    # Add manually_corrected and corrected_at columns if they don't exist
    cursor.execute("PRAGMA table_info(comparisons)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'manually_corrected' not in columns:
        cursor.execute('ALTER TABLE comparisons ADD COLUMN manually_corrected INTEGER DEFAULT 0')
    if 'corrected_at' not in columns:
        cursor.execute('ALTER TABLE comparisons ADD COLUMN corrected_at TEXT')

    # Add bt_score column to book_mentions if it doesn't exist
    cursor.execute("PRAGMA table_info(book_mentions)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'bt_score' not in columns:
        cursor.execute('ALTER TABLE book_mentions ADD COLUMN bt_score REAL')

    # Add bt_score column to books if it doesn't exist
    cursor.execute("PRAGMA table_info(books)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'bt_score' not in columns:
        cursor.execute('ALTER TABLE books ADD COLUMN bt_score REAL')

    # Create genres table for genre labels
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS genres (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')

    # Create book_genres junction table with CASCADE delete
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS book_genres (
            book_id INTEGER NOT NULL,
            genre_id INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (book_id, genre_id),
            FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE,
            FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
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


def get_extraction_counts():
    """Get counts of posts for extraction logging.

    Returns:
        Tuple: (total_with_content, already_extracted, to_process)
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Total posts with content
    cursor.execute('SELECT COUNT(*) FROM posts WHERE content_text IS NOT NULL')
    total_with_content = cursor.fetchone()[0]

    # Posts already extracted
    cursor.execute('SELECT COUNT(*) FROM posts WHERE content_text IS NOT NULL AND books_extracted_at IS NOT NULL')
    already_extracted = cursor.fetchone()[0]

    conn.close()

    to_process = total_with_content - already_extracted
    return (total_with_content, already_extracted, to_process)


def get_posts_for_extraction(reextract=False):
    """Get posts that need book extraction.

    Args:
        reextract: If True, return all posts with content (ignore books_extracted_at).
                   If False, only return posts where books_extracted_at IS NULL.

    Returns:
        List of tuples: (id, url, title, content_html, content_text)
    """
    conn = get_connection()
    cursor = conn.cursor()
    if reextract:
        cursor.execute('''
            SELECT id, url, title, content_html, content_text FROM posts
            WHERE content_text IS NOT NULL
            ORDER BY id
        ''')
    else:
        cursor.execute('''
            SELECT id, url, title, content_html, content_text FROM posts
            WHERE content_text IS NOT NULL AND books_extracted_at IS NULL
            ORDER BY id
        ''')
    posts = cursor.fetchall()
    conn.close()
    return posts


def mark_post_books_extracted(post_id):
    """Mark a post as having had books extracted."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE posts
        SET books_extracted_at = datetime('now')
        WHERE id = ?
    ''', (post_id,))
    conn.commit()
    conn.close()


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


def insert_book_mention_with_sentiment(book_id, post_id, context_text, sentiment_score):
    """Insert a book mention with sentiment score already set, ignoring duplicates."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO book_mentions (book_id, post_id, context_text, sentiment_score)
            VALUES (?, ?, ?, ?)
        ''', (book_id, post_id, context_text, sentiment_score))
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


# ============================================================
# Bradley-Terry Pairwise Ranking Functions
# ============================================================

def get_all_book_mentions():
    """Get all book mentions with their context for sampling.

    Returns list of tuples: (mention_id, book_id, book_title, context_text, post_content)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT bm.id, bm.book_id, b.title, bm.context_text, p.content_text
        FROM book_mentions bm
        JOIN books b ON bm.book_id = b.id
        JOIN posts p ON bm.post_id = p.id
        WHERE p.content_text IS NOT NULL
        ORDER BY bm.id
    ''')
    mentions = cursor.fetchall()
    conn.close()
    return mentions


def get_mentions_for_bt():
    """Get all mention IDs for Bradley-Terry sampling.

    Returns list of mention IDs.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT bm.id
        FROM book_mentions bm
        JOIN posts p ON bm.post_id = p.id
        WHERE p.content_text IS NOT NULL
    ''')
    mentions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return mentions


def get_mention_for_comparison(mention_id):
    """Get a single mention's details for comparison.

    Returns tuple: (mention_id, book_id, book_title, context_text, post_content)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT bm.id, bm.book_id, b.title, bm.context_text, p.content_text
        FROM book_mentions bm
        JOIN books b ON bm.book_id = b.id
        JOIN posts p ON bm.post_id = p.id
        WHERE bm.id = ?
    ''', (mention_id,))
    mention = cursor.fetchone()
    conn.close()
    return mention


def insert_comparison(mention_a_id, mention_b_id, winner_id):
    """Insert a comparison result.

    Args:
        mention_a_id: First mention in comparison
        mention_b_id: Second mention in comparison
        winner_id: ID of winning mention, or None for tie
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO comparisons (mention_a_id, mention_b_id, winner_id)
        VALUES (?, ?, ?)
    ''', (mention_a_id, mention_b_id, winner_id))
    conn.commit()
    conn.close()


def insert_comparisons_batch(comparisons):
    """Insert multiple comparisons in a single transaction.

    Args:
        comparisons: List of tuples (mention_a_id, mention_b_id, winner_id)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT INTO comparisons (mention_a_id, mention_b_id, winner_id)
        VALUES (?, ?, ?)
    ''', comparisons)
    conn.commit()
    conn.close()


def get_all_comparisons():
    """Get all comparisons for Bradley-Terry fitting.

    Returns list of tuples: (mention_a_id, mention_b_id, winner_id)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT mention_a_id, mention_b_id, winner_id
        FROM comparisons
    ''')
    comparisons = cursor.fetchall()
    conn.close()
    return comparisons


def get_comparison_count():
    """Get total number of comparisons."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM comparisons')
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_win_rates_vs_random(top_k_ids):
    """Compute win rate vs random for each top-k item.

    A "vs random" comparison is one where a top-k item faced a non-top-k item.
    Win rate = wins / (wins + losses) for each top-k item.

    Args:
        top_k_ids: Set of mention IDs currently in top-k

    Returns:
        Dict mapping mention_id to dict with:
        - wins: Number of wins against non-top-k items
        - losses: Number of losses against non-top-k items
        - total: Total comparisons vs random
        - win_rate: wins/total (0-1), or None if no comparisons
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get all comparisons
    cursor.execute('''
        SELECT mention_a_id, mention_b_id, winner_id
        FROM comparisons
    ''')
    comparisons = cursor.fetchall()
    conn.close()

    # Convert to set for O(1) lookup
    top_k_set = set(top_k_ids)

    # Track wins and losses for each top-k item
    win_rates = {}
    for mid in top_k_set:
        win_rates[mid] = {'wins': 0, 'losses': 0, 'total': 0, 'win_rate': None}

    for mention_a, mention_b, winner in comparisons:
        # Check if this is a top-k vs non-top-k comparison
        a_in_top = mention_a in top_k_set
        b_in_top = mention_b in top_k_set

        # Skip if both or neither are in top-k (not a validation comparison)
        if a_in_top == b_in_top:
            continue

        # Identify which is the top-k item
        top_item = mention_a if a_in_top else mention_b

        # Skip ties (winner_id is None)
        if winner is None:
            continue

        # Record win or loss
        win_rates[top_item]['total'] += 1
        if winner == top_item:
            win_rates[top_item]['wins'] += 1
        else:
            win_rates[top_item]['losses'] += 1

    # Calculate win rates
    for mid in win_rates:
        total = win_rates[mid]['total']
        if total > 0:
            win_rates[mid]['win_rate'] = win_rates[mid]['wins'] / total

    return win_rates


def update_mention_bt_score(mention_id, bt_score):
    """Update the Bradley-Terry score for a mention."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE book_mentions
        SET bt_score = ?
        WHERE id = ?
    ''', (bt_score, mention_id))
    conn.commit()
    conn.close()


def update_mention_bt_scores_batch(scores):
    """Update Bradley-Terry scores for multiple mentions in a single transaction.

    Args:
        scores: List of tuples (mention_id, bt_score)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany('''
        UPDATE book_mentions
        SET bt_score = ?
        WHERE id = ?
    ''', [(score, mid) for mid, score in scores])
    conn.commit()
    conn.close()


def update_book_bt_scores():
    """Update aggregated Bradley-Terry scores for all books.

    Aggregates mention-level bt_scores to book-level using weighted average,
    where weight is the number of comparisons each mention participated in.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # For each book, calculate weighted average of mention bt_scores
    # Weight = number of comparisons that mention participated in
    cursor.execute('''
        UPDATE books
        SET bt_score = (
            SELECT SUM(bm.bt_score * weight) / SUM(weight)
            FROM (
                SELECT bm.id, bm.book_id, bm.bt_score,
                       (SELECT COUNT(*) FROM comparisons c
                        WHERE c.mention_a_id = bm.id OR c.mention_b_id = bm.id) as weight
                FROM book_mentions bm
                WHERE bm.bt_score IS NOT NULL
            ) bm
            WHERE bm.book_id = books.id AND bm.weight > 0
        )
        WHERE id IN (
            SELECT DISTINCT book_id FROM book_mentions
            WHERE bt_score IS NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


def get_ranked_books_by_bt(top=None, genre=None, min_comparisons=None):
    """Get books ranked by Bradley-Terry score and print formatted results.

    Args:
        top: Limit results to top N books (default: all books)
        genre: Filter by genre
        min_comparisons: Minimum number of comparisons required (default: no filter)

    Returns:
        List of dicts with book info and post URLs
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Build query with optional filters
    query = '''
        SELECT b.id, b.title, b.author, b.bt_score, b.sentiment_score, b.genre,
               (SELECT COUNT(*) FROM comparisons c
                JOIN book_mentions bm ON (c.mention_a_id = bm.id OR c.mention_b_id = bm.id)
                WHERE bm.book_id = b.id) as comparison_count
        FROM books b
        WHERE b.bt_score IS NOT NULL
    '''
    params = []

    if genre:
        query += ' AND b.genre = ?'
        params.append(genre)

    if min_comparisons:
        query += ''' AND (SELECT COUNT(*) FROM comparisons c
                         JOIN book_mentions bm ON (c.mention_a_id = bm.id OR c.mention_b_id = bm.id)
                         WHERE bm.book_id = b.id) >= ?'''
        params.append(min_comparisons)

    query += ' ORDER BY b.bt_score DESC'

    if top:
        query += ' LIMIT ?'
        params.append(top)

    cursor.execute(query, params)
    books = cursor.fetchall()

    if not books:
        print("No books with Bradley-Terry scores found.")
        print("Run 'python main.py rank --comparisons N' first to generate rankings.")
        conn.close()
        return []

    # Get post URLs for each book
    results = []
    for book_id, title, author, bt_score, sentiment_score, book_genre, comparison_count in books:
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
            'bt_score': bt_score,
            'sentiment_score': sentiment_score,
            'genre': book_genre,
            'post_urls': post_urls,
            'comparison_count': comparison_count
        })

    conn.close()

    # Print formatted output
    print("=" * 80)
    print("Marginal Revolution Book Rankings (by Bradley-Terry Score)")
    print("=" * 80)
    print()

    for rank, book in enumerate(results, 1):
        bt = book['bt_score']
        bt_str = f"{bt:.4f}" if bt is not None else "N/A"

        # Print rank, title, and score
        print(f"{rank:3}. {book['title']}")
        print(f"     BT Score: {bt_str}")

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


def has_bt_scores():
    """Check if any books have Bradley-Terry scores."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM books WHERE bt_score IS NOT NULL')
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def get_mention_comparison_counts():
    """Get the number of comparisons each mention has participated in.

    Returns:
        Dict mapping mention_id to comparison count
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Count comparisons for each mention (as either A or B)
    cursor.execute('''
        SELECT mention_id, COUNT(*) as count FROM (
            SELECT mention_a_id as mention_id FROM comparisons
            UNION ALL
            SELECT mention_b_id as mention_id FROM comparisons
        )
        GROUP BY mention_id
    ''')
    counts = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return counts


def get_mentions_with_bt_scores():
    """Get all mentions with their current BT scores.

    Returns:
        Dict mapping mention_id to bt_score (or None if not scored)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, bt_score FROM book_mentions
    ''')
    scores = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return scores


def clear_books_data():
    """Clear all books, book_mentions, and comparisons tables for fresh extraction.

    This is used by --extract-only mode to start with a clean slate.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Delete in order to respect foreign key constraints
    cursor.execute('DELETE FROM comparisons')
    cursor.execute('DELETE FROM book_mentions')
    cursor.execute('DELETE FROM books')

    conn.commit()
    conn.close()

    print("Cleared: comparisons, book_mentions, and books tables.")


def reset_extraction_cache():
    """Reset posts.books_extracted_at to NULL so all posts get reprocessed.

    This is used by --extract-only mode to ensure all posts are re-extracted.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE posts SET books_extracted_at = NULL')
    affected = cursor.rowcount
    conn.commit()
    conn.close()

    print(f"Reset books_extracted_at for {affected} posts.")


# ============================================================
# Genre Labeling Functions
# ============================================================

def get_books_for_genre_labeling():
    """Get all books that need genre labels (have no genres assigned).

    Returns:
        List of tuples: (book_id, book_title, context_text)
        context_text is the combined context from all book_mentions
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get books that have no genres assigned yet
    # Join with book_mentions to get context_text for genre inference
    cursor.execute('''
        SELECT b.id, b.title, GROUP_CONCAT(bm.context_text, '\n\n---\n\n')
        FROM books b
        LEFT JOIN book_mentions bm ON b.id = bm.book_id
        WHERE b.id NOT IN (SELECT DISTINCT book_id FROM book_genres)
        GROUP BY b.id
        ORDER BY b.id
    ''')
    books = cursor.fetchall()
    conn.close()
    return books


def get_all_books_for_genre_labeling():
    """Get ALL books for genre labeling (ignoring existing labels).

    Used with --reset flag to reprocess all books.

    Returns:
        List of tuples: (book_id, book_title, context_text)
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT b.id, b.title, GROUP_CONCAT(bm.context_text, '\n\n---\n\n')
        FROM books b
        LEFT JOIN book_mentions bm ON b.id = bm.book_id
        GROUP BY b.id
        ORDER BY b.id
    ''')
    books = cursor.fetchall()
    conn.close()
    return books


def get_books_count():
    """Get total number of books in database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM books')
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_or_create_genre(genre_name):
    """Get genre ID by name, or create if doesn't exist.

    Args:
        genre_name: Name of the genre

    Returns:
        The genre ID
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Try to get existing genre
    cursor.execute('SELECT id FROM genres WHERE name = ?', (genre_name,))
    result = cursor.fetchone()

    if result:
        conn.close()
        return result[0]

    # Create new genre
    cursor.execute('INSERT INTO genres (name) VALUES (?)', (genre_name,))
    genre_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return genre_id


def add_book_genres(book_id, genre_names):
    """Add genres to a book.

    Args:
        book_id: The book's ID
        genre_names: List of genre name strings
    """
    conn = get_connection()
    cursor = conn.cursor()

    for genre_name in genre_names:
        # Get or create the genre
        cursor.execute('SELECT id FROM genres WHERE name = ?', (genre_name,))
        result = cursor.fetchone()
        if result:
            genre_id = result[0]
        else:
            cursor.execute('INSERT INTO genres (name) VALUES (?)', (genre_name,))
            genre_id = cursor.lastrowid

        # Link book to genre (ignore if already exists)
        cursor.execute('''
            INSERT OR IGNORE INTO book_genres (book_id, genre_id)
            VALUES (?, ?)
        ''', (book_id, genre_id))

    conn.commit()
    conn.close()


def clear_book_genres():
    """Clear all book_genres entries for fresh genre labeling.

    Does NOT delete the genres themselves, just the associations.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM book_genres')
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    print(f"Cleared {affected} book-genre associations.")


def get_genre_labeling_stats():
    """Get statistics about genre labeling progress.

    Returns:
        Dict with stats
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Total books
    cursor.execute('SELECT COUNT(*) FROM books')
    total_books = cursor.fetchone()[0]

    # Books with at least one genre
    cursor.execute('SELECT COUNT(DISTINCT book_id) FROM book_genres')
    books_with_genres = cursor.fetchone()[0]

    # Total genre associations
    cursor.execute('SELECT COUNT(*) FROM book_genres')
    total_associations = cursor.fetchone()[0]

    # Unique genres used
    cursor.execute('SELECT COUNT(DISTINCT genre_id) FROM book_genres')
    unique_genres = cursor.fetchone()[0]

    conn.close()

    return {
        'total_books': total_books,
        'books_with_genres': books_with_genres,
        'books_without_genres': total_books - books_with_genres,
        'total_associations': total_associations,
        'unique_genres': unique_genres
    }


# ============================================================
# Review UI Functions
# ============================================================

def get_comparison_for_review(comparison_id):
    """Get a comparison with full details for review UI.

    Args:
        comparison_id: The comparison ID to fetch

    Returns:
        Dict with comparison details including book titles and context
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT
            c.id,
            c.mention_a_id,
            c.mention_b_id,
            c.winner_id,
            c.created_at,
            c.manually_corrected,
            c.corrected_at,
            ba.title as book_a_title,
            bb.title as book_b_title,
            bma.context_text as context_a,
            bmb.context_text as context_b
        FROM comparisons c
        JOIN book_mentions bma ON c.mention_a_id = bma.id
        JOIN book_mentions bmb ON c.mention_b_id = bmb.id
        JOIN books ba ON bma.book_id = ba.id
        JOIN books bb ON bmb.book_id = bb.id
        WHERE c.id = ?
    ''', (comparison_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        'id': row[0],
        'mention_a_id': row[1],
        'mention_b_id': row[2],
        'winner_id': row[3],
        'created_at': row[4],
        'manually_corrected': bool(row[5]),
        'corrected_at': row[6],
        'book_a_title': row[7],
        'book_b_title': row[8],
        'context_a': row[9],
        'context_b': row[10]
    }


def get_comparison_ids():
    """Get all comparison IDs in order.

    Returns:
        List of comparison IDs sorted by id
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM comparisons ORDER BY id')
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids


def correct_comparison(comparison_id, new_winner_id):
    """Correct a comparison's winner and mark as manually corrected.

    Args:
        comparison_id: The comparison to correct
        new_winner_id: The new winner mention ID
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE comparisons
        SET winner_id = ?,
            manually_corrected = 1,
            corrected_at = datetime('now')
        WHERE id = ?
    ''', (new_winner_id, comparison_id))
    conn.commit()
    conn.close()


def get_review_stats():
    """Get statistics for the review UI analytics section.

    Returns:
        Dict with total comparisons, corrected count, and correction rate
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Total comparisons
    cursor.execute('SELECT COUNT(*) FROM comparisons')
    total = cursor.fetchone()[0]

    # Manually corrected count
    cursor.execute('SELECT COUNT(*) FROM comparisons WHERE manually_corrected = 1')
    corrected = cursor.fetchone()[0]

    conn.close()

    correction_rate = (corrected / total * 100) if total > 0 else 0

    return {
        'total_comparisons': total,
        'manually_corrected': corrected,
        'correction_rate': correction_rate
    }
