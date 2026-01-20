"""
Database operations for Marginal Revolution Book Reviews.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mr_books.db')


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
    """Get books ranked by sentiment score."""
    pass


def find_or_create_book(title, author=None):
    """Find an existing book or create a new one with deduplication."""
    pass


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
