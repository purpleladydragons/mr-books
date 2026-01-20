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


def get_scrape_status():
    """Get the current scrape status and print it to terminal."""
    pass


def get_ranked_books(top=None, genre=None):
    """Get books ranked by sentiment score."""
    pass


def find_or_create_book(title, author=None):
    """Find an existing book or create a new one with deduplication."""
    pass
