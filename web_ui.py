"""
Web UI for Marginal Revolution Book Reviews.
"""

import os
from flask import Flask, render_template_string, request, jsonify
from jinja2 import Environment, BaseLoader

app = Flask(__name__)

# Configuration - can be overridden with environment variables
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')


def get_all_genres():
    """Get all genres with book counts."""
    from db import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT g.name, COUNT(bg.book_id) as book_count
        FROM genres g
        LEFT JOIN book_genres bg ON g.id = bg.genre_id
        GROUP BY g.id
        HAVING book_count > 0
        ORDER BY book_count DESC
    ''')
    genres = cursor.fetchall()
    conn.close()
    return genres


def render_page(content, active_page, **kwargs):
    """Render a page with the base template."""
    template = BASE_TEMPLATE.replace('{{ content }}', content)
    return render_template_string(template, active_page=active_page, **kwargs)


# HTML Templates
BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}MR Book Reviews{% endblock %}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
        }
        html, body {
            margin: 0;
            padding: 0;
            min-height: 100vh;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.7;
            color: #374151;
            background:
                linear-gradient(to bottom, rgba(255,255,255,0.15), rgba(255,255,255,0.1)),
                url('https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Claude_Lorrain_-_Landscape_with_Aeneas_at_Delos_-_Google_Art_Project.jpg/2560px-Claude_Lorrain_-_Landscape_with_Aeneas_at_Delos_-_Google_Art_Project.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .page-wrapper {
            min-height: 100vh;
            padding: 30px 20px 60px;
        }
        /* Floating pill navigation */
        .top-nav {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        .nav-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 10px 24px;
            border-radius: 50px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.8);
        }
        .nav-pill .site-title {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-weight: 700;
            font-size: 14px;
            color: #1f2937;
            padding-right: 12px;
            border-right: 1px solid #e5e7eb;
            margin-right: 4px;
        }
        .nav-pill a {
            color: #6b7280;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            padding: 6px 12px;
            border-radius: 20px;
            transition: all 0.2s;
        }
        .nav-pill a:hover {
            color: #1f2937;
            background: rgba(0, 0, 0, 0.05);
        }
        .nav-pill a.active {
            color: #1f2937;
            background: rgba(0, 0, 0, 0.08);
        }
        /* Main content card */
        .content-card {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 16px;
            box-shadow: 0 8px 40px rgba(0, 0, 0, 0.15), 0 2px 8px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.9);
            padding: 48px 56px;
        }
        .page-title {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-size: 32px;
            font-weight: 400;
            color: #1f2937;
            margin: 0 0 8px 0;
            letter-spacing: -0.02em;
        }
        .page-subtitle {
            font-size: 14px;
            color: #9ca3af;
            margin-bottom: 32px;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        /* Form elements */
        select {
            padding: 10px 14px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            font-size: 14px;
            background: #fafafa;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        select:focus {
            outline: none;
            border-color: #9ca3af;
        }
        input[type="number"],
        input[type="text"] {
            padding: 10px 14px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            font-size: 14px;
            background: #fafafa;
            transition: border-color 0.2s;
        }
        input[type="text"] {
            width: 280px;
        }
        input:focus {
            outline: none;
            border-color: #9ca3af;
        }
        button {
            padding: 10px 20px;
            background: #1f2937;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }
        button:hover {
            background: #374151;
        }
        /* Filters section */
        .filters {
            background: #f9fafb;
            padding: 20px 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            border: 1px solid #f3f4f6;
        }
        .filters form {
            display: flex;
            gap: 16px;
            align-items: center;
            flex-wrap: wrap;
        }
        .filters label {
            font-weight: 500;
            color: #6b7280;
            font-size: 13px;
        }
        .search-hint {
            color: #9ca3af;
            font-size: 13px;
            margin-top: 12px;
            font-style: italic;
        }
        /* Active filters */
        .filter-section {
            margin-bottom: 16px;
        }
        .active-filter {
            background: #1f2937;
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            display: inline-block;
            margin-right: 8px;
            font-size: 13px;
        }
        .active-filter a {
            color: rgba(255,255,255,0.7);
            margin-left: 8px;
            text-decoration: none;
            font-weight: 600;
        }
        .active-filter a:hover {
            color: white;
        }
        /* Stats line */
        .stats {
            color: #9ca3af;
            margin-bottom: 20px;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        /* Book list */
        .book-list {
            border-top: 1px solid #f3f4f6;
        }
        .book-item {
            padding: 20px 0;
            border-bottom: 1px solid #f3f4f6;
        }
        .book-item:last-child {
            border-bottom: none;
        }
        .book-rank {
            display: inline-block;
            width: 36px;
            font-weight: 600;
            color: #9ca3af;
            font-size: 14px;
        }
        .book-title {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-weight: 400;
            color: #1f2937;
            font-size: 17px;
        }
        .book-author {
            color: #6b7280;
            font-size: 14px;
            font-style: italic;
        }
        .book-meta {
            margin-top: 8px;
            font-size: 13px;
            color: #9ca3af;
        }
        .book-score {
            display: inline-block;
            background: #ecfdf5;
            padding: 3px 10px;
            border-radius: 6px;
            font-weight: 500;
            color: #059669;
            font-size: 12px;
        }
        .book-similarity {
            display: inline-block;
            background: #f3f4f6;
            padding: 3px 10px;
            border-radius: 6px;
            margin-left: 8px;
            font-size: 12px;
            color: #6b7280;
        }
        .comparison-badge {
            display: inline-block;
            background: #f3f4f6;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 12px;
            color: #6b7280;
            margin-left: 8px;
        }
        /* Genre tags */
        .genre-tag {
            display: inline-block;
            background: #eff6ff;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 11px;
            color: #3b82f6;
            margin-right: 6px;
            margin-top: 4px;
            text-decoration: none;
            transition: background 0.2s;
        }
        .genre-tag:hover {
            background: #dbeafe;
        }
        /* Book links */
        .book-links {
            margin-top: 10px;
        }
        .book-links a {
            font-size: 12px;
            color: #6b7280;
            text-decoration: none;
            margin-right: 16px;
        }
        .book-links a:hover {
            color: #3b82f6;
            text-decoration: underline;
        }
        /* Genre grid */
        .genre-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 16px;
        }
        .genre-card {
            background: #f9fafb;
            border-radius: 12px;
            padding: 20px;
            text-decoration: none;
            color: inherit;
            transition: all 0.2s;
            border: 1px solid #f3f4f6;
        }
        .genre-card:hover {
            background: #f3f4f6;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        .genre-card h3 {
            margin: 0 0 4px 0;
            color: #1f2937;
            font-size: 15px;
            font-weight: 500;
            text-transform: capitalize;
        }
        .genre-card .count {
            color: #9ca3af;
            font-size: 13px;
        }
        /* No results */
        .no-results {
            padding: 48px;
            text-align: center;
            color: #9ca3af;
        }
        /* Responsive */
        @media (max-width: 768px) {
            .content-card {
                padding: 32px 24px;
                margin: 0 10px;
            }
            .page-title {
                font-size: 26px;
            }
            input[type="text"] {
                width: 100%;
            }
            .filters form {
                flex-direction: column;
                align-items: stretch;
            }
        }
    </style>
</head>
<body>
    <div class="page-wrapper">
        <nav class="top-nav">
            <div class="nav-pill">
                <span class="site-title">MR Books</span>
                <a href="/" class="{{ 'active' if active_page == 'rankings' else '' }}">Rankings</a>
                <a href="/genres" class="{{ 'active' if active_page == 'genres' else '' }}">Genres</a>
                <a href="/search" class="{{ 'active' if active_page == 'search' else '' }}">Search</a>
            </div>
        </nav>
        <main class="content-card">
            {{ content }}
        </main>
    </div>
</body>
</html>
'''

RANKINGS_CONTENT = '''
<h1 class="page-title">Book Rankings</h1>
<p class="page-subtitle">Tyler Cowen's favorites from Marginal Revolution</p>

<div class="filters">
    <form method="GET" action="/">
        <label for="genre">Genre:</label>
        <select name="genre" id="genre">
            <option value="">All Genres</option>
            {% for g, count in genres %}
            <option value="{{ g }}" {{ 'selected' if genre == g else '' }}>{{ g|title }} ({{ count }})</option>
            {% endfor %}
        </select>
        <label for="min_comparisons">Min comparisons:</label>
        <input type="number" id="min_comparisons" name="min_comparisons"
               value="{{ min_comparisons or '' }}" min="0" placeholder="0">
        <label for="top">Show top:</label>
        <input type="number" id="top" name="top"
               value="{{ top or '' }}" min="1" placeholder="All">
        <button type="submit">Filter</button>
    </form>
</div>

{% if genre or min_comparisons %}
<div class="filter-section">
    {% if genre %}
    <span class="active-filter">Genre: {{ genre|title }} <a href="/?{% if min_comparisons %}min_comparisons={{ min_comparisons }}{% endif %}{% if top %}&top={{ top }}{% endif %}">&times;</a></span>
    {% endif %}
    {% if min_comparisons %}
    <span class="active-filter">{{ min_comparisons }}+ comparisons <a href="/?{% if genre %}genre={{ genre }}{% endif %}{% if top %}&top={{ top }}{% endif %}">&times;</a></span>
    {% endif %}
</div>
{% endif %}

<div class="stats">
    Showing {{ books|length }} books{% if genre %} in {{ genre|title }}{% endif %}{% if min_comparisons %} with {{ min_comparisons }}+ comparisons{% endif %}
</div>

<div class="book-list">
    {% if books %}
        {% for book in books %}
        <div class="book-item">
            <span class="book-rank">{{ loop.index }}.</span>
            <span class="book-title">{{ book.title }}</span>
            {% if book.author %}
            <span class="book-author">by {{ book.author }}</span>
            {% endif %}
            <div class="book-meta">
                <span class="book-score">BT: {{ "%.2f"|format(book.bt_score) if book.bt_score is not none else 'N/A' }}</span>
                <span class="comparison-badge">{{ book.comparison_count }} comparisons</span>
            </div>
            {% if book.genres %}
            <div style="margin-top: 5px;">
                {% for g in book.genres[:4] %}
                <a href="/?genre={{ g }}" class="genre-tag">{{ g }}</a>
                {% endfor %}
                {% if book.genres|length > 4 %}
                <span class="genre-tag" style="background: #f0f0f0; color: #888;">+{{ book.genres|length - 4 }} more</span>
                {% endif %}
            </div>
            {% endif %}
            {% if book.post_urls %}
            <div class="book-links">
                {% for url in book.post_urls[:3] %}
                <a href="{{ url }}" target="_blank">{{ url|truncate(50) }}</a>
                {% endfor %}
                {% if book.post_urls|length > 3 %}
                <span style="color: #888; font-size: 12px;">+{{ book.post_urls|length - 3 }} more</span>
                {% endif %}
            </div>
            {% endif %}
        </div>
        {% endfor %}
    {% else %}
        <div class="no-results">No books found matching your filters.</div>
    {% endif %}
</div>
'''

GENRES_CONTENT = '''
<h1 class="page-title">Browse by Genre</h1>
<p class="page-subtitle">{{ genres|length }} genres across {{ total_books }} books</p>

<div class="genre-grid">
    {% for g, count in genres %}
    <a href="/?genre={{ g }}" class="genre-card">
        <h3>{{ g }}</h3>
        <div class="count">{{ count }} book{{ 's' if count != 1 else '' }}</div>
    </a>
    {% endfor %}
</div>
'''

SEARCH_CONTENT = '''
<h1 class="page-title">Search Books</h1>
<p class="page-subtitle">Find books by concept, topic, or theme</p>

<div class="filters">
    <form method="GET" action="/search">
        <input type="text" name="q" value="{{ query or '' }}"
               placeholder="Search for books by concept..." autofocus>
        <label for="genre">Genre:</label>
        <select name="genre" id="genre">
            <option value="">All Genres</option>
            {% for g, count in genres %}
            <option value="{{ g }}" {{ 'selected' if genre == g else '' }}>{{ g|title }} ({{ count }})</option>
            {% endfor %}
        </select>
        <label for="top">Results:</label>
        <input type="number" id="top" name="top"
               value="{{ top or 20 }}" min="1" max="100" style="width: 60px;">
        <button type="submit">Search</button>
    </form>
    <div class="search-hint">Try: "Weimar Republic", "behavioral economics", "artificial intelligence", "Chinese history"</div>
</div>

{% if query %}
{% if genre %}
<div class="filter-section">
    <span class="active-filter">Genre: {{ genre|title }} <a href="/search?q={{ query }}&top={{ top }}">&times;</a></span>
</div>
{% endif %}

<div class="stats">
    Found {{ books|length }} results for "{{ query }}"{% if genre %} in {{ genre|title }}{% endif %} (sorted by BT score)
</div>

<div class="book-list">
    {% if books %}
        {% for book in books %}
        <div class="book-item">
            <span class="book-rank">{{ loop.index }}.</span>
            <span class="book-title">{{ book.title }}</span>
            {% if book.author %}
            <span class="book-author">by {{ book.author }}</span>
            {% endif %}
            <div class="book-meta">
                <span class="book-score">BT: {{ "%.2f"|format(book.bt_score) if book.bt_score is not none else 'N/A' }}</span>
                <span class="book-similarity">similarity: {{ "%.3f"|format(book.similarity) }}</span>
                {% if book.comparison_count %}
                <span class="comparison-badge">{{ book.comparison_count }} comparisons</span>
                {% endif %}
            </div>
            {% if book.genres %}
            <div style="margin-top: 5px;">
                {% for g in book.genres[:4] %}
                <a href="/?genre={{ g }}" class="genre-tag">{{ g }}</a>
                {% endfor %}
                {% if book.genres|length > 4 %}
                <span class="genre-tag" style="background: #f0f0f0; color: #888;">+{{ book.genres|length - 4 }} more</span>
                {% endif %}
            </div>
            {% endif %}
            {% if book.post_urls %}
            <div class="book-links">
                {% for url in book.post_urls[:3] %}
                <a href="{{ url }}" target="_blank">{{ url|truncate(50) }}</a>
                {% endfor %}
                {% if book.post_urls|length > 3 %}
                <span style="color: #888; font-size: 12px;">+{{ book.post_urls|length - 3 }} more</span>
                {% endif %}
            </div>
            {% endif %}
        </div>
        {% endfor %}
    {% else %}
        <div class="no-results">No books found matching "{{ query }}"{% if genre %} in {{ genre|title }}{% endif %}</div>
    {% endif %}
</div>
{% else %}
<div class="book-list">
    <div class="no-results">Enter a search query to find books by concept or topic.</div>
</div>
{% endif %}
'''


@app.route('/')
def rankings():
    """Display book rankings page."""
    from db import init_db, get_connection

    init_db()

    genre = request.args.get('genre', '').strip()
    min_comparisons = request.args.get('min_comparisons', type=int)
    top = request.args.get('top', type=int)

    # Get all genres for dropdown
    genres = get_all_genres()

    conn = get_connection()
    cursor = conn.cursor()

    # Build query using precomputed comparison_count column
    query = '''
        SELECT b.id, b.title, b.author, b.bt_score, b.comparison_count
        FROM books b
        WHERE b.bt_score IS NOT NULL
    '''
    params = []

    if genre:
        query += '''
            AND b.id IN (
                SELECT bg.book_id FROM book_genres bg
                JOIN genres g ON bg.genre_id = g.id
                WHERE LOWER(g.name) = LOWER(?)
            )
        '''
        params.append(genre)

    if min_comparisons:
        query += ' AND b.comparison_count >= ?'
        params.append(min_comparisons)

    query += ' ORDER BY b.bt_score DESC'

    if top:
        query += ' LIMIT ?'
        params.append(top)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    # Get post URLs and genres for each book
    books = []
    for book_id, title, author, bt_score, comparison_count in rows:
        cursor.execute('''
            SELECT DISTINCT p.url
            FROM book_mentions bm
            JOIN posts p ON bm.post_id = p.id
            WHERE bm.book_id = ?
            ORDER BY p.date_published DESC
        ''', (book_id,))
        post_urls = [row[0] for row in cursor.fetchall()]

        cursor.execute('''
            SELECT g.name FROM genres g
            JOIN book_genres bg ON g.id = bg.genre_id
            WHERE bg.book_id = ?
            ORDER BY g.name
        ''', (book_id,))
        book_genres = [row[0] for row in cursor.fetchall()]

        books.append({
            'id': book_id,
            'title': title,
            'author': author,
            'bt_score': bt_score,
            'comparison_count': comparison_count,
            'post_urls': post_urls,
            'genres': book_genres
        })

    conn.close()

    return render_page(
        RANKINGS_CONTENT,
        active_page='rankings',
        books=books,
        genres=genres,
        genre=genre,
        min_comparisons=min_comparisons,
        top=top
    )


@app.route('/genres')
def genres():
    """Display genres browsing page."""
    from db import init_db, get_connection

    init_db()

    genre_list = get_all_genres()

    # Calculate total books with genres
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(DISTINCT book_id) FROM book_genres')
    total_books = cursor.fetchone()[0]
    conn.close()

    return render_page(
        GENRES_CONTENT,
        active_page='genres',
        genres=genre_list,
        total_books=total_books
    )


@app.route('/search')
def search():
    """Display search page."""
    from db import init_db, get_connection

    init_db()

    query = request.args.get('q', '').strip()
    genre = request.args.get('genre', '').strip()
    top = request.args.get('top', 20, type=int)

    # Get all genres for dropdown
    all_genres = get_all_genres()

    books = []
    if query:
        # Import here to avoid loading model on every request
        from embeddings import semantic_search
        books = semantic_search(
            query=query,
            top_k=top,
            genre=genre if genre else None,
            model_name='all-MiniLM-L6-v2',
            similarity_threshold=0.25
        )

        # Add genres and comparison count to each book
        conn = get_connection()
        cursor = conn.cursor()
        for book in books:
            cursor.execute('''
                SELECT g.name FROM genres g
                JOIN book_genres bg ON g.id = bg.genre_id
                WHERE bg.book_id = ?
                ORDER BY g.name
            ''', (book['book_id'],))
            book['genres'] = [row[0] for row in cursor.fetchall()]

            # Use precomputed comparison_count
            cursor.execute('SELECT comparison_count FROM books WHERE id = ?', (book['book_id'],))
            book['comparison_count'] = cursor.fetchone()[0] or 0
        conn.close()

    return render_page(
        SEARCH_CONTENT,
        active_page='search',
        books=books,
        genres=all_genres,
        genre=genre,
        query=query,
        top=top
    )


# API endpoints for future use
@app.route('/api/rankings')
def api_rankings():
    """API endpoint for rankings."""
    from db import init_db, get_connection

    init_db()

    genre = request.args.get('genre', '').strip()
    min_comparisons = request.args.get('min_comparisons', type=int)
    top = request.args.get('top', type=int, default=100)

    conn = get_connection()
    cursor = conn.cursor()

    query = '''
        SELECT b.id, b.title, b.author, b.bt_score, b.comparison_count
        FROM books b
        WHERE b.bt_score IS NOT NULL
    '''
    params = []

    if genre:
        query += '''
            AND b.id IN (
                SELECT bg.book_id FROM book_genres bg
                JOIN genres g ON bg.genre_id = g.id
                WHERE LOWER(g.name) = LOWER(?)
            )
        '''
        params.append(genre)

    if min_comparisons:
        query += ' AND b.comparison_count >= ?'
        params.append(min_comparisons)

    query += ' ORDER BY b.bt_score DESC LIMIT ?'
    params.append(top)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    books = [{
        'id': r[0],
        'title': r[1],
        'author': r[2],
        'bt_score': r[3],
        'comparison_count': r[4]
    } for r in rows]

    return jsonify(books)


@app.route('/api/search')
def api_search():
    """API endpoint for search."""
    from db import init_db

    init_db()

    query = request.args.get('q', '').strip()
    top = request.args.get('top', 20, type=int)

    if not query:
        return jsonify({'error': 'Missing query parameter "q"'}), 400

    from embeddings import semantic_search
    books = semantic_search(
        query=query,
        top_k=top,
        model_name='all-MiniLM-L6-v2',
        similarity_threshold=0.25
    )

    return jsonify(books)


def run_web_ui(host='127.0.0.1', port=5001, debug=False):
    """Run the web UI server.

    Args:
        host: Host to bind to (default: 127.0.0.1, use 0.0.0.0 for external access)
        port: Port to run on (default: 5001)
        debug: Enable debug mode (default: False)
    """
    print(f"\nStarting Marginal Revolution Book Reviews Web UI...")
    print(f"Local:   http://127.0.0.1:{port}")
    if host == '0.0.0.0':
        print(f"Network: http://0.0.0.0:{port}")
    print(f"\nPress Ctrl+C to stop.\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_web_ui(debug=True)
