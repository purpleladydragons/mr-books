"""
Embedding and semantic search functionality for book reviews.
"""

import hashlib
import numpy as np

DEFAULT_MODEL = 'all-MiniLM-L6-v2'
MAX_TEXT_LENGTH = 2000  # Approx 512 tokens


def get_embedding_text(title, author, context_text):
    """Build text for embedding a book.

    Args:
        title: The book's title
        author: Author name (may be None)
        context_text: Combined context_text from all book_mentions

    Returns:
        Combined text for embedding (truncated to MAX_TEXT_LENGTH chars)
    """
    parts = [title]
    if author:
        parts.append(f"by {author}")

    header = " - ".join(parts)

    if context_text:
        # Truncate context to fit within limit
        remaining = MAX_TEXT_LENGTH - len(header) - 2  # 2 for ". "
        truncated_context = context_text[:remaining] if remaining > 0 else ""
        return f"{header}. {truncated_context}"
    else:
        return header


def compute_text_hash(text):
    """Compute SHA256 hash of text for cache invalidation."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def generate_embeddings(model_name=DEFAULT_MODEL, reset=False):
    """Generate embeddings for all books.

    Args:
        model_name: Sentence transformer model to use
        reset: If True, regenerate all embeddings (ignore cache)
    """
    from sentence_transformers import SentenceTransformer
    from db import (
        get_books_for_embedding,
        get_existing_embeddings,
        store_embeddings_batch,
        clear_embeddings,
        init_db
    )

    init_db()

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # Get all books with context
    print("Fetching books from database...")
    books = get_books_for_embedding()
    total_books = len(books)
    print(f"Found {total_books} books.")

    if total_books == 0:
        print("No books found. Run 'python main.py analyze' first.")
        return

    # Get existing embeddings for cache check
    if reset:
        print("Reset flag set - clearing existing embeddings...")
        clear_embeddings(model_name)
        existing = {}
    else:
        existing = get_existing_embeddings(model_name)
        print(f"Found {len(existing)} existing embeddings.")

    # Prepare texts for embedding
    to_embed = []
    for book_id, title, author, context_text in books:
        text = get_embedding_text(title, author, context_text)
        text_hash = compute_text_hash(text)

        # Skip if already embedded with same text
        if book_id in existing and existing[book_id] == text_hash:
            continue

        to_embed.append((book_id, text, text_hash))

    if not to_embed:
        print("All books already have up-to-date embeddings.")
        return

    print(f"Generating embeddings for {len(to_embed)} books...")

    # Batch encode for efficiency
    texts = [t[1] for t in to_embed]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Prepare batch insert
    embeddings_data = []
    for i, (book_id, text, text_hash) in enumerate(to_embed):
        embedding_bytes = embeddings[i].astype(np.float32).tobytes()
        embeddings_data.append((book_id, embedding_bytes, text_hash, model_name))

    # Store in batch
    print("Storing embeddings in database...")
    store_embeddings_batch(embeddings_data)

    print(f"Done! Generated embeddings for {len(to_embed)} books.")


def semantic_search(query, top_k=20, genre=None, min_score=None,
                    model_name=DEFAULT_MODEL, show_scores=False,
                    similarity_threshold=0.3):
    """Perform semantic search for books.

    Args:
        query: Search query string
        top_k: Number of results to return
        genre: Optional genre filter
        min_score: Optional minimum BT score filter
        model_name: Sentence transformer model to use
        show_scores: If True, include similarity scores in output
        similarity_threshold: Minimum similarity to include (default 0.3)

    Returns:
        List of dicts with book info, similarity scores, and post URLs
    """
    from sentence_transformers import SentenceTransformer
    from db import load_embeddings_with_genre, get_embedding_count, get_post_urls_for_book, init_db

    init_db()

    # Check if embeddings exist
    count = get_embedding_count(model_name)
    if count == 0:
        print(f"No embeddings found for model '{model_name}'.")
        print("Run 'python main.py embed' first to generate embeddings.")
        return []

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # Embed the query
    query_embedding = model.encode(query, convert_to_numpy=True).astype(np.float32)

    # Load book embeddings with filters
    print("Searching...")
    rows = load_embeddings_with_genre(model_name, genre, min_score)

    if not rows:
        if genre:
            print(f"No books found with genre '{genre}'.")
        else:
            print("No books found matching filters.")
        return []

    # Compute similarities
    results = []
    for book_id, embedding_blob, title, author, bt_score in rows:
        book_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        # Cosine similarity
        similarity = np.dot(query_embedding, book_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(book_embedding)
        )

        # Filter by similarity threshold
        if similarity >= similarity_threshold:
            results.append({
                'book_id': book_id,
                'title': title,
                'author': author,
                'bt_score': bt_score,
                'similarity': float(similarity)
            })

    # Sort by BT score (highest first), with None values at the end
    results.sort(key=lambda x: (x['bt_score'] is not None, x['bt_score'] or 0), reverse=True)

    # Take top_k and fetch post URLs
    results = results[:top_k]
    for r in results:
        r['post_urls'] = get_post_urls_for_book(r['book_id'])

    return results


def format_search_results(results, show_scores=False):
    """Format search results for display.

    Args:
        results: List of result dicts from semantic_search
        show_scores: If True, show similarity scores
    """
    if not results:
        return

    print(f"\nFound {len(results)} results (sorted by BT score):\n")
    print("=" * 80)

    for i, r in enumerate(results, 1):
        title = r['title']
        author = r['author'] or 'Unknown'
        bt_score = r['bt_score']
        similarity = r['similarity']
        post_urls = r.get('post_urls', [])

        # Format BT score
        bt_str = f"{bt_score:.2f}" if bt_score is not None else "N/A"

        # Build header line
        line = f"{i:3}. {title}"
        if author != 'Unknown':
            line += f" by {author}"
        line += f"  [BT: {bt_str}]"

        if show_scores:
            line += f"  (sim: {similarity:.3f})"

        print(line)

        # Print post URLs indented
        for url in post_urls:
            print(f"        {url}")

        print()  # Blank line between entries

    print("=" * 80)
