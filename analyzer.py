"""
NLP and sentiment analysis module for Marginal Revolution Book Reviews.
"""

import re
from bs4 import BeautifulSoup

# Lazy-load spacy to avoid import errors if model not installed
_nlp = None


def get_nlp():
    """Get the spaCy NLP model, loading it lazily."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
            _nlp = spacy.load('en_core_web_sm')
    return _nlp


def extract_italicized_titles(content_html):
    """Extract potential book titles from italicized text (<em> or <i> tags)."""
    if not content_html:
        return []

    soup = BeautifulSoup(content_html, 'lxml')
    titles = []

    for tag in soup.find_all(['em', 'i']):
        text = tag.get_text(strip=True)
        # Filter out likely non-book italics (too short, too long, or common phrases)
        if text and 3 < len(text) < 200:
            # Skip common non-book italicized phrases
            lower = text.lower()
            skip_phrases = ['et al', 'ibid', 'e.g.', 'i.e.', 'etc.', 'vs.', 'per se']
            if not any(phrase == lower for phrase in skip_phrases):
                titles.append(text)

    return titles


def extract_ner_titles(content_text):
    """Extract potential book titles using spaCy NER (WORK_OF_ART entities)."""
    if not content_text:
        return []

    nlp = get_nlp()
    doc = nlp(content_text[:100000])  # Limit to 100k chars to avoid memory issues

    titles = []
    for ent in doc.ents:
        if ent.label_ == 'WORK_OF_ART':
            text = ent.text.strip()
            if text and 3 < len(text) < 200:
                titles.append(text)

    return titles


def extract_pattern_titles(content_text):
    """Extract potential book titles using common patterns."""
    if not content_text:
        return []

    titles = []

    # Patterns for book mentions
    patterns = [
        r'(?:I |we |he |she |Tyler |Cowen )?(?:recommend|recommends|recommended)\s+["\']([^"\']+)["\']',
        r'(?:just |recently )?(?:finished|read|reading)\s+["\']([^"\']+)["\']',
        r'new book\s+["\']([^"\']+)["\']',
        r'(?:the book|his book|her book|this book)\s+["\']([^"\']+)["\']',
        r'(?:titled|called|entitled)\s+["\']([^"\']+)["\']',
        # Match quoted titles that look like book titles
        r'"([A-Z][^"]{5,100})"',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content_text, re.IGNORECASE)
        for match in matches:
            text = match.strip()
            if text and 3 < len(text) < 200:
                titles.append(text)

    return titles


def get_context_for_title(title, content_text, context_chars=300):
    """Get the surrounding context for a title mention."""
    if not content_text or not title:
        return ""

    # Find the title in the text
    idx = content_text.lower().find(title.lower())
    if idx == -1:
        return content_text[:context_chars] if content_text else ""

    # Get context around the mention
    start = max(0, idx - context_chars // 2)
    end = min(len(content_text), idx + len(title) + context_chars // 2)

    context = content_text[start:end].strip()

    # Clean up partial words at boundaries
    if start > 0:
        context = '...' + context.split(' ', 1)[-1] if ' ' in context else '...' + context
    if end < len(content_text):
        context = context.rsplit(' ', 1)[0] + '...' if ' ' in context else context + '...'

    return context


def clean_title(title):
    """Clean and normalize a book title."""
    if not title:
        return None

    # Strip whitespace and quotes
    title = title.strip().strip('"\'')

    # Remove trailing punctuation except for titles ending with ? or !
    if title and title[-1] in '.,;:':
        title = title[:-1].strip()

    # Skip if too short or too long after cleaning
    if not title or len(title) < 4 or len(title) > 200:
        return None

    # Skip common non-book patterns
    skip_patterns = [
        r'^https?://',  # URLs
        r'^www\.',
        r'^\d+$',  # Just numbers
        r'^[A-Z]{2,}$',  # Just acronyms
    ]
    for pattern in skip_patterns:
        if re.match(pattern, title):
            return None

    return title


def extract_books_from_post(post_id, content_html, content_text):
    """Extract book titles from a post using NLP and pattern matching."""
    from db import find_or_create_book, insert_book_mention

    if not content_text and not content_html:
        return []

    all_titles = set()

    # Method 1: Extract from italicized text (HTML)
    italicized = extract_italicized_titles(content_html)
    all_titles.update(italicized)

    # Method 2: Extract using spaCy NER
    ner_titles = extract_ner_titles(content_text)
    all_titles.update(ner_titles)

    # Method 3: Extract using patterns
    pattern_titles = extract_pattern_titles(content_text)
    all_titles.update(pattern_titles)

    # Process and store each unique title
    extracted_books = []
    for title in all_titles:
        cleaned = clean_title(title)
        if not cleaned:
            continue

        # Find existing book (with fuzzy matching) or create new one
        book_id = find_or_create_book(cleaned)
        if book_id is None:
            continue

        # Get context for this mention
        context = get_context_for_title(cleaned, content_text)

        # Create the book mention
        insert_book_mention(book_id, post_id, context)
        extracted_books.append((book_id, cleaned))

    return extracted_books


def analyze_sentiment(context_text):
    """Analyze sentiment of text using VADER.

    Returns compound score (-1 to 1) or None if text is empty.
    """
    if not context_text or not context_text.strip():
        return None

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(context_text)
    return scores['compound']


def analyze_book_mentions():
    """Analyze sentiment for all unanalyzed book mentions and update book scores."""
    from db import (
        get_unanalyzed_mentions,
        update_mention_sentiment,
        update_book_sentiment
    )

    mentions = get_unanalyzed_mentions()
    total = len(mentions)

    if total == 0:
        print("No unanalyzed book mentions found.")
        return

    print(f"Analyzing sentiment for {total} book mentions...")

    for i, (mention_id, context_text) in enumerate(mentions, 1):
        score = analyze_sentiment(context_text)
        if score is not None:
            update_mention_sentiment(mention_id, score)

        if i % 100 == 0 or i == total:
            print(f"Analyzed {i}/{total} mentions")

    # Now update aggregated book sentiment scores
    print("Updating aggregated book sentiment scores...")
    update_book_sentiment()

    print("Done!")


def categorize_book(book_id, content_text):
    """Categorize a book by genre based on content keywords."""
    pass


def analyze_all_posts():
    """Process all posts to extract books and analyze sentiment."""
    from db import get_posts_with_content

    posts = get_posts_with_content()
    total = len(posts)

    if total == 0:
        print("No posts with content found. Run 'python main.py scrape' first.")
        return

    print(f"Analyzing {total} posts for book mentions...")

    total_books_found = 0
    for i, (post_id, url, title, content_html, content_text) in enumerate(posts, 1):
        books = extract_books_from_post(post_id, content_html, content_text)
        total_books_found += len(books)

        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total} posts, found {total_books_found} book mentions so far")

    print(f"\nExtracted {total_books_found} book mentions from {total} posts.")

    # Now analyze sentiment for all book mentions
    print("\n--- Sentiment Analysis ---")
    analyze_book_mentions()
