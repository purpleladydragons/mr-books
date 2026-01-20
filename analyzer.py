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


def is_interview_or_transcript(text):
    """Detect if text appears to be from an interview or transcript.

    Interview/transcript posts typically have speaker tags like:
    - 'TYLER COWEN:', 'TYLER:', 'TC:'
    - 'GUEST:', 'RUSS ROBERTS:', 'Name:'
    - Lines starting with a name followed by a colon

    Returns True if the text appears to be from an interview/transcript.
    """
    if not text:
        return False

    # Common speaker tag patterns
    speaker_patterns = [
        r'^[A-Z][A-Z\s]+:',  # ALL CAPS NAME:
        r'^[A-Z][a-z]+\s+[A-Z][a-z]+:',  # First Last:
        r'^TYLER\s*(COWEN)?:',  # TYLER: or TYLER COWEN:
        r'^TC:',  # TC:
        r'^GUEST:',
        r'^HOST:',
        r'^INTERVIEWER:',
        r'^Q:',  # Q: for questions
        r'^A:',  # A: for answers
    ]

    # Check if multiple lines match speaker patterns (need at least 2 different speakers)
    lines = text.split('\n')
    speaker_lines = 0
    for line in lines:
        line = line.strip()
        for pattern in speaker_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                speaker_lines += 1
                break

    # If at least 2 speaker-tagged lines, likely a transcript
    return speaker_lines >= 2


def extract_tyler_opinion(text, book_title):
    """Extract only Tyler Cowen's opinion from interview/transcript text.

    Args:
        text: The full context text
        book_title: The book title being discussed

    Returns:
        Tyler's opinion text if found, None if Tyler doesn't express an opinion
    """
    if not text or not book_title:
        return None

    lines = text.split('\n')
    tyler_sections = []
    current_speaker = None
    current_text = []

    # Patterns that indicate Tyler is speaking
    tyler_patterns = [
        r'^TYLER\s*(COWEN)?:',
        r'^TC:',
    ]

    # Patterns that indicate someone else is speaking
    other_speaker_pattern = r'^[A-Z][A-Z\s]*:|^[A-Z][a-z]+\s+[A-Z][a-z]+:'

    for line in lines:
        line_stripped = line.strip()

        # Check if this line starts a new speaker section
        is_tyler = any(re.match(p, line_stripped, re.IGNORECASE) for p in tyler_patterns)
        is_other = re.match(other_speaker_pattern, line_stripped) and not is_tyler

        if is_tyler:
            # Save previous Tyler section if any
            if current_speaker == 'tyler' and current_text:
                tyler_sections.append(' '.join(current_text))
            current_speaker = 'tyler'
            # Remove the speaker tag from the line
            for p in tyler_patterns:
                line_stripped = re.sub(p, '', line_stripped, flags=re.IGNORECASE).strip()
            current_text = [line_stripped] if line_stripped else []
        elif is_other:
            # Save previous Tyler section if any
            if current_speaker == 'tyler' and current_text:
                tyler_sections.append(' '.join(current_text))
            current_speaker = 'other'
            current_text = []
        elif current_speaker:
            # Continue the current speaker's section
            current_text.append(line_stripped)

    # Don't forget the last section
    if current_speaker == 'tyler' and current_text:
        tyler_sections.append(' '.join(current_text))

    if not tyler_sections:
        return None

    # Find sections where Tyler mentions the book
    book_lower = book_title.lower()
    relevant_sections = [s for s in tyler_sections if book_lower in s.lower()]

    if relevant_sections:
        return ' '.join(relevant_sections)

    # If book not mentioned in Tyler's sections, return None
    # This means Tyler didn't express an opinion about this book
    return None


def check_ollama_available(model='llama3.2:3b'):
    """Check if Ollama is available and the model is loaded.

    Args:
        model: The Ollama model to check for

    Returns:
        True if Ollama is available, raises ConnectionError otherwise
    """
    import json
    import urllib.request
    import urllib.error

    try:
        url = 'http://localhost:11434/api/tags'
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode('utf-8'))
            models = [m.get('name', '') for m in result.get('models', [])]
            # Check if requested model is available (handle both 'model' and 'model:tag' formats)
            model_base = model.split(':')[0]
            available = any(m.startswith(model_base) for m in models)
            if not available:
                print(f"Warning: Model '{model}' not found in Ollama. Available models: {models}")
                print(f"  - Pull the model with: ollama pull {model}")
                raise ConnectionError(f"Model '{model}' not available in Ollama")
            return True
    except urllib.error.URLError as e:
        print(f"Error: Could not connect to Ollama at localhost:11434. Is Ollama running?")
        print(f"  - Start Ollama with: ollama serve")
        print(f"  - Make sure model is available: ollama pull {model}")
        raise ConnectionError(f"Ollama connection failed: {e}")


def analyze_sentiment_ollama(context_text, book_title=None, model='llama3.2:3b'):
    """Analyze sentiment using Ollama LLM for better context understanding.

    Args:
        context_text: The text context where the book is mentioned
        book_title: The title of the book (used for interview extraction)
        model: The Ollama model to use (default: llama3.2:3b)

    Returns:
        Normalized score (-1 to 1) or None if:
        - Text is empty
        - Ollama is not available
        - In interview posts where Tyler doesn't express an opinion
    """
    import json
    import urllib.request
    import urllib.error

    if not context_text or not context_text.strip():
        return None

    # Check if this is an interview/transcript
    if is_interview_or_transcript(context_text):
        # Extract only Tyler's opinion
        tyler_text = extract_tyler_opinion(context_text, book_title)
        if tyler_text is None:
            # Tyler doesn't express an opinion about this book
            return None
        analysis_text = tyler_text
    else:
        # Regular post - assume all opinions are Tyler's
        analysis_text = context_text

    # Build prompt for the LLM
    prompt = f"""Analyze Tyler Cowen's sentiment toward the book mentioned in this text.

Text:
{analysis_text}

Rate Tyler Cowen's sentiment toward the book on a scale of 1-10:
- 1-2: Very negative (strongly dislikes, criticizes, warns against)
- 3-4: Negative (dislikes, has significant concerns)
- 5-6: Neutral (mentions without strong opinion, mixed feelings)
- 7-8: Positive (likes, recommends)
- 9-10: Very positive (loves, highly recommends, enthusiastic)

Respond with ONLY a single number from 1 to 10. Do not include any explanation."""

    # Call Ollama API
    try:
        url = 'http://localhost:11434/api/generate'
        data = json.dumps({
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,  # Low temperature for consistent ratings
            }
        }).encode('utf-8')

        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            response_text = result.get('response', '').strip()

            # Parse the numeric response
            # Try to extract a number from the response
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response_text)
            if numbers:
                score = float(numbers[0])
                # Clamp to 1-10 range
                score = max(1, min(10, score))
                # Normalize to -1 to 1 scale: (score - 5.5) / 4.5
                normalized = (score - 5.5) / 4.5
                return round(normalized, 4)

            print(f"Warning: Could not parse LLM response: {response_text}")
            return None

    except urllib.error.URLError as e:
        print(f"Error: Could not connect to Ollama at localhost:11434. Is Ollama running?")
        print(f"  - Start Ollama with: ollama serve")
        print(f"  - Make sure model is available: ollama pull {model}")
        raise ConnectionError(f"Ollama connection failed: {e}")
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None


def analyze_book_mentions(use_ollama=False, model='llama3.2:3b'):
    """Analyze sentiment for all unanalyzed book mentions and update book scores.

    Args:
        use_ollama: If True, use Ollama LLM for sentiment analysis instead of VADER
        model: Ollama model to use (default: llama3.2:3b)
    """
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

    if use_ollama:
        print(f"Analyzing sentiment for {total} book mentions using Ollama ({model})...")
    else:
        print(f"Analyzing sentiment for {total} book mentions using VADER...")

    skipped = 0
    analyzed = 0

    for i, (mention_id, context_text, book_title) in enumerate(mentions, 1):
        try:
            if use_ollama:
                score = analyze_sentiment_ollama(context_text, book_title, model)
            else:
                score = analyze_sentiment(context_text)

            if score is not None:
                update_mention_sentiment(mention_id, score)
                analyzed += 1
            else:
                skipped += 1

        except ConnectionError:
            # Ollama connection failed - abort
            print(f"\nAborting: Ollama is not available. Analyzed {analyzed} mentions before error.")
            return
        except Exception as e:
            print(f"\nError analyzing mention {mention_id}: {e}")
            skipped += 1

        if i % 100 == 0 or i == total:
            print(f"Analyzed {i}/{total} mentions ({analyzed} scored, {skipped} skipped)")

    # Now update aggregated book sentiment scores
    print("Updating aggregated book sentiment scores...")
    update_book_sentiment()

    print(f"Done! Scored {analyzed} mentions, skipped {skipped}.")


def categorize_book(content_text):
    """Categorize a book by genre based on content keywords.

    Uses keyword matching on post content to assign one of:
    history, economics, fiction, biography, science, philosophy, other

    Args:
        content_text: The text content of posts mentioning this book

    Returns:
        A genre string
    """
    if not content_text:
        return 'other'

    text_lower = content_text.lower()

    # Genre keywords - ordered by specificity
    genre_keywords = {
        'biography': [
            'biography', 'biographies', 'autobiograph', 'memoir', 'life of',
            'life story', 'personal history', 'born in', 'childhood',
            'his life', 'her life', 'their life', 'my life'
        ],
        'economics': [
            'econom', 'market', 'gdp', 'inflation', 'monetary', 'fiscal',
            'trade', 'capitalism', 'socialist', 'price', 'supply', 'demand',
            'wealth', 'poverty', 'inequality', 'growth', 'recession',
            'finance', 'banking', 'investment', 'stocks', 'bonds',
            'entrepreneur', 'business', 'corporation'
        ],
        'history': [
            'history', 'histor', 'century', 'ancient', 'medieval', 'war',
            'empire', 'dynasty', 'revolution', 'colonial', 'civilization',
            'world war', 'civil war', 'historical', 'era', 'period',
            'archaeological', 'antiquity'
        ],
        'science': [
            'science', 'scientific', 'physics', 'chemistry', 'biology',
            'evolution', 'genetic', 'quantum', 'neuroscience', 'brain',
            'experiment', 'research', 'hypothesis', 'theory', 'discovery',
            'mathematician', 'mathematics', 'algorithm', 'computer',
            'technology', 'engineering', 'medical', 'disease', 'psychology'
        ],
        'philosophy': [
            'philosophy', 'philosophical', 'ethics', 'moral', 'metaphysics',
            'epistemology', 'ontology', 'existential', 'meaning of life',
            'consciousness', 'free will', 'determinism', 'justice',
            'virtue', 'aesthetic', 'logic', 'reason', 'truth'
        ],
        'fiction': [
            'novel', 'fiction', 'story', 'narrator', 'character',
            'protagonist', 'plot', 'narrative', 'literary', 'literature',
            'tale', 'prose', 'short stories', 'imaginary', 'fictitious'
        ]
    }

    # Count matches for each genre
    genre_scores = {}
    for genre, keywords in genre_keywords.items():
        score = 0
        for keyword in keywords:
            # Count occurrences
            count = text_lower.count(keyword)
            score += count
        genre_scores[genre] = score

    # Find genre with highest score
    best_genre = 'other'
    best_score = 0

    for genre, score in genre_scores.items():
        if score > best_score:
            best_score = score
            best_genre = genre

    # Only return a specific genre if we have at least some keyword matches
    if best_score < 2:
        return 'other'

    return best_genre


def categorize_all_books():
    """Categorize all books that don't have a genre yet."""
    from db import get_books_without_genre, get_book_contexts, update_book_genre

    books = get_books_without_genre()
    total = len(books)

    if total == 0:
        print("No books without genres found.")
        return

    print(f"Categorizing {total} books by genre...")

    for i, (book_id,) in enumerate(books, 1):
        # Get all contexts for this book
        contexts = get_book_contexts(book_id)
        combined_content = ' '.join(contexts)

        # Categorize based on combined content
        genre = categorize_book(combined_content)
        update_book_genre(book_id, genre)

        if i % 100 == 0 or i == total:
            print(f"Categorized {i}/{total} books")

    print("Done!")


def analyze_all_posts(use_ollama=False, model='llama3.2:3b', reextract=False):
    """Process all posts to extract books and analyze sentiment.

    Args:
        use_ollama: If True, use Ollama LLM for sentiment analysis instead of VADER
        model: Ollama model to use (default: llama3.2:3b)
        reextract: If True, re-extract books from all posts (ignore cache)
    """
    from db import get_posts_for_extraction, mark_post_books_extracted

    # Fail fast: Check Ollama availability BEFORE starting book extraction
    if use_ollama:
        print("Checking Ollama availability...")
        try:
            check_ollama_available(model)
            print(f"Ollama is available with model '{model}'.")
        except ConnectionError:
            print("\nAborting: Cannot proceed with --use-ollama when Ollama is not available.")
            return

    posts = get_posts_for_extraction(reextract=reextract)
    total = len(posts)

    if total == 0:
        if reextract:
            print("No posts with content found. Run 'python main.py scrape' first.")
        else:
            print("No new posts to process. All posts have already had books extracted.")
            print("Use --reextract to force re-extraction from all posts.")
        # Still run sentiment analysis and categorization for any unanalyzed mentions
    else:
        if reextract:
            print(f"Re-extracting books from {total} posts (--reextract flag set)...")
        else:
            print(f"Extracting books from {total} new posts...")

        total_books_found = 0
        for i, (post_id, url, title, content_html, content_text) in enumerate(posts, 1):
            books = extract_books_from_post(post_id, content_html, content_text)
            total_books_found += len(books)

            # Mark post as processed
            mark_post_books_extracted(post_id)

            if i % 100 == 0 or i == total:
                print(f"Processed {i}/{total} posts, found {total_books_found} book mentions so far")

        print(f"\nExtracted {total_books_found} book mentions from {total} posts.")

    # Now analyze sentiment for all book mentions
    print("\n--- Sentiment Analysis ---")
    analyze_book_mentions(use_ollama=use_ollama, model=model)

    # Categorize books by genre
    print("\n--- Genre Categorization ---")
    categorize_all_books()
