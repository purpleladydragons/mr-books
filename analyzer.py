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


def analyze_post_with_llm(post_content, model='llama3.2:3b'):
    """Analyze a full post with LLM to extract books and sentiment in one call.

    Args:
        post_content: The full post content_text (not a 300-char snippet)
        model: The Ollama model to use (default: llama3.2:3b)

    Returns:
        List of dicts: [{'book_title': str, 'sentiment_score': float, 'reasoning': str}]
        Returns empty list if:
        - Post is empty
        - LLM call fails
        - JSON parsing fails
        - In interview posts where Tyler doesn't express any opinions
    """
    import json
    import urllib.request
    import urllib.error

    if not post_content or not post_content.strip():
        return []

    # Truncate very long posts to avoid token limits (keep first 8000 chars)
    analysis_text = post_content[:8000] if len(post_content) > 8000 else post_content

    # Detect if this is an interview/transcript
    is_interview = is_interview_or_transcript(post_content)

    if is_interview:
        prompt = f"""You are analyzing a blog post from Tyler Cowen's "Marginal Revolution" blog. This appears to be an interview or transcript with multiple speakers.

IMPORTANT: Only extract books that TYLER COWEN personally expresses an opinion about. Ignore book recommendations from guests or interviewees.

Look for speaker tags like "TYLER:", "TYLER COWEN:", "TC:" to identify Tyler's statements.

Text:
{analysis_text}

Find ALL books mentioned in the post that Tyler Cowen expresses an opinion about. For each book, rate Tyler's sentiment on a scale of 1-10:
- 1-2: Very negative (strongly dislikes, criticizes, warns against)
- 3-4: Negative (dislikes, has significant concerns)
- 5-6: Neutral (mentions without strong opinion, mixed feelings)
- 7-8: Positive (likes, recommends)
- 9-10: Very positive (loves, highly recommends, enthusiastic)

Respond with ONLY a JSON array. Each element should have:
- "book_title": the book title (string)
- "sentiment_score": your rating 1-10 (number)
- "reasoning": brief explanation of Tyler's opinion (string)

If no books are found or Tyler doesn't express opinions about any books, respond with: []

Example response format:
[{{"book_title": "The Great Gatsby", "sentiment_score": 8, "reasoning": "Tyler calls it a masterpiece"}}]"""
    else:
        prompt = f"""You are analyzing a blog post from Tyler Cowen's "Marginal Revolution" blog. This is a regular blog post where Tyler is the author.

Text:
{analysis_text}

Find ALL books mentioned in the post. For each book, rate Tyler Cowen's sentiment on a scale of 1-10:
- 1-2: Very negative (strongly dislikes, criticizes, warns against)
- 3-4: Negative (dislikes, has significant concerns)
- 5-6: Neutral (mentions without strong opinion, mixed feelings)
- 7-8: Positive (likes, recommends)
- 9-10: Very positive (loves, highly recommends, enthusiastic)

Respond with ONLY a JSON array. Each element should have:
- "book_title": the book title (string)
- "sentiment_score": your rating 1-10 (number)
- "reasoning": brief explanation of Tyler's opinion (string)

If no books are found, respond with: []

Example response format:
[{{"book_title": "The Great Gatsby", "sentiment_score": 8, "reasoning": "Tyler calls it a masterpiece"}}]"""

    # Call Ollama API
    try:
        url = 'http://localhost:11434/api/generate'
        data = json.dumps({
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,  # Low temperature for consistent output
            }
        }).encode('utf-8')

        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            response_text = result.get('response', '').strip()

            # Try to parse JSON from the response
            # Handle cases where LLM adds extra text around the JSON
            try:
                # First try direct parse
                books = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON array from the response
                # Look for [...] pattern
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    try:
                        books = json.loads(match.group())
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse JSON from LLM response")
                        return []
                else:
                    # No JSON array found
                    return []

            # Validate and normalize the response
            if not isinstance(books, list):
                return []

            normalized_books = []
            for book in books:
                if not isinstance(book, dict):
                    continue
                if 'book_title' not in book or 'sentiment_score' not in book:
                    continue

                title = book.get('book_title', '').strip()
                if not title or len(title) < 3:
                    continue

                try:
                    score = float(book.get('sentiment_score', 5.5))
                    # Clamp to 1-10 range
                    score = max(1, min(10, score))
                    # Normalize to -1 to 1 scale: (score - 5.5) / 4.5
                    normalized_score = round((score - 5.5) / 4.5, 4)
                except (ValueError, TypeError):
                    normalized_score = 0.0

                normalized_books.append({
                    'book_title': title,
                    'sentiment_score': normalized_score,
                    'reasoning': book.get('reasoning', '')
                })

            return normalized_books

    except urllib.error.URLError as e:
        print(f"Error: Could not connect to Ollama at localhost:11434")
        raise ConnectionError(f"Ollama connection failed: {e}")
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return []


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


def analyze_posts_full_context(posts, model='llama3.2:3b'):
    """Process posts using full-context LLM analysis.

    This mode uses a single LLM call per post to extract all books and their
    sentiment scores together, using the full post content for better understanding.

    Args:
        posts: List of tuples (id, url, title, content_html, content_text)
        model: Ollama model to use
    """
    from db import find_or_create_book, insert_book_mention_with_sentiment, mark_post_books_extracted

    total = len(posts)
    total_books_found = 0
    errors = 0

    print(f"Analyzing {total} posts with full-context LLM mode...")

    for i, (post_id, url, title, content_html, content_text) in enumerate(posts, 1):
        try:
            # Single LLM call returns all books with sentiment
            books = analyze_post_with_llm(content_text, model)

            for book_data in books:
                book_title = book_data['book_title']
                sentiment_score = book_data['sentiment_score']
                reasoning = book_data.get('reasoning', '')

                # Use fuzzy matching for deduplication
                book_id = find_or_create_book(book_title)
                if book_id is None:
                    continue

                # Store full post content as context (or reasoning if provided)
                # Use reasoning as context since it's more concise and directly relevant
                context = reasoning if reasoning else content_text[:500]

                # Insert book mention with sentiment already set
                insert_book_mention_with_sentiment(book_id, post_id, context, sentiment_score)
                total_books_found += 1

            # Mark post as processed
            mark_post_books_extracted(post_id)

        except ConnectionError:
            print(f"\nOllama connection failed. Stopping analysis.")
            return total_books_found, errors
        except Exception as e:
            print(f"\nWarning: Error processing post '{title}': {e}")
            errors += 1
            # Mark post as processed anyway to avoid re-processing on next run
            mark_post_books_extracted(post_id)

        if i % 10 == 0 or i == total:
            print(f"Processed {i}/{total} posts, found {total_books_found} book mentions so far")

    return total_books_found, errors


def analyze_all_posts(use_ollama=False, model='llama3.2:3b', reextract=False, full_context=False):
    """Process all posts to extract books and analyze sentiment.

    Args:
        use_ollama: If True, use Ollama LLM for sentiment analysis instead of VADER
        model: Ollama model to use (default: llama3.2:3b)
        reextract: If True, re-extract books from all posts (ignore cache)
        full_context: If True, use full post content with single LLM call per post (requires --use-ollama)
    """
    from db import get_posts_for_extraction, mark_post_books_extracted, update_book_sentiment

    # full_context requires use_ollama
    if full_context and not use_ollama:
        print("Warning: --full-context requires --use-ollama. Enabling Ollama mode.")
        use_ollama = True

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

        if full_context:
            # Use the new full-context LLM approach
            total_books_found, errors = analyze_posts_full_context(posts, model)
            print(f"\nExtracted {total_books_found} book mentions from {total} posts using full-context LLM.")
            if errors > 0:
                print(f"Encountered {errors} errors during processing.")

            # Update aggregated book sentiment scores
            print("\nUpdating aggregated book sentiment scores...")
            update_book_sentiment()
        else:
            # Use the original extraction approach
            total_books_found = 0
            for i, (post_id, url, title, content_html, content_text) in enumerate(posts, 1):
                books = extract_books_from_post(post_id, content_html, content_text)
                total_books_found += len(books)

                # Mark post as processed
                mark_post_books_extracted(post_id)

                if i % 100 == 0 or i == total:
                    print(f"Processed {i}/{total} posts, found {total_books_found} book mentions so far")

            print(f"\nExtracted {total_books_found} book mentions from {total} posts.")

    # For full_context mode, sentiment is already done. For traditional mode, analyze mentions.
    if not full_context:
        # Now analyze sentiment for all book mentions
        print("\n--- Sentiment Analysis ---")
        analyze_book_mentions(use_ollama=use_ollama, model=model)

    # Categorize books by genre
    print("\n--- Genre Categorization ---")
    categorize_all_books()


# ============================================================
# Bradley-Terry Pairwise Ranking Functions
# ============================================================

def sample_pairs(mention_ids, n_pairs):
    """Randomly sample pairs of book mentions for comparison.

    Args:
        mention_ids: List of mention IDs to sample from
        n_pairs: Number of pairs to sample

    Returns:
        List of tuples: [(mention_a_id, mention_b_id), ...]
    """
    import random

    if len(mention_ids) < 2:
        return []

    pairs = []
    for _ in range(n_pairs):
        # Sample two different mentions
        a, b = random.sample(mention_ids, 2)
        # Always order pair consistently (smaller ID first) to allow deduplication if needed
        if a > b:
            a, b = b, a
        pairs.append((a, b))

    return pairs


def sample_pairs_adaptive(mention_ids, n_pairs, bt_scores, comparison_counts, min_coverage=5):
    """Sample pairs using adaptive/uncertainty sampling.

    Prioritizes pairs where:
    1. Items have fewer than min_coverage comparisons (ensure coverage)
    2. Current BT scores are similar (uncertain rankings)

    Args:
        mention_ids: List of mention IDs to sample from
        n_pairs: Number of pairs to sample
        bt_scores: Dict mapping mention_id to bt_score (None for unscored)
        comparison_counts: Dict mapping mention_id to comparison count
        min_coverage: Minimum comparisons per item before focusing on uncertainty

    Returns:
        Tuple: (pairs, stats) where pairs is List[(mention_a_id, mention_b_id)]
               and stats is dict with sampling statistics
    """
    import random
    import math

    if len(mention_ids) < 2:
        return [], {'random': 0, 'coverage': 0, 'uncertainty': 0}

    # Categorize mentions by coverage
    low_coverage = []  # Items with < min_coverage comparisons
    has_coverage = []  # Items with >= min_coverage comparisons

    for mid in mention_ids:
        count = comparison_counts.get(mid, 0)
        if count < min_coverage:
            low_coverage.append(mid)
        else:
            has_coverage.append(mid)

    # For items with scores, compute uncertainty weights
    # Higher weight = more uncertain (closer scores)
    scored_items = [(mid, bt_scores.get(mid)) for mid in has_coverage
                    if bt_scores.get(mid) is not None]

    pairs = []
    stats = {'random': 0, 'coverage': 0, 'uncertainty': 0}

    for _ in range(n_pairs):
        pair = None

        # Strategy 1: If many items lack coverage, prioritize them (60% chance)
        if low_coverage and random.random() < 0.6:
            # Sample one from low_coverage and one from anywhere
            a = random.choice(low_coverage)
            # Prefer pairing with scored items if available
            if scored_items and random.random() < 0.7:
                b, _ = random.choice(scored_items)
            else:
                b = random.choice([m for m in mention_ids if m != a])
            pair = (a, b)
            stats['coverage'] += 1

        # Strategy 2: Sample based on uncertainty (items with similar scores)
        elif len(scored_items) >= 2 and random.random() < 0.8:
            # Weighted sampling - higher weight for pairs with similar scores
            # Use softmax-like weighting based on score similarity
            pair = _sample_uncertain_pair(scored_items)
            if pair:
                stats['uncertainty'] += 1

        # Strategy 3: Random fallback
        if pair is None:
            a, b = random.sample(mention_ids, 2)
            pair = (a, b)
            stats['random'] += 1

        # Order pair consistently
        if pair[0] > pair[1]:
            pair = (pair[1], pair[0])
        pairs.append(pair)

    return pairs, stats


def _sample_uncertain_pair(scored_items):
    """Sample a pair based on uncertainty (similar BT scores).

    Uses inverse score difference as weight - pairs with similar scores
    are more likely to be sampled.

    Args:
        scored_items: List of tuples (mention_id, bt_score)

    Returns:
        Tuple (mention_a_id, mention_b_id) or None if cannot sample
    """
    import random
    import math

    if len(scored_items) < 2:
        return None

    # For efficiency, don't compute all pairs - sample candidates
    n_candidates = min(50, len(scored_items) * (len(scored_items) - 1) // 2)

    candidates = []
    weights = []

    for _ in range(n_candidates):
        # Sample two different items
        i, j = random.sample(range(len(scored_items)), 2)
        mid_a, score_a = scored_items[i]
        mid_b, score_b = scored_items[j]

        if score_a is None or score_b is None:
            continue

        # Calculate uncertainty weight (inverse of score difference)
        # Add small epsilon to avoid division by zero
        diff = abs(score_a - score_b)
        # Use exponential weighting - much higher weight for close pairs
        weight = math.exp(-5 * diff)  # e^(-5*diff) gives high weight when diff is small

        candidates.append((mid_a, mid_b))
        weights.append(weight)

    if not candidates:
        return None

    # Weighted random choice
    total_weight = sum(weights)
    if total_weight == 0:
        return random.choice(candidates)

    r = random.random() * total_weight
    cumulative = 0
    for pair, weight in zip(candidates, weights):
        cumulative += weight
        if r <= cumulative:
            return pair

    return candidates[-1]  # Fallback


def compute_uncertainty_metric(bt_scores, comparison_counts, min_coverage=5):
    """Compute overall uncertainty metric for the current ranking.

    Higher values indicate more uncertainty in the ranking.

    Args:
        bt_scores: Dict mapping mention_id to bt_score
        comparison_counts: Dict mapping mention_id to comparison count
        min_coverage: Minimum coverage threshold

    Returns:
        Dict with uncertainty metrics
    """
    import math

    scored = [(mid, score) for mid, score in bt_scores.items()
              if score is not None]

    if len(scored) < 2:
        return {'total_uncertainty': float('inf'), 'avg_gap': 0, 'coverage_pct': 0}

    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Compute average gap between adjacent items
    gaps = []
    for i in range(len(scored) - 1):
        gap = abs(scored[i][1] - scored[i + 1][1])
        gaps.append(gap)

    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    # Compute uncertainty as sum of inverse gaps (more uncertainty when gaps are small)
    total_uncertainty = sum(1 / (g + 0.01) for g in gaps)

    # Coverage percentage
    all_mentions = set(bt_scores.keys())
    covered = sum(1 for mid in all_mentions
                  if comparison_counts.get(mid, 0) >= min_coverage)
    coverage_pct = (covered / len(all_mentions) * 100) if all_mentions else 0

    return {
        'total_uncertainty': total_uncertainty,
        'avg_gap': avg_gap,
        'coverage_pct': coverage_pct,
        'n_scored': len(scored),
        'n_low_coverage': len(all_mentions) - covered
    }


def compare_pair_with_llm(mention_a, mention_b, model='llama3.2:3b'):
    """Ask LLM which review is more positive about its book.

    Args:
        mention_a: Tuple (mention_id, book_id, book_title, context_text, post_content)
        mention_b: Tuple (mention_id, book_id, book_title, context_text, post_content)
        model: Ollama model to use

    Returns:
        Tuple (mention_a_id, mention_b_id, winner_id)
        winner_id is None for ties
    """
    import json
    import urllib.request
    import urllib.error

    mention_a_id, book_a_id, title_a, context_a, content_a = mention_a
    mention_b_id, book_b_id, title_b, context_b, content_b = mention_b

    # Use context if available, otherwise use truncated post content
    text_a = context_a if context_a else (content_a[:1000] if content_a else "")
    text_b = context_b if context_b else (content_b[:1000] if content_b else "")

    # Truncate to reasonable length for comparison
    text_a = text_a[:1500] if len(text_a) > 1500 else text_a
    text_b = text_b[:1500] if len(text_b) > 1500 else text_b

    prompt = f"""Compare these two book reviews from Tyler Cowen's blog. Which review expresses a MORE POSITIVE sentiment toward its book?

REVIEW A - About "{title_a}":
{text_a}

REVIEW B - About "{title_b}":
{text_b}

Answer with ONLY one of these options:
- "A" if Review A is more positive about its book
- "B" if Review B is more positive about its book
- "TIE" if they are equally positive or you cannot determine

Your answer (A, B, or TIE):"""

    try:
        url = 'http://localhost:11434/api/generate'
        data = json.dumps({
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,
            }
        }).encode('utf-8')

        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            response_text = result.get('response', '').strip().upper()

            # Parse response
            if 'TIE' in response_text or 'EQUAL' in response_text or 'CANNOT' in response_text:
                return (mention_a_id, mention_b_id, None)
            elif response_text.startswith('A') or 'REVIEW A' in response_text:
                return (mention_a_id, mention_b_id, mention_a_id)
            elif response_text.startswith('B') or 'REVIEW B' in response_text:
                return (mention_a_id, mention_b_id, mention_b_id)
            else:
                # Can't parse, treat as tie
                return (mention_a_id, mention_b_id, None)

    except Exception as e:
        # On error, return tie
        return (mention_a_id, mention_b_id, None)


def compare_pairs_parallel(pairs, model='llama3.2:3b', workers=5, rate_limit=5.0):
    """Compare pairs in parallel using ThreadPoolExecutor.

    Args:
        pairs: List of tuples (mention_a_id, mention_b_id)
        model: Ollama model to use
        workers: Number of concurrent workers
        rate_limit: Max requests per second across all workers

    Returns:
        List of tuples (mention_a_id, mention_b_id, winner_id)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    import time

    from db import get_mention_for_comparison

    # Rate limiter (similar to scraper.py)
    class RateLimiter:
        def __init__(self, rate):
            self.min_interval = 1.0 / rate
            self.last_time = 0
            self.lock = threading.Lock()

        def wait(self):
            with self.lock:
                now = time.time()
                elapsed = now - self.last_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                self.last_time = time.time()

    rate_limiter = RateLimiter(rate_limit)

    def compare_single(pair):
        """Worker function for comparing a single pair."""
        rate_limiter.wait()

        mention_a_id, mention_b_id = pair
        mention_a = get_mention_for_comparison(mention_a_id)
        mention_b = get_mention_for_comparison(mention_b_id)

        if not mention_a or not mention_b:
            return None

        return compare_pair_with_llm(mention_a, mention_b, model)

    results = []
    total = len(pairs)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_pair = {executor.submit(compare_single, pair): pair for pair in pairs}

        completed = 0
        for future in as_completed(future_to_pair):
            completed += 1
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                pass  # Skip failed comparisons

            if completed % 100 == 0 or completed == total:
                print(f"Compared {completed}/{total} pairs ({len(results)} valid)")

    return results


def fit_bradley_terry(comparisons, mention_ids):
    """Fit Bradley-Terry model using choix library.

    Args:
        comparisons: List of tuples (mention_a_id, mention_b_id, winner_id)
        mention_ids: List of all mention IDs (for mapping)

    Returns:
        Dict mapping mention_id to Bradley-Terry score
    """
    import numpy as np

    # Create mapping from mention_id to index
    id_to_idx = {mid: idx for idx, mid in enumerate(mention_ids)}
    idx_to_id = {idx: mid for mid, idx in id_to_idx.items()}
    n_items = len(mention_ids)

    # Convert comparisons to choix format
    # choix expects list of (winner_idx, loser_idx) tuples
    valid_comparisons = []
    ties = 0

    for mention_a_id, mention_b_id, winner_id in comparisons:
        if mention_a_id not in id_to_idx or mention_b_id not in id_to_idx:
            continue

        if winner_id is None:
            # Tie - skip for now (could add tie handling later)
            ties += 1
            continue

        winner_idx = id_to_idx[winner_id]
        loser_idx = id_to_idx[mention_b_id if winner_id == mention_a_id else mention_a_id]
        valid_comparisons.append((winner_idx, loser_idx))

    if not valid_comparisons:
        print("No valid comparisons for Bradley-Terry fitting.")
        return {}

    print(f"Fitting Bradley-Terry model with {len(valid_comparisons)} comparisons ({ties} ties excluded)...")

    try:
        import choix

        # Fit the model using choix's pairwise comparison method
        # Use opt_pairwise for maximum likelihood estimation
        params = choix.opt_pairwise(n_items, valid_comparisons, alpha=0.01)

        # Convert to dict
        scores = {idx_to_id[idx]: float(params[idx]) for idx in range(n_items)}
        return scores

    except Exception as e:
        print(f"Error fitting Bradley-Terry model: {e}")
        return {}


def run_pairwise_ranking(n_comparisons=10000, model='llama3.2:3b', workers=5, adaptive=False):
    """Run pairwise comparison ranking process.

    Args:
        n_comparisons: Number of pairwise comparisons to make
        model: Ollama model to use
        workers: Number of parallel workers
        adaptive: If True, use adaptive/uncertainty sampling with periodic refitting
    """
    from db import (
        get_mentions_for_bt,
        get_all_comparisons,
        get_comparison_count,
        insert_comparisons_batch,
        update_mention_bt_scores_batch,
        update_book_bt_scores,
        get_mention_comparison_counts,
        get_mentions_with_bt_scores
    )

    # Check Ollama availability first
    print("Checking Ollama availability...")
    try:
        check_ollama_available(model)
        print(f"Ollama is available with model '{model}'.")
    except ConnectionError:
        print("\nAborting: Cannot proceed when Ollama is not available.")
        return

    # Get all mention IDs
    mention_ids = get_mentions_for_bt()
    print(f"Found {len(mention_ids)} book mentions for ranking.")

    if len(mention_ids) < 2:
        print("Need at least 2 book mentions for pairwise ranking.")
        return

    # Check existing comparisons
    existing_count = get_comparison_count()
    print(f"Existing comparisons: {existing_count}")

    if adaptive:
        # Run adaptive sampling with periodic refitting
        print(f"\n=== ADAPTIVE SAMPLING MODE ===")
        print(f"Will refit BT model every 1000 comparisons to update uncertainty estimates.")
        _run_adaptive_ranking(mention_ids, n_comparisons, model, workers)
    else:
        # Original random sampling approach
        print(f"Sampling {n_comparisons} pairs for comparison (random)...")
        pairs = sample_pairs(mention_ids, n_comparisons)

        # Run comparisons in parallel
        print(f"Running pairwise comparisons with {workers} workers...")
        results = compare_pairs_parallel(pairs, model=model, workers=workers)

        # Store results
        if results:
            print(f"Storing {len(results)} comparison results...")
            insert_comparisons_batch(results)

        # Final fit
        _fit_and_update_scores(mention_ids)


def _run_adaptive_ranking(mention_ids, n_comparisons, model, workers, refit_interval=1000, min_coverage=5):
    """Run adaptive ranking with periodic BT model refitting.

    Args:
        mention_ids: List of mention IDs
        n_comparisons: Total number of comparisons to make
        model: Ollama model to use
        workers: Number of parallel workers
        refit_interval: Refit BT model every N comparisons
        min_coverage: Minimum comparisons per item before focusing on uncertainty
    """
    from db import (
        get_all_comparisons,
        insert_comparisons_batch,
        update_mention_bt_scores_batch,
        get_mention_comparison_counts,
        get_mentions_with_bt_scores
    )

    total_completed = 0
    total_stats = {'random': 0, 'coverage': 0, 'uncertainty': 0}
    uncertainty_log = []  # Track uncertainty over time

    # Get initial BT scores and comparison counts
    bt_scores = get_mentions_with_bt_scores()
    comparison_counts = get_mention_comparison_counts()

    # If we have existing comparisons but no BT scores, fit initial model
    all_comparisons = get_all_comparisons()
    if all_comparisons and not any(bt_scores.get(mid) is not None for mid in mention_ids):
        print("Fitting initial BT model from existing comparisons...")
        scores = fit_bradley_terry(all_comparisons, mention_ids)
        if scores:
            score_tuples = list(scores.items())
            for i in range(0, len(score_tuples), 1000):
                chunk = score_tuples[i:i+1000]
                update_mention_bt_scores_batch(chunk)
            bt_scores = scores

    # Log initial uncertainty
    initial_metrics = compute_uncertainty_metric(bt_scores, comparison_counts, min_coverage)
    print(f"\nInitial state:")
    print(f"  Scored items: {initial_metrics['n_scored']}")
    print(f"  Low coverage items: {initial_metrics['n_low_coverage']}")
    print(f"  Coverage: {initial_metrics['coverage_pct']:.1f}%")
    if initial_metrics['n_scored'] > 0:
        print(f"  Total uncertainty: {initial_metrics['total_uncertainty']:.2f}")
        print(f"  Avg gap between ranks: {initial_metrics['avg_gap']:.4f}")
    uncertainty_log.append((0, initial_metrics))

    while total_completed < n_comparisons:
        # Calculate batch size (up to refit_interval or remaining)
        batch_size = min(refit_interval, n_comparisons - total_completed)

        print(f"\n--- Batch {total_completed // refit_interval + 1}: {batch_size} comparisons ---")

        # Sample pairs using adaptive strategy
        pairs, batch_stats = sample_pairs_adaptive(
            mention_ids, batch_size, bt_scores, comparison_counts, min_coverage
        )

        # Update totals
        for key in batch_stats:
            total_stats[key] += batch_stats[key]

        print(f"Sampling strategy: {batch_stats['coverage']} coverage, "
              f"{batch_stats['uncertainty']} uncertainty, {batch_stats['random']} random")

        # Run comparisons
        print(f"Running comparisons with {workers} workers...")
        results = compare_pairs_parallel(pairs, model=model, workers=workers)

        # Store results
        if results:
            print(f"Storing {len(results)} comparison results...")
            insert_comparisons_batch(results)

        total_completed += batch_size

        # Update comparison counts
        for a, b, _ in results:
            comparison_counts[a] = comparison_counts.get(a, 0) + 1
            comparison_counts[b] = comparison_counts.get(b, 0) + 1

        # Refit BT model
        print("Refitting Bradley-Terry model...")
        all_comparisons = get_all_comparisons()
        scores = fit_bradley_terry(all_comparisons, mention_ids)

        if scores:
            # Update in-memory scores
            bt_scores = scores

            # Persist to DB
            score_tuples = list(scores.items())
            for i in range(0, len(score_tuples), 1000):
                chunk = score_tuples[i:i+1000]
                update_mention_bt_scores_batch(chunk)

        # Compute and log uncertainty
        metrics = compute_uncertainty_metric(bt_scores, comparison_counts, min_coverage)
        uncertainty_log.append((total_completed, metrics))

        print(f"Progress: {total_completed}/{n_comparisons} comparisons")
        print(f"  Coverage: {metrics['coverage_pct']:.1f}%")
        if metrics['n_scored'] > 0:
            print(f"  Total uncertainty: {metrics['total_uncertainty']:.2f}")
            # Calculate uncertainty reduction
            if len(uncertainty_log) >= 2 and uncertainty_log[-2][1]['total_uncertainty'] > 0:
                prev_uncertainty = uncertainty_log[-2][1]['total_uncertainty']
                reduction = (prev_uncertainty - metrics['total_uncertainty']) / prev_uncertainty * 100
                print(f"  Uncertainty reduction: {reduction:+.1f}%")

    # Final summary
    print("\n" + "=" * 60)
    print("ADAPTIVE SAMPLING COMPLETE")
    print("=" * 60)
    print(f"\nTotal comparisons made: {total_completed}")
    print(f"Sampling strategy breakdown:")
    print(f"  Coverage-focused: {total_stats['coverage']} ({total_stats['coverage']/total_completed*100:.1f}%)")
    print(f"  Uncertainty-focused: {total_stats['uncertainty']} ({total_stats['uncertainty']/total_completed*100:.1f}%)")
    print(f"  Random: {total_stats['random']} ({total_stats['random']/total_completed*100:.1f}%)")

    # Show uncertainty reduction over time
    if len(uncertainty_log) >= 2:
        initial = uncertainty_log[0][1]
        final = uncertainty_log[-1][1]
        if initial['total_uncertainty'] > 0 and final['total_uncertainty'] < float('inf'):
            total_reduction = (initial['total_uncertainty'] - final['total_uncertainty']) / initial['total_uncertainty'] * 100
            print(f"\nUncertainty reduction: {initial['total_uncertainty']:.2f} → {final['total_uncertainty']:.2f} ({total_reduction:+.1f}%)")
            efficiency = total_reduction / total_completed * 1000 if total_completed > 0 else 0
            print(f"Efficiency: {efficiency:.2f}% uncertainty reduction per 1000 comparisons")

    # Aggregate to book level
    from db import update_book_bt_scores
    print("\nAggregating scores to book level...")
    update_book_bt_scores()

    print("\nDone! Bradley-Terry ranking complete.")
    print(f"Run 'python main.py rankings' to see the results.")


def _fit_and_update_scores(mention_ids):
    """Fit BT model and update scores in database."""
    from db import (
        get_all_comparisons,
        update_mention_bt_scores_batch,
        update_book_bt_scores
    )

    # Get all comparisons (including previous runs)
    all_comparisons = get_all_comparisons()
    total_comparisons = len(all_comparisons)
    print(f"Total comparisons available: {total_comparisons}")

    # Fit Bradley-Terry model
    print("\nFitting Bradley-Terry model...")
    scores = fit_bradley_terry(all_comparisons, mention_ids)

    if scores:
        # Update mention scores
        print(f"Updating Bradley-Terry scores for {len(scores)} mentions...")
        score_tuples = list(scores.items())
        # Batch update in chunks of 1000
        for i in range(0, len(score_tuples), 1000):
            chunk = score_tuples[i:i+1000]
            update_mention_bt_scores_batch(chunk)

        # Aggregate to book level
        print("Aggregating scores to book level...")
        update_book_bt_scores()

        print("\nDone! Bradley-Terry ranking complete.")
        print(f"Run 'python main.py rankings' to see the results.")
    else:
        print("No scores computed. Need more comparisons or valid data.")
