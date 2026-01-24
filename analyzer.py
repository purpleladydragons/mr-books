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


def clean_markdown_from_json(text):
    """Remove markdown formatting (* and _) from outside quoted strings in JSON.

    LLMs sometimes return markdown-formatted JSON like:
    ["*Title One*", _"Title Two"_]

    This function cleans up such responses to make them valid JSON.

    Args:
        text: The raw LLM response text

    Returns:
        Cleaned text with markdown removed from outside quotes
    """
    if not text:
        return text

    result = []
    in_quotes = False
    i = 0

    while i < len(text):
        char = text[i]

        # Track if we're inside quoted strings
        if char == '"' and (i == 0 or text[i-1] != '\\'):
            in_quotes = not in_quotes
            result.append(char)
        elif char in '*_' and not in_quotes:
            # Skip markdown characters outside quotes
            pass
        else:
            result.append(char)

        i += 1

    return ''.join(result)


def log_parse_failure_to_debug_file(post_title, post_url, raw_response):
    """Log full LLM response to debug file when JSON parsing fails.

    Writes the complete raw response to debug_parse_failures.log for diagnosis.
    This helps identify whether parse failures are due to extra text, unescaped
    characters, truncation, or other issues.

    Args:
        post_title: Title of the post being processed
        post_url: URL of the post (optional, can be None)
        raw_response: The full raw LLM response text
    """
    import os
    from datetime import datetime

    # Write to debug file in same directory as this script
    debug_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_parse_failures.log')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(debug_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Post Title: {post_title or 'Unknown'}\n")
        if post_url:
            f.write(f"Post URL: {post_url}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Full LLM Response:\n{raw_response}\n")
        f.write(f"{'='*80}\n")


def log_comparison_failure_to_debug_file(title_a, title_b, prompt, raw_response):
    """Log full details to debug file when comparison response cannot be parsed.

    Writes to debug_comparison_failures.log for diagnosis. This helps analyze
    patterns in unparseable responses and improve parsing logic.

    Args:
        title_a: Title of book A in the comparison
        title_b: Title of book B in the comparison
        prompt: The full prompt sent to the LLM
        raw_response: The full raw LLM response text
    """
    import os
    from datetime import datetime

    # Write to debug file in same directory as this script
    debug_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_comparison_failures.log')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(debug_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Book A: {title_a or 'Unknown'}\n")
        f.write(f"Book B: {title_b or 'Unknown'}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Full Prompt:\n{prompt}\n")
        f.write(f"\n{'-'*40}\n")
        f.write(f"Full LLM Response:\n{raw_response}\n")
        f.write(f"{'='*80}\n")


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


def check_gemini_available(api_key=None, model='gemini-2.5-flash'):
    """Check if Gemini API is available.

    Args:
        api_key: The Gemini API key (or read from GEMINI_API_KEY env var)
        model: The Gemini model to use (default: gemini-2.5-flash)

    Returns:
        True if Gemini is available, raises ConnectionError otherwise
    """
    import os

    # Get API key from parameter or environment
    key = api_key or os.environ.get('GEMINI_API_KEY')
    if not key:
        print("Error: Gemini API key not provided.")
        print("  - Set GEMINI_API_KEY environment variable, or")
        print("  - Use --api-key flag")
        raise ConnectionError("Gemini API key not provided")

    # Try a simple API call to verify the key works
    try:
        from google import genai
        client = genai.Client(api_key=key)
        # Simple test - try to generate a short response
        response = client.models.generate_content(
            model=model,
            contents="Say 'ok' if you can hear me."
        )
        return True
    except ImportError:
        print("Error: google-genai package not installed.")
        print("  - Install with: pip install google-genai")
        raise ConnectionError("google-genai package not installed")
    except Exception as e:
        print(f"Error: Could not connect to Gemini API: {e}")
        raise ConnectionError(f"Gemini connection failed: {e}")


def check_provider_available(provider='ollama', model=None, api_key=None):
    """Check if the selected LLM provider is available.

    Args:
        provider: 'ollama' or 'gemini'
        model: Model name (defaults based on provider if not specified)
        api_key: API key for Gemini (ignored for Ollama)

    Returns:
        True if provider is available, raises ConnectionError otherwise
    """
    if provider == 'ollama':
        default_model = model or 'llama3.2:3b'
        return check_ollama_available(default_model)
    elif provider == 'gemini':
        default_model = model or 'gemini-2.5-flash'
        return check_gemini_available(api_key, default_model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'gemini'.")


def call_ollama(prompt, model='llama3.2:3b', temperature=0.1, timeout=120):
    """Call Ollama API with a prompt.

    Args:
        prompt: The prompt to send
        model: Ollama model to use
        temperature: Sampling temperature (default: 0.1 for consistent output)
        timeout: Request timeout in seconds

    Returns:
        Response text from the model

    Raises:
        ConnectionError: If Ollama is not available
    """
    import json
    import urllib.request
    import urllib.error

    url = 'http://localhost:11434/api/generate'
    data = json.dumps({
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': temperature,
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', '').strip()
    except urllib.error.URLError as e:
        raise ConnectionError(f"Ollama connection failed: {e}")


def call_gemini(prompt, model='gemini-2.5-flash', api_key=None, temperature=0.1, timeout=120, max_retries=5):
    """Call Gemini API with a prompt.

    Args:
        prompt: The prompt to send
        model: Gemini model to use (default: gemini-2.5-flash)
        api_key: API key (or read from GEMINI_API_KEY env var)
        temperature: Sampling temperature (default: 0.1 for consistent output)
        timeout: Request timeout in seconds (used as request_timeout)
        max_retries: Number of retries for rate limit errors

    Returns:
        Response text from the model

    Raises:
        ConnectionError: If Gemini API is not available
    """
    import os
    import time

    key = api_key or os.environ.get('GEMINI_API_KEY')
    if not key:
        raise ConnectionError("Gemini API key not provided")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ConnectionError("google-genai package not installed")

    client = genai.Client(api_key=key)

    # Configure generation
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=4096,  # Plenty for a JSON array of book titles
    )

    # Retry logic for rate limits
    for attempt in range(max_retries):
        try:
            print(">", end="", flush=True)  # Request starting
            start = time.time()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            elapsed = time.time() - start
            # Check finish reason - log if not STOP (normal completion)
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason = response.candidates[0].finish_reason
                if str(finish_reason) not in ('STOP', 'FinishReason.STOP'):
                    print(f"Warning: Gemini finish_reason={finish_reason}")
            print(f"[{elapsed:.1f}s]", end="", flush=True)  # Progress with timing
            return response.text.strip()
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit or transient server errors (429, 503, 504, overloaded)
            if 'rate' in error_str or '429' in error_str or 'quota' in error_str or '503' in error_str or '504' in error_str or 'overload' in error_str or 'unavailable' in error_str or 'deadline' in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                    print(f"Transient error ({e}), waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            raise ConnectionError(f"Gemini API error: {e}")

    raise ConnectionError("Gemini API: max retries exceeded")


def call_llm(prompt, provider='ollama', model=None, api_key=None, temperature=0.1, timeout=120):
    """Unified function to call LLM with any supported provider.

    Args:
        prompt: The prompt to send
        provider: 'ollama' or 'gemini'
        model: Model name (defaults based on provider if not specified)
        api_key: API key for Gemini (ignored for Ollama)
        temperature: Sampling temperature
        timeout: Request timeout in seconds

    Returns:
        Response text from the model

    Raises:
        ConnectionError: If provider is not available
        ValueError: If provider is unknown
    """
    if provider == 'ollama':
        default_model = model or 'llama3.2:3b'
        return call_ollama(prompt, default_model, temperature, timeout)
    elif provider == 'gemini':
        default_model = model or 'gemini-2.5-flash'
        return call_gemini(prompt, default_model, api_key, temperature, timeout)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'gemini'.")


def analyze_post_with_llm(post_content, model=None, post_title=None, post_url=None, provider='ollama', api_key=None):
    """Analyze a full post with LLM to extract books and sentiment in one call.

    Args:
        post_content: The full post content_text (not a 300-char snippet)
        model: The model to use (defaults based on provider if not specified)
        post_title: Optional post title for error logging
        post_url: Optional post URL for debug logging
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)

    Returns:
        List of dicts: [{'book_title': str, 'sentiment_score': float, 'reasoning': str}]
        Returns empty list if:
        - Post is empty
        - LLM call fails
        - JSON parsing fails
        - In interview posts where Tyler doesn't express any opinions
    """
    import json

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

    # Call LLM
    try:
        response_text = call_llm(prompt, provider=provider, model=model, api_key=api_key, temperature=0.1, timeout=120)

        # Try to parse JSON from the response
        # Handle cases where LLM adds extra text around the JSON
        books = None
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
                    title_info = f" for post: {post_title}" if post_title else ""
                    truncated = response_text[:500] + ("..." if len(response_text) > 500 else "")
                    print(f"Warning: Could not parse JSON from LLM response{title_info}:\n{truncated}")
                    # Log full response to debug file for diagnosis
                    log_parse_failure_to_debug_file(post_title, post_url, response_text)
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

    except ConnectionError:
        raise
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return []


def analyze_sentiment_llm(context_text, book_title=None, model=None, provider='ollama', api_key=None):
    """Analyze sentiment using LLM for better context understanding.

    Args:
        context_text: The text context where the book is mentioned
        book_title: The title of the book (used for interview extraction)
        model: The model to use (defaults based on provider if not specified)
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)

    Returns:
        Normalized score (-1 to 1) or None if:
        - Text is empty
        - LLM is not available
        - In interview posts where Tyler doesn't express an opinion
    """
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

    # Call LLM
    try:
        response_text = call_llm(prompt, provider=provider, model=model, api_key=api_key, temperature=0.1, timeout=60)

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

    except ConnectionError:
        raise
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None


# Alias for backwards compatibility
def analyze_sentiment_ollama(context_text, book_title=None, model='llama3.2:3b'):
    """Backwards-compatible wrapper for analyze_sentiment_llm with Ollama."""
    return analyze_sentiment_llm(context_text, book_title, model=model, provider='ollama')


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

        except ConnectionError as e:
            # LLM connection failed - abort
            print(f"\nAborting: LLM connection failed: {e}. Analyzed {analyzed} mentions before error.")
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


def analyze_posts_full_context(posts, model=None, provider='ollama', api_key=None):
    """Process posts using full-context LLM analysis.

    This mode uses a single LLM call per post to extract all books and their
    sentiment scores together, using the full post content for better understanding.

    Args:
        posts: List of tuples (id, url, title, content_html, content_text)
        model: The model to use (defaults based on provider if not specified)
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)
    """
    from db import find_or_create_book, insert_book_mention_with_sentiment, mark_post_books_extracted

    total = len(posts)
    total_books_found = 0
    errors = 0

    print(f"Analyzing {total} posts with full-context LLM mode using {provider}...")

    for i, (post_id, url, title, content_html, content_text) in enumerate(posts, 1):
        try:
            # Single LLM call returns all books with sentiment
            books = analyze_post_with_llm(content_text, model=model, post_title=title, post_url=url, provider=provider, api_key=api_key)

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

        except ConnectionError as e:
            print(f"\nLLM connection failed: {e}. Stopping analysis.")
            return total_books_found, errors
        except Exception as e:
            print(f"\nWarning: Error processing post '{title}': {e}")
            errors += 1
            # Mark post as processed anyway to avoid re-processing on next run
            mark_post_books_extracted(post_id)

        if i % 10 == 0 or i == total:
            print(f"Processed {i}/{total} posts, found {total_books_found} book mentions so far")

    return total_books_found, errors


def extract_books_only(post_title, post_content, model=None, post_url=None, provider='ollama', api_key=None):
    """Extract ONLY book titles from a post using LLM (no sentiment/reasoning).

    This function asks the LLM to identify all book titles mentioned in the post,
    returning only the titles without any sentiment analysis.

    Args:
        post_title: The title of the post (book might be mentioned here)
        post_content: The full post content_text
        model: The model to use (defaults based on provider if not specified)
        post_url: Optional post URL for debug logging
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)

    Returns:
        List of book title strings, or empty list if:
        - Post is empty
        - LLM call fails
        - JSON parsing fails
        - In interview posts where Tyler doesn't mention any books
    """
    import json

    if not post_content or not post_content.strip():
        return []

    # Truncate very long posts to avoid token limits (keep first 8000 chars)
    analysis_text = post_content[:8000] if len(post_content) > 8000 else post_content

    # Detect if this is an interview/transcript
    is_interview = is_interview_or_transcript(post_content)

    if is_interview:
        prompt = f"""You are analyzing a blog post from Tyler Cowen's "Marginal Revolution" blog. This appears to be an interview or transcript with multiple speakers.

IMPORTANT: Only extract book titles that TYLER COWEN personally mentions or comments on. Ignore book recommendations from guests or interviewees.

Look for speaker tags like "TYLER:", "TYLER COWEN:", "TC:" to identify Tyler's statements.

List ALL book titles mentioned in this post that Tyler Cowen mentions or comments on.
Include the FULL book title with subtitle if present (e.g., "Book Title: The Subtitle").
If the post title contains a book title and the body mentions a subtitle (e.g., "the subtitle is X"), combine them into one full title.

Return ONLY valid JSON. Do NOT use markdown formatting like * or _ in your response.
Respond with a JSON array of book title strings.
Example: ["The Great Gatsby", "Thinking Fast and Slow: Why We Make Bad Decisions"]

If no books are found or Tyler doesn't mention any books, respond with: []

Post Title: {post_title}

Post Content:
{analysis_text}"""
    else:
        prompt = f"""You are analyzing a blog post from Tyler Cowen's "Marginal Revolution" blog. This is a regular blog post where Tyler is the author.

List ALL book titles mentioned in this post.
Include the FULL book title with subtitle if present (e.g., "Book Title: The Subtitle").
If the post title contains a book title and the body mentions a subtitle (e.g., "the subtitle is X"), combine them into one full title.
Include any italicized titles that appear to be books.

Return ONLY valid JSON. Do NOT use markdown formatting like * or _ in your response.
Respond with a JSON array of book title strings.
Example: ["The Great Gatsby", "Thinking Fast and Slow: Why We Make Bad Decisions"]

If no books are found, respond with: []

Post Title: {post_title}

Post Content:
{analysis_text}"""

    # Call LLM
    try:
        response_text = call_llm(prompt, provider=provider, model=model, api_key=api_key, temperature=0.1, timeout=120)

        # Clean markdown formatting from the response before JSON parsing
        cleaned_response = clean_markdown_from_json(response_text)

        # Try to parse JSON from the response
        # Handle cases where LLM adds extra text around the JSON
        book_titles = None
        try:
            # First try direct parse on cleaned response
            book_titles = json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Try to extract JSON array from the cleaned response
            # Look for [...] pattern
            match = re.search(r'\[.*\]', cleaned_response, re.DOTALL)
            if match:
                try:
                    book_titles = json.loads(match.group())
                except json.JSONDecodeError:
                    pass  # Will fall through to fallback extraction

        # If JSON parsing failed, log warning and return empty list (post will be retried)
        if book_titles is None:
            title_info = f" for post: {post_title}" if post_title else ""
            truncated = response_text[:500] + ("..." if len(response_text) > 500 else "")
            print(f"Warning: Could not parse JSON from LLM response{title_info}:\n{truncated}")
            # Log full response to debug file for diagnosis
            log_parse_failure_to_debug_file(post_title, post_url, response_text)
            return []

        # Validate the response
        if not isinstance(book_titles, list):
            return []

        # Filter and clean the titles
        valid_titles = []
        for title in book_titles:
            if not isinstance(title, str):
                continue
            title = title.strip()
            if title and len(title) >= 3 and len(title) <= 200:
                valid_titles.append(title)

        return valid_titles

    except ConnectionError:
        raise
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return []


def extract_books_from_posts_llm(posts, model=None, workers=5, rate_limit=100.0, provider='ollama', api_key=None):
    """Extract books from posts using LLM (extraction only, no sentiment).

    This function extracts book titles and stores full post content as context_text.
    It does NOT compute sentiment scores - that's left for Bradley-Terry ranking.

    Uses parallel processing with ThreadPoolExecutor for faster extraction.

    Args:
        posts: List of tuples (id, url, title, content_html, content_text)
        model: The model to use (defaults based on provider if not specified)
        workers: Number of concurrent workers (default: 5)
        rate_limit: Max requests per second across all workers (default: 5.0)
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)

    Returns:
        Tuple: (total_books_found, errors)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    import time

    from db import find_or_create_book, insert_book_mention, mark_post_books_extracted

    total = len(posts)
    if total == 0:
        return 0, 0

    print(f"Extracting books from {total} posts with {workers} workers using {provider}...")

    # Rate limiter to control request rate across workers
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

    def extract_single_post(post):
        """Worker function for extracting books from a single post."""
        post_id, url, post_title, content_html, content_text = post
        rate_limiter.wait()

        try:
            # LLM call returns list of book title strings
            book_titles = extract_books_only(post_title, content_text, model=model, post_url=url, provider=provider, api_key=api_key)

            # Build full context: post title + content
            full_context = f"{post_title}\n\n{content_text}" if post_title else content_text

            return {
                'post_id': post_id,
                'post_title': post_title,
                'book_titles': book_titles,
                'full_context': full_context,
                'error': None
            }

        except ConnectionError as e:
            return {
                'post_id': post_id,
                'post_title': post_title,
                'book_titles': [],
                'full_context': None,
                'error': f'connection_error: {e}'
            }
        except Exception as e:
            return {
                'post_id': post_id,
                'post_title': post_title,
                'book_titles': [],
                'full_context': None,
                'error': str(e)
            }

    total_books_found = 0
    errors = 0
    completed = 0
    connection_failed = False

    # Process posts in batches to manage memory and provide progress updates
    batch_size = 50
    results_to_write = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_post = {executor.submit(extract_single_post, post): post for post in posts}

        for future in as_completed(future_to_post):
            completed += 1

            try:
                result = future.result()

                if result['error']:
                    if 'connection_error' in result['error']:
                        # Extract actual error message from 'connection_error: {actual_error}'
                        error_msg = result['error'].replace('connection_error: ', '')
                        print(f"\nLLM connection failed: {error_msg}. Stopping extraction.")
                        connection_failed = True
                        # Cancel remaining futures
                        for f in future_to_post:
                            f.cancel()
                        break
                    else:
                        # Error occurred - do NOT mark as extracted so it gets retried next run
                        print(f"\nWarning: Error processing post '{result['post_title']}': {result['error']}")
                        print("  (Post will be retried on next run)")
                        errors += 1
                        # Do NOT add to results_to_write - leave unextracted for retry
                else:
                    # Success (including 0 books found) - mark as extracted
                    results_to_write.append({
                        'post_id': result['post_id'],
                        'book_titles': result['book_titles'],
                        'full_context': result['full_context']
                    })

            except Exception as e:
                errors += 1

            # Write results to DB in batches (sequential writes to avoid SQLite issues)
            if len(results_to_write) >= batch_size or completed == total or connection_failed:
                for res in results_to_write:
                    # Insert book mentions
                    for book_title in res['book_titles']:
                        book_id = find_or_create_book(book_title)
                        if book_id is not None:
                            insert_book_mention(book_id, res['post_id'], res['full_context'])
                            total_books_found += 1
                    # Mark post as extracted (only for successful extractions)
                    mark_post_books_extracted(res['post_id'])

                results_to_write = []

            # Progress logging
            if completed % 50 == 0 or completed == total:
                print(f"Processed {completed}/{total} posts, found {total_books_found} book mentions so far")

            if connection_failed:
                break

    return total_books_found, errors


def analyze_all_posts(use_ollama=False, model=None, reextract=False, full_context=False, extract_only=False, workers=5, reset=False, provider='ollama', api_key=None):
    """Process all posts to extract books and analyze sentiment.

    Args:
        use_ollama: If True, use LLM for sentiment analysis instead of VADER (deprecated, use provider instead)
        model: The model to use (defaults based on provider if not specified)
        reextract: If True, re-extract books from all posts (ignore cache)
        full_context: If True, use full post content with single LLM call per post (requires LLM)
        extract_only: If True, only extract book titles (no sentiment). Incremental by default.
        workers: Number of concurrent workers for LLM extraction (default: 5)
        reset: If True (with extract_only), clear existing data and reprocess all posts
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)
    """
    from db import (
        get_posts_for_extraction, mark_post_books_extracted, update_book_sentiment,
        clear_books_data, reset_extraction_cache, get_extraction_counts
    )

    # If provider is gemini, implicitly set use_ollama to True (for LLM mode)
    if provider == 'gemini':
        use_ollama = True

    # extract_only requires LLM
    if extract_only and not use_ollama:
        print("Warning: --extract-only requires LLM. Enabling LLM mode.")
        use_ollama = True

    # full_context requires LLM
    if full_context and not use_ollama:
        print("Warning: --full-context requires LLM. Enabling LLM mode.")
        use_ollama = True

    # Fail fast: Check provider availability BEFORE starting book extraction
    if use_ollama:
        print(f"Checking {provider} availability...")
        try:
            check_provider_available(provider, model, api_key)
            model_display = model or ('llama3.2:3b' if provider == 'ollama' else 'gemini-2.5-flash-lite')
            print(f"{provider.capitalize()} is available with model '{model_display}'.")
        except ConnectionError:
            print(f"\nAborting: Cannot proceed when {provider} is not available.")
            return

    # For extract_only mode, only clear data if --reset flag is set
    if extract_only:
        print("\n=== EXTRACT-ONLY MODE ===")
        if reset:
            print("--reset flag set: Clearing existing data for fresh start.")
            print("Clearing existing books data...")
            clear_books_data()
            print("Resetting extraction cache (posts.books_extracted_at)...")
            reset_extraction_cache()
            print("Data cleared. Starting fresh extraction...\n")
        else:
            # Log skipped vs processed counts for incremental mode
            total_with_content, already_extracted, to_process = get_extraction_counts()
            print("Incremental mode: Only processing posts not yet extracted.")
            print(f"  Total posts with content: {total_with_content}")
            print(f"  Already extracted (skipping): {already_extracted}")
            print(f"  To be processed: {to_process}")
            print("Use --reset to clear all data and start fresh.\n")

    posts = get_posts_for_extraction(reextract=reextract)
    total = len(posts)

    if total == 0:
        if reextract or extract_only:
            print("No posts with content found. Run 'python main.py scrape' first.")
        else:
            print("No new posts to process. All posts have already had books extracted.")
            print("Use --reextract to force re-extraction from all posts.")
        # Still run sentiment analysis and categorization for any unanalyzed mentions (unless extract_only)
    else:
        if extract_only:
            print(f"Extracting books from {total} posts (extraction only, no sentiment)...")
            # Use the new extraction-only approach with parallel processing
            total_books_found, errors = extract_books_from_posts_llm(posts, model=model, workers=workers, provider=provider, api_key=api_key)
            print(f"\nExtracted {total_books_found} book mentions from {total} posts.")
            if errors > 0:
                print(f"Encountered {errors} errors during processing.")
            print("\nBook extraction complete. Sentiment scores will be computed via Bradley-Terry ranking.")
            print("Run 'python main.py rank --comparisons N' to generate rankings.")
            return  # Skip sentiment analysis and categorization for extract_only mode

        elif reextract:
            print(f"Re-extracting books from {total} posts (--reextract flag set)...")
        else:
            print(f"Extracting books from {total} new posts...")

        if full_context:
            # Use the new full-context LLM approach
            total_books_found, errors = analyze_posts_full_context(posts, model=model, provider=provider, api_key=api_key)
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


def sample_pairs_topk(mention_ids, n_pairs, bt_scores, top_k):
    """Sample pairs to validate top-k tier membership against random opponents.

    New sampling strategy focused on tier identification (not precise ranking):
    - 80% of comparisons: top-k item vs random item (validate top items deserve their spot)
    - 20% of comparisons: pure random (discover dark horses that might belong in top)

    Items that consistently beat random opponents rise to top; items that lose fall out.
    This is "survival of the fittest" - tier validation, not tournament seeding.

    Args:
        mention_ids: List of mention IDs to sample from
        n_pairs: Number of pairs to sample
        bt_scores: Dict mapping mention_id to bt_score (None for unscored)
        top_k: The target k value (focus on identifying top k items)

    Returns:
        Tuple: (pairs, stats) where pairs is List[(mention_a_id, mention_b_id)]
               and stats is dict with sampling statistics
    """
    import random

    if len(mention_ids) < 2:
        return [], {'top_vs_random': 0, 'random': 0}

    # Get scored items and sort by score
    scored = [(mid, bt_scores.get(mid)) for mid in mention_ids
              if bt_scores.get(mid) is not None]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Define top-k items (the items we're validating)
    top_k_actual = min(top_k, len(scored))
    top_k_items = scored[:top_k_actual] if scored else []
    top_k_ids = set(mid for mid, _ in top_k_items)

    # All items not in top-k (challengers)
    non_top_ids = [mid for mid in mention_ids if mid not in top_k_ids]

    # If no scores yet, use all items as potential top
    if not top_k_items:
        top_k_items = [(mid, None) for mid in mention_ids[:top_k]]
        top_k_ids = set(mid for mid, _ in top_k_items)
        non_top_ids = [mid for mid in mention_ids if mid not in top_k_ids]

    pairs = []
    stats = {'top_vs_random': 0, 'random': 0}

    for _ in range(n_pairs):
        r = random.random()

        # Strategy 1: 80% - Top-k item vs random item (validation)
        # This is the core of the new approach: pit top items against random challengers
        if r < 0.8 and top_k_items and non_top_ids:
            mid_a, _ = random.choice(top_k_items)
            mid_b = random.choice(non_top_ids)
            stats['top_vs_random'] += 1

        # Strategy 2: 20% - Pure random (discover dark horses)
        # Random comparisons can surface items that should be in top-k
        else:
            mid_a, mid_b = random.sample(mention_ids, 2)
            stats['random'] += 1

        # Order pair consistently
        if mid_a > mid_b:
            mid_a, mid_b = mid_b, mid_a
        pairs.append((mid_a, mid_b))

    return pairs, stats


def compute_topk_stability(old_topk, new_topk):
    """Compute how much the top-k set changed between iterations.

    Args:
        old_topk: Set of mention IDs that were in top-k previously
        new_topk: Set of mention IDs that are in top-k now

    Returns:
        Dict with stability metrics:
        - overlap: Number of items in both sets
        - changed: Number of items that entered/left the set
        - stability_pct: Percentage of items that remained (0-100)
    """
    if not old_topk or not new_topk:
        return {'overlap': 0, 'changed': 0, 'stability_pct': 0}

    overlap = len(old_topk & new_topk)
    changed = len(old_topk ^ new_topk)  # Symmetric difference
    stability_pct = (overlap / len(new_topk)) * 100 if new_topk else 0

    return {
        'overlap': overlap,
        'changed': changed,
        'stability_pct': stability_pct
    }


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


def compare_pair_with_llm(mention_a, mention_b, model=None, debug=False, provider='ollama', api_key=None):
    """Ask LLM which review is more positive about its book.

    Args:
        mention_a: Tuple (mention_id, book_id, book_title, context_text, post_content)
        mention_b: Tuple (mention_id, book_id, book_title, context_text, post_content)
        model: The model to use (defaults based on provider if not specified)
        debug: If True, return additional debug info
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)

    Returns:
        If debug=False:
            Tuple (mention_a_id, mention_b_id, winner_id)
            winner_id is None for ties
        If debug=True:
            Dict with keys: result, title_a, title_b, prompt, raw_response, parsed_result, error
    """
    import random
    mention_a_id, book_a_id, title_a, context_a, content_a = mention_a
    mention_b_id, book_b_id, title_b, context_b, content_b = mention_b

    # Use context if available, otherwise use truncated post content
    text_a = context_a if context_a else (content_a[:1000] if content_a else "")
    text_b = context_b if context_b else (content_b[:1000] if content_b else "")

    # Truncate to reasonable length for comparison
    text_a = text_a[:1500] if len(text_a) > 1500 else text_a
    text_b = text_b[:1500] if len(text_b) > 1500 else text_b

    # Randomly swap order to eliminate positional bias
    # Track swap so we can map the answer back correctly
    swapped = random.random() < 0.5
    if swapped:
        prompt_title_a, prompt_title_b = title_b, title_a
        prompt_text_a, prompt_text_b = text_b, text_a
    else:
        prompt_title_a, prompt_title_b = title_a, title_b
        prompt_text_a, prompt_text_b = text_a, text_b

    prompt = f"""Compare these two book reviews from Tyler Cowen's blog. Which review expresses a MORE POSITIVE sentiment toward its book?

IMPORTANT: Each review text may mention MULTIPLE books. You must ONLY consider what Tyler says about the SPECIFIC book named in the heading. Ignore any sentiment about other books mentioned in the same post.

REVIEW A - About "{prompt_title_a}":
{prompt_text_a}

REVIEW B - About "{prompt_title_b}":
{prompt_text_b}

You MUST choose either A or B. There is no tie option. Even if the difference is subtle, one review is always slightly more positive. Look for:
- Superlatives ("best", "excellent", "must-read" vs "good", "interesting", "worth reading")
- Personal endorsements ("I loved", "one of my favorites" vs neutral descriptions)
- Recommendation strength ("highly recommend" vs "might enjoy")
- Criticism level (any negatives mentioned vs pure praise)
- A book merely being listed with no commentary is NEUTRAL/NEGATIVE compared to one with explicit praise

Think step by step:
1. Find the specific sentences about "{prompt_title_a}" in Review A. What sentiment do they express? (1 sentence)
2. Find the specific sentences about "{prompt_title_b}" in Review B. What sentiment do they express? (1 sentence)
3. Compare the two sentiments. Which is more positive?
4. Your final answer (just the letter): A or B"""

    debug_info = {
        'title_a': title_a,
        'title_b': title_b,
        'prompt': prompt,
        'raw_response': None,
        'reasoning': None,
        'parsed_result': None,
        'error': None
    }

    try:
        response_text = call_llm(prompt, provider=provider, model=model, api_key=api_key, temperature=0.1, timeout=60)
        debug_info['raw_response'] = response_text

        # Extract reasoning (everything before the final answer)
        # and parse final answer from the END of the response
        import re
        response_stripped = response_text.strip()

        # Look for standalone A or B at the end of the response
        # Check last line first
        lines = response_stripped.split('\n')
        last_line = lines[-1].strip().upper() if lines else ""
        # Strip markdown formatting (**, *, _) for cleaner parsing
        last_line_clean = re.sub(r'[\*_]+', '', last_line).strip()

        parsed_answer = None

        # Check if last line (cleaned) is just "A" or "B"
        if last_line_clean == 'A':
            parsed_answer = 'A'
        elif last_line_clean == 'B':
            parsed_answer = 'B'
        # Check for "Answer: A" or "Final answer: B" patterns (on cleaned line)
        elif re.search(r'(?:answer|choice|final)[:\s]*([AB])\s*$', last_line_clean, re.IGNORECASE):
            match = re.search(r'(?:answer|choice|final)[:\s]*([AB])\s*$', last_line_clean, re.IGNORECASE)
            parsed_answer = match.group(1).upper()
        # Check last few characters for standalone letter (on full response, stripped of markdown)
        else:
            response_clean = re.sub(r'[\*_]+', '', response_stripped.upper()).strip()
            if re.search(r'\b([AB])\s*$', response_clean):
                match = re.search(r'\b([AB])\s*$', response_clean)
                parsed_answer = match.group(1)
            # Fallback: look for any clear A or B choice in the response
            else:
                # Count mentions of "Review A" vs "Review B" as winner indicators
                a_mentions = len(re.findall(r'review\s*a\s*(?:is|expresses|shows|has)', response_stripped, re.IGNORECASE))
                b_mentions = len(re.findall(r'review\s*b\s*(?:is|expresses|shows|has)', response_stripped, re.IGNORECASE))
                if a_mentions > b_mentions:
                    parsed_answer = 'A'
                elif b_mentions > a_mentions:
                    parsed_answer = 'B'

        # Store reasoning (everything except the final answer line)
        if len(lines) > 1:
            debug_info['reasoning'] = '\n'.join(lines[:-1]).strip()
        else:
            debug_info['reasoning'] = response_stripped

        # Map the LLM's answer back to the original mention IDs
        # If we swapped the order, we need to flip A<->B
        if parsed_answer == 'A':
            # LLM chose position A - map back to original ID
            if swapped:
                # Position A was actually mention_b (we swapped)
                debug_info['parsed_result'] = 'B (swapped)'
                result = (mention_a_id, mention_b_id, mention_b_id)
            else:
                debug_info['parsed_result'] = 'A'
                result = (mention_a_id, mention_b_id, mention_a_id)
        elif parsed_answer == 'B':
            # LLM chose position B - map back to original ID
            if swapped:
                # Position B was actually mention_a (we swapped)
                debug_info['parsed_result'] = 'A (swapped)'
                result = (mention_a_id, mention_b_id, mention_a_id)
            else:
                debug_info['parsed_result'] = 'B'
                result = (mention_a_id, mention_b_id, mention_b_id)
        else:
            # Could not parse answer - skip this comparison rather than add noise
            print(f"Warning: Could not parse A/B from comparison response, skipping. Response: {response_text[:200]}...")
            # Log full details to debug file for analysis
            log_comparison_failure_to_debug_file(title_a, title_b, prompt, response_text)
            debug_info['parsed_result'] = 'SKIPPED (unparseable)'
            result = None

        if debug:
            debug_info['result'] = result
            return debug_info
        return result

    except Exception as e:
        debug_info['error'] = str(e)
        debug_info['parsed_result'] = 'SKIPPED (error)'
        print(f"Warning: Comparison error, skipping: {e}")
        result = None
        if debug:
            debug_info['result'] = result
            return debug_info
        return result


def compare_pairs_parallel(pairs, model=None, workers=5, rate_limit=5.0,
                          debug=False, debug_log=None, save_callback=None, save_interval=500,
                          provider='ollama', api_key=None):
    """Compare pairs in parallel using ThreadPoolExecutor.

    Args:
        pairs: List of tuples (mention_a_id, mention_b_id)
        model: The model to use (defaults based on provider if not specified)
        workers: Number of concurrent workers
        rate_limit: Max requests per second across all workers
        debug: If True, log detailed comparison info
        debug_log: File path to write debug logs (None = print to terminal)
        save_callback: Function to call to save results incrementally (takes list of results)
        save_interval: Save results every N comparisons (default 500)
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)

    Returns:
        List of tuples (mention_a_id, mention_b_id, winner_id)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    import time

    from db import get_mention_for_comparison

    # Set up debug logging
    debug_file = None
    if debug and debug_log:
        debug_file = open(debug_log, 'a')
        debug_file.write(f"\n{'='*80}\n")
        debug_file.write(f"DEBUG LOG - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        debug_file.write(f"{'='*80}\n\n")

    def write_debug(text):
        """Write debug output to file or terminal."""
        if debug_file:
            debug_file.write(text + '\n')
            debug_file.flush()
        else:
            print(text)

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
    comparison_counter = [0]  # Use list for mutable counter in closure
    counter_lock = threading.Lock()

    def compare_single(pair):
        """Worker function for comparing a single pair."""
        rate_limiter.wait()

        mention_a_id, mention_b_id = pair
        mention_a = get_mention_for_comparison(mention_a_id)
        mention_b = get_mention_for_comparison(mention_b_id)

        if not mention_a or not mention_b:
            return None

        return compare_pair_with_llm(mention_a, mention_b, model=model, debug=debug, provider=provider, api_key=api_key)

    def format_debug_output(comparison_num, debug_info):
        """Format debug info for output."""
        lines = []
        lines.append(f"\n{'-'*60}")
        lines.append(f"COMPARISON #{comparison_num}")
        lines.append(f"{'-'*60}")
        lines.append(f"Book A: {debug_info['title_a']}")
        lines.append(f"Book B: {debug_info['title_b']}")
        lines.append("")
        lines.append("PROMPT:")
        lines.append("-" * 40)
        lines.append(debug_info['prompt'])
        lines.append("-" * 40)
        lines.append("")
        lines.append("RAW LLM RESPONSE:")
        lines.append("-" * 40)
        if debug_info['raw_response']:
            lines.append(debug_info['raw_response'])
        else:
            lines.append("(no response)")
        lines.append("-" * 40)
        lines.append("")
        # Show extracted reasoning (chain-of-thought)
        if debug_info.get('reasoning'):
            lines.append("REASONING:")
            lines.append("-" * 40)
            lines.append(debug_info['reasoning'])
            lines.append("-" * 40)
            lines.append("")
        lines.append(f"PARSED RESULT: {debug_info['parsed_result']}")
        if debug_info['error']:
            lines.append(f"ERROR: {debug_info['error']}")
        lines.append(f"{'-'*60}\n")
        return '\n'.join(lines)

    all_results = []  # All results returned at end
    pending_results = []  # Results pending save
    total_saved = [0]  # Track total saved for logging
    total = len(pairs)
    interrupted = [False]  # Track if interrupted

    def save_pending():
        """Save pending results if there are any."""
        nonlocal pending_results
        if pending_results and save_callback:
            save_callback(pending_results)
            total_saved[0] += len(pending_results)
            print(f"Saved {len(pending_results)} comparisons to database ({total_saved[0]} total)")
            pending_results = []

    if debug and workers > 1:
        print("Note: Using --workers 1 recommended with --debug for sequential readable output.")

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_pair = {executor.submit(compare_single, pair): pair for pair in pairs}

            completed = 0
            try:
                for future in as_completed(future_to_pair):
                    completed += 1
                    try:
                        result = future.result()
                        if result:
                            if debug:
                                # result is debug_info dict
                                with counter_lock:
                                    comparison_counter[0] += 1
                                    comp_num = comparison_counter[0]
                                write_debug(format_debug_output(comp_num, result))
                                # Only save if we got a valid comparison (not skipped)
                                if result['result'] is not None:
                                    all_results.append(result['result'])
                                    pending_results.append(result['result'])
                            else:
                                all_results.append(result)
                                pending_results.append(result)
                    except Exception as e:
                        if debug:
                            write_debug(f"\nERROR in comparison: {e}\n")
                        pass  # Skip failed comparisons

                    # Save incrementally every save_interval results
                    if save_callback and len(pending_results) >= save_interval:
                        save_pending()

                    if not debug and (completed % 100 == 0 or completed == total):
                        print(f"Compared {completed}/{total} pairs ({len(all_results)} valid)")

            except KeyboardInterrupt:
                interrupted[0] = True
                print(f"\n\nInterrupted! Saving {len(pending_results)} pending comparisons...")
                # Cancel remaining futures
                for f in future_to_pair:
                    f.cancel()
                raise  # Re-raise after cleanup

        # Save any remaining results after normal completion
        if pending_results:
            save_pending()

        # Calculate and log tie rate and A vs B win rate
        if all_results:
            tie_count = sum(1 for r in all_results if r[2] is None)
            tie_rate = (tie_count / len(all_results)) * 100
            print(f"Tie rate: {tie_rate:.1f}% ({tie_count}/{len(all_results)} comparisons)")

            # Calculate A vs B win rate (excluding ties)
            non_tie_results = [r for r in all_results if r[2] is not None]
            if non_tie_results:
                a_wins = sum(1 for r in non_tie_results if r[2] == r[0])  # winner_id == mention_a_id
                b_wins = len(non_tie_results) - a_wins
                a_rate = (a_wins / len(non_tie_results)) * 100
                b_rate = (b_wins / len(non_tie_results)) * 100
                print(f"A vs B win rate: A={a_rate:.1f}% ({a_wins}), B={b_rate:.1f}% ({b_wins}) - target: 45-55% each")

        if debug:
            write_debug(f"\n{'='*60}")
            write_debug(f"SUMMARY: Completed {len(all_results)} valid comparisons out of {total} pairs")
            if all_results:
                tie_count = sum(1 for r in all_results if r[2] is None)
                tie_rate = (tie_count / len(all_results)) * 100
                write_debug(f"Tie rate: {tie_rate:.1f}% ({tie_count}/{len(all_results)} comparisons)")
                # A vs B win rate in debug output too
                non_tie_results = [r for r in all_results if r[2] is not None]
                if non_tie_results:
                    a_wins = sum(1 for r in non_tie_results if r[2] == r[0])
                    b_wins = len(non_tie_results) - a_wins
                    a_rate = (a_wins / len(non_tie_results)) * 100
                    b_rate = (b_wins / len(non_tie_results)) * 100
                    write_debug(f"A vs B win rate: A={a_rate:.1f}% ({a_wins}), B={b_rate:.1f}% ({b_wins}) - target: 45-55% each")
            write_debug(f"{'='*60}\n")

    except KeyboardInterrupt:
        # Save pending results on Ctrl+C
        if pending_results and save_callback:
            save_callback(pending_results)
            total_saved[0] += len(pending_results)
            print(f"Saved {len(pending_results)} comparisons before exit ({total_saved[0]} total saved)")
        print(f"\nComparisons saved. You can resume later - existing comparisons will be preserved.")
        raise

    finally:
        if debug_file:
            debug_file.close()

    return all_results


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


def run_pairwise_ranking(n_comparisons=10000, model=None, workers=5, adaptive=False, top_k=None,
                         debug=False, debug_log=None, provider='ollama', api_key=None):
    """Run pairwise comparison ranking process.

    Args:
        n_comparisons: Number of pairwise comparisons to make
        model: The model to use (defaults based on provider if not specified)
        workers: Number of parallel workers
        adaptive: If True, use adaptive/uncertainty sampling with periodic refitting
        top_k: If set, focus comparisons on identifying top k items (faster convergence)
        debug: If True, log detailed LLM comparison info
        debug_log: File path to write debug logs (None = print to terminal)
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)
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

    # Check provider availability first
    print(f"Checking {provider} availability...")
    try:
        check_provider_available(provider, model, api_key)
        model_display = model or ('llama3.2:3b' if provider == 'ollama' else 'gemini-2.5-flash-lite')
        print(f"{provider.capitalize()} is available with model '{model_display}'.")
    except ConnectionError:
        print(f"\nAborting: Cannot proceed when {provider} is not available.")
        return

    # Debug mode recommendations
    if debug:
        print("\n=== DEBUG MODE ENABLED ===")
        if workers > 1:
            print("Tip: Use --workers 1 for sequential readable output.")
        if debug_log:
            print(f"Debug output will be written to: {debug_log}")
        else:
            print("Debug output will be printed to terminal.")
        print("")

    # Get all mention IDs
    mention_ids = get_mentions_for_bt()
    print(f"Found {len(mention_ids)} book mentions for ranking.")

    if len(mention_ids) < 2:
        print("Need at least 2 book mentions for pairwise ranking.")
        return

    # Check existing comparisons
    existing_count = get_comparison_count()
    print(f"Existing comparisons: {existing_count}")

    # Validate top_k value
    if top_k is not None and top_k > len(mention_ids):
        print(f"Warning: --top-k {top_k} exceeds total items ({len(mention_ids)}). Using all items.")
        top_k = len(mention_ids)

    if top_k is not None:
        # Run top-k focused sampling with early stopping
        print(f"\n=== TOP-K FOCUSED MODE ===")
        print(f"Focusing on identifying top {top_k} items.")
        print(f"Will refit BT model every 500 comparisons and stop early if top-k stabilizes.")
        _run_topk_ranking(mention_ids, n_comparisons, model, workers, top_k, debug=debug, debug_log=debug_log, provider=provider, api_key=api_key)
    elif adaptive:
        # Run adaptive sampling with periodic refitting
        print(f"\n=== ADAPTIVE SAMPLING MODE ===")
        print(f"Will refit BT model every 1000 comparisons to update uncertainty estimates.")
        _run_adaptive_ranking(mention_ids, n_comparisons, model, workers, debug=debug, debug_log=debug_log, provider=provider, api_key=api_key)
    else:
        # Original random sampling approach
        print(f"Sampling {n_comparisons} pairs for comparison (random)...")
        pairs = sample_pairs(mention_ids, n_comparisons)

        # Create save callback for incremental saving
        def save_results(results_batch):
            """Save comparison results to database."""
            insert_comparisons_batch(results_batch)

        # Run comparisons in parallel with incremental saving
        print(f"Running pairwise comparisons with {workers} workers using {provider}...")
        print(f"Comparisons will be saved every 500 results to avoid losing progress.")
        try:
            results = compare_pairs_parallel(pairs, model=model, workers=workers,
                                            debug=debug, debug_log=debug_log,
                                            save_callback=save_results, save_interval=500,
                                            provider=provider, api_key=api_key)
            # Final fit (results already saved incrementally)
            _fit_and_update_scores(mention_ids)
        except KeyboardInterrupt:
            print("\nRanking interrupted. Run again to continue - existing comparisons are preserved.")
            # Still fit the model with whatever we have
            _fit_and_update_scores(mention_ids)
            return


def _run_adaptive_ranking(mention_ids, n_comparisons, model, workers, refit_interval=1000, min_coverage=5,
                          debug=False, debug_log=None, provider='ollama', api_key=None):
    """Run adaptive ranking with periodic BT model refitting.

    Args:
        mention_ids: List of mention IDs
        n_comparisons: Total number of comparisons to make
        model: The model to use (defaults based on provider if not specified)
        workers: Number of parallel workers
        refit_interval: Refit BT model every N comparisons
        min_coverage: Minimum comparisons per item before focusing on uncertainty
        debug: If True, log detailed LLM comparison info
        debug_log: File path to write debug logs (None = print to terminal)
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)
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

        # Create save callback for incremental saving within batches
        batch_results = []  # Track results for comparison count updates

        def save_results(results_batch):
            """Save comparison results to database and track for count updates."""
            batch_results.extend(results_batch)
            insert_comparisons_batch(results_batch)

        # Run comparisons with incremental saving
        print(f"Running comparisons with {workers} workers using {provider}...")
        try:
            results = compare_pairs_parallel(pairs, model=model, workers=workers,
                                            debug=debug, debug_log=debug_log,
                                            save_callback=save_results, save_interval=500,
                                            provider=provider, api_key=api_key)
        except KeyboardInterrupt:
            print("\n\nRanking interrupted. Fitting model with saved comparisons...")
            # Update comparison counts with what we have
            for a, b, _ in batch_results:
                comparison_counts[a] = comparison_counts.get(a, 0) + 1
                comparison_counts[b] = comparison_counts.get(b, 0) + 1
            # Fit and save final model
            all_comparisons = get_all_comparisons()
            scores = fit_bradley_terry(all_comparisons, mention_ids)
            if scores:
                score_tuples = list(scores.items())
                for i in range(0, len(score_tuples), 1000):
                    chunk = score_tuples[i:i+1000]
                    update_mention_bt_scores_batch(chunk)
            from db import update_book_bt_scores
            update_book_bt_scores()
            print("\nRanking interrupted. Run again to continue - existing comparisons are preserved.")
            return

        total_completed += batch_size

        # Update comparison counts from all batch results
        for a, b, _ in batch_results:
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


def _run_topk_ranking(mention_ids, n_comparisons, model, workers, top_k, refit_interval=500, stable_threshold=3,
                      debug=False, debug_log=None, provider='ollama', api_key=None):
    """Run top-k focused ranking with early stopping on stability.

    Args:
        mention_ids: List of mention IDs
        n_comparisons: Maximum number of comparisons to make
        model: The model to use (defaults based on provider if not specified)
        workers: Number of parallel workers
        top_k: Target k value (focus on identifying top k items)
        refit_interval: Refit BT model every N comparisons (default 500)
        stable_threshold: Stop early if top-k stable for this many consecutive refits (default 3)
        debug: If True, log detailed LLM comparison info
        debug_log: File path to write debug logs (None = print to terminal)
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini (ignored for Ollama)
    """
    from db import (
        get_all_comparisons,
        insert_comparisons_batch,
        update_mention_bt_scores_batch,
        get_mentions_with_bt_scores,
        update_book_bt_scores,
        get_win_rates_vs_random
    )

    total_completed = 0
    total_stats = {'top_vs_random': 0, 'random': 0}
    stability_log = []  # Track stability over time
    consecutive_stable = 0

    # Get initial BT scores
    bt_scores = get_mentions_with_bt_scores()

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

    # Get initial top-k set
    previous_topk = _get_topk_set(bt_scores, top_k)

    print(f"\nInitial state:")
    print(f"  Total items: {len(mention_ids)}")
    print(f"  Items with scores: {sum(1 for s in bt_scores.values() if s is not None)}")
    print(f"  Target top-k: {top_k}")
    print(f"  Current top-tier size: {len(previous_topk)}")

    while total_completed < n_comparisons:
        # Calculate batch size (up to refit_interval or remaining)
        batch_size = min(refit_interval, n_comparisons - total_completed)

        batch_num = total_completed // refit_interval + 1
        print(f"\n--- Batch {batch_num}: {batch_size} comparisons ---")

        # Sample pairs using top-k strategy
        pairs, batch_stats = sample_pairs_topk(mention_ids, batch_size, bt_scores, top_k)

        # Update totals
        for key in batch_stats:
            total_stats[key] += batch_stats[key]

        print(f"Sampling strategy: {batch_stats['top_vs_random']} top-vs-random (80%), "
              f"{batch_stats['random']} random (20%)")

        # Create save callback for incremental saving within batches
        def save_results(results_batch):
            """Save comparison results to database."""
            insert_comparisons_batch(results_batch)

        # Run comparisons with incremental saving
        print(f"Running comparisons with {workers} workers using {provider}...")
        try:
            results = compare_pairs_parallel(pairs, model=model, workers=workers,
                                            debug=debug, debug_log=debug_log,
                                            save_callback=save_results, save_interval=500,
                                            provider=provider, api_key=api_key)
        except KeyboardInterrupt:
            print("\n\nRanking interrupted. Fitting model with saved comparisons...")
            # Fit and save final model
            all_comparisons = get_all_comparisons()
            scores = fit_bradley_terry(all_comparisons, mention_ids)
            if scores:
                score_tuples = list(scores.items())
                for i in range(0, len(score_tuples), 1000):
                    chunk = score_tuples[i:i+1000]
                    update_mention_bt_scores_batch(chunk)
            update_book_bt_scores()
            print("\nRanking interrupted. Run again to continue - existing comparisons are preserved.")
            return

        total_completed += batch_size

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

        # Compute top-k stability
        current_topk = _get_topk_set(bt_scores, top_k)
        stability = compute_topk_stability(previous_topk, current_topk)
        stability_log.append((total_completed, stability))

        # Compute win rates vs random for current top-k
        win_rates = get_win_rates_vs_random(current_topk)

        print(f"Progress: {total_completed}/{n_comparisons} comparisons")
        print(f"  Top-{top_k} stability: {stability['stability_pct']:.1f}% ({stability['overlap']}/{top_k} unchanged)")

        # Log tier movement (items moved in/out)
        if stability['changed'] > 0:
            items_in = current_topk - previous_topk
            items_out = previous_topk - current_topk
            print(f"  Tier movement: {len(items_in)} in, {len(items_out)} out")

        # Log win rate summary for items with at least 5 comparisons
        items_with_enough_data = [mid for mid, wr in win_rates.items()
                                   if wr['total'] >= 5]
        if items_with_enough_data:
            avg_win_rate = sum(win_rates[mid]['win_rate'] for mid in items_with_enough_data
                               if win_rates[mid]['win_rate'] is not None) / len(items_with_enough_data)
            min_wr = min((win_rates[mid]['win_rate'] for mid in items_with_enough_data
                          if win_rates[mid]['win_rate'] is not None), default=0)
            print(f"  Win rate vs random: avg={avg_win_rate*100:.1f}%, min={min_wr*100:.1f}% ({len(items_with_enough_data)} items with 5+ comparisons)")

        # Check for early stopping
        if stability['stability_pct'] >= 100.0:
            consecutive_stable += 1
            print(f"  Consecutive stable refits: {consecutive_stable}/{stable_threshold}")
            if consecutive_stable >= stable_threshold:
                print(f"\n*** EARLY STOPPING: Top-{top_k} has been stable for {stable_threshold} consecutive refits ***")
                break
        else:
            consecutive_stable = 0

        # Update previous top-k for next iteration
        previous_topk = current_topk

    # Final summary
    print("\n" + "=" * 60)
    print("TOP-K TIER VALIDATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal comparisons made: {total_completed}")
    if total_completed < n_comparisons:
        print(f"(Stopped early - requested {n_comparisons})")
    print(f"\nSampling strategy breakdown:")
    if total_completed > 0:
        print(f"  Top-k vs random (validation): {total_stats['top_vs_random']} ({total_stats['top_vs_random']/total_completed*100:.1f}%)")
        print(f"  Pure random (discovery): {total_stats['random']} ({total_stats['random']/total_completed*100:.1f}%)")

    # Show stability history
    if stability_log:
        print(f"\nTop-{top_k} stability history:")
        for completed, stab in stability_log[-5:]:  # Show last 5
            print(f"  After {completed} comparisons: {stab['stability_pct']:.1f}% stable")

    # Show final win rates for top-k items
    final_topk = _get_topk_set(bt_scores, top_k)
    final_win_rates = get_win_rates_vs_random(final_topk)
    items_with_data = [(mid, wr) for mid, wr in final_win_rates.items() if wr['total'] >= 5]
    if items_with_data:
        items_with_data.sort(key=lambda x: x[1]['win_rate'] or 0, reverse=True)
        print(f"\nWin rate vs random for top-{top_k} items (confidence metric):")
        # Show items with lowest win rates (potential false positives)
        low_confidence = [(mid, wr) for mid, wr in items_with_data if (wr['win_rate'] or 0) < 0.6]
        if low_confidence:
            print(f"  Low confidence items (win rate < 60%): {len(low_confidence)}")
            for mid, wr in low_confidence[:5]:  # Show up to 5
                print(f"    ID {mid}: {wr['win_rate']*100:.1f}% ({wr['wins']}/{wr['total']} wins)")
        high_confidence = [(mid, wr) for mid, wr in items_with_data if (wr['win_rate'] or 0) >= 0.8]
        print(f"  High confidence items (win rate >= 80%): {len(high_confidence)}")

    # Aggregate to book level
    print("\nAggregating scores to book level...")
    update_book_bt_scores()

    print(f"\nDone! Top-{top_k} tier validation complete.")
    print(f"Run 'python main.py rankings --top {top_k}' to see the results.")


def _get_topk_set(bt_scores, k):
    """Get the set of mention IDs in the top-k by BT score.

    Args:
        bt_scores: Dict mapping mention_id to bt_score
        k: Number of top items to return

    Returns:
        Set of mention IDs in top-k
    """
    # Filter to items with scores and sort
    scored = [(mid, score) for mid, score in bt_scores.items() if score is not None]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return top-k as a set
    return set(mid for mid, _ in scored[:k])


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


# ============================================================
# Genre Labeling Functions
# ============================================================

def label_book_genre(book_title, context_text, model=None, provider='ollama', api_key=None):
    """Use LLM to determine genres for a single book.

    Args:
        book_title: The book's title
        context_text: Combined context from all mentions (post content)
        model: The model to use (defaults based on provider)
        provider: LLM provider ('ollama' or 'gemini')
        api_key: API key for Gemini

    Returns:
        List of genre strings, or empty list if genre is unclear
    """
    import json

    if not book_title:
        return []

    # Truncate context if too long
    max_context = 6000
    if context_text and len(context_text) > max_context:
        context_text = context_text[:max_context] + "..."

    prompt = f"""You are categorizing books mentioned on Tyler Cowen's economics blog "Marginal Revolution".

Book title: "{book_title}"

Context from blog post(s) where this book was mentioned:
{context_text or "(No context available)"}

Based on the book title and context, assign genres to this book. You can assign MULTIPLE genres per book.

Guidelines:
- Use SPECIFIC genres where possible (e.g., "Chinese history" not just "history", "science fiction" not just "fiction")
- Include BOTH general and specific genres where appropriate (e.g., a Napoleon biography could get "biography", "history", AND "European history")
- If the book's genre is unclear or too niche to categorize, return an empty array []
- Common genres include: biography, memoir, history (and specific: American history, European history, Chinese history, military history, etc.), economics, fiction (and specific: science fiction, literary fiction, mystery, thriller, etc.), philosophy, science (and specific: physics, biology, psychology, etc.), politics, business, self-help, technology, art, music, religion, travel, cooking, sports

Return ONLY a valid JSON array of genre strings. Do NOT use markdown formatting.

Examples:
- For a Napoleon biography: ["biography", "history", "European history", "military history"]
- For a Chinese economics book: ["economics", "China", "Asian studies"]
- For a science fiction novel: ["fiction", "science fiction"]
- For an unclear/niche book: []

Your response (JSON array only):"""

    try:
        response = call_llm(prompt, provider=provider, model=model, api_key=api_key, temperature=0.1)

        # Clean and parse the response
        response = response.strip()
        response = clean_markdown_from_json(response)

        # Try to extract JSON array from response
        try:
            genres = json.loads(response)
            if isinstance(genres, list):
                # Filter to strings only and normalize
                return [str(g).strip().lower() for g in genres if g and str(g).strip()]
        except json.JSONDecodeError:
            # Try to find array in response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                try:
                    genres = json.loads(match.group())
                    if isinstance(genres, list):
                        return [str(g).strip().lower() for g in genres if g and str(g).strip()]
                except json.JSONDecodeError:
                    pass

        # If parsing fails, log and return empty
        print(f"  Warning: Could not parse genres for '{book_title}': {response[:200]}")
        return []

    except ConnectionError as e:
        raise  # Re-raise connection errors
    except Exception as e:
        print(f"  Warning: Error labeling genres for '{book_title}': {e}")
        return []


def label_genres_parallel(books, model=None, workers=5, rate_limit=100.0, provider='ollama', api_key=None):
    """Label genres for multiple books using parallel LLM calls.

    Args:
        books: List of tuples (book_id, book_title, context_text)
        model: The model to use
        workers: Number of concurrent workers
        rate_limit: Max requests per second
        provider: LLM provider
        api_key: API key for Gemini

    Returns:
        Tuple: (books_labeled, books_skipped, errors)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    import time

    from db import add_book_genres

    total = len(books)
    if total == 0:
        return 0, 0, 0

    print(f"Labeling genres for {total} books with {workers} workers using {provider}...")

    # Rate limiter
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

    def label_single_book(book):
        """Worker function for labeling a single book."""
        book_id, book_title, context_text = book
        rate_limiter.wait()

        try:
            genres = label_book_genre(book_title, context_text, model=model, provider=provider, api_key=api_key)
            return {
                'book_id': book_id,
                'book_title': book_title,
                'genres': genres,
                'error': None
            }
        except ConnectionError as e:
            return {
                'book_id': book_id,
                'book_title': book_title,
                'genres': [],
                'error': f'connection_error: {e}'
            }
        except Exception as e:
            return {
                'book_id': book_id,
                'book_title': book_title,
                'genres': [],
                'error': str(e)
            }

    books_labeled = 0
    books_skipped = 0
    errors = 0
    completed = 0
    connection_failed = False

    # Process in batches
    batch_size = 50
    results_to_write = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_book = {executor.submit(label_single_book, book): book for book in books}

        for future in as_completed(future_to_book):
            completed += 1

            try:
                result = future.result()

                if result['error']:
                    if 'connection_error' in result['error']:
                        error_msg = result['error'].replace('connection_error: ', '')
                        print(f"\nLLM connection failed: {error_msg}. Stopping labeling.")
                        connection_failed = True
                        for f in future_to_book:
                            f.cancel()
                        break
                    else:
                        print(f"\n  Warning: Error processing book '{result['book_title']}': {result['error']}")
                        errors += 1
                else:
                    results_to_write.append({
                        'book_id': result['book_id'],
                        'genres': result['genres']
                    })

            except Exception as e:
                errors += 1

            # Write to DB periodically and show progress
            if len(results_to_write) >= batch_size or completed == total:
                for r in results_to_write:
                    if r['genres']:
                        add_book_genres(r['book_id'], r['genres'])
                        books_labeled += 1
                    else:
                        books_skipped += 1
                results_to_write = []

                # Progress update
                print(f"  Progress: {completed}/{total} books processed, {books_labeled} labeled, {books_skipped} skipped")

    return books_labeled, books_skipped, errors


def run_genre_labeling(model=None, workers=5, reset=False, provider='ollama', api_key=None):
    """Main function to run genre labeling on all books.

    Args:
        model: The model to use
        workers: Number of concurrent workers
        reset: If True, clear existing genre labels and reprocess all books
        provider: LLM provider
        api_key: API key for Gemini
    """
    from db import (
        get_books_count,
        get_books_for_genre_labeling,
        get_all_books_for_genre_labeling,
        clear_book_genres,
        get_genre_labeling_stats
    )

    # Check if books exist
    book_count = get_books_count()
    if book_count == 0:
        print("Error: No books found in database.")
        print("Run 'python main.py analyze --extract-only --provider <provider>' first to extract books.")
        return

    print(f"Found {book_count} books in database.")

    # Check provider availability
    print(f"\nChecking {provider} availability...")
    try:
        check_provider_available(provider, model, api_key)
        model_display = model or ('llama3.2:3b' if provider == 'ollama' else 'gemini-2.5-flash')
        print(f"{provider.capitalize()} is available with model '{model_display}'.")
    except ConnectionError as e:
        print(f"\nAborting: Cannot proceed when {provider} is not available.")
        print(f"Error: {e}")
        return

    # Handle reset flag
    if reset:
        print("\n--reset flag set: Clearing existing genre labels...")
        clear_book_genres()
        books = get_all_books_for_genre_labeling()
    else:
        books = get_books_for_genre_labeling()

    to_process = len(books)
    if to_process == 0:
        print("\nAll books already have genre labels.")
        print("Use --reset to clear labels and reprocess all books.")

        # Show current stats
        stats = get_genre_labeling_stats()
        print(f"\nCurrent stats:")
        print(f"  Books with genres: {stats['books_with_genres']}/{stats['total_books']}")
        print(f"  Total genre associations: {stats['total_associations']}")
        print(f"  Unique genres used: {stats['unique_genres']}")
        return

    print(f"\n{to_process} books to process.")

    # Run labeling
    books_labeled, books_skipped, errors = label_genres_parallel(
        books,
        model=model,
        workers=workers,
        provider=provider,
        api_key=api_key
    )

    # Final summary
    print(f"\n{'='*50}")
    print("Genre Labeling Complete")
    print(f"{'='*50}")
    print(f"Books labeled: {books_labeled}")
    print(f"Books skipped (unclear genre): {books_skipped}")
    print(f"Errors: {errors}")

    # Show final stats
    stats = get_genre_labeling_stats()
    print(f"\nFinal stats:")
    print(f"  Books with genres: {stats['books_with_genres']}/{stats['total_books']}")
    print(f"  Total genre associations: {stats['total_associations']}")
    print(f"  Unique genres used: {stats['unique_genres']}")
