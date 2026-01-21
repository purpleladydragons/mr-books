"""
Golden tests for book extraction and rating comparisons.

These are integration tests that call the actual LLM (Ollama or Gemini) to verify
that prompt changes don't break expected behavior. Run these before merging prompt changes.

Usage:
    pytest tests/test_golden.py -v
    pytest tests/test_golden.py -v --provider=gemini --api-key=YOUR_KEY
    pytest tests/test_golden.py -v -k extraction  # Run only extraction tests
    pytest tests/test_golden.py -v -k comparison  # Run only comparison tests
"""

import json
import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzer import extract_books_only, compare_pair_with_llm, check_provider_available


# Load fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"

def load_extraction_fixtures():
    with open(FIXTURES_DIR / "extraction_fixtures.json") as f:
        return json.load(f)

def load_comparison_fixtures():
    with open(FIXTURES_DIR / "comparison_fixtures.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def llm_config(request):
    """Get LLM configuration from command line or environment."""
    provider = request.config.getoption("--provider")
    api_key = request.config.getoption("--api-key") or os.environ.get("GEMINI_API_KEY")
    model = request.config.getoption("--model")

    return {
        "provider": provider,
        "api_key": api_key,
        "model": model
    }


@pytest.fixture(scope="session", autouse=True)
def check_llm_available(llm_config):
    """Check that the LLM provider is available before running tests."""
    provider = llm_config["provider"]
    api_key = llm_config["api_key"]
    model = llm_config["model"]

    try:
        check_provider_available(provider=provider, model=model, api_key=api_key)
    except Exception as e:
        pytest.skip(f"LLM provider '{provider}' not available: {e}")


def normalize_title(title):
    """Normalize a book title for fuzzy matching in tests."""
    import re
    # Remove common prefixes
    title = re.sub(r'^(the|a|an)\s+', '', title.lower())
    # Remove punctuation
    title = re.sub(r'[^\w\s]', '', title)
    # Normalize whitespace
    title = ' '.join(title.split())
    return title


def title_matches(extracted, expected):
    """Check if extracted title matches expected (fuzzy match)."""
    extracted_norm = normalize_title(extracted)
    expected_norm = normalize_title(expected)

    # Exact match after normalization
    if extracted_norm == expected_norm:
        return True

    # Substring match (expected is contained in extracted or vice versa)
    if expected_norm in extracted_norm or extracted_norm in expected_norm:
        return True

    # Check if main words match (at least 60% word overlap)
    extracted_words = set(extracted_norm.split())
    expected_words = set(expected_norm.split())

    if len(expected_words) > 0:
        overlap = len(extracted_words & expected_words) / len(expected_words)
        if overlap >= 0.6:
            return True

    return False


def any_title_matches(extracted_list, expected):
    """Check if any extracted title matches the expected title."""
    return any(title_matches(e, expected) for e in extracted_list)


# ============================================================================
# EXTRACTION GOLDEN TESTS
# ============================================================================

class TestExtractionGolden:
    """Golden tests for book title extraction."""

    @pytest.fixture
    def extraction_fixtures(self):
        return load_extraction_fixtures()

    def test_extraction_001_book_in_title(self, llm_config, extraction_fixtures):
        """Test extraction when book title is clearly in post title."""
        fixture = next(f for f in extraction_fixtures if f["id"] == "extraction_001")

        result = extract_books_only(
            post_title=fixture["post_title"],
            post_content=fixture["post_content"],
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) >= 1, f"Expected at least 1 book, got {len(result)}: {result}"

        # Check that at least one expected book was found
        for expected in fixture["expected_books"]:
            assert any_title_matches(result, expected), \
                f"Expected to find '{expected}' in {result}"

    def test_extraction_002_book_with_subtitle(self, llm_config, extraction_fixtures):
        """Test extraction of book with volume/subtitle."""
        fixture = next(f for f in extraction_fixtures if f["id"] == "extraction_002")

        result = extract_books_only(
            post_title=fixture["post_title"],
            post_content=fixture["post_content"],
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) >= 1, f"Expected at least 1 book, got {len(result)}: {result}"

        # Check main title or fuzzy variations
        all_expected = fixture["expected_books"] + fixture.get("expected_books_fuzzy", [])
        found = any(any_title_matches(result, exp) for exp in all_expected)
        assert found, f"Expected to find one of {all_expected} in {result}"

    def test_extraction_003_biography_with_subtitle(self, llm_config, extraction_fixtures):
        """Test extraction of biography with subtitle in content."""
        fixture = next(f for f in extraction_fixtures if f["id"] == "extraction_003")

        result = extract_books_only(
            post_title=fixture["post_title"],
            post_content=fixture["post_content"],
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) >= 1, f"Expected at least 1 book, got {len(result)}: {result}"

        all_expected = fixture["expected_books"] + fixture.get("expected_books_fuzzy", [])
        found = any(any_title_matches(result, exp) for exp in all_expected)
        assert found, f"Expected to find one of {all_expected} in {result}"

    def test_extraction_004_subtitle_in_content(self, llm_config, extraction_fixtures):
        """Test extraction where subtitle appears in content body."""
        fixture = next(f for f in extraction_fixtures if f["id"] == "extraction_004")

        result = extract_books_only(
            post_title=fixture["post_title"],
            post_content=fixture["post_content"],
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) >= 1, f"Expected at least 1 book, got {len(result)}: {result}"

        all_expected = fixture["expected_books"] + fixture.get("expected_books_fuzzy", [])
        found = any(any_title_matches(result, exp) for exp in all_expected)
        assert found, f"Expected to find one of {all_expected} in {result}"

    def test_extraction_005_multiple_books(self, llm_config, extraction_fixtures):
        """Test extraction of post with multiple books mentioned."""
        fixture = next(f for f in extraction_fixtures if f["id"] == "extraction_005")

        result = extract_books_only(
            post_title=fixture["post_title"],
            post_content=fixture["post_content"],
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) >= 1, f"Expected at least 1 book for multi-book post, got {len(result)}: {result}"

        # For multi-book posts, check that at least one expected book was found
        any_found = any(
            any_title_matches(result, exp)
            for exp in fixture["expected_books_any"]
        )
        assert any_found, \
            f"Expected to find at least one of {fixture['expected_books_any']} in {result}"


# ============================================================================
# COMPARISON GOLDEN TESTS
# ============================================================================

class TestComparisonGolden:
    """Golden tests for pairwise sentiment comparison."""

    @pytest.fixture
    def comparison_fixtures(self):
        return load_comparison_fixtures()

    def _make_mention_tuple(self, review, mention_id=1):
        """Create a mention tuple from review fixture data."""
        # (mention_id, book_id, book_title, context_text, post_content)
        return (
            mention_id,
            mention_id,  # book_id same as mention_id for test
            review["book_title"],
            review["context"],
            review["context"]  # Use context as post_content too
        )

    def test_comparison_001_enthusiastic_vs_neutral(self, llm_config, comparison_fixtures):
        """Test that enthusiastic praise beats qualified/neutral review."""
        fixture = next(f for f in comparison_fixtures if f["id"] == "comparison_001")

        mention_a = self._make_mention_tuple(fixture["review_a"], mention_id=1)
        mention_b = self._make_mention_tuple(fixture["review_b"], mention_id=2)

        result = compare_pair_with_llm(
            mention_a=mention_a,
            mention_b=mention_b,
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        # Result is (mention_a_id, mention_b_id, winner_id)
        assert len(result) == 3, f"Expected 3-tuple, got {result}"

        winner_id = result[2]
        expected = fixture["expected_winner"]

        if expected == "A":
            assert winner_id == 1, \
                f"Expected Review A to win, but got winner_id={winner_id}"
        elif expected == "B":
            assert winner_id == 2, \
                f"Expected Review B to win, but got winner_id={winner_id}"

    def test_comparison_002_excellent_vs_overpriced(self, llm_config, comparison_fixtures):
        """Test that 'excellent book' beats 'overpriced' review."""
        fixture = next(f for f in comparison_fixtures if f["id"] == "comparison_002")

        mention_a = self._make_mention_tuple(fixture["review_a"], mention_id=1)
        mention_b = self._make_mention_tuple(fixture["review_b"], mention_id=2)

        result = compare_pair_with_llm(
            mention_a=mention_a,
            mention_b=mention_b,
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        winner_id = result[2]
        assert winner_id == 1, \
            f"Expected 'excellent book' (A) to beat 'overpriced' (B), got winner_id={winner_id}"

    def test_comparison_003_aplus_vs_neutral(self, llm_config, comparison_fixtures):
        """Test that A+ rating clearly beats neutral review."""
        fixture = next(f for f in comparison_fixtures if f["id"] == "comparison_003")

        mention_a = self._make_mention_tuple(fixture["review_a"], mention_id=1)
        mention_b = self._make_mention_tuple(fixture["review_b"], mention_id=2)

        result = compare_pair_with_llm(
            mention_a=mention_a,
            mention_b=mention_b,
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        winner_id = result[2]
        assert winner_id == 1, \
            f"Expected A+ rating (A) to beat neutral (B), got winner_id={winner_id}"

    def test_comparison_004_both_positive_gradation(self, llm_config, comparison_fixtures):
        """Test comparison of two positive reviews - either A wins or tie acceptable."""
        fixture = next(f for f in comparison_fixtures if f["id"] == "comparison_004")

        mention_a = self._make_mention_tuple(fixture["review_a"], mention_id=1)
        mention_b = self._make_mention_tuple(fixture["review_b"], mention_id=2)

        result = compare_pair_with_llm(
            mention_a=mention_a,
            mention_b=mention_b,
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        winner_id = result[2]
        # For close positive reviews, A or TIE is acceptable
        assert winner_id in [1, None], \
            f"Expected A or TIE for two positive reviews, got winner_id={winner_id} (B)"

    def test_comparison_005_close_call(self, llm_config, comparison_fixtures):
        """Test comparison of similarly positive reviews."""
        fixture = next(f for f in comparison_fixtures if f["id"] == "comparison_005")

        mention_a = self._make_mention_tuple(fixture["review_a"], mention_id=1)
        mention_b = self._make_mention_tuple(fixture["review_b"], mention_id=2)

        result = compare_pair_with_llm(
            mention_a=mention_a,
            mention_b=mention_b,
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        winner_id = result[2]
        # For close positive reviews, A or TIE is acceptable
        # ("one of the best science books ever" vs "An excellent book")
        assert winner_id in [1, None], \
            f"Expected A or TIE for close positive reviews, got winner_id={winner_id} (B)"


