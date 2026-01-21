"""
Pytest configuration and fixtures for golden tests.
"""

import os
import pytest


def pytest_addoption(parser):
    """Add command line options for provider configuration."""
    parser.addoption(
        "--provider",
        action="store",
        default="gemini",
        help="LLM provider to use: ollama or gemini (default: gemini)"
    )
    parser.addoption(
        "--api-key",
        action="store",
        default=None,
        help="API key for Gemini (or set GEMINI_API_KEY env var)"
    )
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Model to use (defaults based on provider)"
    )
    parser.addoption(
        "--verbose-llm",
        action="store_true",
        default=False,
        help="Show LLM inputs and outputs for each test"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "extraction: marks tests as extraction tests"
    )
    config.addinivalue_line(
        "markers", "comparison: marks tests as comparison tests"
    )
