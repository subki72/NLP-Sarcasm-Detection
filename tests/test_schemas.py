"""
Unit tests for TextInput schema and input validation logic.
"""

import pytest
from pydantic import ValidationError

from src.schemas import TextInput


def test_valid_headline_passes():
    """Test that normal headline is accepted and preserved."""
    headline = "Local Man Wins Lottery and Continues Working Normally"
    payload = TextInput(text=headline)
    assert payload.text == headline


def test_whitespace_is_automatically_stripped():
    """Test that leading and trailing whitespaces are cleanly stripped."""
    payload = TextInput(text="   Breaking News: Scientists Discover Water   ")
    assert payload.text == "Breaking News: Scientists Discover Water"


def test_blank_whitespace_only_is_rejected():
    """Test that strings containing only whitespaces are rejected with 422 validation error."""
    with pytest.raises(ValidationError) as exc_info:
        TextInput(text="     \t\n   ")
    assert "cannot be empty or contain only whitespace" in str(exc_info.value)


def test_empty_string_is_rejected():
    """Test that completely empty strings are rejected."""
    with pytest.raises(ValidationError):
        TextInput(text="")


def test_exceeding_max_length_is_rejected():
    """Test that text exceeding 500 characters is rejected to prevent DOS attacks."""
    long_text = "a" * 501
    with pytest.raises(ValidationError) as exc_info:
        TextInput(text=long_text)
    assert "at most 500 characters" in str(exc_info.value)


def test_max_length_boundary_is_accepted():
    """Test that exactly 500 characters text is accepted."""
    exact_text = "a" * 500
    payload = TextInput(text=exact_text)
    assert len(payload.text) == 500


def test_non_string_type_is_rejected():
    """Test that non-string data types (int, list, dict) are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        TextInput(text=98765)
    assert "Input text must be a valid string" in str(exc_info.value)
