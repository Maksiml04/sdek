"""
Configuration file for pytest tests.
Provides fixtures and common setup for all tests.
"""
import pytest
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_session_id():
    """Provide a sample session ID for testing."""
    return "test-session-12345"


@pytest.fixture
def sample_country():
    """Provide a sample country name for testing."""
    return "germany"


@pytest.fixture
def sample_query():
    """Provide a sample user query for testing."""
    return "What documents do I need?"


@pytest.fixture
def sample_chat_history():
    """Provide sample chat history for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "Tell me about benefits"}
    ]
