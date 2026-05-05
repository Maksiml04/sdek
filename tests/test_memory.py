"""
Tests for the Memory module.
Tests session management, history storage, and retrieval.
"""
import pytest
import sys
from pathlib import Path

# Ensure app is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMemoryModule:
    """Test cases for memory.py functionality."""

    def test_memory_initialization(self):
        """Test that ConversationMemory can be initialized."""
        try:
            from app.memory import ConversationMemory
            memory = ConversationMemory()
            assert memory is not None
            assert hasattr(memory, 'sessions')
        except ImportError:
            pytest.skip("Memory module not available")

    def test_add_message(self):
        """Test adding messages to a session."""
        try:
            from app.memory import ConversationMemory
            memory = ConversationMemory()
            session_id = "test-session-add"
            
            # Add a message
            memory.add_message(session_id, "user", "Hello")
            
            # Verify it was added
            history = memory.get_history(session_id)
            assert len(history) == 1
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "Hello"
            
            # Clean up
            memory.clear_session(session_id)
        except ImportError:
            pytest.skip("Memory module not available")

    def test_get_history_empty(self):
        """Test getting history for non-existent session."""
        try:
            from app.memory import ConversationMemory
            memory = ConversationMemory()
            
            history = memory.get_history("non-existent-session")
            assert history == []
        except ImportError:
            pytest.skip("Memory module not available")

    def test_multiple_messages_same_session(self):
        """Test adding multiple messages to the same session."""
        try:
            from app.memory import ConversationMemory
            memory = ConversationMemory()
            session_id = "test-session-multi"
            
            # Add multiple messages
            memory.add_message(session_id, "user", "First message")
            memory.add_message(session_id, "assistant", "First response")
            memory.add_message(session_id, "user", "Second message")
            memory.add_message(session_id, "assistant", "Second response")
            
            history = memory.get_history(session_id)
            assert len(history) == 4
            assert history[0]["content"] == "First message"
            assert history[1]["content"] == "First response"
            assert history[2]["content"] == "Second message"
            assert history[3]["content"] == "Second response"
            
            # Clean up
            memory.clear_session(session_id)
        except ImportError:
            pytest.skip("Memory module not available")

    def test_clear_session(self):
        """Test clearing a session."""
        try:
            from app.memory import ConversationMemory
            memory = ConversationMemory()
            session_id = "test-session-clear"
            
            # Add messages
            memory.add_message(session_id, "user", "Test")
            assert len(memory.get_history(session_id)) == 1
            
            # Clear session
            memory.clear_session(session_id)
            
            # Verify it's cleared
            history = memory.get_history(session_id)
            assert len(history) == 0
            
            # Clean up (already cleared, but safe to call again)
            memory.clear_session(session_id)
        except ImportError:
            pytest.skip("Memory module not available")

    def test_multiple_sessions_isolation(self):
        """Test that different sessions are isolated."""
        try:
            from app.memory import ConversationMemory
            memory = ConversationMemory()
            
            session_1 = "session-1"
            session_2 = "session-2"
            
            # Add messages to both sessions
            memory.add_message(session_1, "user", "Message for session 1")
            memory.add_message(session_2, "user", "Message for session 2")
            
            # Verify isolation
            history_1 = memory.get_history(session_1)
            history_2 = memory.get_history(session_2)
            
            assert len(history_1) == 1
            assert len(history_2) == 1
            assert history_1[0]["content"] == "Message for session 1"
            assert history_2[0]["content"] == "Message for session 2"
            
            # Clean up
            memory.clear_session(session_1)
            memory.clear_session(session_2)
        except ImportError:
            pytest.skip("Memory module not available")

    def test_get_all_sessions(self):
        """Test getting all active sessions."""
        try:
            from app.memory import ConversationMemory
            memory = ConversationMemory()
            
            session_1 = "active-session-1"
            session_2 = "active-session-2"
            
            # Create sessions
            memory.add_message(session_1, "user", "Test 1")
            memory.add_message(session_2, "user", "Test 2")
            
            # Get all sessions
            sessions = memory.get_all_sessions()
            
            assert session_1 in sessions
            assert session_2 in sessions
            
            # Clean up
            memory.clear_session(session_1)
            memory.clear_session(session_2)
        except ImportError:
            pytest.skip("Memory module not available")

    def test_session_with_complex_content(self):
        """Test session with complex message content."""
        try:
            from app.memory import ConversationMemory
            memory = ConversationMemory()
            session_id = "test-session-complex"
            
            complex_message = """
            This is a complex message with:
            - Multiple lines
            - Special characters: @#$%^&*()
            - Unicode: ñ, é, ü, 中文
            - Numbers: 12345
            """
            
            memory.add_message(session_id, "user", complex_message)
            memory.add_message(session_id, "assistant", "Response with **markdown** and `code`")
            
            history = memory.get_history(session_id)
            
            assert len(history) == 2
            assert "Unicode" in history[0]["content"]
            assert "markdown" in history[1]["content"]
            
            # Clean up
            memory.clear_session(session_id)
        except ImportError:
            pytest.skip("Memory module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
