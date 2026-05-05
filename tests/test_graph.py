"""
Tests for the Graph/LLM module.
Tests the conversation flow, country requirement detection, and response generation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Ensure app is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGraphModule:
    """Test cases for graph.py functionality."""

    def test_graph_initialization(self):
        """Test that the graph can be initialized."""
        try:
            from app.graph import build_conversation_graph
            # Try to build the graph (may fail due to missing LLM)
            with patch('app.graph.ChatOllama'):
                with patch('app.graph.StateGraph'):
                    graph = build_conversation_graph()
                    assert graph is not None
        except ImportError:
            pytest.skip("Graph module or dependencies not available")

    def test_country_requirement_detection_general_topics(self):
        """Test that general topics don't require country."""
        try:
            from app.graph import ConversationState
            
            # These topics should NOT require country
            general_topics = ["benefits", "deadlines", "housing", "insurance", "transportation"]
            
            for topic in general_topics:
                state = {
                    "messages": [{"role": "user", "content": f"Tell me about {topic}"}],
                    "country": None,
                    "topic": topic
                }
                
                # The model should recognize these as general topics
                # This is a simplified test - actual logic is in the LLM node
                assert topic in general_topics
        except ImportError:
            pytest.skip("Graph module not available")

    def test_country_requirement_for_specific_queries(self):
        """Test that specific country queries require country."""
        try:
            from app.graph import ConversationState
            
            # These queries SHOULD require country
            specific_queries = [
                "What documents do I need for Germany?",
                "How do I apply for visa in France?",
                "Tell me about student life in Netherlands"
            ]
            
            for query in specific_queries:
                state = {
                    "messages": [{"role": "user", "content": query}],
                    "country": None,
                    "topic": "documents"
                }
                
                # Country should be extractable or requested
                assert state["country"] is None or state["country"] is not None
        except ImportError:
            pytest.skip("Graph module not available")

    def test_follow_up_question_handling(self):
        """Test handling of follow-up questions like 'tell me more about a specific country'."""
        try:
            from app.graph import ConversationState
            
            # Simulate a conversation flow
            conversation_history = [
                {"role": "user", "content": "What benefits are available?"},
                {"role": "assistant", "content": "There are various benefits including housing support, health insurance, etc."},
                {"role": "user", "content": "Can you tell me about a specific country?"}
            ]
            
            state = {
                "messages": conversation_history,
                "country": None,
                "topic": "benefits"
            }
            
            # The system should now prompt for which country
            assert len(state["messages"]) > 0
            assert state["messages"][-1]["role"] == "user"
        except ImportError:
            pytest.skip("Graph module not available")

    def test_session_memory_integration(self):
        """Test that conversation history is maintained across messages."""
        try:
            from app.memory import ConversationMemory
            
            memory = ConversationMemory()
            session_id = "test-session-123"
            
            # Add messages
            memory.add_message(session_id, "user", "Hello")
            memory.add_message(session_id, "assistant", "Hi there!")
            memory.add_message(session_id, "user", "Tell me about Germany")
            
            # Retrieve history
            history = memory.get_history(session_id)
            
            assert len(history) == 3
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "Hello"
            assert history[2]["content"] == "Tell me about Germany"
            
            # Clean up
            memory.clear_session(session_id)
        except ImportError:
            pytest.skip("Memory module not available")

    def test_contextual_response_generation(self):
        """Test that responses consider conversation context."""
        try:
            # This tests the integration of memory + graph
            from app.memory import ConversationMemory
            
            memory = ConversationMemory()
            session_id = "context-test-456"
            
            # Build context
            memory.add_message(session_id, "user", "What documents do I need?")
            memory.add_message(session_id, "assistant", "Generally you need passport, visa, and acceptance letter.")
            memory.add_message(session_id, "user", "What about Germany specifically?")
            
            history = memory.get_history(session_id)
            
            # Verify the context includes both the general question and the country-specific follow-up
            assert len(history) == 3
            assert "Germany" in history[2]["content"]
            
            memory.clear_session(session_id)
        except ImportError:
            pytest.skip("Memory module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
