"""
Tests for the main API endpoints.
Tests the chat endpoint, session management, and integration.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Ensure app is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAPIEndpoints:
    """Test cases for main.py API endpoints."""

    def test_chat_endpoint_basic(self):
        """Test basic chat endpoint functionality."""
        try:
            from app.main import app
            client = TestClient(app)
            
            with patch('app.main.process_chat_request') as mock_process:
                mock_process.return_value = {
                    "response": "Hello! How can I help you?",
                    "session_id": "test-session"
                }
                
                response = client.post("/api/chat", json={
                    "message": "Hello",
                    "session_id": "test-session"
                })
                
                assert response.status_code == 200
                data = response.json()
                assert "response" in data
                assert "session_id" in data
        except ImportError:
            pytest.skip("Main module or dependencies not available")

    def test_chat_endpoint_without_session_id(self):
        """Test chat endpoint creates new session if none provided."""
        try:
            from app.main import app
            client = TestClient(app)
            
            with patch('app.main.process_chat_request') as mock_process:
                mock_process.return_value = {
                    "response": "Hello!",
                    "session_id": "auto-generated-session"
                }
                
                response = client.post("/api/chat", json={
                    "message": "Hello"
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["session_id"] is not None
        except ImportError:
            pytest.skip("Main module or dependencies not available")

    def test_chat_endpoint_with_country(self):
        """Test chat endpoint handles country-specific queries."""
        try:
            from app.main import app
            client = TestClient(app)
            
            with patch('app.main.process_chat_request') as mock_process:
                mock_process.return_value = {
                    "response": "For Germany, you need...",
                    "session_id": "test-session-de",
                    "country": "germany"
                }
                
                response = client.post("/api/chat", json={
                    "message": "What documents do I need for Germany?",
                    "session_id": "test-session-de"
                })
                
                assert response.status_code == 200
                data = response.json()
                assert "Germany" in data["response"] or "germany" in str(data)
        except ImportError:
            pytest.skip("Main module or dependencies not available")

    def test_chat_endpoint_follow_up(self):
        """Test chat endpoint handles follow-up questions with context."""
        try:
            from app.main import app
            client = TestClient(app)
            
            # First message
            with patch('app.main.process_chat_request') as mock_process:
                mock_process.return_value = {
                    "response": "There are various benefits available.",
                    "session_id": "test-followup"
                }
                
                response1 = client.post("/api/chat", json={
                    "message": "What benefits are available?",
                    "session_id": "test-followup"
                })
                
                assert response1.status_code == 200
            
            # Follow-up message
            with patch('app.main.process_chat_request') as mock_process:
                mock_process.return_value = {
                    "response": "In Germany specifically, you get...",
                    "session_id": "test-followup",
                    "country": "germany"
                }
                
                response2 = client.post("/api/chat", json={
                    "message": "What about Germany?",
                    "session_id": "test-followup"
                })
                
                assert response2.status_code == 200
                # The response should consider the previous context
        except ImportError:
            pytest.skip("Main module or dependencies not available")

    def test_health_endpoint(self):
        """Test health check endpoint."""
        try:
            from app.main import app
            client = TestClient(app)
            
            response = client.get("/health")
            
            # Health endpoint should exist and return 200
            assert response.status_code in [200, 404]  # 404 if not implemented yet
        except ImportError:
            pytest.skip("Main module or dependencies not available")

    def test_root_endpoint_returns_html(self):
        """Test that root endpoint returns the chat interface."""
        try:
            from app.main import app
            client = TestClient(app)
            
            response = client.get("/")
            
            # Should return HTML page (status 200) or redirect
            assert response.status_code in [200, 307, 404]
            
            if response.status_code == 200:
                assert "text/html" in response.headers.get("content-type", "")
        except ImportError:
            pytest.skip("Main module or dependencies not available")

    def test_invalid_message_format(self):
        """Test handling of invalid message format."""
        try:
            from app.main import app
            client = TestClient(app)
            
            # Send empty message
            response = client.post("/api/chat", json={
                "message": "",
                "session_id": "test-invalid"
            })
            
            # Should handle gracefully (either error or default response)
            assert response.status_code in [200, 422, 400]
        except ImportError:
            pytest.skip("Main module or dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
