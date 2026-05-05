# Testing Guide

This directory contains comprehensive tests for the application components.

## Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/test_rag.py -v
pytest tests/test_parser.py -v
pytest tests/test_graph.py -v
pytest tests/test_memory.py -v
pytest tests/test_api.py -v
```

### Run specific test function
```bash
pytest tests/test_memory.py::TestMemoryModule::test_add_message -v
```

### Run with coverage report
```bash
pytest --cov=app --cov-report=html
```

Then open `htmlcov/index.html` in your browser to see the coverage report.

## Test Files Overview

### `conftest.py`
Pytest configuration and shared fixtures for all tests.

### `test_rag.py`
Tests for the RAG (Retrieval-Augmented Generation) module:
- Document retrieval
- Country filtering
- Context formatting
- Empty results handling

### `test_parser.py`
Tests for the document parser:
- Country detection from filenames
- Topic detection from content
- Country normalization
- Text splitting
- File parsing

### `test_graph.py`
Tests for the conversation graph and LLM integration:
- Graph initialization
- Country requirement detection
- Follow-up question handling
- Session memory integration
- Contextual responses

### `test_memory.py`
Tests for the conversation memory system:
- Session creation and management
- Adding messages
- History retrieval
- Session clearing
- Multi-session isolation
- Complex content handling

### `test_api.py`
Tests for the FastAPI endpoints:
- Chat endpoint functionality
- Session ID handling
- Country-specific queries
- Follow-up conversations
- Health check endpoint
- HTML interface
- Error handling

## Writing New Tests

When adding new tests, follow these guidelines:

1. Use descriptive test names: `test_<feature>_<scenario>`
2. Use fixtures from `conftest.py` when possible
3. Mock external dependencies (LLM, embeddings, etc.)
4. Include both success and failure scenarios
5. Clean up resources after tests (sessions, files, etc.)
6. Add docstrings explaining what each test verifies

Example:

```python
def test_feature_with_country(self):
    """Test that feature works correctly with country specified."""
    # Arrange
    session_id = "test-session"
    country = "germany"
    
    # Act
    result = some_function(session_id, country)
    
    # Assert
    assert result is not None
    assert result.country == country
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions snippet
- name: Install dependencies
  run: pip install -r requirements-test.txt

- name: Run tests
  run: pytest --cov=app --cov-fail-under=80
```

## Troubleshooting

### Import errors
Make sure you're running tests from the project root:
```bash
cd /workspace
pytest
```

### Missing dependencies
```bash
pip install -r requirements-test.txt
pip install -r requirements.txt  # if exists
```

### Tests failing due to external services
Most tests use mocking, but if some don't:
```bash
pytest -m "not requires_external_service"
```
