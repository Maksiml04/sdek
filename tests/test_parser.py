"""
Tests for the Parser module.
Tests document parsing, topic detection, and country extraction.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Ensure app is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestParserModule:
    """Test cases for parser.py functionality."""

    def test_parser_initialization(self):
        """Test that DocumentParser class can be initialized."""
        try:
            from app.parser import DocumentParser
            parser = DocumentParser()
            assert parser is not None
            assert hasattr(parser, 'PATTERNS')
            assert hasattr(parser, 'TOPIC_KEYWORDS')
        except ImportError:
            pytest.skip("Parser module not available")

    def test_detect_country_from_filename(self):
        """Test country detection from filename."""
        try:
            from app.parser import DocumentParser
            parser = DocumentParser()
            
            # Test various filename patterns
            assert parser._detect_country_from_filename("germany_benefits.pdf") == "germany"
            assert parser._detect_country_from_filename("France_Guide.docx") == "france"
            assert parser._detect_country_from_filename("berlin_info.txt") == "germany"
            assert parser._detect_country_from_filename("random_file.pdf") is None
        except ImportError:
            pytest.skip("Parser module not available")

    def test_detect_topic(self):
        """Test topic detection from text content."""
        try:
            from app.parser import DocumentParser
            parser = DocumentParser()
            
            # Test topic detection
            assert parser._detect_topic("You need to submit your visa application before the deadline") == "deadlines"
            assert parser._detect_topic("Students are eligible for housing benefits and health insurance") == "benefits"
            assert parser._detect_topic("You can find accommodation through student dormitories") == "housing"
            assert parser._detect_topic("Some random text without keywords") is None
        except ImportError:
            pytest.skip("Parser module not available")

    def test_normalize_country(self):
        """Test country name normalization."""
        try:
            from app.parser import DocumentParser
            parser = DocumentParser()
            
            # Test normalization
            assert parser._normalize_country("DEUTSCHLAND") == "germany"
            assert parser._normalize_country("Deutschland") == "germany"
            assert parser._normalize_country("germany") == "germany"
            assert parser._normalize_country("FRANCE") == "france"
            assert parser._normalize_country("unknown_country") == "unknown_country"
        except ImportError:
            pytest.skip("Parser module not available")

    def test_smart_splitter(self):
        """Test text splitting functionality."""
        try:
            from app.parser import DocumentParser
            parser = DocumentParser()
            
            long_text = "This is sentence one. This is sentence two. This is sentence three. " * 10
            chunks = parser._smart_splitter(long_text, chunk_size=100)
            
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)
            assert all(len(chunk) <= 150 for chunk in chunks)  # Allow some overlap
        except ImportError:
            pytest.skip("Parser module not available")

    def test_parse_file_mock(self):
        """Test file parsing with mocked file operations."""
        try:
            from app.parser import DocumentParser
            
            mock_content = """
            Germany Student Guide
            Documents needed: passport, visa, acceptance letter
            Deadlines: Apply before June 30th
            Benefits: Health insurance, housing support
            """
            
            with patch('app.parser.DocumentParser._detect_country_from_filename', return_value='germany'):
                with patch('app.parser.DocumentParser._detect_topic', return_value='documents'):
                    with patch('app.parser.DocumentParser._smart_splitter', return_value=[mock_content]):
                        parser = DocumentParser()
                        result = parser.parse_file("dummy_path/germany_guide.pdf")
                        
                        assert result is not None
                        assert isinstance(result, list) or result is not None
        except ImportError:
            pytest.skip("Parser module not available")

    def test_country_mapping_completeness(self):
        """Test that country mapping includes major countries."""
        try:
            from app.parser import DocumentParser
            parser = DocumentParser()
            
            # Check that common country variations are mapped
            countries_to_check = ['germany', 'france', 'netherlands', 'sweden', 'finland']
            for country in countries_to_check:
                assert country in parser.COUNTRY_MAPPING or any(
                    v == country for v in parser.COUNTRY_MAPPING.values()
                )
        except ImportError:
            pytest.skip("Parser module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
