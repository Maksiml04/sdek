"""
Tests for the RAG (Retrieval-Augmented Generation) module.
Tests document retrieval and context generation with the new OOP RAGService class.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Ensure app is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRAGService:
    """Test cases for rag.py RAGService class functionality."""

    def test_rag_service_initialization(self):
        """Test that RAGService class can be initialized."""
        try:
            from app.rag import RAGService
            
            # Test default initialization
            with patch('app.rag.Chroma'):
                with patch('app.rag.FastEmbedEmbeddings'):
                    rag = RAGService()
                    assert rag is not None
                    assert rag.db_path == "./chroma_db"
                    
            # Test custom initialization
            with patch('app.rag.Chroma'):
                with patch('app.rag.FastEmbedEmbeddings'):
                    rag_custom = RAGService(db_path="./test_db", embedding_model="test-model")
                    assert rag_custom.db_path == "./test_db"
                    assert rag_custom.embedding_model_name == "test-model"
        except ImportError as e:
            pytest.skip(f"RAG module or dependencies not available: {e}")

    def test_get_embedding_model(self):
        """Test embedding model creation and caching."""
        try:
            from app.rag import RAGService
            
            mock_embedding = Mock()
            
            with patch('app.rag.Chroma'):
                with patch('app.rag.FastEmbedEmbeddings', return_value=mock_embedding) as mock_factory:
                    rag = RAGService()
                    
                    # First call should create the model
                    result1 = rag.get_embedding_model()
                    assert result1 == mock_embedding
                    mock_factory.assert_called_once()
                    
                    # Second call should return cached model
                    result2 = rag.get_embedding_model()
                    assert result2 == mock_embedding
                    # Should still be called only once (cached)
                    assert mock_factory.call_count == 1
        except ImportError:
            pytest.skip("RAG module or dependencies not available")

    def test_init_vectorstore_new(self):
        """Test vectorstore initialization when DB doesn't exist."""
        try:
            from app.rag import RAGService
            from langchain_core.documents import Document
            
            mock_docs = [
                Document(page_content="Test content", metadata={"country": "germany"})
            ]
            mock_split_docs = [
                Document(page_content="Chunk 1", metadata={"chunk_id": 0}),
                Document(page_content="Chunk 2", metadata={"chunk_id": 1})
            ]
            mock_vectorstore = Mock()
            
            with patch('app.rag.os.path.exists', return_value=False):
                with patch('app.rag.FastEmbedEmbeddings'):
                    with patch('app.rag.CdekStartParser') as MockParser:
                        MockParser.return_value.parse_all.return_value = mock_docs
                        
                        with patch('app.rag.RecursiveCharacterTextSplitter') as MockSplitter:
                            MockSplitter.return_value.split_documents.return_value = mock_split_docs
                            
                            with patch('app.rag.Chroma.from_documents', return_value=mock_vectorstore) as mock_from_docs:
                                rag = RAGService()
                                result = rag.init_vectorstore()
                                
                                assert result == mock_vectorstore
                                MockParser.return_value.parse_all.assert_called_once()
                                mock_from_docs.assert_called_once()
                                mock_vectorstore.persist.assert_called_once()
        except ImportError:
            pytest.skip("RAG module or dependencies not available")

    def test_init_vectorstore_existing(self):
        """Test loading existing vectorstore."""
        try:
            from app.rag import RAGService
            
            mock_vectorstore = Mock()
            
            with patch('app.rag.os.path.exists', return_value=True):
                with patch('app.rag.FastEmbedEmbeddings'):
                    with patch('app.rag.Chroma', return_value=mock_vectorstore) as mock_chroma:
                        rag = RAGService()
                        result = rag.init_vectorstore()
                        
                        assert result == mock_vectorstore
                        # Should load from existing directory, not create new
                        assert mock_chroma.called
        except ImportError:
            pytest.skip("RAG module or dependencies not available")

    def test_retrieve_context_with_country(self):
        """Test context retrieval with country filter."""
        try:
            from app.rag import RAGService
            
            mock_docs = [
                Mock(page_content="Germany specific info", metadata={"country": "germany"}),
                Mock(page_content="More Germany info", metadata={"country": "germany"})
            ]
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search.return_value = mock_docs
            
            with patch('app.rag.FastEmbedEmbeddings'):
                with patch('app.rag.Chroma'):
                    rag = RAGService()
                    rag._vectorstore = mock_vectorstore
                    
                    context = rag.retrieve_context(query="benefits", country="germany", k=2)
                    
                    assert "Germany specific info" in context
                    assert "More Germany info" in context
                    # Verify filter was applied
                    mock_vectorstore.similarity_search.assert_called_with(
                        "benefits", k=2, filter={"country": "germany"}
                    )
        except ImportError:
            pytest.skip("RAG module or dependencies not available")

    def test_retrieve_context_without_country(self):
        """Test context retrieval without country filter."""
        try:
            from app.rag import RAGService
            
            mock_docs = [
                Mock(page_content="General benefits info", metadata={})
            ]
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search.return_value = mock_docs
            
            with patch('app.rag.FastEmbedEmbeddings'):
                with patch('app.rag.Chroma'):
                    rag = RAGService()
                    rag._vectorstore = mock_vectorstore
                    
                    context = rag.retrieve_context(query="deadlines", country=None)
                    
                    assert "General benefits info" in context
                    # Verify no filter was applied
                    mock_vectorstore.similarity_search.assert_called_with(
                        "deadlines", k=3, filter={}
                    )
        except ImportError:
            pytest.skip("RAG module or dependencies not available")

    def test_retrieve_context_fallback(self):
        """Test fallback to unfiltered search when filtered search returns nothing."""
        try:
            from app.rag import RAGService
            
            mock_empty = []
            mock_fallback = [Mock(page_content="Fallback result", metadata={})]
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search.side_effect = [mock_empty, mock_fallback]
            
            with patch('app.rag.FastEmbedEmbeddings'):
                with patch('app.rag.Chroma'):
                    rag = RAGService()
                    rag._vectorstore = mock_vectorstore
                    
                    context = rag.retrieve_context(query="rare topic", country="germany")
                    
                    assert "Fallback result" in context
                    # Should have been called twice: first with filter, then without
                    assert mock_vectorstore.similarity_search.call_count == 2
        except ImportError:
            pytest.skip("RAG module or dependencies not available")

    def test_retrieve_documents(self):
        """Test retrieve_documents method returns Document objects."""
        try:
            from app.rag import RAGService
            from langchain_core.documents import Document
            
            mock_docs = [
                Document(page_content="Doc 1", metadata={"source": "file1.txt"}),
                Document(page_content="Doc 2", metadata={"source": "file2.txt"})
            ]
            mock_vectorstore = Mock()
            mock_vectorstore.similarity_search.return_value = mock_docs
            
            with patch('app.rag.FastEmbedEmbeddings'):
                with patch('app.rag.Chroma'):
                    rag = RAGService()
                    rag._vectorstore = mock_vectorstore
                    
                    results = rag.retrieve_documents(query="test", country="france", k=2)
                    
                    assert len(results) == 2
                    assert all(isinstance(doc, Document) for doc in results)
                    assert results[0].page_content == "Doc 1"
        except ImportError:
            pytest.skip("RAG module or dependencies not available")

    def test_singleton_pattern(self):
        """Test that get_rag_service returns singleton instance."""
        try:
            from app.rag import get_rag_service, RAGService
            
            with patch('app.rag.Chroma'):
                with patch('app.rag.FastEmbedEmbeddings'):
                    # Reset singleton
                    import app.rag as rag_module
                    rag_module._rag_service = None
                    
                    service1 = get_rag_service()
                    service2 = get_rag_service()
                    
                    # Should return same instance
                    assert service1 is service2
                    assert isinstance(service1, RAGService)
        except ImportError:
            pytest.skip("RAG module or dependencies not available")

    def test_backward_compatibility_functions(self):
        """Test that legacy functions still work."""
        try:
            from app.rag import init_vectorstore, retrieve_context, get_rag_service
            
            mock_vectorstore = Mock()
            mock_docs = [Mock(page_content="Legacy test content", metadata={})]
            mock_vectorstore.similarity_search.return_value = mock_docs
            
            # Test with existing DB path (loads existing vectorstore)
            with patch('app.rag.os.path.exists', return_value=True):
                with patch('app.rag.Chroma', return_value=mock_vectorstore):
                    with patch('app.rag.FastEmbedEmbeddings'):
                        # Reset singleton
                        import app.rag as rag_module
                        rag_module._rag_service = None
                        
                        # Test init_vectorstore
                        vs = init_vectorstore()
                        assert vs == mock_vectorstore
                        
                        # Test retrieve_context
                        context = retrieve_context("test query", "germany")
                        assert "Legacy test content" in context
        except ImportError:
            pytest.skip("RAG module or dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
