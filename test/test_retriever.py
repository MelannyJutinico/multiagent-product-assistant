# test/test_retriever.py
import pytest
from unittest.mock import MagicMock, patch
from langchain.docstore.document import Document
from app.agents.retriever import RetrieverAgent
from app.services.vector_store import VectorStoreService
import os
# Fixtures / Accesorios
@pytest.fixture
def mock_vector_store():
    """Creates a mock VectorStoreService"""
    mock = MagicMock(spec=VectorStoreService)
    mock.search.return_value = [
        Document(page_content="Product with 2-year warranty", metadata={"source": "prod1.txt"}),
        Document(page_content="Extended warranty available", metadata={"source": "prod2.txt"})
    ]
    return mock

@pytest.fixture
def retriever(mock_vector_store):
    """Provides a RetrieverAgent with mocked dependencies"""
    return RetrieverAgent(vector_store=mock_vector_store)

# Unit Tests / Pruebas unitarias
def test_retrieve_documents(retriever, mock_vector_store):
    """Test document retrieval functionality"""
    # Test data / Datos de prueba
    test_query = "warranty"
    
    # Execute / Ejecutar
    results = retriever.retrieve(test_query)
    
    # Verify / Verificar
    mock_vector_store.search.assert_called_once_with(test_query)
    assert len(results) == 2
    assert isinstance(results[0], Document)
    assert "warranty" in results[0].page_content.lower()

def test_empty_query_handling(retriever):
    """Test empty query handling """
    with pytest.raises(ValueError):
        retriever.retrieve("")

# Integration Tests 
@pytest.mark.integration
def test_real_retrieval():
    """Test integration with real VectorStoreService"""
    vector_store = VectorStoreService()
    
    assert os.path.exists(vector_store.index_path), \
        f"indice doesnt exist {vector_store.index_path}. Execute index_documents.py first"
    
    vector_store.load_index()
    assert vector_store.db is not None, "VectorStore not initializec."
    
    retriever = RetrieverAgent(vector_store=vector_store)
    test_query = "headphones"
    results = retriever.retrieve(test_query)
    
    assert len(results) > 0, "No documents retrieved"
    assert any("headphones" in doc.page_content.lower() for doc in results), \
        "Any document does not contain the query term"

    related_terms = ["headphone", "sound", "audio"]
    assert any(
        any(term in doc.page_content.lower() for term in [test_query] + related_terms)
        for doc in results
    ), "Documents do not contain related terms"

@pytest.mark.integration
def test_unknown_term_handling():
    """Test handling of unknown terms """
    vector_store = VectorStoreService()
    retriever = RetrieverAgent(vector_store=vector_store)
    
    results = retriever.retrieve("nonexistentterm123")
    assert not any("xyz123abc" in doc.page_content.lower() for doc in results)