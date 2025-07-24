# test/test_retriever.py
import pytest
from unittest.mock import MagicMock, patch
from langchain.docstore.document import Document
from app.agents.retriever import RetrieverAgent
from app.services.vector_store import VectorStoreService
import os

# ----- Fixtures -----
@pytest.fixture
def mock_vector_store():
    """Mock VectorStoreService with predefined search results"""
    mock = MagicMock(spec=VectorStoreService)
    mock.search.return_value = [
        Document(
            page_content="Premium wireless headphones with noise cancellation",
            metadata={"source": "prod1.txt", "score": 0.85}
        ),
        Document(
            page_content="Bluetooth headphones with 30-hour battery life",
            metadata={"source": "prod2.txt", "score": 0.78}
        )
    ]
    return mock

@pytest.fixture
def retriever(mock_vector_store):
    """RetrieverAgent instance with mocked dependencies"""
    return RetrieverAgent(vector_store=mock_vector_store)

# ----- Unit Tests -----
def test_retrieve_documents(retriever, mock_vector_store):
    """Test successful document retrieval with query processing"""
    # Setup
    test_query = "noise cancelling headphones"
    
    # Execution
    results = retriever.retrieve(test_query)
    
    # Verification
    mock_vector_store.search.assert_called_once_with(test_query.lower())  # Verify preprocessing
    assert len(results) == 2
    assert results[0].metadata["score"] >= results[1].metadata["score"]  # Verify reranking

def test_empty_query_handling(retriever):
    """Test proper error handling for empty queries"""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("   ")

# ----- Integration Tests -----
@pytest.mark.integration
def test_semantic_retrieval():
    """Test end-to-end retrieval with actual vector store"""
    # Setup
    vector_store = VectorStoreService()
    
    # Verify index exists
    if not os.path.exists(vector_store.index_path):
        pytest.skip("Vector index not found. Skipping integration test.")
    
    # Initialize components
    vector_store.load_index()
    retriever = RetrieverAgent(vector_store=vector_store)
    
    # Test semantic search
    query = "audio devices with long battery"
    results = retriever.retrieve(query)
    
    # Verification
    assert len(results) > 0, "No documents retrieved"
    assert all(isinstance(doc, Document) for doc in results)
    
    # Verify reranking worked
    scores = [doc.metadata.get("score", 0) for doc in results]
    assert scores == sorted(scores, reverse=True), "Documents not properly reranked"

@pytest.mark.integration
def test_unknown_term_handling():
    """Test retrieval behavior with non-existent terms"""
    vector_store = VectorStoreService()
    retriever = RetrieverAgent(vector_store=vector_store)
    
    if not os.path.exists(vector_store.index_path):
        pytest.skip("Vector index not found. Skipping integration test.")
    
    # Test with completely unrelated term
    results = retriever.retrieve("nonexistenttermxyz123")
    
    # Should return empty list or low-score documents
    if len(results) > 0:
        low_score_threshold = 0.3
        assert all(
            doc.metadata.get("score", 0) < low_score_threshold 
            for doc in results
        ), "Unexpected high scores for irrelevant documents"

# ----- Performance Tests -----
@pytest.mark.performance
def test_retrieval_latency():
    """Test retrieval meets acceptable latency standards"""
    vector_store = VectorStoreService()
    retriever = RetrieverAgent(vector_store=vector_store)
    
    if not os.path.exists(vector_store.index_path):
        pytest.skip("Vector index not found. Skipping integration test.")
    
    import time
    start_time = time.time()
    
    # Warm-up call
    retriever.retrieve("headphones")
    
    # Measured call
    query = "wireless audio devices"
    results = retriever.retrieve(query)
    
    elapsed = time.time() - start_time
    assert elapsed < 0.5, f"Retrieval took {elapsed:.2f}s (>500ms threshold)"
    
    print(f"\nRetrieval latency: {elapsed*1000:.2f}ms for {len(results)} documents")