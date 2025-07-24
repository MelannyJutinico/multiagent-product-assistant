# app/agents/retriever.py
from typing import List
from langchain.docstore.document import Document
from app.services.vector_store import VectorStoreService
from langchain_core.runnables import RunnableLambda

class RetrieverAgent:
    def __init__(self, vector_store: VectorStoreService):
        """Initialize the RetrieverAgent with a vector store service.
        
        Args:
            vector_store: Initialized VectorStoreService instance
        """
        self.vector_store = vector_store
        self._setup_pipeline()
        
    def _setup_pipeline(self):
        """Configure the LangChain Runnable pipeline for document retrieval.
        
        The pipeline consists of:
        1. Query preprocessing
        2. Document retrieval
        3. Result reranking
        """
        self.search_chain = (
            RunnableLambda(self._preprocess_query)
            | RunnableLambda(self._retrieve_docs)
            | RunnableLambda(self._rerank_docs)
        )

    def _retrieve_docs(self, query: str) -> List[Document]:
        """Retrieve documents from the vector store for a given query.
        
        Args:
            query: Preprocessed search query
            
        Returns:
            List of relevant documents with scores in metadata
        """
        return self.vector_store.search(query)
    
    def retrieve(self, query: str) -> List[Document]:
        """Main interface for document retrieval.
        
        Args:
            query: User's search query
            
        Returns:
            List of reranked relevant documents
            
        Raises:
            ValueError: If query is empty or whitespace-only
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return self.search_chain.invoke(query)
    
    def _preprocess_query(self, query: str) -> str:
        """Normalize and clean the search query.
        
        Args:
            query: Raw input query
            
        Returns:
            Cleaned and normalized query string
        """
        return query.strip().lower()
    
    def _rerank_docs(self, docs: List[Document]) -> List[Document]:
        """Sort documents by combined relevance score.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Top 3 documents sorted by score in descending order
        """
        return sorted(
            docs, 
            key=lambda d: d.metadata.get('score', 0), 
            reverse=True
        )[:3]