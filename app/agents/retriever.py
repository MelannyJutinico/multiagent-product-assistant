# app/agents/retriever.py
from typing import List
from langchain.docstore.document import Document
from app.services.vector_store import VectorStoreService

class RetrieverAgent:
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents based on the query."""
        return self.vector_store.search(query)