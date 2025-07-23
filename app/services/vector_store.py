# app/services/vector_store.py
from langchain_community.vectorstores import FAISS
import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings 
from app.config import settings

class VectorStoreService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        self.db = None
        self.index_path = "data/vector_store.index"  
        self._configure_pickle()
        
    def _configure_pickle(self):
        """Configure pickle settings to avoid warnings for the example."""
        import warnings
        warnings.filterwarnings("ignore", message=".*deterministic.*")
        
        # This is only for demonstration purposes; in production, is necessary to use a more secure protocol.
        pickle.DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL

    def index_documents(self, documents):
        """Index documents into the vector store."""
        self.db = FAISS.from_documents(documents, self.embeddings)
        self._save_index()
    
    def _save_index(self):
        """Save the index to the specified path with security checks."""
        self.db.save_local(self.index_path)
    
    def load_index(self):
        """Charge the index from the specified path."""
        if os.path.exists(self.index_path):
            try:
                self.db = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Necesario en versiones recientes
                )
            except Exception as e:
                raise ValueError(f"Error cargando índice: {str(e)}")

    
    def search(self, query: str, top_k: int = None):
        if not self.db:
            self.load_index() 
        if not self.db:
            raise ValueError("Vector store not initialized. Please index documents first.")
        return self.db.similarity_search(query, k=top_k or settings.TOP_K)