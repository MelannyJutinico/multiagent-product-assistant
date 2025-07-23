# scripts/index_documents.py
import os
from langchain.docstore.document import Document
from app.services.vector_store import VectorStoreService

def load_documents():
    docs = []
    for filename in os.listdir("data/product_docs"):
        with open(f"data/product_docs/{filename}", "r", encoding='utf-8') as f:
            content = f.read()
        docs.append(Document(
            page_content=content,
            metadata={"source": filename}
        ))
    return docs

if __name__ == "__main__":
    vector_store = VectorStoreService()
    documents = load_documents()
    vector_store.index_documents(documents)
    print(f"Indexados {len(documents)} documentos. Índice guardado en {vector_store.index_path}")