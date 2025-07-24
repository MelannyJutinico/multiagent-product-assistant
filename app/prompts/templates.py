# app/prompts/templates.py
from typing import List
from langchain.docstore.document import Document

def format_context_with_sources(docs: List[Document]) -> str:
    """Formats documents with metadata into a consistent string format.
    
    Args:
        docs: List of Document objects with metadata
        
    Returns:
        String with formatted documents and sources
    """
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n"
        f"{doc.page_content}"
        for i, doc in enumerate(docs))