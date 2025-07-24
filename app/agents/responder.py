# app/agents/responder.py
from pathlib import Path
from typing import List, Dict
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.services.llm_service import LLMService
from app.prompts.templates import format_context_with_sources
import logging

class ResponderAgent:
    """Agent responsible for generating natural language responses using LLM.
    
    Handles:
    - Prompt templating and management
    - Context formatting
    - Response generation
    - Error handling
    """
    
    def __init__(self, llm_service: LLMService):
        """Initialize the ResponderAgent with required services.
        
        Args:
            llm_service: Initialized LLM service instance
        """
        self.llm_service = llm_service
        self.prompts = self._load_prompt_templates()
        self.response_chain = self._build_response_pipeline()
        
        # Configure logging
        self.logger = logging.getLogger(__name__)

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from markdown files.
        
        Returns:
            Dictionary mapping prompt names to their content
        """
        prompt_dir = Path(__file__).parent.parent / "prompts" / "responder"
        try:
            return {
                "system": (prompt_dir / "system.md").read_text(encoding="utf-8"),
                "no_context": (prompt_dir / "no_context.md").read_text(encoding="utf-8"),
                "error": (prompt_dir / "error.md").read_text(encoding="utf-8")
            }
        except FileNotFoundError as e:
            self.logger.error(f"Prompt files not found: {e}")
            raise

    def _build_response_pipeline(self):
        """Construct the LangChain processing pipeline.
        
        Returns:
            Configured Runnable pipeline
        """
        return (
            {"context": self._format_context, "query": RunnablePassthrough()}
            | self._create_prompt_template()
            | self.llm_service.get_llm()
            | StrOutputParser()
        )

    def _create_prompt_template(self):
        """Create LangChain prompt template from loaded prompts."""
        return ChatPromptTemplate.from_messages([
            ("system", self.prompts["system"]),
            ("human", "{query}")
        ])

    def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """Generate response based on user query and retrieved documents.
        
        Args:
            query: User's natural language question
            context_docs: List of relevant documents retrieved
            
        Returns:
            Generated response string
        """
        if not query.strip():
            return self._handle_empty_query()
            
        if not context_docs:
            return self.prompts["no_context"].format(query=query)
            
        try:
            return self.response_chain.invoke({
                "query": query,
                "context": context_docs
            })
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            return self.prompts["error"]
        
    def _format_context(self, docs: List[Document]) -> str:
        """Formatea los documentos conservando sus metadatos."""
        formatted_context = []
        for doc in docs.get("context"):
            # Asegúrate de incluir el contenido y los metadatos
            formatted_context.append(
                f"Contenido: {doc.page_content}\nFuente: {doc.metadata.get('source', 'unknown')}"
            )
        return "\n\n".join(formatted_context)

    
    def _handle_empty_query(self) -> str:
        """Handle cases where query is empty or whitespace.
        
        Returns:
            Appropriate error response
        """
        return "Please provide a valid question about our products."