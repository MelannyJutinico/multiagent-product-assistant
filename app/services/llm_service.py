# app/services/llm_service.py
from langchain_openai import ChatOpenAI  
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from app.config import settings
from langchain.schema import HumanMessage, SystemMessage
import logging
import os 

class LLMService:
    """Service wrapper for LLM operations with caching and configuration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._validate_api_key()
        self._configure_cache()
        self.llm = self._initialize_llm()
        
        
    def _validate_api_key(self):
        """Verify OpenAI API key is present."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Please set it in your environment variables"
            )
            
    def _configure_cache(self):
        """Setup caching to reduce duplicate LLM calls."""
        set_llm_cache(InMemoryCache())
        self.logger.info("LLM response caching enabled")

    def _initialize_llm(self):
        """Initialize the LLM with production-grade settings."""
        return ChatOpenAI  (
            model=settings.LLM_MODEL,
            temperature=0.3,  # Balanced creativity/accuracy
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            api_key=settings.OPENAI_API_KEY
        )

    def get_llm(self):
        """Get the configured LLM instance."""
        return self.llm

    
    def generate(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """
        Generate response using chat format.
        
        Args:
            prompt: User input/question
            system_message: Optional system instruction
            **kwargs: Additional LLM parameters
        """
        try:
            messages = []
            
            if system_message:
                messages.append(SystemMessage(content=system_message))
            
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages, **kwargs)
            self.logger.debug(f"Generated response for prompt: {prompt[:100]}...")
            return response.content  
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            raise

    def generate_chat(self, messages: list, **kwargs) -> str:
        """Generate response with full message history."""
        try:
            response = self.llm.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            self.logger.error(f"Chat generation failed: {str(e)}")
            raise