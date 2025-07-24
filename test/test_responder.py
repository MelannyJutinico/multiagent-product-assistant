# app/tests/agents/test_responder.py
import pytest
from unittest.mock import MagicMock, patch
from langchain.docstore.document import Document
from app.agents.responder import ResponderAgent
from app.services.llm_service import LLMService

@pytest.fixture
def mock_llm_service():
    service = MagicMock(spec=LLMService)
    service.get_llm.return_value = MagicMock()
    return service

@pytest.fixture
def responder_agent(mock_llm_service):
    with patch("pathlib.Path.read_text") as mock_read:
        mock_read.return_value = "template content"
        return ResponderAgent(mock_llm_service)

class TestResponderAgent:
    def test_initialization_loads_templates(self, responder_agent):
        assert responder_agent.prompts == {
            "system": "template content",
            "no_context": "template content",
            "error": "template content"
        }

    def test_generate_response_with_valid_input(self, responder_agent, mock_llm_service):
   
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Mocked response"
        responder_agent.response_chain = mock_chain

        
        test_query = "What's the warranty period?"
        test_docs = [
            Document(page_content="Warranty info", metadata={"source": "doc1.pdf"}),
            Document(page_content="Product specs", metadata={"source": "doc2.pdf"})
        ]

        
        response = responder_agent.generate_response(test_query, test_docs)

        mock_chain.invoke.assert_called_once()
        assert response == "Mocked response"

    def test_generate_response_with_empty_query(self, responder_agent):
        response = responder_agent.generate_response("   ", [])
        assert response == "Please provide a valid question about our products."

    def test_generate_response_with_no_context(self, responder_agent):
        response = responder_agent.generate_response("Valid query", [])
        assert response == "template content"

    def test_generate_response_handles_errors(self, responder_agent):
        # Configurar el mock de la cadena
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Test error")
        responder_agent.response_chain = mock_chain
        
        # Mockear el logger y ejecutar
        with patch.object(responder_agent.logger, 'error') as mock_logger:
            response = responder_agent.generate_response("Valid query", [
                Document(page_content="content", metadata={})
            ])
            
            # Verificaciones
            assert response == "template content"
            mock_logger.assert_called_once_with("Response generation failed: Test error")
            mock_chain.invoke.assert_called_once()

    def test_format_context_properly_structures_documents(self, responder_agent):
        test_docs = {
            "context": [
                Document(page_content="Content 1", metadata={"source": "doc1.pdf"}),
                Document(page_content="Content 2", metadata={"source": "doc2.pdf"})
            ]
        }

        formatted = responder_agent._format_context(test_docs)
        
        expected = (
            "Contenido: Content 1\nFuente: doc1.pdf\n\n"
            "Contenido: Content 2\nFuente: doc2.pdf"
        )
        assert formatted == expected

    def test_format_context_with_missing_metadata(self, responder_agent):
        test_docs = {
            "context": [
                Document(page_content="Content", metadata={})
            ]
        }

        formatted = responder_agent._format_context(test_docs)
        assert "Fuente: unknown" in formatted

    @patch("app.agents.responder.ChatPromptTemplate.from_messages")
    def test_prompt_template_creation(self, mock_prompt, responder_agent):
        responder_agent._create_prompt_template()
        mock_prompt.assert_called_once_with([
            ("system", "template content"),
            ("human", "{query}")
        ])