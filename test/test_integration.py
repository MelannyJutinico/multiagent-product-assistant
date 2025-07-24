# app/tests/agents/test_integration.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain.docstore.document import Document
from app.agents.orchestrator import Orchestrator, AgentState
from app.agents.retriever import RetrieverAgent
from app.agents.responder import ResponderAgent
import os

@pytest.fixture
def mock_agents():
    retriever = MagicMock(spec=RetrieverAgent)
    responder = MagicMock(spec=ResponderAgent)
    return retriever, responder

@pytest.fixture
def orchestrator(mock_agents):
    retriever, responder = mock_agents
    return Orchestrator(retriever, responder)

class TestIntegration:
    def test_initialization(self, orchestrator, mock_agents):
        retriever, responder = mock_agents
        assert orchestrator.retriever == retriever
        assert orchestrator.responder == responder
        assert hasattr(orchestrator, 'workflow')

    @patch.object(Orchestrator, '_create_workflow')
    def test_workflow_creation(self, mock_create, mock_agents):
        retriever, responder = mock_agents
        orchestrator = Orchestrator(retriever, responder)
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_success(self, orchestrator):
     
        test_docs = [Document(page_content="test", metadata={"source": "test.pdf"})]
        orchestrator.retriever.retrieve.return_value = test_docs
        orchestrator.responder.generate_response.return_value = "Test response"
        
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = {
            "query": "test query",
            "documents": test_docs,
            "response": "Test response"
        }
        orchestrator.workflow = mock_workflow

        result = await orchestrator.process_query("test query")
        # Verificar
        assert result == {
            "response": "Test response",
            "sources": [{"source_name": "test.pdf"}]
        }
        mock_workflow.ainvoke.assert_called_once_with({"query": "test query"})

    @pytest.mark.asyncio
    async def test_process_query_failure(self, orchestrator):
    
      

        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.side_effect = Exception("Test error")
        orchestrator.workflow = mock_workflow

        with patch.object(orchestrator.logger, 'error') as mock_logger:
            result = await orchestrator.process_query("failing query")
            
            assert result == {
                "error": "Failed to process query",
                "details": "Test error"
            }
            mock_logger.assert_called_once()

    def test_retrieve_documents(self, orchestrator):
   
        test_docs = [Document(page_content="doc1")]
        orchestrator.retriever.retrieve.return_value = test_docs

        state = AgentState(query="test query", documents=[], response="")
        result = orchestrator._retrieve_documents(state)

        assert result == {"documents": test_docs}
        orchestrator.retriever.retrieve.assert_called_once_with("test query")

    def test_generate_response(self, orchestrator):

        test_docs = [Document(page_content="doc1")]
        orchestrator.responder.generate_response.return_value = "Test response"

        
        state = AgentState(query="test query", documents=test_docs, response="")
        result = orchestrator._generate_response(state)

        
        assert result == {"response": "Test response"}
        orchestrator.responder.generate_response.assert_called_once_with(
            "test query", test_docs
        )

    def test_workflow_structure(self, orchestrator):
  
        assert hasattr(orchestrator.workflow, 'nodes')
        assert 'retrieve' in orchestrator.workflow.nodes
        assert 'respond' in orchestrator.workflow.nodes