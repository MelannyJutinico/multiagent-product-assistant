# app/agents/orchestrator.py
from langgraph.graph import StateGraph  # Updated import
from typing import TypedDict, List, Dict
from langchain.docstore.document import Document
from app.agents.retriever import RetrieverAgent
from app.agents.responder import ResponderAgent
import logging

class AgentState(TypedDict):
    """State representation for the agent workflow.
    
    Attributes:
        query: The user's original query string
        documents: List of retrieved documents
        response: Generated response from the LLM
    """
    query: str
    documents: List[Document]
    response: str

class Orchestrator:
    """Coordinates the multi-agent RAG workflow using LangGraph.
    
    Responsibilities:
    1. Manages the execution flow between Retriever and Responder agents
    2. Maintains state throughout the pipeline
    3. Handles error cases and logging
    """
    
    def __init__(self, retriever: RetrieverAgent, responder: ResponderAgent):
        """Initialize the Orchestrator with agent dependencies.
        
        Args:
            retriever: Initialized RetrieverAgent instance
            responder: Initialized ResponderAgent instance
        """
        self.retriever = retriever
        self.responder = responder
        self.logger = logging.getLogger(__name__)
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Construct and configure the LangGraph state machine.
        
        Returns:
            Configured StateGraph ready for execution
        """
        workflow = StateGraph(AgentState)  # Updated to StateGraph

        # Add nodes to the workflow
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("respond", self._generate_response)
        
        # Define the execution flow
        workflow.add_edge("retrieve", "respond")
        
        # Configure start and end points
        workflow.set_entry_point("retrieve")
        workflow.set_finish_point("respond")
        
        return workflow.compile()

    async def process_query(self, query: str) -> Dict[str, any]:
        """Execute the full RAG pipeline for a user query.
        
        Args:
            query: User's natural language question
            
        Returns:
            Dictionary containing:
            - response: Generated answer
            - sources: List of source documents used
            - error: Optional error message
        """
        try:
            result = await self.workflow.ainvoke({"query": query})
        
            return {
                "response": result["response"],
                "sources": [
                    {
                        "source_name": doc.metadata.get("source", "unknown")
                    }
                    for doc in result["documents"]
                ]
            }
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            return {
                "error": "Failed to process query",
                "details": str(e)
            }

    def _retrieve_documents(self, state: AgentState) -> Dict[str, List[Document]]:
        """Execute document retrieval step.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary with key "documents" containing retrieved docs
        """
        documents = self.retriever.retrieve(state["query"])
        self.logger.debug(f"Retrieved {len(documents)} documents")
        return {"documents": documents}

    def _generate_response(self, state: AgentState) -> Dict[str, str]:
        """Generate LLM response based on retrieved documents.
        
        Args:
            state: Current workflow state containing query and documents
            
        Returns:
            Dictionary with key "response" containing generated answer
        """
        response = self.responder.generate_response(
            state["query"],
            state["documents"]
        )
        self.logger.debug(f"Generated response: {response[:100]}...")
        return {"response": response}