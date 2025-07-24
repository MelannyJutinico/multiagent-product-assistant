# app/main.py
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import time
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.agents.retriever import RetrieverAgent
from app.agents.responder import ResponderAgent
from app.agents.orchestrator import Orchestrator

# Initialize services and agents
vector_store = VectorStoreService()
llm_service = LLMService()
retriever = RetrieverAgent(vector_store)
responder = ResponderAgent(llm_service)
orchestrator = Orchestrator(retriever, responder)


app = FastAPI(
    title="Zubale Product Query Bot API",
    description="""## RAG-powered Product Information Service
    
**Key Features:**
- Semantic search across product documentation
- AI-generated responses grounded in retrieved context
- Source citation for all information provided""",
    version="1.0.0",
    openapi_tags=[{
        "name": "queries",
        "description": "Product information retrieval endpoints"
    }]
)

class QueryRequest(BaseModel):
    """Request model for product queries"""
    user_id: str = Field(
        ...,
        example="usr_12345",
        description="Unique identifier for the user making the request"
    )
    query: str = Field(
        ...,
        min_length=3,
        max_length=200,
        example="What products have extended warranty?",
        description="Natural language question about products"
    )

class SourceDocument(BaseModel):
    """Metadata about retrieved source documents"""
    source_name: str = Field(..., example="prod_manual.pdf")

class QueryResponse(BaseModel):
    """Standardized response format"""
    user_id: str
    response: str
    sources: List[SourceDocument]
    processing_time_ms: Optional[float]
    model_version: str = Field(default="v1.0")

@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["queries"],
    summary="Submit product query",
    responses={
        200: {
            "description": "Successful response with generated answer",
            "content": {
                "application/json": {
                    "example": {
                        "user_id": "usr_12345",
                        "response": "The X3 model has 2-year warranty [Doc 1]",
                        "sources": [
                            {"source_name": "x3_specs.pdf"}
                        ],
                        "processing_time_ms": 423.1,
                        "model_version": "v1.0"
                    }
                }
            }
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {"detail": "Query must be 3-200 characters"}
                }
            }
        },
        500: {
            "description": "Internal processing error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error"}
                }
            }
        }
    }
)
async def handle_query(
    request: QueryRequest = Body(
        ...,
        examples={
            "normal": {
                "summary": "Standard warranty query",
                "value": {
                    "user_id": "usr_12345",
                    "query": "What's the warranty period?"
                }
            },
            "technical": {
                "summary": "Technical specification query",
                "value": {
                    "user_id": "eng_001",
                    "query": "Bluetooth version of Model X3"
                }
            }
        }
    )
):
    """Process natural language product queries using RAG pipeline.
    
    Typical flow:
    1. Semantic search retrieves relevant product documents
    2. LLM generates response grounded in retrieved context
    3. Response includes traceable source references
    
    Rate Limit: 5 requests/minute per user
    """
    try:
        start_time = time.time()
        result = await orchestrator.process_query(request.query)
   
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        sources = [{"source_name": source} if isinstance(source, str) else source 
                  for source in result.get("sources", [])]
        
        return {
            "user_id": request.user_id,
            "response": result["response"],
            "sources": sources,
            "processing_time_ms": (time.time() - start_time) * 1000
        }
    except Exception as e:
        logging.error(f"Endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")