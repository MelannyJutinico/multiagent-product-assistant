# Zubale Product Query Bot – RAG + LangGraph

This repository is a technical challenge solution for the **AI Engineer** role at Zubale. It implements a microservice that answers product-related questions using a **Retrieval-Augmented Generation (RAG)** pipeline, a **multi-agent architecture** powered by **LangGraph**, and a REST API built with **FastAPI**.

---

## Features

- Semantic document search using FAISS and HuggingFace embeddings
- Natural language answer generation using OpenAI LLM (gpt-3.5-turbo)
- Modular agent architecture with Retriever and Responder agents
- Prompt templates with injected context and source metadata
- RESTful API endpoint for external integration
- Unit testing with `pytest` and `pytest-asyncio`
- Dockerized for local or cloud deployment
- Environment-driven configuration using `.env` file

---

## Requirements

- Python 3.10+
- OpenAI API key
- Docker (optional)
- `pip install -r requirements.txt`

---

## Project Structure

├── app/
│ ├── agents/ # Retriever and Responder agents
│ ├── services/ # LLM and vector store services
│ ├── prompts/responder/ # Prompt templates (.md files)
│ ├── main.py # FastAPI application
│ └── config.py # Environment configuration
│
├── data/product_docs/ # Sample product documents (.txt)
├── scripts/index_documents.py # Document indexing script
├── test/ # Unit tests
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt


---

## Environment Setup

Create a file named `credentials.env` inside the `app/` folder with the following variables:

    OPENAI_API_KEY=sk-xxx
    LLM_MODEL=gpt-3.5-turbo
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    TOP_K=3

**Indexing Product Documents**
    Place your .txt files inside the data/product_docs/ folder.

Then run:
    python scripts/index_documents.py

This will generate a FAISS vector store index at data/vector_store.index.

## Running the API

Option 1: Local (Uvicorn)

    uvicorn app.main:app --reload

Option 2: Docker

    docker-compose up --build

## Example Request

POST /query

{
  "user_id": "usr_12345",
  "query": "Does the X3 model support Bluetooth 5.0?"
}

## Example Response:

{
  "user_id": "usr_12345",
  "response": "Yes, the X3 model supports Bluetooth 5.0 [source: x3_specs.txt]",
  "sources": [
    {"source_name": "x3_specs.txt"}
  ],
  "processing_time_ms": 412.2,
  "model_version": "v1.0"
}

# Running Tests

Ensure pytest-asyncio is installed for async test support.
