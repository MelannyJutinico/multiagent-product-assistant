from .config import settings  


def init_app():
    from fastapi import FastAPI
    app = FastAPI(title="Product Query Bot")
    return app

__all__ = ["settings", "RetrieverAgent", "ResponderAgent", "QueryRequest", "init_app"]