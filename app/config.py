# app/config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

ruta_env = os.path.join(os.path.dirname(__file__), "credentials.env")
load_dotenv(ruta_env)

class Settings(BaseSettings):
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    TOP_K: int = int(os.getenv("TOP_K", 3))
    MAX_RETRY: int = int(os.getenv("MAX_RETRY", 3))
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
  
  

settings = Settings()


