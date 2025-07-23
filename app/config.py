from dotenv import load_dotenv 
import os

class Settings:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    DATA_PATH = "data/product_data"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TOP_K = int(os.getenv("TOP_K", 3))
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

settings = Settings()


