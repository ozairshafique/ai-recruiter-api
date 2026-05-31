from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """ Settings for the application, loaded from environment variables or .env file.
    """

    app_name: str = "AI Recruiter API"
    api_version: str = "1.0.0"
    debug: bool = False

    # Grok API settings
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0

    # Cohere API settings
    cohere_api_key: str
    cohere_model: str = "embed-multilingual-v3.0"


    # LangChain settings
    langchain_api_key: str
    langchain_tracing_v2: bool = True
    langchain_project: str = "AI-Recruiter-API"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # FAISS settings
    faiss_index_path: str = "./faiss_index"

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval settings
    top_k: int = 5

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """ Get the application settings, cached for performance.
    """
    return Settings()