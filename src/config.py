from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings with environment-based configuration.
    """
    
    # API Key
    OPENAI_API_KEY: str
    
    # Model Name
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Embedding Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # RAG Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVE: int = 10
    TOP_K_RERANK: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Using lru_cache ensures we only load settings once.
    """
    return Settings()