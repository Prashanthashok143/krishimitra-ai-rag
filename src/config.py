from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "KrishiMitra AI"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    
    GEMINI_API_KEY: str
    
    MONGO_URI: str
    MONGODB_DATABASE_NAME: str ="KrishiMitra_db"
    MONGODB_COLLECTION_NAME: str = "agricultural_advisories"
    MONGODB_VECTOR_INDEX_NAME: str = "krishimitra_vector_index"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Cached getter so settings are parsed once into memory
    rather than re-reading the filesystem on every dependency injection.
    """
    return Settings()
    