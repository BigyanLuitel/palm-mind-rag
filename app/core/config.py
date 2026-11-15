from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    redis_url: str
    database_url: str

    groq_api_key: SecretStr
    groq_model: str = "openai/gpt-oss-120b"

    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    retrieval_score_margin: float = Field(default=0.2, ge=0.0, le=1.0)
    booking_ttl_seconds: int = Field(default=1_800, ge=60)
    chat_history_length: int = Field(default=10, ge=1)
    chat_ttl_seconds: int = Field(default=86_400, ge=60)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, ge=1)

    @model_validator(mode="after")
    def check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
