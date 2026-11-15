import pytest
from pydantic import ValidationError

from app.core.config import Settings

REQUIRED = {
    "qdrant_url": "http://localhost:6333",
    "redis_url": "redis://localhost:6379",
    "database_url": "postgresql+asyncpg://u:p@localhost/db",
    "groq_api_key": "gsk_test_secret",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*REQUIRED, "chunk_size", "chunk_overlap", "groq_model"):
        monkeypatch.delenv(name.upper(), raising=False)


def make(**overrides: object) -> Settings:
    return Settings(_env_file=None, **{**REQUIRED, **overrides})  # type: ignore[arg-type]


def test_defaults() -> None:
    settings = make()
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.groq_model == "openai/gpt-oss-120b"
    assert settings.embedding_dim == 384


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValidationError):
        make(chunk_size=200, chunk_overlap=200)


def test_chunk_size_has_a_floor() -> None:
    with pytest.raises(ValidationError):
        make(chunk_size=50)


def test_missing_required_value_fails() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_api_key_is_not_printed() -> None:
    settings = make()
    assert "gsk_test_secret" not in repr(settings)
    assert "gsk_test_secret" not in str(settings)
