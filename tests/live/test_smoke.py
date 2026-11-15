"""Smoke tests against the REAL Groq model.

Run them on purpose with:  pytest -m live
They are skipped when there is no GROQ_API_KEY. Qdrant, Redis and SQL are still
fakes here, and so is the embedder: only the language model is real.
"""

from collections.abc import AsyncIterator
from datetime import date

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.services.booking import BookingDraft
from app.services.booking_extractor import analyze_message
from app.services.llm import GroqClient, create_groq_client
from app.services.rag import RagService
from app.services.vector_store import VectorStore
from tests.fakes import NOW, FakeEmbedder

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    try:
        settings = Settings()  # type: ignore[call-arg]  # reads .env / environment
    except ValidationError:
        pytest.skip("No .env or environment settings, so no GROQ_API_KEY")
    if not settings.groq_api_key.get_secret_value().strip():
        pytest.skip("GROQ_API_KEY is empty")
    return settings


@pytest.fixture
async def llm(real_settings: Settings) -> AsyncIterator[GroqClient]:  # type: ignore[override]
    """Replaces the fake `llm` fixture from conftest with the real model."""
    yield create_groq_client(real_settings)


async def test_the_model_finds_name_and_email(llm: GroqClient) -> None:
    analysis = await analyze_message(
        llm,
        "My name is Ram Sharma, ram@example.com",
        BookingDraft(),
        awaiting_confirmation=False,
        now=NOW,
    )
    assert analysis.name == "Ram Sharma"
    assert analysis.email == "ram@example.com"


async def test_the_model_resolves_a_relative_date(llm: GroqClient) -> None:
    analysis = await analyze_message(
        llm,
        "I want to book an interview next Tuesday at 2pm",
        BookingDraft(),
        awaiting_confirmation=False,
        now=NOW,
    )
    assert analysis.intent == "booking"
    assert analysis.date is not None
    parsed = date.fromisoformat(analysis.date)
    assert parsed.weekday() == 1  # Tuesday
    assert parsed > NOW.date()
    assert analysis.time == "14:00"


async def test_the_model_answers_from_the_document(
    llm: GroqClient, store: VectorStore, embedder: FakeEmbedder
) -> None:
    from uuid import uuid4

    chunks = ["Refunds are accepted within 30 days of purchase."]
    await store.add_chunks(uuid4(), "policy.txt", chunks, embedder.embed_texts(chunks))
    rag = RagService(embedder, store, llm, 5, 0.25, 0.2)  # type: ignore[arg-type]
    result = await rag.answer("How many days do I have for refunds accepted?", [])
    assert "30" in result.answer
    assert result.sources[0].filename == "policy.txt"


async def test_booking_conversation_end_to_end(
    llm: GroqClient,
    client: httpx.AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async def say(text: str, session_id: str | None = None) -> dict[str, str]:
        body = {"message": text, **({"session_id": session_id} if session_id else {})}
        response = await client.post("/api/v1/chat", json=body)
        assert response.status_code == 200, response.text
        return response.json()  # type: ignore[no-any-return]

    first = await say("I want to book an interview")
    assert first["booking_status"] == "collecting"
    sid = first["session_id"]

    second = await say("My name is Ram Sharma and my email is ram@example.com", sid)
    assert second["booking_status"] == "collecting"

    third = await say("next Tuesday at 2pm", sid)
    assert third["booking_status"] == "confirming"
    assert "2:00 PM" in third["answer"]

    done = await say("yes", sid)
    assert done["booking_status"] == "confirmed"

    rows = (await client.get("/api/v1/bookings")).json()
    assert len(rows) == 1 and rows[0]["email"] == "ram@example.com"

    stray = await say("yes")  # a new session: must not start a booking
    assert stray["booking_status"] == "none"