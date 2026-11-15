"""Shared test doubles. No network, no real services."""

import asyncio
import hashlib
import json
import math
import re
import time as clock_time
from datetime import date, datetime, time
from typing import Any

from app.core.config import Settings
from app.services.booking import TIMEZONE, BookingDraft
from app.services.booking_flow import BookingFlow
from app.services.booking_repository import DuplicateBookingError
from app.services.booking_state import BookingStateStore
from app.services.llm import LLMMessage

# Sunday 15 November 2026, 10:00 in Kathmandu.
NOW = datetime(2026, 11, 15, 10, 0, tzinfo=TIMEZONE)
COLLECTION = "test_docs"
DIM = 384


class ScriptedLLM:
    """A fake LLM with three behaviours, picked by the kind of request.

    - JSON mode (booking analysis): returns `script[message]` as JSON, `{}` if the
      message is not in the script (meaning intent "other").
    - Query rewriting: returns `rewrite`, or the question unchanged.
    - Anything else (the RAG answer): returns `reply`.

    `calls` records every request. Set `error` to make every call raise it.
    """

    def __init__(
        self,
        script: dict[str, dict[str, Any] | str] | None = None,
        *,
        reply: str = "Answer from the documents.",
        rewrite: str | None = None,
    ) -> None:
        self.script = script or {}
        self.reply = reply
        self.rewrite = rewrite
        self.error: Exception | None = None
        self.calls: list[list[LLMMessage]] = []

    @property
    def rewrite_calls(self) -> list[list[LLMMessage]]:
        return [c for c in self.calls if "rewrite the user's last question" in c[0].content]

    @property
    def answer_calls(self) -> list[list[LLMMessage]]:
        return [c for c in self.calls if c[0].content.startswith("You answer questions")]

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        if json_mode:
            text = messages[-1].content.split("User message: ", 1)[-1]
            scripted = self.script.get(text, {})
            return scripted if isinstance(scripted, str) else json.dumps(scripted)
        if "rewrite the user's last question" in messages[0].content:
            if self.rewrite is not None:
                return self.rewrite
            last = messages[-1].content.split("Last question: ", 1)[1]
            return last.split("\n\nStandalone question:")[0]
        return self.reply


class FakeEmbedder:
    """Bag-of-words hashing: texts sharing words get similar vectors.

    Deterministic and instant, so retrieval tests need no model download.
    `delay` slows it down (for race tests); `fail` makes it raise.
    """

    dim = DIM

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.fail = False
        self.text_calls = 0
        self.queries: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.text_calls += 1
        if self.delay:
            clock_time.sleep(self.delay)  # runs in a worker thread, so this is fine
        if self.fail:
            raise RuntimeError("embedding failed")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FakeBookingRepository:
    """Same rule as the database: one booking per session, date and time."""

    def __init__(self) -> None:
        self.saved: list[tuple[str, BookingDraft]] = []

    async def save(self, session_id: str, draft: BookingDraft) -> None:
        key: tuple[str, date | None, time | None] = (
            session_id,
            draft.booking_date,
            draft.booking_time,
        )
        for saved_session, saved in self.saved:
            if (saved_session, saved.booking_date, saved.booking_time) == key:
                raise DuplicateBookingError()
        self.saved.append((session_id, draft))


def build_flow(
    llm: ScriptedLLM, states: BookingStateStore, repository: Any
) -> BookingFlow:
    return BookingFlow(
        llm,  # type: ignore[arg-type]
        states,
        repository,
        clock=lambda: NOW,
    )


def make_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        qdrant_url="http://localhost:6333",
        redis_url="redis://localhost:6379",
        database_url="sqlite+aiosqlite://",
        groq_api_key="test-key",
    )


async def never_finishes() -> None:
    await asyncio.sleep(60)