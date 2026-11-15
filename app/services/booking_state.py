from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.services.booking import BookingDraft


class BookingState(BaseModel):
    """A booking that is being collected through the chat."""

    draft: BookingDraft = Field(default_factory=BookingDraft)
    awaiting_confirmation: bool = False


class BookingStateStore:
    """Keeps each chat session's unfinished booking in Redis, with a timeout."""

    def __init__(self, client: Redis, ttl_seconds: int) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"booking:{session_id}"

    async def get(self, session_id: str) -> BookingState | None:
        raw = await self._client.get(self._key(session_id))
        return BookingState.model_validate_json(raw) if raw else None

    async def save(self, session_id: str, state: BookingState) -> None:
        await self._client.set(
            self._key(session_id), state.model_dump_json(), ex=self._ttl_seconds
        )

    async def clear(self, session_id: str) -> None:
        await self._client.delete(self._key(session_id))