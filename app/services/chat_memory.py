from typing import Literal

from pydantic import BaseModel
from redis.asyncio import Redis

from app.core.config import Settings


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


def create_redis_client(settings: Settings) -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=True)


class ChatMemory:
    """Keeps the last few question/answer pairs of each chat session in Redis."""

    def __init__(self, client: Redis, max_turns: int, ttl_seconds: int) -> None:
        self._client = client
        self._max_turns = max_turns
        self._ttl_seconds = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    async def get(self, session_id: str) -> list[ChatMessage]:
        raw = await self._client.lrange(self._key(session_id), 0, -1)
        return [ChatMessage.model_validate_json(item) for item in raw]

    async def add_exchange(
        self, session_id: str, user_text: str, assistant_text: str
    ) -> None:
        key = self._key(session_id)
        messages = [
            ChatMessage(role="user", content=user_text),
            ChatMessage(role="assistant", content=assistant_text),
        ]
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *(message.model_dump_json() for message in messages))
            pipe.ltrim(key, -2 * self._max_turns, -1)
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()

    async def clear(self, session_id: str) -> None:
        await self._client.delete(self._key(session_id))