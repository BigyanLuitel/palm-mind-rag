from typing import Any, Literal, Protocol

import groq
from groq import AsyncGroq
from pydantic import BaseModel

from app.core.config import Settings


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMUnavailableError(Exception):
    """The LLM provider could not be reached or returned an error."""


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str: ...


class GroqClient:
    def __init__(self, client: AsyncGroq, model: str) -> None:
        self._client = client
        self._model = model

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        options: dict[str, Any] = {}
        if json_mode:
            options["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[message.model_dump() for message in messages],
                temperature=temperature,
                **options,
            )
        except groq.APIError as exc:
            raise LLMUnavailableError(str(exc)) from exc
        return response.choices[0].message.content or ""


def create_groq_client(settings: Settings) -> GroqClient:
    client = AsyncGroq(
        api_key=settings.groq_api_key.get_secret_value(),
        timeout=30.0,
        max_retries=2,
    )
    return GroqClient(client, settings.groq_model)