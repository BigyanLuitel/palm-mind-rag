from pydantic import BaseModel

from app.services.booking_flow import BookingFlow, BookingStatus
from app.services.chat_memory import ChatMemory
from app.services.rag import RagService, Source


class ChatResult(BaseModel):
    answer: str
    sources: list[Source]
    booking_status: BookingStatus


class ChatService:
    """One chat turn: booking flow first, otherwise answer from the documents."""

    def __init__(self, memory: ChatMemory, rag: RagService, flow: BookingFlow) -> None:
        self._memory = memory
        self._rag = rag
        self._flow = flow

    async def reply(self, session_id: str, message: str) -> ChatResult:
        history = await self._memory.get(session_id)
        turn = await self._flow.handle(session_id, message)

        if turn.handled:
            answer, sources = turn.reply, []
        else:
            result = await self._rag.answer(message, history)
            answer = result.answer
            if turn.reminder:
                answer = f"{answer}\n\n{turn.reminder}"
            sources = result.sources

        await self._memory.add_exchange(session_id, message, answer)
        return ChatResult(
            answer=answer, sources=sources, booking_status=turn.status
        )

    async def clear(self, session_id: str) -> None:
        await self._memory.clear(session_id)
        await self._flow.reset(session_id)