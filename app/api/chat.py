from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field, StringConstraints

from app.api.deps import ChatServiceDep
from app.services.booking_flow import BookingStatus
from app.services.rag import Source

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

SessionId = Annotated[
    str, Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
]
Message = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


class ChatRequest(BaseModel):
    message: Message
    session_id: SessionId | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]
    booking_status: BookingStatus


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    session_id = request.session_id or uuid4().hex
    result = await service.reply(session_id, request.message)
    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        sources=result.sources,
        booking_status=result.booking_status,
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat(session_id: SessionId, service: ChatServiceDep) -> Response:
    await service.clear(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)