from datetime import date, datetime, time
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from app.api.deps import BookingRepositoryDep

router = APIRouter(prefix="/api/v1/bookings", tags=["bookings"])


class BookingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: str
    name: str
    email: str
    booking_date: date
    booking_time: time
    created_at: datetime


@router.get("", response_model=list[BookingOut])
async def list_bookings(repository: BookingRepositoryDep) -> list[BookingOut]:
    return [BookingOut.model_validate(b) for b in await repository.list_recent()]