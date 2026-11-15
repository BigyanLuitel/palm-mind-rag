from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.db import Booking
from app.services.booking import BookingDraft


class DuplicateBookingError(Exception):
    """This session already booked this exact date and time."""


class BookingRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save(self, session_id: str, draft: BookingDraft) -> Booking:
        if (
            draft.name is None
            or draft.email is None
            or draft.booking_date is None
            or draft.booking_time is None
        ):
            raise ValueError("Cannot save an incomplete booking")

        booking = Booking(
            session_id=session_id,
            name=draft.name,
            email=draft.email,
            booking_date=draft.booking_date,
            booking_time=draft.booking_time,
        )
        async with self._session_factory() as session:
            session.add(booking)
            try:
                await session.commit()
            except IntegrityError as exc:
                raise DuplicateBookingError() from exc
        return booking

    async def list_recent(self, limit: int = 100) -> list[Booking]:
        query = select(Booking).order_by(Booking.created_at.desc()).limit(limit)
        async with self._session_factory() as session:
            return list((await session.execute(query)).scalars())