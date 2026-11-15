import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from app.services.booking import (
    TIMEZONE,
    BookingAnalysis,
    BookingDraft,
    apply_analysis,
)
from app.services.booking_extractor import analyze_message
from app.services.booking_repository import BookingRepository, DuplicateBookingError
from app.services.booking_state import BookingState, BookingStateStore
from app.services.llm import LLMClient

BookingStatus = Literal["none", "collecting", "confirming", "confirmed", "cancelled"]

_QUESTIONS = {
    "name": "What's your full name?",
    "email": "What's your email address?",
    "date": "Which date would you like for the interview?",
    "time": "What time suits you? Please give an exact time, like 3 PM.",
}


@dataclass(frozen=True)
class BookingTurn:
    handled: bool  # True: `reply` is the whole answer. False: answer from documents.
    status: BookingStatus
    reply: str = ""
    reminder: str = ""  # added after a document answer while a booking is open


_BARE_REPLIES = frozenset(
    {"yes", "no", "ok", "okay", "sure", "yeah", "yep", "yup", "nope", "nah", "y", "n"}
)


_CANCEL_WORDS = re.compile(
    r"\b(cancel|cancell?ed|forget (it|about)|never ?mind|no longer|"
    r"(don't|dont|do not) want|stop|abort|call (it )?off|drop it|scrap)\b"
)


def _asks_to_cancel(message: str) -> bool:
    """True only if the user's own words ask to cancel (the LLM can be wrong)."""
    return _CANCEL_WORDS.search(message.lower().replace("\u2019", "'")) is not None


def _is_bare_reply(message: str) -> bool:
    """True for a lone yes/no/ok, which must never start a booking by itself."""
    return re.sub(r"[^a-z ]", "", message.lower()).strip() in _BARE_REPLIES


def _now() -> datetime:
    return datetime.now(TIMEZONE)


class BookingFlow:
    """Decides what each chat message means for an interview booking."""

    def __init__(
        self,
        llm: LLMClient,
        states: BookingStateStore,
        repository: BookingRepository,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._llm = llm
        self._states = states
        self._repository = repository
        self._clock = clock

    async def handle(self, session_id: str, message: str) -> BookingTurn:
        now = self._clock()
        state = await self._states.get(session_id)
        if state is None and _is_bare_reply(message):
            return BookingTurn(handled=False, status="none")
        draft = state.draft if state else BookingDraft()
        awaiting = state.awaiting_confirmation if state else False
        analysis = await analyze_message(self._llm, message, draft, awaiting, now)

        if analysis.intent == "cancel" and not _asks_to_cancel(message):
            # Cancelling throws the booking away, so the user's words must ask for
            # it. A model once labelled "My name is Ram..." as a cancel.
            intent = "booking" if _has_booking_content(analysis) else "other"
            analysis = analysis.model_copy(update={"intent": intent})

        if analysis.intent == "cancel":
            return await self._cancel(session_id, state)

        if state is not None and analysis.intent == "other":
            if not _has_booking_content(analysis):
                return await self._remind(session_id, state)

        if state is None and analysis.intent != "booking":
            return BookingTurn(handled=False, status="none")

        return await self._advance(session_id, state, analysis, now)

    async def reset(self, session_id: str) -> None:
        await self._states.clear(session_id)

    async def _advance(
        self,
        session_id: str,
        state: BookingState | None,
        analysis: BookingAnalysis,
        now: datetime,
    ) -> BookingTurn:
        draft = state.draft if state else BookingDraft()
        awaiting = state.awaiting_confirmation if state else False
        updated, problems = apply_analysis(draft, analysis, now)
        unchanged = updated == draft and not problems

        if awaiting and unchanged and analysis.confirmation == "yes":
            return await self._confirm(session_id, updated)

        if awaiting and unchanged and analysis.confirmation == "no":
            await self._states.save(session_id, BookingState(draft=updated))
            return BookingTurn(
                handled=True,
                status="collecting",
                reply=(
                    "Okay, nothing has been booked. "
                    "What would you like to change: name, email, date or time?"
                ),
            )

        if updated.is_complete():
            await self._states.save(
                session_id, BookingState(draft=updated, awaiting_confirmation=True)
            )
            return BookingTurn(
                handled=True,
                status="confirming",
                reply=" ".join([*problems, _summary(updated)]),
            )

        await self._states.save(session_id, BookingState(draft=updated))
        return BookingTurn(
            handled=True,
            status="collecting",
            reply=_collecting_reply(
                draft, updated, analysis, problems, is_new=state is None
            ),
        )

    async def _confirm(self, session_id: str, draft: BookingDraft) -> BookingTurn:
        try:
            await self._repository.save(session_id, draft)
        except DuplicateBookingError:
            await self._states.clear(session_id)
            return BookingTurn(
                handled=True,
                status="none",
                reply=(
                    "You have already booked that date and time, "
                    "so nothing new was added."
                ),
            )
        await self._states.clear(session_id)
        return BookingTurn(
            handled=True,
            status="confirmed",
            reply=(
                f"Your interview is booked for {_when(draft)}, "
                f"under {draft.name} ({draft.email})."
            ),
        )

    async def _cancel(self, session_id: str, state: BookingState | None) -> BookingTurn:
        if state is None:
            return BookingTurn(
                handled=True, status="none", reply="There is no booking in progress."
            )
        await self._states.clear(session_id)
        return BookingTurn(handled=True, status="cancelled", reply="Booking cancelled.")

    async def _remind(self, session_id: str, state: BookingState) -> BookingTurn:
        await self._states.save(session_id, state)  # keeps the timeout fresh
        if state.awaiting_confirmation:
            return BookingTurn(
                handled=False,
                status="confirming",
                reminder=(
                    "Your interview booking is waiting for confirmation. "
                    "Reply yes to confirm it, or no to change it."
                ),
            )
        return BookingTurn(
            handled=False,
            status="collecting",
            reminder=(
                "Your interview booking is still in progress. "
                f"I still need your {', '.join(state.draft.missing())}."
            ),
        )


def _has_booking_content(analysis: BookingAnalysis) -> bool:
    return any(
        value is not None
        for value in (
            analysis.name,
            analysis.email,
            analysis.date,
            analysis.time,
            analysis.confirmation,
        )
    )


def _when(draft: BookingDraft) -> str:
    assert draft.booking_date is not None and draft.booking_time is not None
    day = draft.booking_date.strftime("%A, %d %B %Y")
    clock = draft.booking_time.strftime("%I:%M %p").lstrip("0")
    return f"{day} at {clock}"


def _summary(draft: BookingDraft) -> str:
    return (
        "Here is your booking:\n"
        f"- Name: {draft.name}\n"
        f"- Email: {draft.email}\n"
        f"- When: {_when(draft)}\n"
        "Shall I confirm it? (yes/no)"
    )


def _collecting_reply(
    before: BookingDraft,
    after: BookingDraft,
    analysis: BookingAnalysis,
    problems: list[str],
    is_new: bool,
) -> str:
    parts = list(problems)
    if not problems and is_new and after == before:
        parts.append("Sure, I can book an interview for you.")
    if after.name and not before.name:
        parts.append(f"Thanks, {after.name.split()[0]}.")
    parts.append(_QUESTIONS[_next_field(after, analysis)])
    return " ".join(parts)


def _next_field(after: BookingDraft, analysis: BookingAnalysis) -> str:
    """Ask again for a field the user just got wrong, else the first missing one."""
    tried = {
        "name": analysis.name,
        "email": analysis.email,
        "date": analysis.date,
        "time": analysis.time,
    }
    missing = after.missing()
    retry = [field for field in missing if tried[field] is not None]
    return (retry or missing)[0]
