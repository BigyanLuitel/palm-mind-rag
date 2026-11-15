from typing import Any

import pytest

from app.services.booking_flow import BookingFlow
from app.services.booking_state import BookingStateStore
from tests.fakes import FakeBookingRepository, ScriptedLLM, build_flow

START = "I want to book an interview"
FULL = "I'm Ram Sharma, ram@example.com, next Tuesday at 2pm"
SESSION = "s1"


def booking(**fields: Any) -> dict[str, Any]:
    return {"intent": "booking", **fields}


def make(
    states: BookingStateStore,
    repository: FakeBookingRepository,
    extra: dict[str, dict[str, Any] | str] | None = None,
) -> tuple[BookingFlow, ScriptedLLM]:
    script: dict[str, dict[str, Any] | str] = {
        START: booking(),
        "My name is Ram Sharma, ram@example.com": booking(
            name="Ram Sharma", email="ram@example.com"
        ),
        FULL: booking(
            name="Ram Sharma",
            email="ram@example.com",
            date="2026-11-17",
            time="14:00",
        ),
        "yes": booking(confirmation="yes"),
        "no": booking(confirmation="no"),
        "yes but make it 4pm": booking(confirmation="yes", time="16:00"),
        "make it Wednesday": booking(date="2026-11-18"),
        "cancel that": {"intent": "cancel"},
        "my email is nope": booking(email="nope"),
        "last Monday": booking(date="2026-11-10"),
        "today at 9am": booking(date="2026-11-15", time="09:00"),
        "sounds good": booking(confirmation="yes"),
    }
    script.update(extra or {})
    llm = ScriptedLLM(script)
    return build_flow(llm, states, repository), llm


async def test_booking_request_starts_collecting(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    turn = await flow.handle(SESSION, START)
    assert turn.handled and turn.status == "collecting"
    assert "Sure, I can book an interview" in turn.reply
    assert "What's your full name?" in turn.reply
    assert await states.get(SESSION) is not None


async def test_name_and_email_then_asks_for_the_date(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, START)
    turn = await flow.handle(SESSION, "My name is Ram Sharma, ram@example.com")
    assert turn.status == "collecting"
    assert "Thanks, Ram." in turn.reply
    assert "Which date" in turn.reply


async def test_invalid_email_is_rejected_and_asked_again(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, START)
    turn = await flow.handle(SESSION, "my email is nope")
    assert "'nope' is not a valid email address." in turn.reply
    assert "What's your email address?" in turn.reply


async def test_past_date_is_rejected_with_a_message(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, START)
    turn = await flow.handle(SESSION, "last Monday")
    assert "That date is in the past." in turn.reply
    assert turn.status == "collecting"


async def test_time_already_passed_today_is_asked_again(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, START)
    turn = await flow.handle(SESSION, "today at 9am")
    assert "That time has already passed today." in turn.reply
    assert "What time suits you?" in turn.reply


async def test_all_details_lead_to_a_summary(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    turn = await flow.handle(SESSION, FULL)
    assert turn.status == "confirming"
    assert "Ram Sharma" in turn.reply
    assert "ram@example.com" in turn.reply
    assert "Tuesday, 17 November 2026 at 2:00 PM" in turn.reply
    assert "(yes/no)" in turn.reply
    assert repository.saved == []


async def test_yes_saves_and_clears_the_state(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, FULL)
    turn = await flow.handle(SESSION, "yes")
    assert turn.status == "confirmed"
    assert "Tuesday, 17 November 2026 at 2:00 PM" in turn.reply
    assert len(repository.saved) == 1
    assert repository.saved[0][0] == SESSION
    assert await states.get(SESSION) is None


async def test_yes_with_a_change_does_not_save(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, FULL)
    turn = await flow.handle(SESSION, "yes but make it 4pm")
    assert turn.status == "confirming"
    assert "4:00 PM" in turn.reply
    assert repository.saved == []


async def test_no_goes_back_to_collecting(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, FULL)
    turn = await flow.handle(SESSION, "no")
    assert turn.status == "collecting"
    assert "nothing has been booked" in turn.reply
    assert repository.saved == []
    state = await states.get(SESSION)
    assert state is not None and not state.awaiting_confirmation


async def test_a_change_without_yes_shows_the_summary_again(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, FULL)
    turn = await flow.handle(SESSION, "make it Wednesday")
    assert turn.status == "confirming"
    assert "Wednesday, 18 November 2026" in turn.reply
    assert repository.saved == []


async def test_cancel_clears_the_booking(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, FULL)
    turn = await flow.handle(SESSION, "cancel that")
    assert turn.status == "cancelled"
    assert turn.reply == "Booking cancelled."
    assert await states.get(SESSION) is None


async def test_cancel_without_a_booking(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    turn = await flow.handle(SESSION, "cancel that")
    assert turn.handled and turn.status == "none"
    assert turn.reply == "There is no booking in progress."


async def test_document_question_while_collecting_gets_a_reminder(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, START)
    turn = await flow.handle(SESSION, "What is the refund policy?")
    assert not turn.handled
    assert turn.status == "collecting"
    assert "still in progress" in turn.reminder
    assert "I still need your name, email, date, time." in turn.reminder


async def test_document_question_while_confirming_gets_a_reminder(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, FULL)
    turn = await flow.handle(SESSION, "What is the refund policy?")
    assert not turn.handled
    assert turn.status == "confirming"
    assert "waiting for confirmation" in turn.reminder
    assert repository.saved == []


async def test_question_without_a_booking_is_not_handled(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    turn = await flow.handle(SESSION, "What is the refund policy?")
    assert not turn.handled
    assert turn.status == "none"
    assert turn.reminder == ""


@pytest.mark.parametrize(
    "message", ["yes", "No", "ok", "Yes!", "  sure  ", "y", "nope", "Yeah."]
)
async def test_a_bare_reply_never_starts_a_booking(
    message: str, states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    # Worst case: the LLM claims this is a booking with a "yes".
    flow, llm = make(states, repository, {message: booking(confirmation="yes")})
    turn = await flow.handle(SESSION, message)
    assert not turn.handled
    assert turn.status == "none"
    assert llm.calls == []  # the guard runs before the LLM is asked
    assert await states.get(SESSION) is None
    assert repository.saved == []


async def test_yes_is_ignored_when_nothing_was_asked(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, START)
    turn = await flow.handle(SESSION, "sounds good")
    assert turn.status == "collecting"
    assert repository.saved == []


async def test_booking_the_same_slot_twice_is_rejected(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, FULL)
    await flow.handle(SESSION, "yes")
    await flow.handle(SESSION, FULL)
    turn = await flow.handle(SESSION, "yes")
    assert turn.status == "none"
    assert "already booked" in turn.reply
    assert len(repository.saved) == 1


async def test_two_sessions_do_not_share_a_booking(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository)
    await flow.handle("alice", FULL)
    turn = await flow.handle("bob", "What is the refund policy?")
    assert not turn.handled and turn.status == "none"


async def test_booking_state_has_a_timeout(
    states: BookingStateStore, repository: FakeBookingRepository, redis: Any
) -> None:
    flow, _ = make(states, repository)
    await flow.handle(SESSION, START)
    assert 0 < await redis.ttl(f"booking:{SESSION}") <= 1_800


async def test_garbage_from_the_llm_is_treated_as_not_a_booking(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository, {"hello there": "sorry, I cannot help"})
    turn = await flow.handle(SESSION, "hello there")
    assert not turn.handled and turn.status == "none"
