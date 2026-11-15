from typing import Any

import pytest

from app.services.booking_state import BookingStateStore
from tests.fakes import FakeBookingRepository
from tests.test_booking_flow import FULL, SESSION, START, make

DETAILS = "My name is Ram Sharma, ram@example.com"


async def test_details_wrongly_labelled_cancel_do_not_cancel(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    # The real model once did exactly this.
    wrong: dict[str, Any] = {
        "intent": "cancel",
        "name": "Ram Sharma",
        "email": "ram@example.com",
    }
    flow, _ = make(states, repository, {DETAILS: wrong})
    await flow.handle(SESSION, START)
    turn = await flow.handle(SESSION, DETAILS)
    assert turn.status == "collecting"
    assert "Thanks, Ram." in turn.reply
    state = await states.get(SESSION)
    assert state is not None and state.draft.name == "Ram Sharma"


async def test_a_question_wrongly_labelled_cancel_keeps_the_booking(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    question = "What is the refund policy?"
    flow, _ = make(states, repository, {question: {"intent": "cancel"}})
    await flow.handle(SESSION, START)
    turn = await flow.handle(SESSION, question)
    assert not turn.handled
    assert turn.status == "collecting"
    assert await states.get(SESSION) is not None


async def test_a_wrong_cancel_while_confirming_does_not_lose_the_summary(
    states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository, {"sounds fine": {"intent": "cancel"}})
    await flow.handle(SESSION, FULL)
    turn = await flow.handle(SESSION, "sounds fine")
    assert turn.status == "confirming"
    state = await states.get(SESSION)
    assert state is not None and state.awaiting_confirmation


@pytest.mark.parametrize(
    "message",
    [
        "cancel",
        "Cancel it please",
        "never mind",
        "nevermind",
        "forget it",
        "I don\u2019t want it anymore",
        "no longer needed",
        "stop",
        "call it off",
    ],
)
async def test_real_cancel_requests_still_cancel(
    message: str, states: BookingStateStore, repository: FakeBookingRepository
) -> None:
    flow, _ = make(states, repository, {message: {"intent": "cancel"}})
    await flow.handle(SESSION, START)
    turn = await flow.handle(SESSION, message)
    assert turn.status == "cancelled"
    assert await states.get(SESSION) is None
