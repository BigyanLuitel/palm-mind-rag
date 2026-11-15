from typing import Any

import httpx

from app.services.llm import LLMUnavailableError
from app.services.rag import NOT_FOUND_ANSWER
from tests.fakes import ScriptedLLM
from tests.test_api_documents import upload

REFUND_QUESTION = "refunds accepted within 30 days?"
FULL = "I'm Ram Sharma, ram@example.com, next Tuesday at 2pm"
BOOKING_SCRIPT: dict[str, dict[str, Any]] = {
    "I want to book an interview": {"intent": "booking"},
    FULL: {
        "intent": "booking",
        "name": "Ram Sharma",
        "email": "ram@example.com",
        "date": "2026-11-17",
        "time": "14:00",
    },
    "yes": {"intent": "booking", "confirmation": "yes"},
}


async def chat(
    client: httpx.AsyncClient, message: str, session_id: str | None = None
) -> httpx.Response:
    body: dict[str, Any] = {"message": message}
    if session_id is not None:
        body["session_id"] = session_id
    return await client.post("/api/v1/chat", json=body)


async def test_a_session_id_is_created_when_missing(client: httpx.AsyncClient) -> None:
    response = await chat(client, "hello")
    assert response.status_code == 200
    body = response.json()
    assert len(body["session_id"]) >= 8
    assert body["answer"] == NOT_FOUND_ANSWER  # nothing uploaded yet
    assert body["sources"] == []
    assert body["booking_status"] == "none"


async def test_answers_come_from_the_uploaded_document(
    client: httpx.AsyncClient, llm: ScriptedLLM
) -> None:
    await upload(client, content=b"Refunds are accepted within 30 days of purchase.")
    llm.reply = "You have 30 days."
    body = (await chat(client, REFUND_QUESTION)).json()
    assert body["answer"] == "You have 30 days."
    assert body["sources"][0]["filename"] == "policy.txt"


async def test_second_turn_uses_the_first_as_history(
    client: httpx.AsyncClient, llm: ScriptedLLM
) -> None:
    await upload(client, content=b"Refunds are accepted within 30 days of purchase.")
    llm.reply = "30 days."
    session_id = (await chat(client, REFUND_QUESTION)).json()["session_id"]
    llm.rewrite = "Do refunds accepted within 30 days apply to sale items?"
    await chat(client, "and for sale items?", session_id)

    assert len(llm.rewrite_calls) == 1
    transcript = llm.rewrite_calls[0][-1].content
    assert REFUND_QUESTION in transcript and "30 days." in transcript
    assert any(m.content == REFUND_QUESTION for m in llm.answer_calls[-1])


async def test_sessions_do_not_see_each_others_history(
    client: httpx.AsyncClient, llm: ScriptedLLM
) -> None:
    await chat(client, "secret question from alice", "alice")
    await chat(client, "hello", "bob")
    assert llm.rewrite_calls == []  # bob has no history, so nothing to rewrite


async def test_bad_chat_requests_are_422(client: httpx.AsyncClient) -> None:
    assert (await chat(client, "   ")).status_code == 422
    assert (await chat(client, "")).status_code == 422
    assert (await chat(client, "x" * 2001)).status_code == 422
    assert (await chat(client, "hi", "bad id!")).status_code == 422
    assert (await client.post("/api/v1/chat", json={})).status_code == 422


async def test_llm_outage_becomes_503(
    client: httpx.AsyncClient, llm: ScriptedLLM
) -> None:
    llm.error = LLMUnavailableError("provider down")
    response = await chat(client, "hello")
    assert response.status_code == 503
    assert "provider down" not in response.text


async def test_full_booking_conversation_is_saved(
    client: httpx.AsyncClient, llm: ScriptedLLM
) -> None:
    llm.script.update(BOOKING_SCRIPT)
    first = (await chat(client, "I want to book an interview")).json()
    session_id = first["session_id"]
    assert first["booking_status"] == "collecting"

    summary = (await chat(client, FULL, session_id)).json()
    assert summary["booking_status"] == "confirming"
    assert "Tuesday, 17 November 2026 at 2:00 PM" in summary["answer"]
    assert (await client.get("/api/v1/bookings")).json() == []  # not saved yet

    done = (await chat(client, "yes", session_id)).json()
    assert done["booking_status"] == "confirmed"

    rows = (await client.get("/api/v1/bookings")).json()
    assert len(rows) == 1
    assert rows[0]["name"] == "Ram Sharma"
    assert rows[0]["email"] == "ram@example.com"
    assert rows[0]["booking_date"] == "2026-11-17"
    assert rows[0]["booking_time"].startswith("14:00")
    assert rows[0]["session_id"] == session_id


async def test_booking_the_same_slot_twice_saves_one_row(
    client: httpx.AsyncClient, llm: ScriptedLLM
) -> None:
    llm.script.update(BOOKING_SCRIPT)
    session_id = "repeat-customer"
    for _ in range(2):
        await chat(client, FULL, session_id)
        last = (await chat(client, "yes", session_id)).json()
    assert "already booked" in last["answer"]
    assert len((await client.get("/api/v1/bookings")).json()) == 1


async def test_a_lone_yes_never_creates_a_booking(
    client: httpx.AsyncClient, llm: ScriptedLLM
) -> None:
    llm.script.update(BOOKING_SCRIPT)
    body = (await chat(client, "yes")).json()
    assert body["booking_status"] == "none"
    assert (await client.get("/api/v1/bookings")).json() == []


async def test_deleting_a_chat_clears_history_and_open_booking(
    client: httpx.AsyncClient, llm: ScriptedLLM
) -> None:
    llm.script.update(BOOKING_SCRIPT)
    session_id = (await chat(client, "I want to book an interview")).json()[
        "session_id"
    ]
    still_open = (await chat(client, "What now?", session_id)).json()
    assert still_open["booking_status"] == "collecting"

    assert (await client.delete(f"/api/v1/chat/{session_id}")).status_code == 204

    after = (await chat(client, "What now?", session_id)).json()
    assert after["booking_status"] == "none"
    assert len(llm.rewrite_calls) == 1  # only before the delete, when history existed


async def test_deleting_with_a_bad_session_id_is_422(client: httpx.AsyncClient) -> None:
    assert (await client.delete("/api/v1/chat/bad!id")).status_code == 422