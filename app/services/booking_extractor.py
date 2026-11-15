import json
import logging
import re
from datetime import datetime

from pydantic import ValidationError

from app.services.booking import BookingAnalysis, BookingDraft
from app.services.llm import LLMClient, LLMMessage

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You help schedule job interviews.
Today is {today} (timezone Asia/Kathmandu). The current time is {clock}.

Read the user's latest message and reply with ONE JSON object and nothing else,
with exactly these keys:
- "intent": "booking" if the user wants to book or schedule an interview, is giving
  details for such a booking, or is answering a question about one; "cancel" if the
  user says they no longer want to book (cancel, forget it, never mind); otherwise
  "other".
- "name": the user's full name if they stated it, else null.
- "email": the user's email address if they stated it, else null.
- "date": the interview date as YYYY-MM-DD if they stated one (work out words like
  "tomorrow" or "next Friday" from today's date), else null.
- "time": the interview time as 24-hour HH:MM if they stated a specific time, else
  null. Vague times such as "morning" are null.
- "confirmation": "yes" or "no" only if the assistant had just asked the user to
  confirm the booking and the user answered, else null. Never use "no" unless the
  user actually declined.

Use only what the user actually said. Never guess or invent values. Never copy values
from "Already collected" unless the user repeats or changes them."""


async def analyze_message(
    llm: LLMClient,
    message: str,
    draft: BookingDraft,
    awaiting_confirmation: bool,
    now: datetime,
) -> BookingAnalysis:
    """Ask the LLM what this message says about a booking."""
    system = _SYSTEM_PROMPT.format(
        today=now.strftime("%A, %Y-%m-%d"), clock=now.strftime("%H:%M")
    )
    collected = draft.model_dump(mode="json")
    user = (
        f"Already collected: {json.dumps(collected)}\n"
        f"The assistant just asked the user to confirm the booking: "
        f"{'yes' if awaiting_confirmation else 'no'}\n"
        f"User message: {message}"
    )
    raw = await llm.complete(
        [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user),
        ],
        temperature=0.0,
        json_mode=True,
    )
    analysis = _parse(raw)
    if not awaiting_confirmation:
        # The model cannot answer a question we never asked: ignore its guess.
        analysis.confirmation = None
    return analysis


def _parse(raw: str) -> BookingAnalysis:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        logger.warning("Booking analysis had no JSON: %r", raw[:200])
        return BookingAnalysis()
    try:
        return BookingAnalysis.model_validate(json.loads(match.group()))
    except (json.JSONDecodeError, ValidationError):
        logger.warning("Booking analysis was not valid: %r", raw[:200])
        return BookingAnalysis()
