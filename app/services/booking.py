import re
from datetime import date, datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, field_validator

TIMEZONE = ZoneInfo("Asia/Kathmandu")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_DAYS_AHEAD = 365


class BookingDraft(BaseModel):
    """What we know so far about an interview booking."""

    name: str | None = None
    email: str | None = None
    booking_date: date | None = None
    booking_time: time | None = None

    def missing(self) -> list[str]:
        labels = {
            "name": self.name,
            "email": self.email,
            "date": self.booking_date,
            "time": self.booking_time,
        }
        return [label for label, value in labels.items() if value is None]

    def is_complete(self) -> bool:
        return not self.missing()


class BookingAnalysis(BaseModel):
    """What the LLM found in one user message (not yet checked by our rules)."""

    intent: Literal["booking", "cancel", "other"] = "other"
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None
    confirmation: Literal["yes", "no"] | None = None

    @field_validator("name", "email", "date", "time", "confirmation", mode="before")
    @classmethod
    def empty_means_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a"}:
            return None
        return value


def apply_analysis(
    draft: BookingDraft, analysis: BookingAnalysis, now: datetime
) -> tuple[BookingDraft, list[str]]:
    """Merge new values into the draft. Invalid values are dropped with a message."""
    updated = draft.model_copy()
    problems: list[str] = []

    if analysis.name is not None:
        name = " ".join(analysis.name.split())
        if 2 <= len(name) <= 200:
            updated.name = name
        else:
            problems.append("That name doesn't look right.")

    if analysis.email is not None:
        email = analysis.email.strip().lower()
        if len(email) <= 320 and EMAIL_PATTERN.match(email):
            updated.email = email
        else:
            problems.append(f"'{analysis.email}' is not a valid email address.")

    if analysis.date is not None:
        parsed_date = _parse_date(analysis.date)
        today = now.date()
        if parsed_date is None:
            problems.append("I couldn't understand that date.")
        elif parsed_date < today:
            problems.append("That date is in the past.")
        elif parsed_date > today + timedelta(days=MAX_DAYS_AHEAD):
            problems.append("That date is more than a year away.")
        else:
            updated.booking_date = parsed_date

    if analysis.time is not None:
        parsed_time = _parse_time(analysis.time)
        if parsed_time is None:
            problems.append("I couldn't understand that time.")
        else:
            updated.booking_time = parsed_time

    if (
        updated.booking_date == now.date()
        and updated.booking_time is not None
        and updated.booking_time <= now.time()
    ):
        problems.append("That time has already passed today.")
        updated.booking_time = None

    return updated, problems


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _parse_time(value: str) -> time | None:
    try:
        parsed = time.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed.replace(second=0, microsecond=0, tzinfo=None)