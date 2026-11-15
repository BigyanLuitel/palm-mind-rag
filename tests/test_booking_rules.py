from datetime import date, time

from app.services.booking import (
    BookingAnalysis,
    BookingDraft,
    apply_analysis,
)
from tests.fakes import NOW


def apply(
    draft: BookingDraft | None = None, **fields: str
) -> tuple[BookingDraft, list[str]]:
    return apply_analysis(
        draft or BookingDraft(), BookingAnalysis(intent="booking", **fields), NOW
    )


def test_valid_values_fill_the_draft() -> None:
    draft, problems = apply(
        name="  Ram   Sharma ",
        email=" Ram@Example.COM ",
        date="2026-11-17",
        time="14:00",
    )
    assert problems == []
    assert draft.name == "Ram Sharma"
    assert draft.email == "ram@example.com"
    assert draft.booking_date == date(2026, 11, 17)
    assert draft.booking_time == time(14, 0)
    assert draft.is_complete()


def test_missing_lists_what_is_still_needed() -> None:
    draft, _ = apply(name="Ram Sharma")
    assert draft.missing() == ["email", "date", "time"]


def test_short_name_is_rejected() -> None:
    draft, problems = apply(name="R")
    assert draft.name is None
    assert problems == ["That name doesn't look right."]


def test_bad_emails_are_rejected() -> None:
    for bad in ("nope", "a@b", "a b@c.com", "@c.com", "a@@c.com"):
        draft, problems = apply(email=bad)
        assert draft.email is None, bad
        assert problems, bad


def test_past_date_is_rejected() -> None:
    draft, problems = apply(date="2026-11-14")
    assert draft.booking_date is None
    assert problems == ["That date is in the past."]


def test_today_is_allowed() -> None:
    draft, problems = apply(date="2026-11-15")
    assert draft.booking_date == date(2026, 11, 15)
    assert problems == []


def test_date_more_than_a_year_away_is_rejected() -> None:
    draft, problems = apply(date="2027-12-01")
    assert draft.booking_date is None
    assert problems == ["That date is more than a year away."]


def test_garbage_date_and_time_are_rejected() -> None:
    draft, problems = apply(date="next tuesday", time="3pm")
    assert draft.booking_date is None
    assert draft.booking_time is None
    assert problems == [
        "I couldn't understand that date.",
        "I couldn't understand that time.",
    ]


def test_seconds_and_timezone_are_dropped_from_time() -> None:
    draft, _ = apply(time="14:00:30")
    assert draft.booking_time == time(14, 0)
    draft, _ = apply(time="14:00+05:45")
    assert draft.booking_time == time(14, 0)
    assert draft.booking_time.tzinfo is None


def test_time_that_already_passed_today_is_rejected() -> None:
    draft, problems = apply(date="2026-11-15", time="09:00")
    assert draft.booking_time is None
    assert problems == ["That time has already passed today."]


def test_later_time_today_is_allowed() -> None:
    draft, problems = apply(date="2026-11-15", time="15:00")
    assert draft.booking_time == time(15, 0)
    assert problems == []


def test_same_time_tomorrow_is_allowed() -> None:
    draft, problems = apply(date="2026-11-16", time="09:00")
    assert draft.booking_time == time(9, 0)
    assert problems == []


def test_invalid_value_never_overwrites_a_good_one() -> None:
    existing = BookingDraft(name="Ram Sharma", email="ram@example.com")
    draft, problems = apply(existing, email="broken")
    assert draft.email == "ram@example.com"
    assert problems


def test_apply_does_not_change_the_input_draft() -> None:
    existing = BookingDraft()
    apply(existing, name="Ram Sharma")
    assert existing.name is None


def test_empty_words_from_the_llm_mean_none() -> None:
    analysis = BookingAnalysis(
        name="null", email="N/A", date="", time="None", confirmation="  "
    )
    assert analysis.name is None
    assert analysis.email is None
    assert analysis.date is None
    assert analysis.time is None
    assert analysis.confirmation is None