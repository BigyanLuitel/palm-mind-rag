import random

import pytest

from app.services.chunker import fixed_chunks, sentence_chunks

SAMPLE = (
    "Refunds are accepted within 30 days. Items must be unused! "
    "Do you need a receipt? Yes, always.\n\nShipping takes 3 to 5 days. "
    "International orders may take longer."
)


def test_fixed_chunks_respect_size_and_overlap() -> None:
    text = "abcdefghij" * 10  # 100 characters
    chunks = fixed_chunks(text, size=30, overlap=10)
    assert all(len(c) <= 30 for c in chunks)
    assert chunks[0][-10:] == chunks[1][:10]


def test_fixed_chunks_short_text_is_one_chunk() -> None:
    assert fixed_chunks("hello world", size=100, overlap=10) == ["hello world"]


def test_fixed_chunks_empty_text() -> None:
    assert fixed_chunks("", size=100, overlap=10) == []
    assert fixed_chunks("   \n  ", size=100, overlap=10) == []


def test_sentence_chunks_keep_sentences_whole() -> None:
    chunks = sentence_chunks(SAMPLE, size=80, overlap=0)
    assert all(len(c) <= 80 for c in chunks)
    for chunk in chunks:
        assert chunk[-1] in ".!?"


def test_sentence_chunks_overlap_repeats_last_sentence() -> None:
    chunks = sentence_chunks(SAMPLE, size=90, overlap=40)
    assert len(chunks) >= 2
    shared = "Do you need a receipt?"
    assert chunks[0].endswith(shared)
    assert chunks[1].startswith(shared)


def test_sentence_chunks_split_a_huge_word() -> None:
    chunks = sentence_chunks("x" * 250, size=100, overlap=0)
    assert [len(c) for c in chunks] == [100, 100, 50]


@pytest.mark.parametrize("func", [fixed_chunks, sentence_chunks])
def test_overlap_must_be_smaller_than_size(func) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        func("some text", size=100, overlap=100)


def _random_text(rng: random.Random) -> str:
    pieces = ["a", "bb", "hello", "world.", "Yes!", "Why?", "\n\n", " ", "x" * 150]
    return "".join(rng.choice(pieces) + rng.choice(["", " "]) for _ in range(120))


@pytest.mark.parametrize("func", [fixed_chunks, sentence_chunks])
def test_fuzz_no_empty_chunk_and_none_too_long(func) -> None:  # type: ignore[no-untyped-def]
    rng = random.Random(1234)
    for _ in range(300):
        text = _random_text(rng)
        size = rng.randint(100, 400)
        overlap = rng.randint(0, size - 1)
        chunks = func(text, size, overlap)
        assert all(c.strip() for c in chunks), "empty chunk"
        assert all(len(c) <= size for c in chunks), "chunk longer than size"


def test_fuzz_sentence_chunks_lose_no_words() -> None:
    rng = random.Random(99)
    for _ in range(200):
        text = _random_text(rng)
        size = rng.randint(100, 400)
        chunks = sentence_chunks(text, size, rng.randint(0, size - 1))
        kept = {word for chunk in chunks for word in chunk.split()}
        for word in text.split():
            if len(word) <= size:
                assert word in kept
