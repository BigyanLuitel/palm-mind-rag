import re


def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Cut text into windows of `size` characters; neighbours share `overlap`."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def sentence_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Pack whole sentences into chunks of at most `size` characters."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    pieces = _split_long(_split_sentences(text), size)
    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and len(" ".join([*current, piece])) > size:
            chunks.append(" ".join(current))
            current = _tail(current, overlap)
            while current and len(" ".join([*current, piece])) > size:
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text)
    return [" ".join(p.split()) for p in parts if p.strip()]


def _split_long(sentences: list[str], size: int) -> list[str]:
    """A sentence longer than `size` is split by words so it still fits."""
    out: list[str] = []
    for sentence in sentences:
        if len(sentence) <= size:
            out.append(sentence)
            continue
        line = ""
        for word in sentence.split():
            if len(word) > size:  # e.g. a huge URL: cut it by characters
                if line:
                    out.append(line)
                    line = ""
                out.extend(fixed_chunks(word, size, 0))
            elif line and len(line) + 1 + len(word) > size:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            out.append(line)
    return out


def _tail(sentences: list[str], overlap: int) -> list[str]:
    """The last sentences that fit inside `overlap` characters."""
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        total += len(sentence) + (1 if tail else 0)
        if total > overlap:
            break
        tail.insert(0, sentence)
    return tail
