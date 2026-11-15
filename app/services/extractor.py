from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class UnsupportedFileTypeError(Exception):
    """The file is not a .pdf or .txt."""


class UnreadableFileError(Exception):
    """The file is the right type but we could not get text out of it."""


def extract_text(filename: str, data: bytes) -> str:
    """Return the plain text of a .pdf or .txt file."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        text = _read_txt(data)
    elif suffix == ".pdf":
        text = _read_pdf(data)
    else:
        raise UnsupportedFileTypeError("Only .pdf and .txt files are supported.")

    text = text.strip()
    if not text:
        raise UnreadableFileError(
            "No text found. The file may be empty or a scanned image."
        )
    return text


def _read_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnreadableFileError("The text file is not valid UTF-8.") from exc


def _read_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise UnreadableFileError("Password-protected PDFs are not supported.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise UnreadableFileError("The PDF is corrupted or unreadable.") from exc
    return "\n\n".join(pages)
