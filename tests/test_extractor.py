from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.services.extractor import (
    UnreadableFileError,
    UnsupportedFileTypeError,
    extract_text,
)


def test_txt_is_decoded_and_stripped() -> None:
    assert extract_text("notes.txt", b"  hello world \n") == "hello world"


def test_txt_with_bom_and_unicode() -> None:
    data = "\ufeffनमस्ते world".encode()
    assert extract_text("a.txt", data) == "नमस्ते world"


def test_extension_is_case_insensitive() -> None:
    assert extract_text("NOTES.TXT", b"hi") == "hi"


def test_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("report.docx", b"data")


def test_empty_txt_is_unreadable() -> None:
    with pytest.raises(UnreadableFileError):
        extract_text("empty.txt", b"   \n ")


def test_non_utf8_txt_is_unreadable() -> None:
    with pytest.raises(UnreadableFileError):
        extract_text("bad.txt", b"\xff\xfe\x00bad\x80")


def test_corrupt_pdf_is_unreadable() -> None:
    with pytest.raises(UnreadableFileError):
        extract_text("broken.pdf", b"this is not a pdf at all")


def test_blank_pdf_is_unreadable() -> None:
    """A page with no text behaves like a scanned document."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(UnreadableFileError):
        extract_text("scan.pdf", buffer.getvalue())
