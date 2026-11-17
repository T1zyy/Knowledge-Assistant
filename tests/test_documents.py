from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.documents import extract, split, tokenizer, validate_file
from app.errors import ServiceError


def test_format_validation():
    assert validate_file(r"C:\incoming\Guide.MD", "text/plain") == ("Guide.MD", "text/markdown")
    with pytest.raises(ServiceError) as exc:
        validate_file("image.png", "image/png")
    assert exc.value.status == 415


@pytest.mark.parametrize(
    ("blob", "media_type", "code"),
    [
        (b"\xef\xbb\xbf  ", "text/plain", "empty_document"),
        (b"invalid\xff", "text/plain", "invalid_encoding"),
        (b"a\0b", "text/plain", "invalid_text"),
        (b"text not PDF", "application/pdf", "invalid_pdf_signature"),
        (b"%PDF-1.4\nbroken", "application/pdf", "invalid_pdf"),
    ],
)
def test_extraction_failures(blob, media_type, code):
    with pytest.raises(ServiceError) as exc:
        extract(blob, media_type, max_pdf_pages=20, max_chars=1000)
    assert exc.value.code == code


def test_scan_without_text_is_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ServiceError) as exc:
        extract(output.getvalue(), "application/pdf", max_pdf_pages=20, max_chars=1000)
    assert exc.value.code == "no_pdf_text"


def test_token_chunking_preserves_cyrillic_and_emoji(settings):
    settings.chunk_tokens = 32
    settings.chunk_overlap = 6
    text = "Служебная заявка 🧪 содержит номер и описание. " * 24
    fragments = split(text.encode(), "text/plain", settings)
    assert len(fragments) > 2
    for item in fragments:
        assert "\ufffd" not in item.text
        assert item.text in text
        assert len(tokenizer().encode_ordinary(item.text)) <= settings.chunk_tokens
