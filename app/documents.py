import io
import re
from functools import lru_cache
from pathlib import PurePosixPath

import tiktoken
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import Settings
from app.errors import ServiceError
from app.models import Fragment

MEDIA_TYPES = {
    ".pdf": ("application/pdf",),
    ".txt": ("text/plain",),
    ".md": ("text/markdown", "text/plain", "text/x-markdown"),
    ".markdown": ("text/markdown", "text/plain", "text/x-markdown"),
}


def validate_file(filename: str | None, content_type: str | None) -> tuple[str, str]:
    basename = PurePosixPath((filename or "").replace("\\", "/")).name
    ext = PurePosixPath(basename).suffix.lower()
    supported = MEDIA_TYPES.get(ext, ())
    mime = (content_type or "").partition(";")[0].lower().strip()
    if not supported or mime not in (*supported, "", "application/octet-stream"):
        raise ServiceError(415, "unsupported_file_type", "Upload a PDF, TXT or Markdown file")
    if len(basename) > 255 or any(not char.isprintable() for char in basename):
        raise ServiceError(422, "invalid_filename", "Invalid file name")
    return basename, supported[0]


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract(content: bytes, media_type: str, max_pdf_pages: int, max_chars: int) -> list[Fragment]:
    if media_type == "application/pdf":
        if not content.startswith(b"%PDF-"):
            raise ServiceError(415, "invalid_pdf_signature", "File is not a PDF")
        pages = []
        size = 0
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise ServiceError(422, "encrypted_pdf", "Encrypted PDFs are not supported")
            if len(reader.pages) > max_pdf_pages:
                raise ServiceError(413, "too_many_pages", "PDF exceeds the page limit")
            for number, page in enumerate(reader.pages, 1):
                if page.get_contents() is not None:
                    text = clean((page.extract_text() or "").replace("\x00", ""))
                    size += len(text)
                    if size > max_chars:
                        raise ServiceError(413, "document_too_long", "Extracted text is too long")
                    if text:
                        pages.append(Fragment(text, number))
        except (PdfReadError, ValueError, TypeError, KeyError, RecursionError) as exc:
            raise ServiceError(422, "invalid_pdf", "PDF text extraction failed") from exc
        if not pages:
            raise ServiceError(422, "no_pdf_text", "PDF has no text layer")
        return pages
    try:
        text = clean(content.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ServiceError(422, "invalid_encoding", "TXT and Markdown must be UTF-8") from exc
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        raise ServiceError(422, "invalid_text", "File contains binary control characters")
    if len(text) > max_chars:
        raise ServiceError(413, "document_too_long", "Extracted text is too long")
    if not text:
        raise ServiceError(422, "empty_document", "Document has no text")
    return [Fragment(text)]


@lru_cache(maxsize=2)
def tokenizer(name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def split(content: bytes, media_type: str, settings: Settings) -> list[Fragment]:
    pages = extract(content, media_type, settings.max_pdf_pages, settings.max_chunks * 12_000)
    encoding = tokenizer()
    fragments = []
    for page in pages:
        tokens = encoding.encode_ordinary(page.text)
        text, offsets = encoding.decode_with_offsets(tokens)
        offsets.append(len(text))
        start = 0
        while start < len(tokens):
            end = min(start + settings.chunk_tokens, len(tokens))
            while end > start:
                piece = text[offsets[start] : offsets[end]].strip()
                if (
                    offsets[end - 1] != offsets[end]
                    and len(encoding.encode_ordinary(piece)) <= settings.chunk_tokens
                ):
                    break
                end -= 1
            piece = text[offsets[start] : offsets[end]].strip()
            if piece:
                fragments.append(Fragment(piece, page.page))
            if len(fragments) > settings.max_chunks:
                raise ServiceError(413, "too_many_chunks", "Document exceeds the chunk limit")
            if end == len(tokens):
                break
            new_start = max(start + 1, end - settings.chunk_overlap)
            while offsets[new_start] == offsets[new_start - 1]:
                new_start += 1
            start = new_start
    if not fragments:
        raise ServiceError(422, "empty_document", "Document has no text")
    return fragments
