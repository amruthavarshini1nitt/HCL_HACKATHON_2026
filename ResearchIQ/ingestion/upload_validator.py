"""Step A1: validate uploaded PDFs and register them (duplicate-safe)."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from config import settings
from database.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
# Active content that has no place in a research paper. Best-effort byte scan (streams may be compressed);
# PyMuPDF never executes scripts, so this is defence in depth.
ACTIVE_MARKERS = (b"/Launch", b"/JavaScript", b"/JS ", b"/JS(", b"/RichMedia", b"/SubmitForm")


class UploadError(Exception):
    """Validation failure with a message that is safe to show to users."""


@dataclass
class ValidatedPDF:
    sha256: str
    page_count: int
    size_bytes: int
    metadata: dict[str, str]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_filename(name: str) -> str:
    """Strip directories and unsafe characters from a client-supplied filename."""
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "paper.pdf"
    return base[:120]


def validate_pdf_bytes(data: bytes, filename: str) -> ValidatedPDF:
    """Check extension, size, signature, encryption, readability and page count."""
    if not filename.lower().endswith(".pdf"):
        raise UploadError("Only .pdf files are accepted.")
    if not data:
        raise UploadError("The file is empty.")
    max_bytes = settings.max_file_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise UploadError(f"File exceeds the {settings.max_file_mb} MB limit.")
    if not data[:1024].lstrip().startswith(PDF_MAGIC):
        raise UploadError("File content is not a valid PDF (bad signature).")
    for marker in ACTIVE_MARKERS:
        if marker in data:
            raise UploadError("The PDF contains active content (scripts or launch actions) and was rejected.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:  # noqa: BLE001 - PyMuPDF raises several types
        logger.warning("unreadable PDF %s", safe_filename(filename), exc_info=True)
        raise UploadError("The PDF is malformed or unreadable.") from None
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise UploadError("Encrypted or password-protected PDFs are not supported.")
        pages = doc.page_count
        if pages < 1:
            raise UploadError("The PDF has no pages.")
        if pages > settings.max_pages:
            raise UploadError(f"The PDF has {pages} pages; the limit is {settings.max_pages}.")
        try:
            doc.load_page(0).get_text()
        except Exception:  # noqa: BLE001
            raise UploadError("The PDF's first page cannot be read.") from None
        meta = {k: (v or "") for k, v in (doc.metadata or {}).items()}
    finally:
        doc.close()
    return ValidatedPDF(sha256=sha256_bytes(data), page_count=pages, size_bytes=len(data), metadata=meta)


@dataclass
class UploadResult:
    filename: str
    ok: bool
    paper_id: str | None = None
    message: str = ""
    duplicate_of: str | None = None


def register_upload(data: bytes, filename: str, store: MetadataStore) -> UploadResult:
    """Validate, de-duplicate by SHA-256, save to disk and register in the catalogue."""
    fname = safe_filename(filename)
    try:
        v = validate_pdf_bytes(data, filename)
    except UploadError as e:
        return UploadResult(fname, False, message=str(e))
    existing = store.get_paper_by_hash(v.sha256)
    if existing:
        return UploadResult(fname, False, paper_id=existing["paper_id"],
                            duplicate_of=existing["paper_id"],
                            message=f"Duplicate of {existing['paper_id']} ({existing['original_filename']}).")
    settings.papers_dir.mkdir(parents=True, exist_ok=True)
    path = settings.papers_dir / f"{v.sha256[:16]}_{fname}"
    path.write_bytes(data)
    try:
        pid = store.register_paper(
            original_filename=fname, sha256=v.sha256, storage_path=str(path), page_count=v.page_count,
            title=(v.metadata.get("title") or None))
    except ValueError:  # lost a race with a concurrent upload of the same file
        path.unlink(missing_ok=True)
        dup = store.get_paper_by_hash(v.sha256)
        return UploadResult(fname, False, paper_id=dup and dup["paper_id"],
                            duplicate_of=dup and dup["paper_id"], message="Duplicate paper.")
    return UploadResult(fname, True, paper_id=pid, message="Registered.")
