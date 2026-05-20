"""
Lab report OCR — PyMuPDF render → preprocess → Tesseract → clean text → JSON storage.

Types:
    OCRExtractionResult, PageExtractionDetail, ExtractionMetadata

Functions:
    pdf_to_images           — PDF pages to PIL images at configurable DPI
    preprocess_image        — grayscale, sharpen, threshold for Tesseract
    extract_text_from_image — single-page OCR
    clean_text              — normalize whitespace and junk characters
    extract_text_from_pdf   — full pipeline; optional save to data/ocr/{user_id}/
    save_extraction_for_user / load_latest_extraction — persistence
    parse_lab_report_text   — convenience: OCR text → biomarkers via tools.biomarkers
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageFilter, ImageOps

from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _configure_tesseract() -> None:
    """Apply optional Tesseract binary path from settings."""
    cmd = get_settings().tesseract_cmd
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


_configure_tesseract()

# Default render resolution for PDF → image (higher improves OCR on small text)
DEFAULT_PDF_DPI = 200
DEFAULT_TESSERACT_LANG = "eng"
PREPROCESSING_STEPS = ("grayscale", "sharpen", "threshold")


class OCRExtractionResult(TypedDict):
    """Structured result returned by PDF extraction."""

    text: str
    page_count: int
    metadata: dict[str, Any]
    success: bool
    storage_path: str | None


@dataclass
class PageExtractionDetail:
    """Per-page OCR diagnostics."""

    page_number: int
    char_count: int
    success: bool
    error: str | None = None


@dataclass
class ExtractionMetadata:
    """Metadata persisted with each user extraction."""

    source_file: str
    extracted_at: str
    page_count: int
    pages_succeeded: int
    pages_failed: int
    total_characters: int
    preprocessing: list[str] = field(default_factory=lambda: list(PREPROCESSING_STEPS))
    tesseract_lang: str = DEFAULT_TESSERACT_LANG
    pdf_dpi: int = DEFAULT_PDF_DPI
    page_details: list[PageExtractionDetail] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def pdf_to_images(
    pdf_path: str | Path,
    dpi: int = DEFAULT_PDF_DPI,
) -> list[Image.Image]:
    """
    Convert each page of a PDF into a PIL RGB image using PyMuPDF.

    Args:
        pdf_path: Path to the input PDF file.
        dpi: Target resolution for rendering (72 DPI is PDF default).

    Returns:
        List of PIL images, one per successfully rendered page.
        Returns an empty list if the file is missing or cannot be opened.
    """
    path = Path(pdf_path)
    if not path.is_file():
        logger.error("PDF not found: %s", path)
        return []

    images: list[Image.Image] = []
    doc: fitz.Document | None = None
    page_total = 0

    try:
        doc = fitz.open(path)
        page_total = doc.page_count
        logger.info("Rendering PDF %s (%d pages) at %d DPI", path.name, page_total, dpi)

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for index in range(page_total):
            page_number = index + 1
            try:
                page = doc.load_page(index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                images.append(image)
                logger.debug("Rendered page %d/%d (%dx%d)", page_number, page_total, pixmap.width, pixmap.height)
            except Exception as exc:
                logger.warning(
                    "Failed to render page %d of %s: %s",
                    page_number,
                    path.name,
                    exc,
                    exc_info=True,
                )
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", path, exc, exc_info=True)
        return []
    finally:
        if doc is not None:
            doc.close()

    logger.info("PDF %s: rendered %d/%d pages", path.name, len(images), page_total)
    return images


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Apply OCR-oriented preprocessing: grayscale, sharpen, threshold.

    Args:
        image: Source PIL image (RGB or other mode).

    Returns:
        Preprocessed single-bit-style image suitable for Tesseract.
    """
    gray = image.convert("L")
    sharpened = gray.filter(ImageFilter.SHARPEN)
    # Autocontrast improves threshold separation on uneven scans
    contrasted = ImageOps.autocontrast(sharpened)
    thresholded = contrasted.point(lambda px: 255 if px > 140 else 0, mode="1")
    return thresholded.convert("L")


def extract_text_from_image(
    image: str | Path | Image.Image,
    *,
    lang: str = DEFAULT_TESSERACT_LANG,
    apply_preprocessing: bool = True,
) -> str:
    """
    Extract text from a single image via Tesseract OCR.

    Args:
        image: File path or in-memory PIL image (e.g. from pdf_to_images).
        lang: Tesseract language code(s).
        apply_preprocessing: When True, run preprocess_image before OCR.

    Returns:
        Raw OCR text for the image (empty string on failure).
    """
    pil_image: Image.Image | None = None
    source_label = "image"

    try:
        if isinstance(image, Image.Image):
            pil_image = image.copy()
            source_label = "PIL.Image"
        else:
            path = Path(image)
            source_label = path.name
            if not path.is_file():
                logger.error("Image file not found: %s", path)
                return ""
            pil_image = Image.open(path)
            pil_image.load()
    except Exception as exc:
        logger.error("Failed to load image %s: %s", source_label, exc, exc_info=True)
        return ""

    try:
        ocr_input = preprocess_image(pil_image) if apply_preprocessing else pil_image
        text = pytesseract.image_to_string(ocr_input, lang=lang)
        logger.debug("OCR on %s: %d characters", source_label, len(text))
        return text
    except pytesseract.TesseractNotFoundError:
        logger.error(
            "Tesseract executable not found. Install tesseract-ocr and ensure it is on PATH."
        )
        return ""
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", source_label, exc, exc_info=True)
        return ""
    finally:
        if pil_image is not None and not isinstance(image, Image.Image):
            pil_image.close()


def clean_text(text: str) -> str:
    """
    Normalize OCR output into readable concatenated text.

    - Strips trailing/leading whitespace per line
    - Collapses runs of blank lines
    - Removes non-printable control characters
    """
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)

    joined = "\n".join(lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    joined = re.sub(r"[^\S\n]+", " ", joined)
    return joined.strip()


def extract_text_from_pdf(
    pdf_path: str | Path,
    *,
    user_id: str | None = None,
    dpi: int = DEFAULT_PDF_DPI,
    lang: str = DEFAULT_TESSERACT_LANG,
    store: bool = True,
) -> OCRExtractionResult:
    """
    Extract and clean text from all pages of a PDF.

    Pipeline: pdf_to_images → preprocess → Tesseract → clean_text → optional persist.

    Args:
        pdf_path: Path to the PDF file.
        user_id: When set and store=True, persist results under data/ocr/{user_id}/.
        dpi: Rendering resolution passed to pdf_to_images.
        lang: Tesseract language code(s).
        store: Persist extraction JSON for user_id when provided.

    Returns:
        OCRExtractionResult with cleaned text, page_count, metadata, and success flag.
    """
    path = Path(pdf_path)
    metadata = ExtractionMetadata(
        source_file=str(path.resolve()),
        extracted_at=datetime.now(timezone.utc).isoformat(),
        page_count=0,
        pages_succeeded=0,
        pages_failed=0,
        total_characters=0,
        pdf_dpi=dpi,
        tesseract_lang=lang,
    )

    if not path.is_file():
        metadata.errors.append(f"PDF not found: {path}")
        return _build_result("", metadata, success=False, storage_path=None)

    settings = get_settings()
    dpi = dpi or settings.ocr_pdf_dpi
    lang = lang or settings.ocr_tesseract_lang
    metadata.pdf_dpi = dpi
    metadata.tesseract_lang = lang

    page_images = pdf_to_images(path, dpi=dpi)
    metadata.page_count = len(page_images)

    if not page_images:
        metadata.errors.append("No pages rendered from PDF")
        return _build_result("", metadata, success=False, storage_path=None)

    page_texts: list[str] = []
    for index, page_image in enumerate(page_images):
        page_number = index + 1
        try:
            raw = extract_text_from_image(page_image, lang=lang, apply_preprocessing=True)
            cleaned_page = clean_text(raw)
            page_texts.append(cleaned_page)
            detail = PageExtractionDetail(
                page_number=page_number,
                char_count=len(cleaned_page),
                success=True,
            )
            metadata.pages_succeeded += 1
            if not cleaned_page:
                metadata.warnings.append(f"Page {page_number}: no text detected")
        except Exception as exc:
            detail = PageExtractionDetail(
                page_number=page_number,
                char_count=0,
                success=False,
                error=str(exc),
            )
            metadata.pages_failed += 1
            metadata.errors.append(f"Page {page_number}: {exc}")
            logger.warning("OCR failed on page %d of %s: %s", page_number, path.name, exc)
        metadata.page_details.append(detail)

    full_text = clean_text("\n\n".join(t for t in page_texts if t))
    metadata.total_characters = len(full_text)

    success = metadata.pages_succeeded > 0 and bool(full_text)
    if metadata.pages_failed and metadata.pages_succeeded:
        metadata.warnings.append(
            f"Partial extraction: {metadata.pages_failed} page(s) failed"
        )

    storage_path: str | None = None
    if store and user_id:
        try:
            saved = save_extraction_for_user(user_id, full_text, metadata)
            storage_path = str(saved)
            logger.info("Stored OCR extraction for user %s at %s", user_id, saved)
        except Exception as exc:
            metadata.errors.append(f"Storage failed: {exc}")
            logger.error("Failed to store OCR for user %s: %s", user_id, exc, exc_info=True)

    return _build_result(full_text, metadata, success=success, storage_path=storage_path)


def save_extraction_for_user(
    user_id: str,
    text: str,
    metadata: ExtractionMetadata,
) -> Path:
    """
    Persist cleaned OCR text and metadata under data/ocr/{user_id}/.

    Args:
        user_id: Target user identifier.
        text: Cleaned concatenated OCR text.
        metadata: Extraction diagnostics and provenance.

    Returns:
        Path to the written JSON file.
    """
    settings = get_settings()
    user_dir = settings.data_dir / "ocr" / _sanitize_user_id(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    source_stem = Path(metadata.source_file).stem
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = user_dir / f"{timestamp}_{source_stem}.json"

    payload = {
        "user_id": user_id,
        "text": text,
        "page_count": metadata.page_count,
        "metadata": _metadata_to_dict(metadata),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _append_latest_pointer(user_dir, out_path)
    return out_path


def load_latest_extraction(user_id: str) -> dict[str, Any] | None:
    """
    Load the most recent OCR extraction for a user.

    Args:
        user_id: Target user identifier.

    Returns:
        Parsed JSON dict or None if no extractions exist.
    """
    settings = get_settings()
    pointer = settings.data_dir / "ocr" / _sanitize_user_id(user_id) / "latest.json"
    if not pointer.is_file():
        return None
    try:
        return json.loads(pointer.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load latest OCR for %s: %s", user_id, exc)
        return None


def parse_lab_report_text(text: str, *, user_id: str | None = None) -> dict[str, Any]:
    """
    Parse OCR text into a structured health profile via deterministic extraction.

    Args:
        text: Cleaned OCR output.
        user_id: When set, persist profile under data/health/{user_id}/.

    Returns:
        Dict with raw_text and validated health profile.
    """
    from tools.biomarkers import build_health_profile

    profile = build_health_profile(text, user_id=user_id, source="ocr", store=bool(user_id))
    return {"raw_text": text, "profile": profile}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_user_id(user_id: str) -> str:
    """Restrict user_id to safe path segment characters."""
    cleaned = re.sub(r"[^\w\-]", "_", user_id.strip())
    return cleaned or "unknown_user"


def _metadata_to_dict(metadata: ExtractionMetadata) -> dict[str, Any]:
    data = asdict(metadata)
    data["page_details"] = [asdict(p) for p in metadata.page_details]
    return data


def _build_result(
    text: str,
    metadata: ExtractionMetadata,
    *,
    success: bool,
    storage_path: str | None,
) -> OCRExtractionResult:
    return {
        "text": text,
        "page_count": metadata.page_count,
        "metadata": _metadata_to_dict(metadata),
        "success": success,
        "storage_path": storage_path,
    }


def _append_latest_pointer(user_dir: Path, extraction_path: Path) -> None:
    """Write a small index file pointing at the latest extraction."""
    pointer = user_dir / "latest.json"
    pointer.write_text(
        json.dumps(
            {
                "path": str(extraction_path),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
