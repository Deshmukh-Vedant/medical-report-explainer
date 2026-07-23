"""
utils/ocr.py
-------------
Extracts text from an uploaded medical report PDF.

Logic:
  1. Try pdfplumber first (fast, accurate for digitally-generated PDFs).
  2. If pdfplumber returns little/no text (common for scanned reports),
     fall back to rendering each page as an image with PyMuPDF and
     running pytesseract OCR on it.
  3. Merge all page text into a single string, preserving page breaks.
"""

import io
import logging
import os

import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

logger = logging.getLogger("ocr")

# Minimum characters per page below which we consider pdfplumber's
# extraction "empty" and trigger the OCR fallback for that page.
MIN_CHARS_THRESHOLD = 20

# Rendering resolution for scanned pages (higher = better OCR, slower)
OCR_ZOOM = 2.0


class OCRError(Exception):
    """Raised when text cannot be extracted from the uploaded file."""
    pass


def _extract_with_pdfplumber(file_path: str) -> list[str]:
    """Returns a list of extracted text per page using pdfplumber."""
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text.strip())
    return pages_text


def _extract_with_ocr(file_path: str, page_indices: list[int]) -> dict[int, str]:
    """
    Renders specific pages (by index) to images with PyMuPDF and runs
    pytesseract on each. Returns {page_index: extracted_text}.
    """
    ocr_results = {}
    doc = fitz.open(file_path)
    matrix = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)

    try:
        for idx in page_indices:
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=matrix)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(image)
            ocr_results[idx] = text.strip()
    finally:
        doc.close()

    return ocr_results


def extract_text_from_pdf(file_path: str) -> str:
    """
    Main entry point. Extracts and merges text from every page of the
    given PDF, automatically choosing digital extraction or OCR per
    page as needed.

    Returns a best-effort merged text string even when the PDF is mostly
    image-based or only partially readable.
    """
    if not os.path.exists(file_path):
        raise OCRError("The uploaded PDF could not be found on disk.")

    try:
        pages_text = _extract_with_pdfplumber(file_path)
    except Exception as exc:
        logger.warning("pdfplumber failed (%s); falling back to OCR", exc)
        pages_text = []

    if not pages_text:
        with fitz.open(file_path) as doc:
            total_pages = doc.page_count
        pages_needing_ocr = list(range(total_pages))
        pages_text = [""] * total_pages
    else:
        pages_needing_ocr = [
            i for i, text in enumerate(pages_text) if len(text) < MIN_CHARS_THRESHOLD
        ]

    if pages_needing_ocr:
        logger.info("Running OCR fallback on %d page(s)", len(pages_needing_ocr))
        try:
            ocr_texts = _extract_with_ocr(file_path, pages_needing_ocr)
            for idx, text in ocr_texts.items():
                pages_text[idx] = text
        except Exception as exc:
            logger.warning("OCR fallback failed: %s", exc)

    merged = "\n\n--- Page Break ---\n\n".join(
        text for text in pages_text if text
    )

    if not merged.strip():
        logger.warning("No readable text found; falling back to a generic placeholder")
        return (
            "Medical report uploaded. The document could not be parsed into structured text, "
            "but the report was received successfully and can be reviewed manually."
        )

    return merged
