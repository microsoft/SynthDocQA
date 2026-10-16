# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""PDF content loading utilities.

Two main access patterns:

  images (vision models)
    -- pdf_to_page_texts()  : extract all page texts cheaply (used for page selection)
    -- render_page_images() : render a specific list of page indices as base64 PNG

  text (text-only models)
    -- pdf_to_text()        : full concatenated text, optionally sliced to pages

PyMuPDF (fitz) is already a dependency of the synthpdf package.
Each public function opens and closes its own fitz document so that
worker threads do not share document handles.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 144  # 2x the fitz default; good balance of quality vs payload


# ---------------------------------------------------------------------------
# Per-page text extraction (cheap — used for page selection)
# ---------------------------------------------------------------------------

def pdf_to_page_texts(pdf_path: Path) -> list[str]:
    """Extract text from every page and return as a list (one string per page).

    Fast: fitz text extraction for a 200-page PDF typically takes < 1 second.
    Thread-safe: opens a fresh document handle.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    texts = [page.get_text() for page in doc]
    doc.close()
    return texts


# ---------------------------------------------------------------------------
# On-demand image rendering for selected pages
# ---------------------------------------------------------------------------

def render_page_images(
    pdf_path: Path,
    page_indices: list[int],
    dpi: int = _DEFAULT_DPI,
) -> list[str]:
    """Render specific PDF pages as base64-encoded PNG strings.

    Parameters
    ----------
    pdf_path:     Path to the PDF file.
    page_indices: 0-based page numbers to render (must be valid indices).
    dpi:          Render resolution.  144 DPI gives ~1190x918 px for A4 pages.

    Returns
    -------
    List of base64 PNG strings, one per requested page, in the order given.
    Thread-safe: opens a fresh document handle per call.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: list[str] = []
    for idx in page_indices:
        if idx < 0 or idx >= n_pages:
            logger.warning("Page index %d out of range for %s (%d pages)", idx, pdf_path.name, n_pages)
            continue
        pixmap = doc[idx].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        b64 = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
        images.append(b64)

    doc.close()
    return images


# ---------------------------------------------------------------------------
# Full-document text (for text-mode models)
# ---------------------------------------------------------------------------

def pdf_to_text(
    pdf_path: Path,
    page_indices: Optional[list[int]] = None,
) -> str:
    """Extract text from a PDF.

    If *page_indices* is given, only those pages are included (in order).
    Otherwise all pages are extracted and joined with a page-break marker.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    indices = page_indices if page_indices is not None else list(range(len(doc)))
    parts: list[str] = []
    for idx in indices:
        if 0 <= idx < len(doc):
            parts.append(doc[idx].get_text())
    doc.close()
    return "\n\n--- Page Break ---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Per-document cache (main thread only — NOT shared across worker threads)
# ---------------------------------------------------------------------------

class PDFPageCache:
    """Cache per-page texts for one document.

    page_texts is loaded once in the main thread and shared read-only with
    worker threads (safe under CPython's GIL).  Images are rendered on
    demand per question inside worker threads using separate fitz handles.
    """

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = pdf_path
        self._page_texts: Optional[list[str]] = None
        self._full_text: Optional[str] = None

    @property
    def page_texts(self) -> list[str]:
        """Per-page text list, loaded on first access."""
        if self._page_texts is None:
            self._page_texts = pdf_to_page_texts(self.pdf_path)
            logger.debug(
                "Extracted text from %d pages of %s",
                len(self._page_texts), self.pdf_path.name,
            )
        return self._page_texts

    @property
    def n_pages(self) -> int:
        return len(self.page_texts)

    def full_text(self, page_indices: Optional[list[int]] = None) -> str:
        """Return concatenated text.  If page_indices is None, return all pages."""
        if page_indices is None:
            if self._full_text is None:
                self._full_text = "\n\n--- Page Break ---\n\n".join(self.page_texts)
            return self._full_text
        return pdf_to_text(self.pdf_path, page_indices=page_indices)

    def render_images(self, page_indices: list[int], dpi: int = _DEFAULT_DPI) -> list[str]:
        """Render selected pages as base64 PNG strings (opens fresh fitz handle)."""
        return render_page_images(self.pdf_path, page_indices, dpi=dpi)
