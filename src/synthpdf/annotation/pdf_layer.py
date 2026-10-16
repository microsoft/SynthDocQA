# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""PDF-layer inline annotation renderer.

Uses PyMuPDF (fitz) to apply ``AnnotationSpec`` objects directly onto a
rendered PDF.  Each annotation type maps to a specific PyMuPDF operation:

  highlight      -- page.add_highlight_annot()   (coloured PDF highlight)
  text_color     -- add_highlight_annot() + optional add_strikeout_annot()
  circle         -- page.draw_oval()             (drawn on page content)
  margin_callout -- page.insert_textbox()        (text drawn in left margin)
  sticky_note    -- page.add_text_annot()        (PDF sticky-note icon)
  score_callout  -- draw_oval() + insert_text()  (badge + comment at bottom)

All operations modify the PDF in-place (same path).  The function returns a
dict mapping annotation_id -> page_number (0-based) so the manifest can
record which page each annotation landed on.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from synthpdf.annotation.text_picker import AnnotationSpec

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_NAMED_COLORS: dict[str, tuple[float, float, float]] = {
    "yellow": (1.0, 0.95, 0.0),
    "orange": (1.0, 0.60, 0.0),
    "green":  (0.55, 0.90, 0.55),
    "pink":   (1.0, 0.70, 0.85),
    "blue":   (0.15, 0.35, 0.90),
    "teal":   (0.00, 0.55, 0.55),
    "red":    (0.80, 0.05, 0.05),
    "purple": (0.50, 0.00, 0.75),
}


def _to_rgb(color: str) -> tuple[float, float, float]:
    """Convert a colour name or #RRGGBB hex string to an (r, g, b) tuple in [0, 1]."""
    lower = color.lower().strip()
    if lower in _NAMED_COLORS:
        return _NAMED_COLORS[lower]
    if lower.startswith("#") and len(lower) == 7:
        try:
            h = lower[1:]
            return (int(h[0:2], 16) / 255.0,
                    int(h[2:4], 16) / 255.0,
                    int(h[4:6], 16) / 255.0)
        except ValueError:
            pass
    return (0.0, 0.0, 0.0)  # fallback: black


# ---------------------------------------------------------------------------
# Per-type annotation appliers
# ---------------------------------------------------------------------------

def _apply_highlight(page: Any, rect: Any, spec: AnnotationSpec) -> None:
    rgb = _to_rgb(spec.color)
    annot = page.add_highlight_annot(rect)
    annot.set_colors(stroke=rgb)
    annot.update()


def _apply_text_color(page: Any, rect: Any, spec: AnnotationSpec) -> None:
    """Coloured highlight (simulates coloured text) + optional strikethrough."""
    rgb = _to_rgb(spec.color)
    annot = page.add_highlight_annot(rect)
    annot.set_colors(stroke=rgb)
    annot.update()
    if spec.extra.get("strikethrough"):
        sa = page.add_strikeout_annot(rect)
        sa.set_colors(stroke=rgb)
        sa.update()


def _apply_circle(page: Any, rect: Any, spec: AnnotationSpec) -> None:
    """Draw a slightly inflated oval around the matched text rect."""
    import fitz
    rgb = _to_rgb(spec.color)
    oval_rect = fitz.Rect(
        rect.x0 - 5, rect.y0 - 4,
        rect.x1 + 5, rect.y1 + 4,
    )
    page.draw_oval(oval_rect, color=rgb, width=1.5)


def _apply_margin_callout(page: Any, rect: Any, spec: AnnotationSpec) -> None:
    """Insert comment text in the left margin, underline the anchor text in
    the body, and draw a thin connector line between them.

    Visual anatomy:
      [margin note text]  ----  [underlined anchor text in body]
    """
    import fitz
    rgb = _to_rgb(spec.color)
    comment = spec.extra.get("comment", spec.text[:60])

    # 1. Underline the anchor text in the document body so the reader can
    #    see which phrase the margin note refers to.
    underline_annot = page.add_underline_annot(rect)
    underline_annot.set_colors(stroke=rgb)
    underline_annot.update()

    # 2. Write the comment in the left margin, vertically centred on the
    #    anchor text's midpoint.
    margin_right = 54.0
    mid_y = (rect.y0 + rect.y1) / 2
    margin_rect = fitz.Rect(2, mid_y - 36, margin_right, mid_y + 36)
    rc = page.insert_textbox(
        margin_rect,
        comment,
        fontsize=6.5,
        color=rgb,
        fontname="helv",
        align=0,
    )
    if rc < 0:
        # Overflow — extend downward
        margin_rect = fitz.Rect(2, mid_y - 10, margin_right, mid_y + 80)
        page.insert_textbox(margin_rect, comment, fontsize=6.5, color=rgb,
                            fontname="helv", align=0)

    # 3. Draw a thin horizontal connector line from the right edge of the
    #    margin area to the left edge of the anchor text rect, at the
    #    vertical midpoint of the anchor text.
    line_y = mid_y
    page.draw_line(
        fitz.Point(margin_right + 2, line_y),
        fitz.Point(rect.x0 - 2, line_y),
        color=rgb,
        width=0.5,
    )


def _apply_sticky_note(page: Any, rect: Any, spec: AnnotationSpec) -> None:
    """Add a PDF sticky-note icon above the matched text."""
    import fitz
    rgb = _to_rgb(spec.color)
    note_text = spec.extra.get("note", spec.text[:120])
    point = fitz.Point(rect.x0, rect.y0 - 6)
    annot = page.add_text_annot(point, note_text, icon="Note")
    annot.set_colors(stroke=rgb)
    annot.update()


def _apply_score_callout(page: Any, spec: AnnotationSpec) -> None:
    """Draw a score badge oval + comment text near the bottom of the page."""
    import fitz
    rgb = _to_rgb(spec.color)
    score   = spec.extra.get("score", "7/10")
    comment = spec.extra.get("comment", "")

    pw = page.rect.width
    ph = page.rect.height

    # Badge oval — positioned center-right, ~80 pts from bottom
    bx = pw * 0.52
    by = ph - 80.0
    badge = fitz.Rect(bx - 18, by - 11, bx + 18, by + 11)
    page.draw_oval(badge, color=rgb, width=1.5)

    # Score text centred inside the badge
    page.insert_text(
        fitz.Point(bx - 13, by + 4),
        score,
        fontsize=9,
        color=rgb,
        fontname="helv",
    )

    # Comment text to the right of the badge
    if comment:
        comment_rect = fitz.Rect(bx + 22, by - 14, pw - 25, by + 24)
        page.insert_textbox(
            comment_rect,
            comment,
            fontsize=9,
            color=rgb,
            fontname="helv",
            align=0,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_inline_annotations(
    pdf_path: Path,
    specs: list[AnnotationSpec],
) -> dict[str, int]:
    """Apply inline annotations to *pdf_path* (in-place) using PyMuPDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.  Modified in-place.
    specs:
        Annotation specs produced by ``pick_annotation_specs()``.

    Returns
    -------
    dict[str, int]
        Maps ``annotation_id`` to the 0-based page number where the
        annotation was placed.  IDs not present in the result were not
        placed (text not found in PDF).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.warning("PyMuPDF (fitz) not installed; inline annotation pass skipped.")
        return {}

    doc = fitz.open(str(pdf_path))
    page_map: dict[str, int] = {}

    for spec in specs:
        try:
            if spec.type == "score_callout":
                # Always placed on the last page — no text search required
                last_page = doc[-1]
                _apply_score_callout(last_page, spec)
                page_map[spec.annotation_id] = last_page.number
                continue

            # Search every page for the target phrase
            placed = False
            for page in doc:
                hits = page.search_for(spec.text)
                if not hits:
                    continue

                rect = hits[0]  # annotate the first occurrence only

                if spec.type == "highlight":
                    _apply_highlight(page, rect, spec)
                elif spec.type == "text_color":
                    _apply_text_color(page, rect, spec)
                elif spec.type == "circle":
                    _apply_circle(page, rect, spec)
                elif spec.type == "margin_callout":
                    _apply_margin_callout(page, rect, spec)
                elif spec.type == "sticky_note":
                    _apply_sticky_note(page, rect, spec)
                else:
                    log.warning("Unknown inline annotation type %r; skipping.", spec.type)
                    break

                page_map[spec.annotation_id] = page.number
                placed = True
                break  # first occurrence only

            if not placed and spec.text:
                log.debug(
                    "Inline annotation %r: phrase %r not found in PDF.",
                    spec.annotation_id, spec.text[:40],
                )

        except Exception as exc:
            log.warning(
                "Failed to apply inline annotation %r (%s): %s",
                spec.annotation_id, spec.type, exc,
            )

    # Save to a sibling temp file, then atomically replace the original.
    # PyMuPDF raises "save to original must be incremental" when the path
    # passed to save() matches the path the file was opened from.
    import os
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf", dir=pdf_path.parent)
    os.close(tmp_fd)
    try:
        doc.save(tmp_path, garbage=3, deflate=True)
        doc.close()
        os.replace(tmp_path, str(pdf_path))
    except Exception:
        doc.close()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return page_map
