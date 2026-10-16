# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Optional PDF annotation pass.

Applies PDF-level annotations (highlights, sticky notes, freetext boxes) to an
already-rendered PDF.

Two back-ends are supported:

  * PyMuPDF (fitz)  — preferred.  Supports text-search-based placement so
                      annotations appear at the exact location of the target
                      phrase.  Handles ``sticky_note``, ``freetext``, and
                      ``highlight`` annotation types.

  * pypdf           — fallback.  Places basic sticky-note annotations at a
                      fixed position on page 0.

Annotation specs are read from ``variant.render_payload["pdf_annotations"]``
(new format from ``variants/annotations.py``) with a legacy fallback that
synthesises a basic sticky-note from ``variant.ground_truth``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from synthpdf.types import Variant


# ---------------------------------------------------------------------------
# Back-end: PyMuPDF (fitz)
# ---------------------------------------------------------------------------

def _apply_fitz(
    pdf_path: Path,
    annotation_variants: list[Variant],
    output_path: Path,
) -> dict[str, Any]:
    """Apply rich PDF annotations using PyMuPDF."""
    import fitz  # type: ignore[import]

    doc = fitz.open(str(pdf_path))
    annotations_applied: list[dict[str, Any]] = []

    for variant in annotation_variants:
        # New format: explicit pdf_annotations list in render_payload
        ann_specs: list[dict[str, Any]] = variant.render_payload.get("pdf_annotations", [])

        # Legacy fallback: synthesise a sticky-note from ground_truth
        if not ann_specs:
            gt = variant.ground_truth
            ann_type = gt.get("annotation_type", "callout")
            target_text = gt.get("target_text", "")
            if target_text:
                ann_specs = [{
                    "type": "sticky_note",
                    "target_text": target_text,
                    "note": f"[{ann_type.upper()}] {target_text[:120]}",
                    "color": [1.0, 1.0, 0.0],
                }]

        for spec in ann_specs:
            spec_type = spec.get("type", "sticky_note")
            target_text = spec.get("target_text", "")

            placed = False
            for page in doc:
                # Search for target phrase on this page
                if target_text:
                    hits = page.search_for(target_text)
                else:
                    hits = []

                if hits:
                    rect = hits[0]
                elif not placed and page.number == 0:
                    # Fallback: top-right corner of first page
                    pw, ph = page.rect.width, page.rect.height
                    rect = fitz.Rect(pw - 170, 55, pw - 20, 75)
                else:
                    continue

                if spec_type == "sticky_note":
                    point = fitz.Point(rect.x0, rect.y0)
                    note_text = spec.get("note", "")
                    annot = page.add_text_annot(point, note_text)
                    color = spec.get("color", [1.0, 1.0, 0.0])
                    annot.set_colors(stroke=tuple(color))
                    annot.update()

                elif spec_type == "freetext":
                    pw = page.rect.width
                    box_w = min(200, max(80, pw - rect.x0 - 20))
                    # Place box just below the found text
                    box_rect = fitz.Rect(
                        min(rect.x0, pw - box_w - 10),
                        rect.y1 + 4,
                        min(rect.x0 + box_w, pw - 10),
                        rect.y1 + 32,
                    )
                    text = spec.get("text", spec.get("note", ""))
                    fill_color = tuple(spec.get("fill_color", [1.0, 1.0, 0.9]))
                    border_color = tuple(spec.get("border_color", [0.8, 0.5, 0.0]))
                    text_color = tuple(spec.get("text_color", [0.5, 0.1, 0.0]))
                    fontsize = int(spec.get("fontsize", 8))
                    try:
                        annot = page.add_freetext_annot(
                            box_rect, text,
                            fontsize=fontsize,
                            text_color=text_color,
                            fill_color=fill_color,
                            border_color=border_color,
                        )
                    except TypeError:
                        # Older PyMuPDF API (< 1.18)
                        annot = page.add_freetext_annot(box_rect, text, fontsize=fontsize)
                        annot.set_colors(stroke=border_color, fill=fill_color)
                    annot.update()

                elif spec_type == "highlight":
                    annot = page.add_highlight_annot(rect)
                    color = spec.get("color", [1.0, 1.0, 0.0])
                    annot.set_colors(stroke=tuple(color))
                    note = spec.get("note", "")
                    if note:
                        annot.set_info(content=note)
                    annot.update()

                else:
                    # Unknown type — skip gracefully
                    continue

                annotations_applied.append({
                    "variant_id": variant.id,
                    "type": spec_type,
                    "target": target_text[:60],
                    "page": page.number,
                })
                placed = True
                if target_text:
                    break  # annotate only the first occurrence

    doc.save(str(output_path), garbage=3, deflate=True)
    doc.close()

    return {
        "backend": "fitz",
        "annotations_applied": annotations_applied,
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Back-end: pypdf (fallback)
# ---------------------------------------------------------------------------

def _apply_pypdf(
    pdf_path: Path,
    annotation_variants: list[Variant],
    output_path: Path,
) -> dict[str, Any]:
    """Apply basic sticky-note annotations using pypdf."""
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DictionaryObject,
        FloatObject,
        NameObject,
        TextStringObject,
    )

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    annotations_applied: list[dict[str, Any]] = []

    if not writer.pages:
        with open(output_path, "wb") as f:
            writer.write(f)
        return {"backend": "pypdf", "annotations_applied": [], "output_path": str(output_path)}

    page = writer.pages[0]
    y_cursor = 700.0  # stack annotations down the page

    for variant in annotation_variants:
        specs = variant.render_payload.get("pdf_annotations", [])
        if specs:
            entries = [
                {
                    "type": s.get("type", "sticky_note"),
                    "text": s.get("note", s.get("text", "")),
                    "target": s.get("target_text", ""),
                }
                for s in specs
            ]
        else:
            gt = variant.ground_truth
            ann_type = gt.get("annotation_type", "callout")
            target = gt.get("target_text", "")
            entries = [{
                "type": ann_type,
                "text": f"[{ann_type.upper()}] {target[:120]}",
                "target": target,
            }]

        for entry in entries:
            annotation = DictionaryObject({
                NameObject("/Type"):     NameObject("/Annot"),
                NameObject("/Subtype"):  NameObject("/Text"),
                NameObject("/Contents"): TextStringObject(entry["text"][:200]),
                NameObject("/Rect"): ArrayObject([
                    FloatObject(50),
                    FloatObject(y_cursor),
                    FloatObject(200),
                    FloatObject(y_cursor + 18),
                ]),
                NameObject("/Open"): NameObject("/false"),
                NameObject("/Name"): NameObject("/Note"),
            })
            if "/Annots" not in page:
                page[NameObject("/Annots")] = ArrayObject()
            page["/Annots"].append(annotation)  # type: ignore[attr-defined]
            annotations_applied.append({
                "variant_id": variant.id,
                "type": entry["type"],
                "target": entry["target"][:60],
            })
            y_cursor -= 22.0  # move down for next annotation

    with open(output_path, "wb") as f:
        writer.write(f)

    return {
        "backend": "pypdf",
        "annotations_applied": annotations_applied,
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def apply_pdf_annotations(
    pdf_path: Path,
    annotation_variants: list[Variant],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Apply PDF-level annotations from *annotation_variants* to *pdf_path*.

    Tries PyMuPDF (fitz) first for rich, text-search-positioned annotations.
    Falls back to pypdf for basic sticky notes if fitz is not available.

    Annotation specs are read from ``variant.render_payload["pdf_annotations"]``
    (produced by ``variants/annotations.py``).  Variants that lack this field
    fall back to a basic sticky-note synthesised from ``variant.ground_truth``.

    If *output_path* is None the PDF is modified in-place.

    Returns a metadata dict with keys ``backend``, ``annotations_applied``,
    and ``output_path``.
    """
    if output_path is None:
        output_path = pdf_path

    # Try PyMuPDF first
    try:
        return _apply_fitz(pdf_path, annotation_variants, output_path)
    except ImportError:
        pass
    except Exception as exc:
        return {"error": f"fitz annotation failed: {exc}", "output_path": str(output_path)}

    # Fallback: pypdf
    try:
        return _apply_pypdf(pdf_path, annotation_variants, output_path)
    except ImportError:
        return {
            "error": "Neither PyMuPDF nor pypdf is installed; skipping PDF annotation pass.",
            "output_path": str(output_path),
        }
    except Exception as exc:
        return {"error": f"pypdf annotation failed: {exc}", "output_path": str(output_path)}
