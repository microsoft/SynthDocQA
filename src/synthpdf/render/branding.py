# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Logo branding support for synthpdf PDF generation.

Provides BrandingSpec (what logo to draw + where) and factory functions for
per-backend rendering callbacks / HTML snippets.

Usage (ReportLab):
    spec = BrandingSpec(logo_path=Path("amazon.png"), placement="header", position="right")
    on_first, on_later = make_rl_callbacks(spec)
    doc.build(flowables, onFirstPage=on_first, onLaterPages=on_later)

Usage (HTML renderers):
    snippet = make_html_logo_snippet(spec)
    full_html = f"... <body>{snippet}{html_body}</body> ..."
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BrandingSpec:
    """Specifies a logo and its placement on PDF pages."""

    logo_path: Path
    placement: str = "header"     # "header" | "footer" | "both"
    position: str = "right"       # "left" | "center" | "right"
    pages: str = "all"            # "all" | "first" | "rest"
    logo_height_pt: float = 30.0  # logo height in ReportLab points (~1 cm)

    def __post_init__(self) -> None:
        self.logo_path = Path(self.logo_path)
        if self.placement not in ("header", "footer", "both"):
            raise ValueError(f"BrandingSpec.placement must be header|footer|both, got {self.placement!r}")
        if self.position not in ("left", "center", "right"):
            raise ValueError(f"BrandingSpec.position must be left|center|right, got {self.position!r}")
        if self.pages not in ("all", "first", "rest"):
            raise ValueError(f"BrandingSpec.pages must be all|first|rest, got {self.pages!r}")


def select_logo(logos_dir: Path | str, rng: Any) -> Path | None:
    """Pick a random PNG logo from *logos_dir*. Returns None if folder is empty."""
    logos_dir = Path(logos_dir)
    pngs = sorted(logos_dir.glob("*.png"))
    if not pngs:
        return None
    idx = int(rng.integers(0, len(pngs)))
    return pngs[idx]


def make_rl_callbacks(spec: BrandingSpec):
    """Return ``(on_first_page, on_later_pages)`` ReportLab canvas callbacks.

    Both callbacks draw the logo according to *spec*. Pass them directly to
    ``SimpleDocTemplate.build(flowables, onFirstPage=..., onLaterPages=...)``.
    """
    from reportlab.lib.units import cm
    from reportlab.lib.pagesizes import A4

    # Must match the left/right/top/bottom margins in ReportLabRenderer
    MARGIN = 2 * cm
    HEADER_GAP = 4   # points above content-area top edge
    FOOTER_GAP = 4   # points from page bottom edge

    logo_str = str(spec.logo_path)
    h = spec.logo_height_pt

    # Pre-compute logo width from aspect ratio
    try:
        from PIL import Image as PILImage
        with PILImage.open(logo_str) as im:
            iw, ih = im.size
        aspect = iw / ih if ih else 3.0
    except Exception:
        aspect = 3.0   # fallback: assume a wide horizontal logo
    w = h * aspect

    page_w, page_h = A4

    def _x() -> float:
        if spec.position == "left":
            return MARGIN
        if spec.position == "right":
            return page_w - MARGIN - w
        return (page_w - w) / 2   # center

    def _draw(canvas: Any, is_first: bool) -> None:
        # Apply pages filter
        if spec.pages == "first" and not is_first:
            return
        if spec.pages == "rest" and is_first:
            return

        x = _x()
        canvas.saveState()
        try:
            if spec.placement in ("header", "both"):
                # Bottom of logo sits just above the content-area top
                y_hdr = page_h - MARGIN + HEADER_GAP
                canvas.drawImage(
                    logo_str, x, y_hdr,
                    width=w, height=h,
                    preserveAspectRatio=True, mask="auto",
                )
            if spec.placement in ("footer", "both"):
                # Logo starts FOOTER_GAP from the very bottom of the page
                canvas.drawImage(
                    logo_str, x, FOOTER_GAP,
                    width=w, height=h,
                    preserveAspectRatio=True, mask="auto",
                )
        except Exception:
            pass   # never break PDF rendering because of branding
        canvas.restoreState()

    def on_first_page(canvas: Any, doc: Any) -> None:
        _draw(canvas, is_first=True)

    def on_later_pages(canvas: Any, doc: Any) -> None:
        _draw(canvas, is_first=False)

    return on_first_page, on_later_pages


def make_html_logo_snippet(spec: BrandingSpec, logo_height_px: int | None = None) -> str:
    """Return an HTML snippet with a fixed-position logo for xhtml2pdf / WeasyPrint.

    The snippet uses ``position: fixed`` so the logo appears on every rendered
    page in the same location.  The logo is embedded as a base64 data URI to
    avoid file:// URI path resolution failures in xhtml2pdf on Windows
    (drive-letter paths like file:///Q:/... fail to load in xhtml2pdf's handler).

    The logo height is taken from ``spec.logo_height_pt`` (converted to px) by
    default.  Pass ``logo_height_px`` explicitly to override.

    Inline CSS ``height`` / ``width`` / ``max-width`` on the ``<img>`` element are
    used (not HTML attributes) so they take precedence over any global stylesheet
    rules such as ``img { height: auto; max-width: 100%; }`` that would otherwise
    scale the logo to fill its container.
    """
    import base64

    # Convert pt → px (CSS reference pixel: 96 dpi → 1pt = 96/72 px ≈ 1.333 px)
    if logo_height_px is None:
        logo_height_px = max(20, round(spec.logo_height_pt * 96 / 72))

    pos_css_map = {
        "left":   "left: 1.5cm;",
        "right":  "right: 1.5cm;",
        "center": "left: 50%; transform: translateX(-50%);",
    }
    pos_css = pos_css_map.get(spec.position, "right: 1.5cm;")

    # Embed as base64 data URI — works in all backends without file path issues
    try:
        b64 = base64.b64encode(spec.logo_path.read_bytes()).decode("ascii")
        logo_uri = f"data:image/png;base64,{b64}"
    except Exception:
        logo_uri = spec.logo_path.resolve().as_uri()  # fallback to file URI

    # Use inline style (not HTML attributes) so the height/width rules override
    # any global  img { height: auto; max-width: 100%; }  stylesheet rules that
    # would otherwise scale the logo to the full container width.
    img_style = (
        f"height:{logo_height_px}px !important; "
        f"width:auto !important; "
        f"max-width:none !important; "
        f"opacity:0.85;"
    )

    snippets: list[str] = []

    if spec.placement in ("header", "both"):
        snippets.append(
            f'<div style="position:fixed; top:0.2cm; {pos_css} z-index:999;">'
            f'<img src="{logo_uri}" style="{img_style}" />'
            f"</div>"
        )

    if spec.placement in ("footer", "both"):
        snippets.append(
            f'<div style="position:fixed; bottom:0.2cm; {pos_css} z-index:999;">'
            f'<img src="{logo_uri}" style="{img_style}" />'
            f"</div>"
        )

    return "\n".join(snippets)
