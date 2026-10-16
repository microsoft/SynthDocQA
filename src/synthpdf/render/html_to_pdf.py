# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""HTML to PDF renderer.

Primary backend: Playwright (headless Chromium - best CSS fidelity).
Fallback 1: WeasyPrint.
Fallback 2: xhtml2pdf (two-pass CSS sanitizer + nuclear strip).

When xhtml2pdf passes 1 and 2 both fail, the exception propagates to the
caller (doc_builder.py), which logs a warning and skips that page.

Public API
----------
render_html_to_pdf(html, output_path, backend="playwright") -> dict
merge_page_pdfs(page_pdfs, output_path) -> dict
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_html_to_pdf(
    html: str | Path,
    output_path: Path,
    backend: Literal["playwright", "weasyprint", "xhtml2pdf"] = "playwright",
    viewport: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Render a single-page HTML document to PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backends_to_try: list[str]
    if backend == "playwright":
        backends_to_try = ["playwright", "weasyprint", "xhtml2pdf"]
    elif backend == "weasyprint":
        backends_to_try = ["weasyprint", "xhtml2pdf"]
    else:
        backends_to_try = ["xhtml2pdf"]

    last_exc: Exception | None = None
    for b in backends_to_try:
        try:
            if b == "playwright":
                return _playwright_render(html, output_path, viewport=viewport)
            elif b == "weasyprint":
                return _weasyprint_render(html, output_path)
            else:
                return _xhtml2pdf_render(html, output_path)
        except Exception as exc:
            log.warning("Backend %r failed: %s — trying next", b, exc)
            last_exc = exc

    raise RuntimeError(
        f"All HTML->PDF backends failed for {output_path}. Last error: {last_exc}"
    ) from last_exc


def merge_page_pdfs(page_pdfs: list[Path], output_path: Path) -> dict[str, Any]:
    """Merge multiple single-page PDFs into one multi-page document."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for p in page_pdfs:
        writer.append(str(p))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        writer.write(fh)

    return {"page_count": len(page_pdfs)}


# ---------------------------------------------------------------------------
# Playwright renderer
# ---------------------------------------------------------------------------

def _playwright_render(
    html: str | Path,
    output_path: Path,
    viewport: dict[str, int] | None = None,
) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    _vp = viewport or {"width": 794, "height": 1123}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport=_vp)
            if isinstance(html, Path):
                page.goto(html.resolve().as_uri(), wait_until="networkidle", timeout=30_000)
            else:
                page.set_content(html, wait_until="networkidle")
            page.pdf(path=str(output_path), print_background=True, prefer_css_page_size=True)
        finally:
            browser.close()

    return {"page_count": 1, "backend": "playwright"}


# ---------------------------------------------------------------------------
# WeasyPrint renderer (fallback 1)
# ---------------------------------------------------------------------------

def _weasyprint_render(html: str | Path, output_path: Path) -> dict[str, Any]:
    import weasyprint  # type: ignore[import]

    if isinstance(html, Path):
        doc = weasyprint.HTML(filename=str(html.resolve()))
    else:
        doc = weasyprint.HTML(string=html)

    doc.write_pdf(str(output_path))
    return {"page_count": 1, "backend": "weasyprint"}


# ---------------------------------------------------------------------------
# xhtml2pdf renderer (fallback 2) — 2-pass strategy
# ---------------------------------------------------------------------------
#
# Pass 1: CSS sanitizer  — preserves styling; handles most templates
# Pass 2: Nuclear strip  — removes all CSS + unwraps block wrappers
#
# If both passes fail, the exception propagates. doc_builder.py catches it
# and skips the page with a warning — no bad output is written.

_REMOVE_CSS_PROPS = [
    "box-shadow", "transform", "filter", "backdrop-filter", "transition", "animation",
    "grid-template-columns", "grid-template-rows", "grid-template", "grid-auto-flow",
    "grid-column", "grid-row", "grid-area", "gap", "column-gap", "row-gap", "grid-gap",
    "flex-direction", "flex-wrap", "align-items", "align-self", "align-content",
    "justify-content", "justify-items", "justify-self", "flex", "flex-grow", "flex-shrink",
    "flex-basis", "order", "place-items", "place-content",
    "writing-mode", "text-orientation",
    "user-select", "-webkit-user-select",
    "-webkit-print-color-adjust", "print-color-adjust", "color-adjust",
    "pointer-events",
    "overflow", "overflow-x", "overflow-y",
    "z-index",
    "height", "min-height", "max-height",
    "top", "right", "bottom", "left",
]

_UNWRAP_TAGS = frozenset([
    "div", "section", "article", "header", "footer", "main", "aside", "nav",
    "figure", "figcaption", "details", "summary",
])


def _sanitize_for_xhtml2pdf(html_str: str) -> str:
    """Sanitize HTML/CSS to the subset xhtml2pdf can handle without crashing."""

    # 1. rgba/hsla -> rgb/hsl
    html_str = re.sub(
        r'rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\d.]+\s*\)',
        r'rgb(\1, \2, \3)', html_str,
    )
    html_str = re.sub(
        r'hsla\(\s*(\d+)\s*,\s*([\d.]+%?)\s*,\s*([\d.]+%?)\s*,\s*[\d.]+\s*\)',
        r'hsl(\1, \2, \3)', html_str,
    )

    # 2. 8-digit hex -> 6-digit
    html_str = re.sub(r'(#[0-9a-fA-F]{6})[0-9a-fA-F]{2}\b', r'\1', html_str)

    # 3. CSS custom properties
    html_str = re.sub(
        r'var\(--[\w-]+\s*,\s*([^)]+)\)', lambda m: m.group(1).strip(), html_str,
    )
    html_str = re.sub(r'var\(--[\w-]+\)', 'inherit', html_str)
    html_str = re.sub(r'--[\w-]+\s*:[^;{}]+;', '', html_str)

    # 4. Gradients -> first solid color
    def _first_gradient_color(m: re.Match) -> str:
        inner = m.group(0)
        color = re.search(r'(#[0-9a-fA-F]{3,6}|rgb\(\d+,\s*\d+,\s*\d+\))', inner)
        return color.group(1) if color else '#cccccc'
    html_str = re.sub(
        r'(?:linear|radial)-gradient\((?:[^()]*|\([^)]*\))*\)',
        _first_gradient_color, html_str,
    )

    # 5. display: grid/flex -> block
    html_str = re.sub(
        r'\bdisplay\s*:\s*(?:inline-)?(?:grid|flex)\b', 'display: block', html_str,
    )

    # 6. position: non-static -> removed
    html_str = re.sub(
        r'\bposition\s*:\s*(?:absolute|relative|fixed|sticky)\s*;?', '', html_str,
    )

    # 7. @page: strip 'size:' but keep 'margin: 0'
    html_str = re.sub(
        r'(@page\b[^{]*\{)([^}]*)(\})',
        lambda m: m.group(1) + re.sub(r'\bsize\s*:[^;]*;?\s*', '', m.group(2)) + m.group(3),
        html_str, flags=re.DOTALL,
    )
    html_str = re.sub(r'@page\b[^{]*\{\s*\}', '', html_str)

    # 8. Remove standalone ::before / ::after rules
    html_str = re.sub(
        r'[^{}]*[^,\s{]::(?:before|after)\s*\{[^}]*\}', '', html_str, flags=re.DOTALL,
    )

    # 9. Remove unsupported CSS properties
    for prop in _REMOVE_CSS_PROPS:
        html_str = re.sub(rf'(?<![a-zA-Z0-9-]){re.escape(prop)}\s*:[^;{{}}]*;', '', html_str)

    # 10. REPLACE wide pixel widths with 495px (A4 content width ~595pt - margins)
    #
    # CRITICAL: replace, do NOT delete.
    # Templates set `.page { width: 794px }`. Deleting leaves .page with no width,
    # so inner `width: 100%` resolves as 100% of unknown = 'x' -> PmlKeepInFrame crash.
    # Replacing with 495px preserves the chain: 100% of 495px = 495px (valid float).
    html_str = re.sub(
        r'(?<![a-zA-Z0-9-])width\s*:\s*(?:[5-9]\d{2}|[1-9]\d{3,})px\s*;?',
        'width: 495px;', html_str,
    )
    html_str = re.sub(
        r'(?<![a-zA-Z0-9-])width\s*:\s*100vw\s*;?', 'width: 495px;', html_str,
    )

    # 11. Remove calc() declarations
    html_str = re.sub(
        r'(?<![a-zA-Z0-9-])[\w-]+\s*:\s*[^;{}]*\bcalc\s*\([^)]*\)[^;{}]*;', '', html_str,
    )

    # 12. Remove em-unit letter-spacing (xhtml2pdf logs "Not a float '0.05em'")
    html_str = re.sub(r'(?<![a-zA-Z0-9-])letter-spacing\s*:[^;{}]*em\s*;', '', html_str)

    # 13. Cap large pixel font-sizes to prevent oversized text.
    #
    # Templates designed for full-browser viewports use large heading fonts
    # (36px, 60px, 72px, ...). In xhtml2pdf the layout containers that clip
    # this text (overflow:hidden, fixed height) have been stripped, so the
    # text overflows and fills the entire page. Cap all px font-sizes to 20px
    # while keeping smaller sizes intact — this preserves visual hierarchy
    # (heading > subheading > body) without any text overflowing its column.
    def _cap_font_size(m: re.Match) -> str:
        val = float(m.group(1))
        capped = min(val, 20.0)
        # Keep .0 values as integers for cleaner CSS
        return f'font-size: {int(capped) if capped == int(capped) else capped}px'
    html_str = re.sub(
        r'(?<![a-zA-Z0-9-])font-size\s*:\s*(\d+(?:\.\d+)?)px',
        _cap_font_size,
        html_str,
    )
    # vw/vh/vmin font sizes (viewport-relative) — convert to a fixed 14px
    # since xhtml2pdf has no concept of viewport dimensions.
    html_str = re.sub(
        r'(?<![a-zA-Z0-9-])font-size\s*:\s*[\d.]+v[wh][^;{]*;?',
        'font-size: 14px;',
        html_str,
    )

    return html_str


def _strip_styles_for_xhtml2pdf(html_str: str) -> str:
    """Nuclear fallback: strip ALL CSS, SVG, and block-wrapper tags.

    Unwrapping div/section/header/footer/main etc. places tables directly
    in <body> so xhtml2pdf always knows their containing-block width
    (= page content width, ~495pt for A4).
    """
    # 1. Remove all <style> blocks
    html_str = re.sub(
        r'<style\b[^>]*>.*?</style>', '', html_str, flags=re.DOTALL | re.IGNORECASE,
    )

    # 2. Remove all inline style="" attributes
    html_str = re.sub(
        r'\bstyle\s*=\s*(?:"[^"]*"|\'[^\']*\')', '', html_str, flags=re.IGNORECASE,
    )

    # 3. Unwrap structural block elements (remove tags, keep content)
    for tag in _UNWRAP_TAGS:
        html_str = re.sub(rf'<{tag}\b[^>]*>', '', html_str, flags=re.IGNORECASE)
        html_str = re.sub(rf'</{tag}\s*>', '', html_str, flags=re.IGNORECASE)

    # 4. Remove <svg> elements (no SVG renderer in xhtml2pdf)
    html_str = re.sub(
        r'<svg\b[^>]*>.*?</svg>', '', html_str, flags=re.DOTALL | re.IGNORECASE,
    )

    # 5. Remove <img src="file://..."> (Windows file:// URIs fail in xhtml2pdf)
    html_str = re.sub(
        r'<img\b[^>]*\bsrc\s*=\s*(?:"file://[^"]*"|\'file://[^\']*\')[^>]*/?>',
        '', html_str, flags=re.IGNORECASE,
    )

    # 6. Normalize residual color functions
    html_str = re.sub(
        r'rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\d.]+\s*\)',
        r'rgb(\1, \2, \3)', html_str,
    )
    html_str = re.sub(
        r'hsla\(\s*(\d+)\s*,\s*([\d.]+%?)\s*,\s*([\d.]+%?)\s*,\s*[\d.]+\s*\)',
        r'hsl(\1, \2, \3)', html_str,
    )
    html_str = re.sub(r'(#[0-9a-fA-F]{6})[0-9a-fA-F]{2}\b', r'\1', html_str)

    # 7. Add width="100%" to tables (belt-and-suspenders after unwrap)
    html_str = re.sub(
        r'<table\b(?![^>]*\bwidth\s*=)', '<table width="100%"',
        html_str, flags=re.IGNORECASE,
    )

    return html_str


_XHTML2PDF_LAYOUT_ERRORS = ("PmlKeepInFrame", "row heights", "Declaration group")


def _xhtml2pdf_render(html: str | Path, output_path: Path) -> dict[str, Any]:
    """Two-pass xhtml2pdf renderer. Raises on failure — caller skips the page."""
    from xhtml2pdf import pisa  # type: ignore[import]

    if isinstance(html, Path):
        html_str = html.read_text(encoding="utf-8")
        base_url = str(html.resolve().parent) + "/"
    else:
        html_str = html
        base_url = None

    def _xhtml2pdf_pass(candidate: str) -> dict[str, Any]:
        with open(output_path, "wb") as fh:
            result = pisa.CreatePDF(candidate, dest=fh, default_css="", path=base_url)
        if result.err:
            raise RuntimeError(f"xhtml2pdf error code {result.err}")
        return {"page_count": 1, "backend": "xhtml2pdf"}

    def _is_layout_error(exc: Exception) -> bool:
        return any(m in str(exc) for m in _XHTML2PDF_LAYOUT_ERRORS)

    # Pass 1: full CSS sanitizer
    try:
        return _xhtml2pdf_pass(_sanitize_for_xhtml2pdf(html_str))
    except Exception as exc:
        if not _is_layout_error(exc):
            raise
        log.warning("xhtml2pdf pass-1 layout crash (%s), retrying with nuclear strip", str(exc)[:80])

    # Pass 2: nuclear strip (all CSS removed + structural wrappers unwrapped)
    return _xhtml2pdf_pass(_strip_styles_for_xhtml2pdf(html_str))
