# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Multi-page template document builder.

``build_mixed_document()`` has the same return signature as
``compose.markdown_builder.build_document()``:

    (pdf_path, variants_used, ground_truth_dict)

so the CLI can swap between the two builders transparently.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import jinja2
import numpy as np

from synthpdf.rng import fork_rng
from synthpdf.template_engine.adapters import fill_template_variables
from synthpdf.template_engine.index import TemplateIndex
from synthpdf.render.html_to_pdf import merge_page_pdfs, render_html_to_pdf
from synthpdf.types import DocumentSpec, Variant


def _dedup_qa(qa_iter: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return QA candidates with duplicate questions removed (first occurrence kept)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for qa in qa_iter:
        key = qa.get("question", "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(qa)
    return out

log = logging.getLogger(__name__)


def _logo_data_uri(logo_path: Path) -> str | None:
    """Return a base64 data URI for *logo_path*, or None if the file cannot be read."""
    import base64
    try:
        data = logo_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Synthetic brand identities
# (company_name, primary_color, accent_color, tagline)
# ---------------------------------------------------------------------------

_BRANDS = [('Sample Organization 01', '#1a237e', '#3949ab', 'Synthetic document'), ('Sample Organization 02', '#b71c1c', '#e53935', 'Synthetic document'), ('Sample Organization 03', '#00695c', '#00897b', 'Synthetic document'), ('Sample Organization 04', '#4a148c', '#7b1fa2', 'Synthetic document'), ('Sample Organization 05', '#0d47a1', '#1565c0', 'Synthetic document'), ('Sample Organization 06', '#212121', '#424242', 'Synthetic document'), ('Sample Organization 07', '#e65100', '#f57c00', 'Synthetic document'), ('Sample Organization 08', '#1b5e20', '#2e7d32', 'Synthetic document'), ('Sample Organization 09', '#880e4f', '#ad1457', 'Synthetic document'), ('Sample Organization 10', '#006064', '#00838f', 'Synthetic document'), ('Sample Organization 11', '#3e2723', '#5d4037', 'Synthetic document'), ('Sample Organization 12', '#1565c0', '#1976d2', 'Synthetic document'), ('Sample Organization 13', '#1a237e', '#283593', 'Synthetic document'), ('Sample Organization 14', '#004d40', '#00695c', 'Synthetic document'), ('Sample Organization 15', '#37474f', '#546e7a', 'Synthetic document'), ('Sample Organization 16', '#f57f17', '#f9a825', 'Synthetic document'), ('Sample Organization 17', '#263238', '#37474f', 'Synthetic document'), ('Sample Organization 18', '#bf360c', '#d84315', 'Synthetic document'), ('Sample Organization 19', '#1a237e', '#0d47a1', 'Synthetic document'), ('Sample Organization 20', '#4e342e', '#6d4c41', 'Synthetic document')]


def _pick_brand(rng: np.random.Generator) -> dict[str, str]:
    """Select a random brand identity for this document."""
    idx = int(rng.integers(0, len(_BRANDS)))
    name, primary, accent, tagline = _BRANDS[idx]
    words = name.split()
    initials = "".join(w[0] for w in words if w)[:2].upper()
    return {
        "name": name,
        "primary": primary,
        "accent": accent,
        "tagline": tagline,
        "initials": initials,
    }


def _make_brand_bar_html(
    brand: dict[str, str],
    page_num: int,
    total_pages: int,
    logo_path: "Path | None" = None,
) -> str:
    """Return a self-contained HTML header bar with the brand logo and name.

    When *logo_path* is provided the actual logo image is shown; otherwise the
    two-letter monogram square is used as a fallback.  Uses a pure-table layout
    so it renders correctly in xhtml2pdf, WeasyPrint, and Playwright.
    """
    primary = brand["primary"]
    initials = brand["initials"]
    page_label = f"Page {page_num} of {total_pages}"

    # Build the left identity cell: real logo image or monogram fallback
    _logo_uri = _logo_data_uri(logo_path) if logo_path is not None and logo_path.exists() else None
    if _logo_uri is not None:
        identity_cell = (
            f'<td style="padding:4px 8px 4px 10px; vertical-align:middle;">'
            f'  <img src="{_logo_uri}" height="28"'
            f'       style="vertical-align:middle; display:block;" />'
            f'</td>'
        )
    else:
        identity_cell = (
            f'<td width="42" style="padding:5px 8px 5px 10px; vertical-align:middle;">'
            f'  <table cellspacing="0" cellpadding="0" style="border-collapse:collapse;">'
            f'  <tr><td width="32" height="32" align="center" valign="middle"'
            f'      style="background:{primary}; color:#ffffff; font-family:Arial,sans-serif;'
            f'             font-size:12px; font-weight:700; padding:0;">'
            f'    {initials}'
            f'  </td></tr></table>'
            f'</td>'
        )

    return (
        f'<table width="100%" cellspacing="0" cellpadding="0" '
        f'style="background:#f4f4f4; border-bottom:2px solid {primary}; '
        f'margin-bottom:12px; border-collapse:collapse;">'
        f'<tr>'
        + identity_cell
        # Company name cell
        + f'<td style="padding:5px 8px; vertical-align:middle; '
        f'           font-family:Arial,sans-serif; font-size:10px; font-weight:700; '
        f'           color:{primary}; letter-spacing:0.8px;">'
        f'  {brand["name"]}'
        f'</td>'
        # Page number cell (right-aligned)
        + f'<td style="padding:5px 12px 5px 8px; text-align:right; vertical-align:middle; '
        f'           font-family:Arial,sans-serif; font-size:8px; color:#888888; '
        f'           white-space:nowrap;">'
        f'  {page_label}'
        f'</td>'
        f'</tr></table>'
    )


def _inject_brand_bar(html_str: str, bar_html: str) -> str:
    """Inject *bar_html* immediately after the opening <body ...> tag."""
    import re
    # Match <body> with optional attributes
    match = re.search(r'<body[^>]*>', html_str, re.IGNORECASE)
    if match:
        insert_pos = match.end()
        return html_str[:insert_pos] + "\n" + bar_html + "\n" + html_str[insert_pos:]
    # No <body> found — prepend to the whole string
    return bar_html + "\n" + html_str


def _split_pdf_to_pages(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Split a multi-page PDF into individual single-page PDFs."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    pages: list[Path] = []
    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)
        p = output_dir / f"content_{i:03d}.pdf"
        with open(p, "wb") as fh:
            writer.write(fh)
        pages.append(p)
    return pages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_mixed_document(
    doc_spec: DocumentSpec,
    variant_pool: dict[str, list[Variant]],
    template_index: "TemplateIndex | None",
    output_dir: Path,
    doc_rng: np.random.Generator,
    recipe: Any,
    html_backend: str = "xhtml2pdf",
    md_backend: str = "reportlab",
    template_fraction: float = 0.10,
    branding_spec: Any = None,
    css_file: "Path | None" = None,
    title_pages_dir: "Path | None" = None,
    page_layout: str = "portrait",
) -> tuple[Path, list[Variant], dict[str, Any]]:
    """Build a PDF that mixes markdown content with Jinja2 template pages.

    Structure of every document:
      - Page 1: styled title page (HTML-rendered)
      - Remaining pages: markdown content with ~template_fraction of pages
        replaced by randomly-selected Jinja2 template pages.

    Parameters
    ----------
    html_backend:
        Backend for HTML -> PDF (title page + template pages).
    md_backend:
        Backend for markdown -> PDF (main content).
    template_fraction:
        Fraction of content pages to replace with Jinja2 templates (default 0.10).
    """
    from synthpdf.compose.markdown_builder import build_document
    from synthpdf.render.md_to_pdf import render_markdown_to_pdf

    pages_dir = output_dir / "_tmp_pages" / doc_spec.doc_id
    pages_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pick one brand identity for this document
    brand_rng = fork_rng(doc_rng, "branding")
    brand = _pick_brand(brand_rng)
    log.info("Brand: %s (%s)", brand["name"], brand["primary"])

    logo_path: "Path | None" = getattr(branding_spec, "logo_path", None)

    # 2. Build markdown content
    md_rng = fork_rng(doc_rng, "markdown_content")
    md_path, variants_used, gt = build_document(
        doc_spec, variant_pool, output_dir, recipe, md_rng
    )

    # 3. Render markdown -> multi-page content PDF
    content_pdf = pages_dir / "content.pdf"
    render_markdown_to_pdf(
        md_path, content_pdf,
        backend=md_backend,
        css_file=css_file,
        branding_spec=branding_spec,
    )

    # 4. Split content PDF into individual pages
    content_pages = _split_pdf_to_pages(content_pdf, pages_dir)
    if not content_pages:
        content_pages = []

    # 5. Replace random ~template_fraction of content pages with Jinja2 template pages
    n_replace = max(1, round(len(content_pages) * template_fraction))
    template_positions: list[int] = []
    if n_replace > 0 and template_index is not None and len(template_index) > 0:
        all_positions = list(range(len(content_pages)))
        chosen = doc_rng.choice(len(all_positions), size=min(n_replace, len(all_positions)), replace=False)
        template_positions = sorted(int(i) for i in chosen)

        industry = _infer_industry(doc_spec, template_index)
        all_pool_variants = (
            list(variant_pool.get("tables", []))
            + list(variant_pool.get("images", []))
            + list(variant_pool.get("texts", []))
        )

        # Pre-select a diverse sequence of templates for ALL positions at once.
        # Calling select_sequence(1, ...) per-position always requested "cover_page"
        # which meant only 1 template was ever used.
        seq_rng = fork_rng(doc_rng, "template_sequence")
        entries_seq = template_index.select_sequence(
            len(template_positions), industry, seq_rng
        )

        for i, pos in enumerate(template_positions):
            if not entries_seq:
                break
            entry = entries_seq[i % len(entries_seq)]
            tpl_rng = fork_rng(doc_rng, f"tpl_{pos}")
            page_rng = fork_rng(doc_rng, f"tpl_page_{pos}")
            page_variants = _sample_variants(all_pool_variants, 3, page_rng)

            page_num = pos + 1
            total_pages = len(content_pages)

            try:
                var_ctx = fill_template_variables(
                    template_html=entry.html_path,
                    spec=entry.spec,
                    variants=page_variants,
                    rng=page_rng,
                    doc_title=doc_spec.title or doc_spec.doc_id,
                    page_number=page_num,
                )
                html_str = _render_jinja(entry.html_path, var_ctx)
                # Do NOT inject the brand bar into Jinja2 template pages.
                # Templates already have their own full-page layout (794×1123 px,
                # overflow:hidden) and include brand elements via context variables
                # (institution_name, logo_text, brand_name etc.).  Injecting an
                # extra bar pushes content down, causing overflow and visual clutter.

                tpl_html_path = pages_dir / f"tpl_{pos:03d}.html"
                tpl_html_path.write_text(html_str, encoding="utf-8")
                tpl_pdf_path = pages_dir / f"tpl_{pos:03d}.pdf"
                render_html_to_pdf(tpl_html_path, tpl_pdf_path, backend=html_backend)
                content_pages[pos] = tpl_pdf_path
                variants_used.extend(page_variants)
            except Exception as exc:
                log.warning("Template replacement at page %d failed (%s): %s",
                            page_num, entry.template_id, exc)

    # 6. Prepend title page, then merge
    # Title page templates are all landscape-designed (1440x900); always render in landscape.
    title_rng = fork_rng(doc_rng, "title_page")
    from synthpdf.template_engine.title_page_renderer import render_title_page
    title_pdf = render_title_page(
        doc_spec=doc_spec,
        brand=brand,
        pages_dir=pages_dir,
        doc_rng=title_rng,
        page_layout="landscape",
        title_pages_dir=title_pages_dir,
    )
    if title_pdf is not None:
        content_pages = [title_pdf] + content_pages

    pdf_path = output_dir / "pdfs" / f"{doc_spec.doc_id}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    merge_result = merge_page_pdfs(content_pages, pdf_path)

    gt["brand"] = brand["name"]
    gt["brand_primary_color"] = brand["primary"]
    gt["render_mode"] = "mixed"
    gt["has_title_page"] = title_pdf is not None
    gt["page_count"] = merge_result["page_count"]
    gt["template_pages"] = [p + 1 for p in template_positions]

    # Deduplicate variants
    seen: set[str] = set()
    unique_variants = []
    for v in variants_used:
        if v.id not in seen:
            seen.add(v.id)
            unique_variants.append(v)

    return pdf_path, unique_variants, gt




# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ItemFirstEnvironment(jinja2.Environment):
    """Jinja2 Environment that prefers dict key access over attribute access.

    The default Jinja2 Environment calls getattr(obj, attr) first, which means
    dict built-in methods like .items(), .keys(), .values() shadow any stored
    keys with the same name.  For example, if a _SmartItem dict has an "items"
    key, template code ``{% for nav in category.items %}`` returns the bound
    method dict.items, not the stored list — crashing with
    "builtin_function_or_method object is not iterable".

    Reversing the lookup order (key first, then attr) fixes all such cases.
    """

    def getattr(self, obj: Any, attribute: str) -> Any:
        # Try key access first so stored values shadow built-in methods.
        try:
            return obj[attribute]
        except (KeyError, TypeError, AttributeError):
            pass
        try:
            return getattr(obj, attribute)
        except AttributeError:
            pass
        return self.undefined(obj=obj, name=attribute)


def _render_jinja(html_path: Path, context: dict[str, Any]) -> str:
    """Render a Jinja2 HTML template, silently ignoring any missing variables."""
    source = html_path.read_text(encoding="utf-8")
    env = _ItemFirstEnvironment(
        loader=jinja2.BaseLoader(),
        autoescape=False,
        undefined=jinja2.ChainableUndefined,   # attr access on undefined → ''
    )
    tmpl = env.from_string(source)
    return tmpl.render(**context)


def _sample_variants(
    pool: list[Variant],
    n: int,
    rng: np.random.Generator,
) -> list[Variant]:
    if not pool:
        return []
    n = min(n, len(pool))
    indices = list(rng.choice(len(pool), size=n, replace=False))
    return [pool[int(i)] for i in indices]


def _infer_industry(
    doc_spec: DocumentSpec,
    template_index: TemplateIndex,
) -> str | None:
    """Map recipe_name to the closest industry in the index."""
    industries = template_index.available_industries
    if not industries:
        return None
    recipe_lower = doc_spec.recipe_name.lower()
    for ind in industries:
        # e.g. recipe "financial" matches industry "financial_services_wealth"
        if ind.split("_")[0] in recipe_lower:
            return ind
    return None


def _deduplicate(variants: list[Variant]) -> list[Variant]:
    seen: set[str] = set()
    result: list[Variant] = []
    for v in variants:
        if v.id not in seen:
            seen.add(v.id)
            result.append(v)
    return result
