# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Title page renderer: randomly picks one of the title_page templates and renders via Playwright."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jinja2
import numpy as np

from synthpdf.render.html_to_pdf import render_html_to_pdf
from synthpdf.types import DocumentSpec

log = logging.getLogger(__name__)

_PAGE_SIZES: dict[str, str] = {
    "landscape": "1440px 900px",
    "portrait":  "794px 1123px",
}

_VIEWPORTS: dict[str, dict[str, int]] = {
    "landscape": {"width": 1440, "height": 900},
    "portrait":  {"width": 794,  "height": 1123},
}

_REPORT_TYPES: dict[str, str] = {
    "financial":  "Financial Report",
    "data":       "Data Report",
    "legal":      "Legal Document",
    "technical":  "Technical Specification",
    "spec":       "Technical Specification",
    "hr":         "Human Resources",
    "onboard":    "Onboarding Guide",
    "executive":  "Executive Briefing",
    "dashboard":  "Executive Dashboard",
    "research":   "Research Report",
    "compliance": "Compliance Report",
    "audit":      "Audit Report",
    "marketing":  "Marketing Presentation",
    "operations": "Operations Manual",
    "training":   "Training Material",
    "contract":   "Contract Document",
    "meeting":    "Meeting Notes",
    "product":    "Product Catalog",
}

_SUBTITLES: dict[str, str] = {
    "financial":  "Financial Performance and Strategic Outlook",
    "data":       "Data Analysis and Insights",
    "legal":      "Comprehensive Legal Documentation",
    "technical":  "Technical Specifications and Implementation Guide",
    "spec":       "Technical Specifications and Implementation Guide",
    "hr":         "Human Resources and Organizational Policy",
    "onboard":    "Employee Onboarding and Integration Program",
    "executive":  "Executive Summary and Key Performance Indicators",
    "dashboard":  "Performance Metrics and Strategic Indicators",
    "research":   "Research Findings, Analysis and Recommendations",
    "compliance": "Regulatory Compliance and Risk Assessment",
    "audit":      "Internal Audit Findings and Recommendations",
    "marketing":  "Brand Strategy and Campaign Performance",
    "operations": "Operational Procedures and Best Practices",
    "training":   "Learning Objectives and Assessment Criteria",
    "contract":   "Terms, Conditions and Agreement Details",
    "meeting":    "Discussion Points, Decisions and Action Items",
    "product":    "Product Portfolio and Pricing Guide",
}


def render_title_page(
    doc_spec: DocumentSpec,
    brand: dict[str, str],
    pages_dir: Path,
    doc_rng: np.random.Generator,
    page_layout: str = "landscape",
    title_pages_dir: Path | None = None,
) -> Path | None:
    """Render a random title page template to a single-page PDF.

    Parameters
    ----------
    doc_spec:
        Document identity (title, recipe_name used for context).
    brand:
        Brand dict with keys: name, primary, accent, tagline, initials.
    pages_dir:
        Temp directory where HTML + PDF will be written.
    doc_rng:
        Per-document RNG for reproducibility.
    page_layout:
        ``"landscape"`` (1440x900) or ``"portrait"`` (794x1123).
    title_pages_dir:
        Directory containing ``title_*.html`` templates.
        Returns None silently if not set or empty.

    Returns
    -------
    Path to the rendered single-page PDF, or None on failure.
    """
    if title_pages_dir is None or not title_pages_dir.exists():
        return None

    templates = sorted(title_pages_dir.glob("title_*.html"))
    if not templates:
        log.warning("No title_*.html templates found in %s", title_pages_dir)
        return None

    idx = int(doc_rng.integers(0, len(templates)))
    template_path = templates[idx]

    page_size_css = _PAGE_SIZES.get(page_layout, _PAGE_SIZES["landscape"])
    viewport      = _VIEWPORTS.get(page_layout, _VIEWPORTS["landscape"])

    recipe = (doc_spec.recipe_name or "").lower()
    report_type = "Annual Report"
    subtitle     = "Comprehensive Analysis and Strategic Outlook"
    for key, label in _REPORT_TYPES.items():
        if key in recipe:
            report_type = label
            subtitle    = _SUBTITLES.get(key, subtitle)
            break

    context: dict[str, Any] = {
        "company_name":   brand["name"],
        "primary_color":  brand["primary"],
        "accent_color":   brand["accent"],
        "tagline":        brand["tagline"],
        "initials":       brand["initials"],
        "document_title": doc_spec.title or doc_spec.doc_id,
        "subtitle":       subtitle,
        "year":           "2025",
        "report_type":    report_type,
        "page_size_css":  page_size_css,
    }

    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        autoescape=False,
        undefined=jinja2.ChainableUndefined,
    )
    try:
        source   = template_path.read_text(encoding="utf-8")
        html_str = env.from_string(source).render(**context)
    except Exception as exc:
        log.warning("Title page Jinja2 render failed (%s): %s", template_path.name, exc)
        return None

    html_path = pages_dir / "title_page.html"
    html_path.write_text(html_str, encoding="utf-8")

    pdf_path = pages_dir / "title_page.pdf"
    try:
        render_html_to_pdf(html_path, pdf_path, backend="playwright", viewport=viewport)
    except Exception as exc:
        log.warning("Title page PDF render failed (%s): %s", template_path.name, exc)
        return None

    log.info("Title page rendered: %s  layout=%s", template_path.name, page_layout)
    return pdf_path
