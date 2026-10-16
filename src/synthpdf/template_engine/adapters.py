# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Content adapters: fill Jinja2 template variables with synthetic content.

Strategy
--------
1. Call ``extract_variables(template_html)`` to discover every variable.
2. For each variable, dispatch on inferred type + name-pattern heuristics to
   generate plausible synthetic content.
3. Return a flat ``{var_name: value}`` dict ready for ``jinja2.Template.render(**ctx)``.

When variant ground truth is provided, table content (rows, schema) is drawn
from it before falling back to synthetic data pools.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np

from synthpdf.template_engine.extractor import extract_variables
from synthpdf.types import Variant


# ---------------------------------------------------------------------------
# Synthetic data pools
# ---------------------------------------------------------------------------

_COMPANIES = ['Sample Organization 01', 'Sample Organization 02', 'Sample Organization 03', 'Sample Organization 04', 'Sample Organization 05', 'Sample Organization 06', 'Sample Organization 07', 'Sample Organization 08', 'Sample Organization 09', 'Sample Organization 10', 'Sample Organization 11', 'Sample Organization 12', 'Sample Organization 13', 'Sample Organization 14', 'Sample Organization 15', 'Sample Organization 16']

_SECTORS = [
    "Financial Services", "Technology", "Healthcare", "Manufacturing",
    "Energy & Resources", "Consumer Goods", "Real Estate", "Logistics",
    "Professional Services", "Insurance", "Retail", "Media & Entertainment",
]

_TITLES = [
    "Annual Performance Review {year}",
    "Strategic Outlook Q{q} {year}",
    "Market Analysis Report",
    "Executive Summary — {year}",
    "Operational Excellence Report",
    "Investment Portfolio Overview",
    "Risk Assessment Summary {year}",
    "Business Intelligence Dashboard",
    "Quarterly Financial Highlights",
    "Competitive Landscape Analysis",
]

_PARAGRAPHS = [
    (
        "The organization demonstrated strong performance across all key metrics during the "
        "reporting period. Revenue growth exceeded projections by 12%, driven primarily by "
        "expansion in core market segments and successful product launches."
    ),
    (
        "Market conditions continue to favor our strategic positioning. The competitive "
        "landscape has shifted in response to macroeconomic pressures, creating new "
        "opportunities for differentiated value creation across the portfolio."
    ),
    (
        "Operational efficiency initiatives delivered measurable improvements in cost "
        "management and resource allocation. The digital transformation roadmap remains "
        "on track with key milestones achieved ahead of schedule."
    ),
    (
        "Risk management frameworks have been enhanced to address emerging challenges "
        "in the regulatory environment. Compliance metrics remain above industry benchmarks "
        "across all business units and geographic regions."
    ),
    (
        "Customer satisfaction scores reached an all-time high, reflecting the impact of "
        "service quality improvements and enhanced engagement programs. Net promoter scores "
        "increased by 18 points year-over-year across all segments."
    ),
    (
        "Capital allocation decisions reflect our disciplined approach to balancing growth "
        "investment with returns to shareholders. Free cash flow generation remains robust, "
        "supporting both organic and inorganic strategic opportunities."
    ),
    (
        "The supply chain resilience program has significantly reduced exposure to single-"
        "source dependencies. Dual-sourcing strategies are now in place for 87% of "
        "critical components, up from 64% in the prior year."
    ),
]

_QUOTES = [
    "We are positioned to capture emerging opportunities while maintaining our commitment to sustainable growth.",
    "The data clearly indicates a positive trajectory across all key performance indicators.",
    "Our strategic investments are beginning to yield measurable returns for all stakeholders.",
    "Innovation remains at the core of our competitive differentiation and long-term value creation.",
    "The resilience of our business model has been validated through challenging market conditions.",
    "We have built a foundation that enables us to move quickly and decisively when opportunities arise.",
]

_FIRST_NAMES = ["Alexandra", "Benjamin", "Catherine", "David", "Elena", "Francis",
                "Grace", "Henry", "Isabella", "James", "Katherine", "Liam"]
_LAST_NAMES = ["Morrison", "Chen", "Anderson", "Williams", "Thompson", "Garcia",
               "Martinez", "Johnson", "Davis", "Wilson", "Taylor", "Brown"]

_ROLES = [
    "Chief Executive Officer", "Chief Financial Officer", "Chief Operating Officer",
    "Managing Director", "Senior Vice President", "Executive Director",
    "Head of Strategy", "Director of Operations", "Chief Strategy Officer",
    "President & CEO", "Group CFO", "Chief Investment Officer",
]

_KPI_LABELS = [
    "Revenue Growth", "Operating Margin", "Market Share", "Customer Retention",
    "ROI", "EBITDA", "Net Income", "Efficiency Ratio", "Debt/Equity",
    "Employee NPS", "Pipeline Value", "Cost Reduction", "CAGR",
    "Return on Assets", "Gross Margin", "Free Cash Flow",
]

_KPI_VALUES = [
    "+14.2%", "$2.8B", "34.6%", "89.3%", "18.7%", "2.4×", "$147M",
    "+12.3%", "0.45", "72 pts", "$890M", "−8.4%", "$1.2T", "99.1%",
    "+22%", "$4.6B", "41%", "3.1×",
]

_NAV_TABS = [
    "Overview", "Performance", "Markets", "Operations", "Risk",
    "Outlook", "Finance", "Strategy", "Analytics", "Portfolio",
    "Clients", "Governance",
]

_TIMELINE_LABELS = [
    "Foundation", "Launch", "Expansion", "Acceleration", "Maturity",
    "Innovation", "Transformation", "Scale", "Optimization", "Future Vision",
    "Milestone", "Partnership", "Acquisition",
]

_TIMELINE_MILESTONES = [
    "Platform Launch", "Series A Funding", "Global Expansion",
    "Product Redesign", "IPO", "Strategic Acquisition",
    "Regulatory Approval", "New Market Entry", "Digital Transformation",
    "Operational Restructuring", "Partnership Agreement",
]

_SECTION_HEADINGS = [
    "Performance Overview", "Market Dynamics", "Strategic Initiatives",
    "Financial Highlights", "Operational Metrics", "Risk Assessment",
    "Growth Opportunities", "Key Findings", "Executive Summary",
    "Outlook & Guidance", "Industry Context", "Investment Thesis",
]

_FOOTER_TEXTS = [
    "CONFIDENTIAL — FOR INTERNAL USE ONLY",
    "© 2024 ALL RIGHTS RESERVED",
    "STRATEGIC INTELLIGENCE REPORT",
    "PROPRIETARY AND CONFIDENTIAL",
    "NOT FOR EXTERNAL DISTRIBUTION",
]

_CHART_TITLES = [
    "Revenue by Quarter", "Market Share Distribution", "Growth Trajectory",
    "Portfolio Allocation", "Performance vs Target", "YoY Comparison",
    "Regional Breakdown", "Segment Analysis",
]

_STATUS_VALUES = ["Active", "Pending", "Closed", "In Progress", "Complete", "Planned"]

_TAG_CLASSES = ["tag-primary", "tag-warning", "tag-success", "tag-danger", "tag-info"]
_TAG_TEXTS = ["GROWTH", "RISK", "OPPORTUNITY", "ALERT", "PRIORITY",
              "REVIEW", "ACTION", "HIGH", "MEDIUM", "LOW"]

_COL_LABEL_MAP = {
    "year": "Year", "date": "Date", "name": "Name", "title": "Title",
    "desc": "Description", "description": "Description", "value": "Value",
    "amount": "Amount ($)", "status": "Status", "milestone": "Milestone",
    "event": "Event", "category": "Category", "investment": "Investment",
    "region": "Region", "type": "Type", "score": "Score", "rank": "Rank",
    "change": "Change", "target": "Target", "actual": "Actual",
}

# 400x300 dark navy PNG placeholder (base64) — works with all PDF renderers
_PLACEHOLDER_IMAGE_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAZAAAAEsCAIAAABi1XKVAAADJklEQVR42u3aoQ2AMBRF0VfCFvVd"
    "oqvU4xgGh2dQLBtAwzkTkJLcfPFKbT0AM1g8ASBYAIIFCBaAYAEIFiBYAIIFIFiAYAEIFoBgAYIF"
    "IFgAggUIFoBgAQgWIFgAggUgWIBgAQgWgGABggUgWACCBQgWgGABggUgWACCBQgWgGABCBYgWACC"
    "BSBYgGABCBaAYAGCBSBYAIIFCBaAYAEIFiBYAIIFIFiAYAEIFoBgAYIFIFgAggUIFoBgAYIFIFgA"
    "ggUIFoBgAQgWIFgAggUgWIBgAQgWgGABggUgWACCBQgWgGABCBYgWACCBSBYgGABCBaAYAGCBSBY"
    "AIIFCBaAYAGCBSBYAIIFCBaAYAEIFiBYAIIFIFiAYAEIFoBgAYIFIFgAggUIFoBgAQgWIFgAggUg"
    "WIBgAQgWgGABggUgWACCBQgWgGABggUgWACCBQgWgGABCBYgWACCBSBYgGABCBaAYAGCBSBYAIIF"
    "CBaAYAEIFiBYAIIFIFiAYAEIFsDT6gn+bGz7jJ99nYd/58ICECwAwQIEC+CLSm3dKwAuLADBAmKH"
    "ReywYoeFCwtAsADBAhAsgNhhAS4sAMECiB0WscPChQUgWACCBQgWQOywAFxYgGABxA6L2GHFDgsX"
    "FiBYAIIFIFhA7LAAXFiAYAEIFoBgAYIFIFgAggUIFoBgAQgWIFgAggUgWIBgAQgWgGABggUgWACC"
    "BQgWgGABCBYgWACCBSBYgGABCBaAYAGCBSBYgGABCBaAYAGCBSBYAIIFCBaAYAEIFiBYAIIFIFiA"
    "YAEIFoBgAYIFIFgAggUIFoBgAQgWIFgAggUgWIBgAQgWgGABggUgWIBgAQgWgGABggUgWACCBQgW"
    "gGABCBYgWACCBSBYgGABCBaAYAGCBSBYAIIFCBaAYAEIFiBYAIIFIFiAYAEIFiBYngAQLADBAgQL"
    "QLAABAsQLADBAhAsQLAABAtAsADBAhAsAMECBAtAsAAECxAsAMECECxAsAAEC0CwAMECECwAwQIE"
    "C0CwAMECECwAwQIEC0CwAAQLECwAwQIQLECwAAQLQLAAwQIQLADBAgQL4D03DIMQyKbEjiMAAAAASUVORK5CYII="
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fill_template_variables(
    template_html: str | Path,
    spec: dict[str, Any],
    variants: list[Variant] | None = None,
    rng: np.random.Generator | None = None,
    doc_title: str | None = None,
    page_number: int = 1,
) -> dict[str, Any]:
    """Return a context dict ``{variable_name: value}`` for Jinja2 rendering.

    Parameters
    ----------
    template_html:
        HTML template source or path.
    spec:
        Template spec JSON (used for color palette, industry, page_role, etc.).
    variants:
        Optional list of pre-generated content variants to source data from.
    rng:
        NumPy RNG for reproducibility.
    doc_title:
        Document-level title to use for ``page_title`` variables.
    page_number:
        1-based page number used for footer / pagination variables.
    """
    if rng is None:
        rng = np.random.default_rng()

    var_types = extract_variables(template_html)
    year = 2024
    palette = spec.get("color_palette", {})

    table_variants = [v for v in (variants or []) if v.kind == "table"]
    text_variants  = [v for v in (variants or []) if v.kind == "text"]

    context: dict[str, Any] = {}
    for var_name, var_type in var_types.items():
        context[var_name] = _fill_one(
            var_name=var_name,
            var_type=var_type,
            spec=spec,
            palette=palette,
            rng=rng,
            doc_title=doc_title,
            page_number=page_number,
            year=year,
            table_variants=table_variants,
            text_variants=text_variants,
        )

    return context


# ---------------------------------------------------------------------------
# Internal dispatcher
# ---------------------------------------------------------------------------

def _fill_one(
    var_name: str,
    var_type: str,
    spec: dict[str, Any],
    palette: dict[str, Any],
    rng: np.random.Generator,
    doc_title: str | None,
    page_number: int,
    year: int,
    table_variants: list[Variant],
    text_variants: list[Variant],
) -> Any:
    # Seed a stdlib random so we get non-numpy randomness for choices/samples
    py_rng = random.Random(int(rng.integers(0, 2**31)))

    if var_type == "url":
        return _PLACEHOLDER_IMAGE_URI

    if var_type == "color":
        colors = [
            palette.get("primary", "#1b263b"),
            palette.get("accent", "#415a77"),
            palette.get("secondary", "#778da9"),
        ]
        return py_rng.choice(colors)

    if var_type == "number":
        return _fill_number(var_name.lower(), page_number, year, py_rng)

    if var_type == "list":
        return _fill_list(
            var_name.lower(), spec, palette, rng, py_rng, year, table_variants
        )

    # Default: string
    return _fill_string(
        var_name=var_name,
        name=var_name.lower(),
        spec=spec,
        palette=palette,
        py_rng=py_rng,
        doc_title=doc_title,
        page_number=page_number,
        year=year,
        text_variants=text_variants,
    )


# ---------------------------------------------------------------------------
# Number filling
# ---------------------------------------------------------------------------

def _fill_number(name: str, page_number: int, year: int, py_rng: random.Random) -> int | float:
    if "year" in name or name == "report_year":
        return year - py_rng.randint(0, 3)
    if "page" in name or name in ("page_number", "section_number"):
        return page_number
    if "height" in name or "width" in name or "px" in name:
        return py_rng.randint(40, 120)
    if "count" in name or "num" in name or "number" in name:
        return py_rng.randint(3, 20)
    return py_rng.randint(10, 100)


# ---------------------------------------------------------------------------
# String filling
# ---------------------------------------------------------------------------

def _fill_string(
    var_name: str,
    name: str,
    spec: dict[str, Any],
    palette: dict[str, Any],
    py_rng: random.Random,
    doc_title: str | None,
    page_number: int,
    year: int,
    text_variants: list[Variant],
) -> str:
    industry = spec.get("industry", "").replace("_", " ").title()

    # --- Column header (timeline_col_year, data_col_amount, etc.) ---
    if "_col_" in name or name.startswith("col_"):
        col_key = name.split("_col_")[-1] if "_col_" in name else name[4:]
        col_key = col_key.replace("_", "")
        return _COL_LABEL_MAP.get(col_key, col_key.replace("_", " ").title())

    # --- Page / document title ---
    if name in ("page_title", "title", "doc_title", "main_title", "document_title"):
        if doc_title:
            return doc_title
        return py_rng.choice(_TITLES).format(year=year, q=py_rng.randint(1, 4))

    if name.endswith("_title") or name in ("section_heading", "section_title", "heading"):
        return py_rng.choice(_SECTION_HEADINGS)

    if name.endswith("_heading"):
        return py_rng.choice(_SECTION_HEADINGS)

    # --- Subtitles / descriptions ---
    if "subtitle" in name or "tagline" in name:
        return py_rng.choice(_PARAGRAPHS)[:100]

    if "description" in name or "summary" in name or "overview" in name:
        return py_rng.choice(_PARAGRAPHS)[:150]

    # --- Body paragraphs ---
    if "paragraph" in name or name.startswith("body_") or name == "intro_text":
        return py_rng.choice(_PARAGRAPHS)

    if name == "body_text":
        return " ".join(py_rng.sample(_PARAGRAPHS, min(2, len(_PARAGRAPHS))))

    # --- Quotes ---
    if "quote" in name and "author" not in name and "role" not in name and "attribution" not in name:
        return py_rng.choice(_QUOTES)

    if "quote_author" in name or (name.endswith("_author") and "quote" in name):
        fn = py_rng.choice(_FIRST_NAMES)
        ln = py_rng.choice(_LAST_NAMES)
        return f"{fn} {ln}"

    if "quote_role" in name or ("_role" in name and "page" not in name):
        return py_rng.choice(_ROLES)

    if "attribution" in name:
        return f"— {py_rng.choice(_COMPANIES)}"

    # --- KPI / metric values ---
    if name.endswith("_value") or name in ("value", "metric_value", "callout_value", "stat_value"):
        return py_rng.choice(_KPI_VALUES)

    if name.endswith("_label") or name in ("label", "metric_label", "callout_label", "stat_label"):
        return py_rng.choice(_KPI_LABELS)

    # --- KPI trend / delta ---
    if "trend" in name and "class" not in name:
        direction = py_rng.choice(["+", "−"])
        return f"{direction}{py_rng.randint(1, 30)}%"

    if "trend_class" in name:
        return py_rng.choice(["positive", "neutral", "negative"])

    if "delta" in name or "change" in name:
        sign = py_rng.choice(["+", "−"])
        return f"{sign}{py_rng.randint(1, 25)}%"

    # --- Badge ---
    if "badge_value" in name:
        return py_rng.choice(_KPI_VALUES)
    if "badge_label" in name:
        return py_rng.choice(_KPI_LABELS)

    # --- Institution / company / org ---
    if any(x in name for x in ("institution", "company", "org_name", "brand_name",
                               "firm_name", "corp_name")):
        return py_rng.choice(_COMPANIES)

    # --- Category / classification ---
    if any(x in name for x in ("category", "classification", "sector")):
        return py_rng.choice(["CONFIDENTIAL", "SECTOR ANALYSIS", "STRATEGIC REPORT",
                              industry.upper(), "INTERNAL REVIEW"])

    # --- Footer text ---
    if "footer" in name:
        if any(x in name for x in ("left", "classification", "company", "label")):
            return py_rng.choice(_FOOTER_TEXTS)
        if any(x in name for x in ("right", "page", "number", "num")):
            return str(page_number)
        if any(x in name for x in ("center", "date", "period")):
            return f"Q{py_rng.randint(1, 4)} {year}"
        return py_rng.choice(_FOOTER_TEXTS)

    # --- Page number (standalone) ---
    if name == "page_number":
        return str(page_number)

    if "page_indicator" in name or "indicator" in name:
        return f"Page {page_number}"

    # --- Chart title ---
    if "chart_title" in name or "chart_heading" in name:
        return py_rng.choice(_CHART_TITLES)

    # --- Section number ---
    if "section_number" in name or "section_num" in name:
        return str(py_rng.randint(1, 9))

    # --- Date / period / quarter ---
    if any(x in name for x in ("date", "period", "quarter", "month")):
        return f"Q{py_rng.randint(1, 4)} {year}"

    # --- Report type ---
    if "report_type" in name or "doc_type" in name or "report_name" in name:
        return "ANNUAL REPORT"

    # --- Alt text ---
    if name.endswith("_alt") or name.endswith("_alt_text"):
        return "Strategic overview visualization"

    # --- Timestamp / version ---
    if "version" in name:
        return f"v{py_rng.randint(1, 5)}.{py_rng.randint(0, 9)}"

    if "timestamp" in name or "updated" in name:
        return f"March {py_rng.randint(1, 28)}, {year}"

    # --- Tag row label ---
    if "tag_row_label" in name or name == "tags_label":
        return py_rng.choice(["TAGS", "FILTERS", "CATEGORIES", "FOCUS AREAS"])

    # --- Timeline section header ---
    if "timeline_section" in name or "timeline_heading" in name:
        return py_rng.choice(["Key Milestones", "Strategic Timeline", "Historical Performance"])

    # --- Color (fallback) ---
    if "color" in name:
        return palette.get("accent", "#415a77")

    # --- Hero / cover elements ---
    if "hero_subtitle" in name:
        return py_rng.choice(_PARAGRAPHS)[:100]

    if "category_label" in name:
        return py_rng.choice(["EXCLUSIVE REPORT", "SECTOR INSIGHT", "ANNUAL OVERVIEW",
                              industry.upper()])

    # --- Default fallback ---
    return py_rng.choice(_SECTION_HEADINGS)


# ---------------------------------------------------------------------------
# List filling
# ---------------------------------------------------------------------------

class _SmartItem(dict):
    """A dict that renders as plain text when used directly in Jinja2 ``{{ item }}``.

    Templates that loop with ``{% for item in items %}{{ item }}{% endfor %}``
    get the most meaningful text field instead of Python's dict repr
    ``{'text': '...', 'label': '...'}``.

    Structured access (``{{ item.text }}``, ``{{ item.label }}``) still works
    because we inherit from dict.

    ``__missing__`` ensures that any key not explicitly set returns a sensible
    placeholder string instead of raising KeyError.  With _ItemFirstEnvironment
    this means ``row.likelihood`` (where "likelihood" is not in the dict) calls
    ``obj["likelihood"]`` → ``__missing__`` → returns a string, so Jinja2 never
    creates a ChainableUndefined.  That prevents the indirect crash where
    ``bool(ChainableUndefined)`` calls ``__len__`` which raises UndefinedError.
    """

    _TEXT_KEYS = ("text", "label", "value", "title", "milestone_title", "milestone_desc")

    # Heuristic defaults: if a requested key matches one of these patterns,
    # return a plausible placeholder rather than a bare "".
    _RISK_WORDS   = ("likelihood", "probability", "frequency", "occurrence")
    _IMPACT_WORDS = ("impact", "severity", "consequence", "magnitude")
    _STATUS_WORDS = ("status", "state", "stage", "phase", "class", "style")
    _OWNER_WORDS  = ("owner", "lead", "author", "manager", "assigned", "contact")
    _MONEY_WORDS  = ("revenue", "amount", "cost", "budget", "value", "gwp",
                     "reserves", "incurred", "pml")
    _RATE_WORDS   = ("ratio", "rate", "pct", "percent", "margin", "growth",
                     "delta", "change", "progress", "bar_pct")

    def __missing__(self, key: str) -> str:
        """Return a sensible placeholder for any key not stored in the dict."""
        k = key.lower()
        if any(w in k for w in self._RISK_WORDS):
            return "Medium"
        if any(w in k for w in self._IMPACT_WORDS):
            return "Moderate"
        if any(w in k for w in self._STATUS_WORDS):
            return ""          # CSS class / inline style — blank is safest
        if any(w in k for w in self._OWNER_WORDS):
            return "J. Smith"
        if any(w in k for w in self._MONEY_WORDS):
            return "$1.2M"
        if k in ("percent", "pct", "bar_pct", "progress", "completion"):
            # Must be numeric — templates often do arithmetic: {{ 100 - item.percent }}
            return 42
        if any(w in k for w in self._RATE_WORDS):
            return "12%"
        if k in ("id", "row_id", "ref", "ref_id", "number", "num"):
            return "—"
        # Generic: return an em-dash so the table cell isn't visually empty
        return "—"

    def __str__(self) -> str:
        for key in self._TEXT_KEYS:
            if key in self:
                return str(self[key])
        if self:
            return str(next(iter(self.values())))
        return ""

    def __repr__(self) -> str:  # Jinja2 uses repr() in some debug paths
        return self.__str__()


def _make_subitems(py_rng: random.Random) -> list[dict[str, Any]]:
    """Return a small list of sub-items used as nested lists inside parent dicts.

    Covers the common nested keys templates access: .items, .entries,
    .children, .members so that _ItemFirstEnvironment can find a real list
    regardless of which key name the template uses.
    """
    items = py_rng.sample(_NAV_TABS, min(py_rng.randint(2, 4), len(_NAV_TABS)))
    return [
        _SmartItem({
            "label": item,
            "name": item,
            "text": item,
            "page": str(py_rng.randint(1, 50)),
            "active": j == 0,
            "title": item,
            "description": "",
        })
        for j, item in enumerate(items)
    ]


def _fill_list(
    name: str,
    spec: dict[str, Any],
    palette: dict[str, Any],
    rng: np.random.Generator,
    py_rng: random.Random,
    year: int,
    table_variants: list[Variant],
) -> list[dict[str, Any]]:
    n = py_rng.randint(3, 6)
    primary = palette.get("primary", "#1b263b")
    accent  = palette.get("accent", "#415a77")
    muted   = palette.get("muted", "#778da9")
    secondary = palette.get("secondary", "#778da9")
    color_cycle = [primary, accent, muted, secondary]

    # --- KPI cards / metric cards ---
    if any(x in name for x in ("kpi", "metric", "stat", "card", "indicator")):
        labels = py_rng.sample(_KPI_LABELS, min(n, len(_KPI_LABELS)))
        return [
            _SmartItem({
                "value": py_rng.choice(_KPI_VALUES),
                "label": labels[i],
                "text": labels[i],
                "color": color_cycle[i % len(color_cycle)],
                "icon": "",
                "delta": f"+{py_rng.randint(1, 30)}%",
                "trend": f"+{py_rng.randint(1, 30)}%",
                "trend_class": py_rng.choice(["positive", "positive", "neutral", "negative"]),
            })
            for i in range(n)
        ]

    # --- Nested nav categories / sidebar sections: [{title, items: [{label, active}]}] ---
    # Must be checked BEFORE the plain "nav" / "section" branches below to avoid
    # returning flat dicts whose dict.items() method is exposed as a non-iterable
    # when the template does  {% for item in section.items %}  or
    # {% for nav in category.items %}  → 'builtin_function_or_method' crash.
    if "categor" in name or name.endswith("sections"):
        categories = py_rng.sample(_NAV_TABS[:8], min(3, len(_NAV_TABS)))
        return [
            _SmartItem({
                "title": cat,
                "label": cat,
                "text": cat,
                # Provide all common nested-list aliases so templates using
                # .items, .entries, .children, .members all find a real list.
                "items":    _make_subitems(py_rng),
                "entries":  _make_subitems(py_rng),
                "children": _make_subitems(py_rng),
                "members":  _make_subitems(py_rng),
            })
            for cat in categories
        ]

    # --- Process steps: [{number, icon, title, description, color, details}] ---
    if "step" in name:
        _step_icons = ["I", "II", "III", "IV", "V", "VI"]
        return [
            _SmartItem({
                "number": str(i + 1),
                "icon": _step_icons[i % len(_step_icons)],
                "title": py_rng.choice(_SECTION_HEADINGS),
                "description": py_rng.choice(_PARAGRAPHS)[:60],
                "text": py_rng.choice(_SECTION_HEADINGS),
                "color": color_cycle[i % len(color_cycle)],
                "details": _make_subitems(py_rng),  # for step.details loops
            })
            for i in range(4)
        ]

    # --- Icon/grid items: [{label, description, icon_bg, ...}] ---
    if "grid" in name:
        return [
            _SmartItem({
                "label": py_rng.choice(_SECTION_HEADINGS),
                "description": py_rng.choice(_PARAGRAPHS)[:80],
                "icon_bg": color_cycle[i % len(color_cycle)],
                "text": py_rng.choice(_SECTION_HEADINGS),
            })
            for i in range(n)
        ]

    # --- Teams / groups with nested member lists ---
    if any(x in name for x in ("team", "group", "department", "unit")):
        return [
            _SmartItem({
                "title": py_rng.choice(_SECTION_HEADINGS),
                "label": py_rng.choice(_SECTION_HEADINGS),
                "text": py_rng.choice(_SECTION_HEADINGS),
                "members": _make_subitems(py_rng),
                "items":   _make_subitems(py_rng),
                "entries": _make_subitems(py_rng),
            })
            for _ in range(py_rng.randint(2, 4))
        ]

    # --- Navigation tabs ---
    if any(x in name for x in ("tab", "nav")):
        tabs = py_rng.sample(_NAV_TABS, min(n, len(_NAV_TABS)))
        return [
            _SmartItem({"label": t, "text": t, "active": i == 0})
            for i, t in enumerate(tabs)
        ]

    # --- Timeline entries / milestones ---
    if any(x in name for x in ("timeline", "milestone", "entry", "entries")):
        n_entries = py_rng.randint(4, 7)
        return [
            _SmartItem({
                "year": str(year - (n_entries - i) + 1),
                "label": py_rng.choice(_TIMELINE_LABELS),
                "text": py_rng.choice(_TIMELINE_MILESTONES),
                "milestone_title": py_rng.choice(_TIMELINE_MILESTONES),
                "milestone_desc": py_rng.choice(_PARAGRAPHS)[:80],
                "investment": f"${py_rng.randint(1, 99)}M",
                "status": py_rng.choice(_STATUS_VALUES),
                "active": i == n_entries - 1,
                "node_label": py_rng.choice(_TIMELINE_LABELS),
                "node_year": str(year - (n_entries - i) + 1),
            })
            for i in range(n_entries)
        ]

    # --- Chart bars ---
    if any(x in name for x in ("bar", "chart", "histogram")):
        return [
            _SmartItem({
                "label": f"Q{i+1}",
                "text": f"Q{i+1}",
                "height_primary": py_rng.randint(40, 120),
                "height_secondary": py_rng.randint(20, 80),
                "value": py_rng.randint(10, 100),
                "color": color_cycle[i % len(color_cycle)],
            })
            for i in range(4)
        ]

    # --- Data table rows ---
    if any(x in name for x in ("row", "table", "data")) and "nodes" not in name:
        # Use real variant ground truth if available
        if table_variants:
            v = table_variants[int(rng.integers(0, len(table_variants)))]
            gt = v.ground_truth
            cells_list = gt.get("cells", [])
            if cells_list and len(cells_list) >= 1:
                rows = cells_list if isinstance(cells_list[0], list) else [cells_list]
                sample_rows = rows[:min(8, len(rows))]
                return [_SmartItem({"cells": [str(c) for c in row], "text": " | ".join(str(c) for c in row)}) for row in sample_rows]
        # Synthetic fallback
        return [
            _SmartItem({
                "cells": [
                    f"Item {i+1}",
                    py_rng.choice(["A", "B", "C", "D"]),
                    f"${py_rng.randint(100, 9999):,}",
                    py_rng.choice(_STATUS_VALUES),
                ],
                "text": f"Item {i+1}",
            })
            for i in range(min(n, 8))
        ]

    # --- Icon + text grid ---
    if any(x in name for x in ("icon", "feature", "highlight", "point")):
        return [
            _SmartItem({
                "icon": py_rng.choice(["●", "■", "▲", "◆", "★"]),
                "title": py_rng.choice(_SECTION_HEADINGS),
                "text": py_rng.choice(_PARAGRAPHS)[:80],
                "value": py_rng.choice(_KPI_VALUES),
            })
            for _ in range(n)
        ]

    # --- Donut / pie / radial charts: [{percent (int), label, color, description}] ---
    # percent MUST be an int — templates do arithmetic: {{ 100 - donut.percent }}
    if any(x in name for x in ("donut", "pie", "radial", "ring", "gauge")):
        return [
            _SmartItem({
                "percent":     py_rng.randint(20, 85),
                "label":       py_rng.choice(_KPI_LABELS),
                "text":        py_rng.choice(_KPI_LABELS),
                "value":       f"{py_rng.randint(20, 85)}%",
                "color":       color_cycle[i % len(color_cycle)],
                "description": py_rng.choice(_PARAGRAPHS)[:60],
            })
            for i in range(py_rng.randint(3, 4))
        ]

    # --- Tags ---
    if "tag" in name:
        return [
            _SmartItem({
                "text": py_rng.choice(_TAG_TEXTS),
                "label": py_rng.choice(_TAG_TEXTS),
                "class": py_rng.choice(_TAG_CLASSES),
            })
            for _ in range(py_rng.randint(3, 6))
        ]

    # --- Bullet / list items ---
    if any(x in name for x in ("bullet", "list", "point", "item", "section")):
        return [
            _SmartItem({
                "text": py_rng.choice(_PARAGRAPHS)[:100],
                "label": py_rng.choice(_SECTION_HEADINGS),
            })
            for _ in range(n)
        ]

    # --- Generic fallback ---
    return [
        _SmartItem({
            "label": py_rng.choice(_SECTION_HEADINGS),
            "value": py_rng.choice(_KPI_VALUES),
            "text": py_rng.choice(_PARAGRAPHS)[:80],
        })
        for _ in range(n)
    ]
