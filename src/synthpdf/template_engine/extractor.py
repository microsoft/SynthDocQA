# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Jinja2 template variable extractor.

Discovers all template variables that must be filled before rendering,
and infers a semantic type for each so that adapters can generate
appropriate synthetic content.

Inferred types
--------------
  "string"  — short or long text (default)
  "url"     — image or resource URL (``*_url``, ``*_src``)
  "color"   — CSS hex color (``*_color``, ``*_hex``)
  "number"  — plain integer / year / count
  "list"    — iterable used in a ``{% for x in LIST %}`` loop
"""
from __future__ import annotations

import re
from pathlib import Path

import jinja2
import jinja2.meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_variables(html: str | Path) -> dict[str, str]:
    """Return ``{var_name: inferred_type}`` for all undeclared template variables.

    Parameters
    ----------
    html:
        HTML template source string or path to file.

    Returns
    -------
    Dict mapping each top-level Jinja2 variable to its inferred type.
    Loop variables (the iteration item, e.g. ``tab`` in
    ``{% for tab in nav_tabs %}``) are **not** included — only the
    iterable (``nav_tabs``) appears, typed as ``"list"``.
    """
    if isinstance(html, Path):
        try:
            source = html.read_text(encoding="utf-8")
        except OSError:
            return {}
    else:
        source = html

    env = jinja2.Environment()
    try:
        ast = env.parse(source)
    except jinja2.TemplateSyntaxError:
        return {}

    undeclared: set[str] = jinja2.meta.find_undeclared_variables(ast)
    loop_iterables: set[str] = _find_loop_iterables(source)

    result: dict[str, str] = {}
    for var in sorted(undeclared):
        if var in loop_iterables:
            result[var] = "list"
        else:
            result[var] = _infer_type(var)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_loop_iterables(source: str) -> set[str]:
    """Extract variable names used as loop iterables in ``{% for x in VAR %}``."""
    pattern = re.compile(r'\{%-?\s*for\s+\w+\s+in\s+(\w+)\s*[-+]?%\}')
    return set(pattern.findall(source))


def _infer_type(var_name: str) -> str:
    """Heuristically map a variable name to a data type."""
    name = var_name.lower()

    # Image / resource URLs
    if name.endswith("_url") or name.endswith("_src"):
        return "url"

    # CSS colors
    if name.endswith("_color") or name.endswith("_hex"):
        return "color"

    # Column headers always string — check before generic "_year" rule
    if "_col_" in name or name.startswith("col_"):
        return "string"

    # Plain numeric values (never a formatted KPI string)
    if name in ("page_number", "section_number", "report_year"):
        return "number"
    if (
        name.endswith("_year")
        or name.endswith("_count")
        or name.endswith("_num")
        or name.endswith("_number")
        or name.endswith("_height")
        or name.endswith("_width")
        or name.endswith("_px")
    ):
        return "number"

    # Loop iterables (explicit suffixes)
    if (
        name.endswith("_items")
        or name.endswith("_list")
        or name.endswith("_rows")
        or name.endswith("_nodes")
        or name.endswith("_tabs")
        or name.endswith("_bars")
        or name.endswith("_cards")
        or name.endswith("_entries")
        or name.endswith("_tags")
        or name.endswith("_cols")
        or name.endswith("_points")
        or name.endswith("_sections")
        or name.endswith("_metrics")
    ):
        return "list"

    return "string"
