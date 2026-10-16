# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Template index: load spec JSONs and select templates by page_role / industry.

A TemplateEntry is only indexed if its corresponding HTML file exists in the
templates directory.  This allows the index to work correctly whether 5 or 100
HTML files have been generated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Canonical document-flow roles
# ---------------------------------------------------------------------------

# Typical first pages for a multi-page document
_OPENER_ROLES = [
    "cover_page",
    "executive_summary",
    "table_of_contents",
    "letter_from_ceo",
]

# Body pages that are cycled through
_BODY_ROLES = [
    "data_narrative",
    "kpi_dashboard",
    "financial_highlights",
    "product_showcase",
    "timeline_page",
    "section_divider",
    "risk_matrix",
    "appendix_data",
    "comparison_table",
    "case_study",
    "infographic_story",
    "map_or_geographic",
    "methodology_page",
    "process_flow",
    "regulatory_disclosure",
    "team_or_org_chart",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TemplateEntry:
    """One indexable template (spec + HTML file)."""

    template_id: str
    html_path: Path
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def page_role(self) -> str:
        return self.spec.get("page_role", "")

    @property
    def industry(self) -> str:
        return self.spec.get("industry", "")

    @property
    def layout(self) -> str:
        return self.spec.get("layout", {}).get("name", "")

    @property
    def density(self) -> str:
        return self.spec.get("density", "")


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class TemplateIndex:
    """Index of available Jinja2 HTML templates.

    Usage::

        idx = TemplateIndex(specs_dir, templates_dir)
        entry = idx.select("kpi_dashboard", industry="financial_services_wealth", rng=rng)
        pages  = idx.select_sequence(n_pages=6, industry="...", rng=rng)
    """

    def __init__(self, specs_dir: Path, templates_dir: Path) -> None:
        self._entries: list[TemplateEntry] = []
        self._by_role: dict[str, list[TemplateEntry]] = {}
        self._by_industry: dict[str, list[TemplateEntry]] = {}
        self._load(specs_dir, templates_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, specs_dir: Path, templates_dir: Path) -> None:
        for spec_file in sorted(specs_dir.glob("*.json")):
            try:
                spec = json.loads(spec_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            tmpl_id: str = spec.get("id", spec_file.stem)
            html_path = templates_dir / f"{tmpl_id}.html"
            if not html_path.exists():
                continue   # HTML not yet generated — skip

            entry = TemplateEntry(
                template_id=tmpl_id,
                html_path=html_path,
                spec=spec,
            )
            self._entries.append(entry)
            self._by_role.setdefault(entry.page_role, []).append(entry)
            self._by_industry.setdefault(entry.industry, []).append(entry)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def select(
        self,
        page_role: str,
        industry: str | None = None,
        rng: np.random.Generator | None = None,
        exclude_ids: set[str] | None = None,
    ) -> TemplateEntry | None:
        """Pick one template matching *page_role* (preferring *industry*).

        Priority order:
          1. page_role match + industry match
          2. page_role match (any industry)
          3. Any available template (random fallback)
        """
        if rng is None:
            rng = np.random.default_rng()
        excl = exclude_ids or set()

        # Candidates with matching role
        role_cands = [e for e in self._by_role.get(page_role, []) if e.template_id not in excl]

        if role_cands and industry:
            ind_cands = [e for e in role_cands if e.industry == industry]
            if ind_cands:
                role_cands = ind_cands

        if role_cands:
            return role_cands[int(rng.integers(0, len(role_cands)))]

        # Fallback: any template not already used
        fallback = [e for e in self._entries if e.template_id not in excl]
        if not fallback:
            fallback = list(self._entries)
        if not fallback:
            return None
        return fallback[int(rng.integers(0, len(fallback)))]

    def select_sequence(
        self,
        n_pages: int,
        industry: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[TemplateEntry]:
        """Return an ordered list of *n_pages* templates for one document.

        The sequence follows a logical document flow:
          • Page 1:    opener role (cover_page / executive_summary)
          • Pages 2-N: body roles cycled in shuffled order
        """
        if rng is None:
            rng = np.random.default_rng()

        if n_pages <= 0:
            return []

        # Build role sequence — always pick a random opener first, then cycle
        # through shuffled body roles.  No special-casing for n_pages=1/2 so
        # even single-page requests get a randomly chosen role instead of
        # always "cover_page".
        opener = _OPENER_ROLES[int(rng.integers(0, len(_OPENER_ROLES)))]
        roles: list[str] = [opener]
        body = list(_BODY_ROLES)
        rng.shuffle(body)
        for i in range(n_pages - 1):
            roles.append(body[i % len(body)])

        used: set[str] = set()
        result: list[TemplateEntry] = []
        for role in roles:
            entry = self.select(role, industry=industry, rng=rng, exclude_ids=used)
            if entry is not None:
                result.append(entry)
                used.add(entry.template_id)

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def available_roles(self) -> list[str]:
        return sorted(self._by_role)

    @property
    def available_industries(self) -> list[str]:
        return sorted(self._by_industry)
