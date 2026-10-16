# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Complex table variant generators.

Produces HTML tables with advanced structural features that stress-test
document extraction systems:

  LLM-driven (default when client available):
    llm_jagged         – LLM designs the full merged-cell structure; renderer
                         applies rowspan/colspan exactly as specified; QA is
                         derived from the LLM-supplied merge_facts.

  Static fallbacks (used when no LLM client is provided):
    merged_cells       – rowspan groups rows by category (department, region…)
    multi_row_headers  – two-tier column headers using colspan grouping
    column_groups      – three-level column headers (channel > geo > metric)
    group_summary_rows – inline subtotal rows embedded per data group
    nested_table       – mini-tables inside data cells
    footnoted_table    – cells with <sup> markers + footnote block below
    matrix_table       – symmetric correlation/heatmap with colour coding
    ragged_colspan     – section-break rows spanning all columns mid-table
    stacked_comparison – side-by-side original vs. revised values
    transposed_metrics – metrics-as-rows, time-periods-as-columns (pivoted)
"""
from __future__ import annotations

import json as _json
import re
import uuid
from typing import Any

import numpy as np

from synthpdf.rng import fork_rng
from synthpdf.types import Variant
from synthpdf.variants.tables import _esc, _fmt_cell, _parse_numeric, _derived_answers

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

_TA = (
    'border="0" cellspacing="0" cellpadding="0" '
    'width="100%" style="border-collapse:collapse; table-layout:fixed; '
    'font-size:7.5pt; word-wrap:break-word;"'
)
_B = (
    "font-size:7.5pt; padding:4px 6px; vertical-align:top; "
    "border:1px solid #cccccc; text-align:left; "
    "word-wrap:break-word; overflow-wrap:break-word;"
)

# Header tiers
_TH1 = _B + " background-color:#1f4e79; color:#ffffff; font-weight:bold; border-color:#1a3f63;"
_TH2 = _B + " background-color:#2e75b6; color:#ffffff; font-weight:bold; border-color:#2360a0;"
_TH3 = _B + " background-color:#dce6f1; font-weight:bold; border-color:#aaaaaa;"

# Data rows
_DE  = _B + " background-color:#f4f8ff;"
_DO  = _B + " background-color:#ffffff;"

# Special rows
_SUB = _B + " background-color:#e8f0fe; font-weight:bold; border-top:1px solid #2e75b6;"
_TOT = _B + " background-color:#dce6f1; font-weight:bold; border-top:2px solid #1f4e79;"
_SEC = _B + " background-color:#1f4e79; color:#ffffff; font-weight:bold; text-align:left; padding:6px;"
_MRG = _B + " background-color:#dce6f1; font-weight:bold; text-align:center; vertical-align:middle;"

# Nested / footnote
_IN_TA = (
    'border="0" cellspacing="0" cellpadding="0" width="100%" '
    'style="border-collapse:collapse; table-layout:fixed; font-size:7pt; word-wrap:break-word;"'
)
_IN_TH = "font-size:7pt; padding:2px 4px; border:1px solid #aaaaaa; background-color:#dce6f1; font-weight:bold; word-wrap:break-word;"
_IN_TD = "font-size:7pt; padding:2px 4px; border:1px solid #cccccc; word-wrap:break-word; overflow-wrap:break-word;"
_FN    = "font-size:7pt; color:#444444; padding:3px 0; border:none;"


def _uid() -> str:
    return str(uuid.uuid4())[:8]


def _rbg(i: int) -> str:
    return _DE if i % 2 == 0 else _DO


def _bg_attr(style: str) -> str:
    """Return ' bgcolor="#rrggbb"' if style contains background-color, else ''."""
    m = re.search(r"background-color\s*:\s*(#[0-9a-fA-F]{3,8})", style)
    return f' bgcolor="{m.group(1)}"' if m else ""


# ---------------------------------------------------------------------------
# Low-level HTML cell builders
# ---------------------------------------------------------------------------

def _th(text: str, cs: int = 1, rs: int = 1, style: str = _TH3, w: str = "") -> str:
    span = (f' colspan="{cs}"' if cs > 1 else "") + (f' rowspan="{rs}"' if rs > 1 else "")
    wa = f' width="{w}"' if w else ""
    bg = _bg_attr(style)
    return f"<th{span}{wa}{bg} style=\"{style}\">{_esc(text)}</th>"


def _td(val: Any, cs: int = 1, rs: int = 1, style: str = _DE, w: str = "") -> str:
    span = (f' colspan="{cs}"' if cs > 1 else "") + (f' rowspan="{rs}"' if rs > 1 else "")
    wa = f' width="{w}"' if w else ""
    bg = _bg_attr(style)
    return f"<td{span}{wa}{bg} style=\"{style}\">{_esc(_fmt_cell(val))}</td>"


def _td_raw(content: str, cs: int = 1, rs: int = 1, style: str = _DE) -> str:
    """Like _td but passes raw HTML content (for nested tables, superscripts…)."""
    span = (f' colspan="{cs}"' if cs > 1 else "") + (f' rowspan="{rs}"' if rs > 1 else "")
    bg = _bg_attr(style)
    return f"<td{span}{bg} style=\"{style}\">{content}</td>"


def _th_raw(content: str, cs: int = 1, rs: int = 1, style: str = _TH3) -> str:
    """Like _th but passes raw HTML content (for superscripts in header cells)."""
    span = (f' colspan="{cs}"' if cs > 1 else "") + (f' rowspan="{rs}"' if rs > 1 else "")
    bg = _bg_attr(style)
    return f"<th{span}{bg} style=\"{style}\">{content}</th>"


def _count_header_cols(headers: list) -> int:
    """Count total leaf columns by summing colspan values in the first header row."""
    if not headers:
        return 8
    first = headers[0]
    if not isinstance(first, list):
        return 8
    return max(1, sum(max(1, int(c.get("colspan", 1))) for c in first if isinstance(c, dict)))


# ---------------------------------------------------------------------------
# Type 1: Merged cells — rowspan grouping by department
# ---------------------------------------------------------------------------

def _gen_merged_cells(rng: np.random.Generator) -> tuple[str, dict]:
    """Two-level rowspan nesting: Department → Sub-team → Employee.

    Column 1 (Department) spans all employees in that dept.
    Column 2 (Sub-team) spans all employees within the sub-team.
    Both span simultaneously, creating a staircase of overlapping rowspans.
    """
    dept_structure: dict[str, dict[str, list[tuple[str, str]]]] = {
        "Engineering": {
            "Backend":   [("Backend Engineer", "L4"), ("Senior Backend Eng.", "L5"), ("Staff Engineer", "L7")],
            "Frontend":  [("Frontend Engineer", "L4"), ("Senior Frontend Eng.", "L5")],
            "DevOps":    [("Site Reliability Eng.", "L5"), ("Platform Engineer", "L6")],
        },
        "Marketing": {
            "Digital":   [("SEO Analyst", "L3"), ("Paid Media Mgr", "L4"), ("Digital Director", "L6")],
            "Brand":     [("Brand Designer", "L3"), ("Brand Strategist", "L5")],
        },
        "Finance": {
            "Accounting":[("Accountant", "L3"), ("Senior Accountant", "L4"), ("Controller", "L6")],
            "FP&A":      [("Financial Analyst", "L4"), ("Sr. FP&A Analyst", "L5"), ("Finance Director", "L7")],
        },
        "Sales": {
            "Enterprise":[("Enterprise AE", "L5"), ("Senior AE", "L6"), ("VP Sales", "L7")],
            "SMB":       [("SMB Rep", "L3"), ("Account Manager", "L4")],
            "Enablement":[("Sales Trainer", "L4"), ("Enablement Lead", "L5")],
        },
        "Operations": {
            "Logistics": [("Ops Analyst", "L3"), ("Logistics Mgr", "L5")],
            "Process":   [("Process Engineer", "L4"), ("Process Director", "L6")],
            "Compliance":[("Compliance Analyst", "L3"), ("Compliance Lead", "L5")],
        },
    }
    salary_by_level = {
        "L3": (58_000, 72_000), "L4": (78_000, 95_000),
        "L5": (100_000, 125_000), "L6": (130_000, 158_000), "L7": (165_000, 210_000),
    }
    fnames = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
              "Irene", "Jack", "Karen", "Leon", "Maya", "Noah", "Olivia", "Paul",
              "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander"]
    lnames = ["Smith", "Jones", "Williams", "Brown", "Davis", "Wilson", "Taylor",
              "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Garcia",
              "Clark", "Lewis", "Robinson", "Walker", "Hall", "Young"]

    dept_names = list(dept_structure.keys())
    rng.shuffle(dept_names)
    n_depts = int(rng.integers(3, 5))
    chosen_depts = dept_names[:n_depts]

    rows_html: list[str] = []
    logical: list[dict] = []
    all_salaries: list[int] = []
    ni = 0

    for dept in chosen_depts:
        sub_teams = dept_structure[dept]
        sub_names = list(sub_teams.keys())
        n_subs = int(rng.integers(2, len(sub_names) + 1))
        chosen_subs = sub_names[:n_subs]

        # Count total employees in this dept for outer rowspan
        dept_employees: list[tuple[str, str, str, str, int]] = []  # (sub, name, title, level, salary)
        for sub in chosen_subs:
            roles = sub_teams[sub]
            n_people = int(rng.integers(2, min(4, len(roles) + 1)))
            for k in range(n_people):
                title, level = roles[k % len(roles)]
                lo, hi = salary_by_level.get(level, (70_000, 90_000))
                salary = int(rng.integers(lo, hi))
                all_salaries.append(salary)
                name = f"{fnames[ni % len(fnames)]} {lnames[(ni + 7) % len(lnames)]}"
                ni += 1
                dept_employees.append((sub, name, title, level, salary))

        dept_rs = len(dept_employees)  # outer rowspan for Department column

        # Emit rows grouped by sub-team
        ei = 0
        for sub in chosen_subs:
            sub_people = [(n, t, lv, s) for (sb, n, t, lv, s) in dept_employees if sb == sub]
            sub_rs = len(sub_people)  # inner rowspan for Sub-team column

            for pi, (name, title, level, salary) in enumerate(sub_people):
                row_style = _rbg(ni + pi)
                cells = ""
                if ei == 0 and pi == 0:
                    # First cell of entire dept: emit both dept and sub-team cells
                    cells += _td(dept, rs=dept_rs, style=_MRG)
                    cells += _td(sub, rs=sub_rs, style=_MRG + " background-color:#dce6f1;")
                elif pi == 0:
                    # First cell of a new sub-team within dept: emit sub-team only
                    cells += _td(sub, rs=sub_rs, style=_MRG + " background-color:#dce6f1;")
                # else: dept and sub-team cells are covered by rowspan — omit both

                cells += (
                    _td(name, style=row_style)
                    + _td(title, style=row_style)
                    + _td(level, style=row_style)
                    + _td(f"${salary:,}", style=row_style)
                )
                rows_html.append(f"<tr>{cells}</tr>")
                logical.append({
                    "dept": dept, "sub_team": sub, "name": name,
                    "title": title, "level": level, "salary": salary,
                })
                ei += 1

        # Sub-total row per department
        dept_sal = sum(s for (sb, n, t, lv, s) in dept_employees)
        rows_html.append(
            "<tr>"
            + _td(f"{dept} Total", cs=2, style=_SUB)
            + _td("", style=_SUB) + _td("", style=_SUB)
            + _td(f"${dept_sal:,}", style=_SUB)
            + "</tr>"
        )

    grand_total = sum(all_salaries)
    rows_html.append(
        "<tr>"
        + _td("GRAND TOTAL", cs=2, style=_TOT)
        + _td("", style=_TOT) + _td("", style=_TOT)
        + _td(f"${grand_total:,}", style=_TOT)
        + "</tr>"
    )

    caption = "Employee Directory — Department / Sub-Team / Role"
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>"
        "<tr>"
        + _th("Department", w="14%") + _th("Sub-Team", w="14%")
        + _th("Name", w="22%") + _th("Title", w="28%")
        + _th("Level", w="8%") + _th("Annual Salary", w="14%")
        + "</tr>"
        + "\n".join(rows_html)
        + "</table>"
    )
    gt = {
        "complex_type": "merged_cells",
        "caption": caption,
        "description": (
            "Two-level rowspan nesting: Department column spans all employees in the dept; "
            "Sub-Team column spans employees within each sub-team. Both rowspans are active "
            "simultaneously, creating a staircase merge pattern across 6 columns."
        ),
        "schema": ["Department", "Sub-Team", "Name", "Title", "Level", "Annual Salary"],
        "logical_rows": logical,
        "structure": {
            "header_rows": 1, "data_rows": len(logical), "cols": 6,
            "has_rowspan": True, "has_colspan": True,
            "rowspan_levels": 2,
            "merge_groups": [
                {
                    "dept": d,
                    "n_rows": sum(1 for r in logical if r["dept"] == d),
                    "sub_teams": list({r["sub_team"] for r in logical if r["dept"] == d}),
                }
                for d in chosen_depts
            ],
        },
        "derived_answers": {
            "total_salary": grand_total,
            "avg_salary": round(grand_total / max(len(logical), 1), 2),
            "n_employees": len(logical),
            "dept_totals": {
                d: sum(r["salary"] for r in logical if r["dept"] == d)
                for d in chosen_depts
            },
        },
        "qa_hints": [
            "Which department has the highest total salary budget?",
            "How many sub-teams does the Engineering department have?",
            "What is the salary for the Staff Engineer in the Backend sub-team?",
            "Which sub-team has the most employees?",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# Type 2: Multi-row headers — two-tier with colspan
# ---------------------------------------------------------------------------

def _gen_multi_row_headers(rng: np.random.Generator) -> tuple[str, dict]:
    """Quarterly financial table with two-row header: quarter (colspan=2) over Revenue/Expenses."""
    regions = ["North America", "EMEA", "Asia Pacific", "Latin America", "Central Europe"]
    n_regions = int(rng.integers(4, 6))
    chosen_regions = regions[:n_regions]

    quarters = ["Q1", "Q2", "Q3", "Q4"]
    n_qtrs = int(rng.integers(3, 5))
    chosen_qtrs = quarters[:n_qtrs]

    n_leaf = n_qtrs * 2  # Revenue + Expenses per quarter
    n_cols = 1 + n_leaf  # Region + leaf cols

    # Header row 1: Region (rowspan=2) + Q1..Qn (colspan=2 each)
    hrow1 = "<tr>" + _th("Region", rs=2) + "".join(_th(q, cs=2, style=_TH2) for q in chosen_qtrs) + "</tr>"
    # Header row 2: Revenue, Expenses × n_qtrs
    hrow2 = "<tr>" + "".join(_th("Revenue") + _th("Expenses") for _ in chosen_qtrs) + "</tr>"

    data_rows_html: list[str] = []
    logical: list[dict] = []

    for ri, region in enumerate(chosen_regions):
        s = _rbg(ri)
        cells = [_td(region, style=s)]
        row_data: dict = {"region": region, "values": {}}
        for q in chosen_qtrs:
            rev = round(float(rng.uniform(1_000_000, 8_000_000)), -3)
            exp = round(float(rng.uniform(rev * 0.55, rev * 0.85)), -3)
            cells.append(_td(f"${rev/1e6:.2f}M", style=s))
            cells.append(_td(f"${exp/1e6:.2f}M", style=s))
            row_data["values"][q] = {"revenue": rev, "expenses": exp, "profit": rev - exp}
        data_rows_html.append("<tr>" + "".join(cells) + "</tr>")
        logical.append(row_data)

    caption = "Quarterly Financial Performance by Region"
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>"
        + hrow1 + hrow2
        + "\n".join(data_rows_html)
        + "</table>"
    )
    gt = {
        "complex_type": "multi_row_headers",
        "caption": caption,
        "description": (
            "Two-tier column headers: top row has quarter labels spanning two columns each "
            "(Revenue, Expenses). Region column spans both header rows via rowspan=2."
        ),
        "schema": {"quarters": chosen_qtrs, "leaf_cols": ["Revenue", "Expenses"]},
        "logical_rows": logical,
        "structure": {
            "header_rows": 2, "data_rows": len(logical),
            "cols": n_cols, "has_rowspan": True, "has_colspan": True,
            "header_levels": 2,
        },
        "qa_hints": [
            "Which region had the highest profit in Q2?",
            "Which quarter shows the largest expense-to-revenue ratio across all regions?",
            "Compare year-to-date revenue between EMEA and Asia Pacific.",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# Type 3: Column groups — three-level headers
# ---------------------------------------------------------------------------

def _gen_column_groups(rng: np.random.Generator) -> tuple[str, dict]:
    """Product sales table with 4-level headers: Division > Channel > Geography > Metric.

    Row 1: Product (rowspan=4) | Division A (colspan=8) | Division B (colspan=8)
    Row 2: Online (colspan=4) | Retail (colspan=4) per Division
    Row 3: Americas (colspan=2) | EMEA (colspan=2) per Channel
    Row 4: Units | Rev($K) per Geography — leaf columns
    """
    products = [
        "Enterprise Suite", "Cloud Backup", "Analytics Pro", "Security Shield",
        "Data Connector", "Mobile Platform", "DevOps Toolkit", "AI Assistant",
    ]
    n_products = int(rng.integers(4, 7))
    chosen = products[:n_products]

    # 4-level hierarchy: Division > Channel > Geography > Metric
    # Row 1: Product (rowspan=4) | Division A (colspan=8) | Division B (colspan=8)
    # Row 2: Online (colspan=4) | Retail (colspan=4)  ×2
    # Row 3: Americas (colspan=2) | EMEA (colspan=2)  ×4
    # Row 4: Units | Rev($K)  ×8  (16 leaf columns total + 1 label = 17 cols)
    divisions  = ["Core Products", "Cloud Services"]
    channels   = ["Online", "Retail"]
    geos       = ["Americas", "EMEA"]
    leaf_cols  = ["Units", "Rev($K)"]

    n_divs  = 2
    n_chs   = len(channels)   # 2
    n_geos  = len(geos)       # 2
    n_leafs = len(leaf_cols)  # 2
    # Total leaf columns per division = 2 channels × 2 geos × 2 metrics = 8
    div_span = n_chs * n_geos * n_leafs   # 8
    ch_span  = n_geos * n_leafs           # 4
    geo_span = n_leafs                    # 2

    # Header row 1
    hrow1 = (
        "<tr>"
        + _th("Product", rs=4, w="16%")
        + "".join(_th(d, cs=div_span, style=_TH1) for d in divisions)
        + "</tr>"
    )
    # Header row 2: Online/Retail for each division
    hrow2 = (
        "<tr>"
        + "".join(
            _th(ch, cs=ch_span, style=_TH2)
            for _ in divisions for ch in channels
        )
        + "</tr>"
    )
    # Header row 3: Americas/EMEA for each channel×division
    hrow3 = (
        "<tr>"
        + "".join(
            _th(geo, cs=geo_span, style=_TH3)
            for _ in divisions for _ in channels for geo in geos
        )
        + "</tr>"
    )
    # Header row 4: leaf metrics for every geo×channel×division
    n_combos = n_divs * n_chs * n_geos   # 8 combos, each with 2 leaf cols
    hrow4 = "<tr>" + (_th(leaf_cols[0]) + _th(leaf_cols[1])) * n_combos + "</tr>"

    data_rows_html: list[str] = []
    logical: list[dict] = []

    for pi, product in enumerate(chosen):
        s = _rbg(pi)
        cells = [_td(product, style=s)]
        row_vals: dict = {"product": product, "data": {}}
        for div in divisions:
            for ch in channels:
                for geo in geos:
                    units = int(rng.integers(100, 5000))
                    rev   = round(units * float(rng.uniform(0.8, 6.5)), 1)
                    cells += [_td(f"{units:,}", style=s), _td(f"${rev:,.1f}", style=s)]
                    row_vals["data"][f"{div}|{ch}|{geo}"] = {"units": units, "rev_k": rev}
        data_rows_html.append("<tr>" + "".join(cells) + "</tr>")
        logical.append(row_vals)

    # Totals row
    total_cells = [_td("TOTALS", style=_TOT)]
    for div in divisions:
        for ch in channels:
            for geo in geos:
                t_units = sum(r["data"][f"{div}|{ch}|{geo}"]["units"] for r in logical)
                t_rev   = sum(r["data"][f"{div}|{ch}|{geo}"]["rev_k"]  for r in logical)
                total_cells += [_td(f"{t_units:,}", style=_TOT), _td(f"${t_rev:,.1f}", style=_TOT)]
    data_rows_html.append("<tr>" + "".join(total_cells) + "</tr>")

    n_cols = 1 + n_divs * n_chs * n_geos * n_leafs
    caption = "Product Sales — Division / Channel / Geography / Metric (4-Level Header)"
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>"
        + hrow1 + hrow2 + hrow3 + hrow4
        + "\n".join(data_rows_html)
        + "</table>"
    )
    gt = {
        "complex_type": "column_groups",
        "caption": caption,
        "description": (
            "Four-level column header hierarchy: Division > Channel > Geography > Metric. "
            "Product column uses rowspan=4 to span all header rows. "
            f"Total of {n_cols} columns ({n_divs} divisions × {n_chs} channels "
            f"× {n_geos} geographies × {n_leafs} metrics + 1 label)."
        ),
        "schema": {
            "divisions": divisions, "channels": channels, "geos": geos, "leaf_cols": leaf_cols,
        },
        "logical_rows": logical,
        "structure": {
            "header_rows": 4, "data_rows": len(logical), "cols": n_cols,
            "has_rowspan": True, "has_colspan": True, "header_levels": 4,
        },
        "qa_hints": [
            "Which product has the highest online revenue in the Core Products division for Americas?",
            "How many leaf metric columns does each Geography sub-header span?",
            "Compare total retail units between Americas and EMEA across both divisions.",
            "Which Channel-Geography combination yields the best revenue per unit overall?",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# Type 4: Group summary rows — inline subtotals per category
# ---------------------------------------------------------------------------

def _gen_group_summary_rows(rng: np.random.Generator) -> tuple[str, dict]:
    """Budget table with section-header rows and in-line subtotals per cost category."""
    cost_groups: dict[str, list[str]] = {
        "Technology":    ["Cloud Infrastructure", "Software Licenses", "Hardware Refresh", "IT Security Tools"],
        "Personnel":     ["Engineering Salaries", "Benefits & Insurance", "Recruiting", "Training & Development"],
        "Facilities":    ["Office Rent", "Utilities", "Maintenance & Repairs", "Cleaning Services"],
        "Marketing":     ["Digital Advertising", "Events & Trade Shows", "Content Production", "PR & Communications"],
    }
    n_groups = int(rng.integers(3, 5))
    chosen_groups = list(cost_groups.keys())[:n_groups]

    n_cols = 5
    header = (
        "<tr>"
        + _th("Line Item") + _th("Q1 Budget") + _th("Q1 Actual") + _th("Variance") + _th("% Var")
        + "</tr>"
    )

    rows_html: list[str] = []
    logical_groups: list[dict] = []
    grand_budget = grand_actual = 0.0

    for group in chosen_groups:
        items = cost_groups[group]
        n_items = int(rng.integers(2, min(4, len(items) + 1)))
        chosen_items = items[:n_items]

        # Section header row
        rows_html.append(
            f"<tr><td colspan=\"{n_cols}\" style=\"{_SEC}\">{_esc(group)}</td></tr>"
        )

        group_budget = group_actual = 0.0
        group_items: list[dict] = []

        for ii, item in enumerate(chosen_items):
            budget = round(float(rng.uniform(20_000, 200_000)), -2)
            actual = round(budget * float(rng.uniform(0.8, 1.25)), -2)
            var = actual - budget
            pct_var = round(var / budget * 100, 1) if budget else 0.0
            group_budget += budget
            group_actual += actual
            s = _rbg(ii)
            rows_html.append(
                "<tr>"
                + _td(item, style=s) + _td(f"${budget:,.0f}", style=s)
                + _td(f"${actual:,.0f}", style=s) + _td(f"${var:+,.0f}", style=s)
                + _td(f"{pct_var:+.1f}%", style=s)
                + "</tr>"
            )
            group_items.append({"item": item, "budget": budget, "actual": actual, "variance": var})

        # Subtotal row
        g_var = group_actual - group_budget
        g_pct = round(g_var / group_budget * 100, 1) if group_budget else 0.0
        rows_html.append(
            "<tr>"
            + _td(f"{group} Total", style=_SUB) + _td(f"${group_budget:,.0f}", style=_SUB)
            + _td(f"${group_actual:,.0f}", style=_SUB) + _td(f"${g_var:+,.0f}", style=_SUB)
            + _td(f"{g_pct:+.1f}%", style=_SUB)
            + "</tr>"
        )
        grand_budget += group_budget
        grand_actual += group_actual
        logical_groups.append({"group": group, "items": group_items,
                                "subtotal_budget": group_budget, "subtotal_actual": group_actual})

    # Grand total
    gt_var = grand_actual - grand_budget
    gt_pct = round(gt_var / grand_budget * 100, 1) if grand_budget else 0.0
    rows_html.append(
        "<tr>"
        + _td("GRAND TOTAL", style=_TOT) + _td(f"${grand_budget:,.0f}", style=_TOT)
        + _td(f"${grand_actual:,.0f}", style=_TOT) + _td(f"${gt_var:+,.0f}", style=_TOT)
        + _td(f"{gt_pct:+.1f}%", style=_TOT)
        + "</tr>"
    )

    caption = "Q1 Budget vs. Actuals by Cost Category"
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>" + header + "\n".join(rows_html) + "</table>"
    )
    gt = {
        "complex_type": "group_summary_rows",
        "caption": caption,
        "description": (
            "Data is grouped by cost category; each group is preceded by a full-width section-header "
            "row (colspan=5, dark background) and followed by a bold subtotal row. "
            "A grand-total row closes the table."
        ),
        "logical_groups": logical_groups,
        "structure": {
            "header_rows": 1, "n_groups": len(chosen_groups),
            "cols": n_cols, "has_section_headers": True, "has_subtotals": True,
        },
        "derived_answers": {
            "grand_budget": grand_budget, "grand_actual": grand_actual,
            "overall_variance": grand_actual - grand_budget,
            "overall_pct_variance": round((grand_actual - grand_budget) / grand_budget * 100, 1),
        },
        "qa_hints": [
            "Which cost category is most over budget?",
            "What is the total variance across all categories?",
            "Which line item has the largest individual overspend?",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# Type 5: Nested table — mini-table inside a cell
# ---------------------------------------------------------------------------

def _gen_nested_table(rng: np.random.Generator) -> tuple[str, dict]:
    """Project status table with TWO nested inner tables per row.

    Column 3 (Sprint History): inner table with 2-tier headers —
      Row 1: Sprint (rowspan=2) | Delivery (colspan=2) | Quality (colspan=2)
      Row 2: Velocity | Done | Bugs | Regressions
    Column 4 (Milestone Plan): inner table with rowspan merging inside data cells.
    Column 5 (Risk Log): inner table with colour-coded severity cells.
    """
    projects = [
        ("Alpha",   "Platform Modernisation",    "On Track"),
        ("Beta",    "Mobile App Redesign",        "At Risk"),
        ("Gamma",   "Data Warehouse Migration",   "On Track"),
        ("Delta",   "API Gateway Rollout",        "Delayed"),
        ("Epsilon", "Security Compliance",        "On Track"),
        ("Zeta",    "AI Model Deployment",        "At Risk"),
    ]
    n_teams = int(rng.integers(3, 5))
    chosen = projects[:n_teams]

    _ST_ON  = _B + " color:#1e8449; font-weight:bold;"
    _ST_AT  = _B + " color:#d4ac0d; font-weight:bold;"
    _ST_DEL = _B + " color:#cb4335; font-weight:bold;"
    status_styles = {"On Track": _ST_ON, "At Risk": _ST_AT, "Delayed": _ST_DEL}

    _RISK_H = _B + " background-color:#e74c3c; color:#fff; font-weight:bold;"
    _RISK_M = _B + " background-color:#f39c12; color:#fff;"
    _RISK_L = _B + " background-color:#27ae60; color:#fff;"

    # Outer header: 2 tiers
    # Row 1: Team (rs=2) | Project (rs=2) | Sprint History (cs=2, rs=1) | Status (rs=2) | Risk Log (rs=2)
    outer_h1 = (
        "<tr>"
        + _th("Team", rs=2, w="7%")
        + _th("Project", rs=2, w="18%")
        + _th("Sprint History", cs=1, w="30%", rs=2)
        + _th("Milestone Plan", w="25%", rs=2)
        + _th("Status", rs=2, w="9%")
        + _th("Risk Log", rs=2, w="11%")
        + "</tr>"
    )

    rows_html: list[str] = []
    logical: list[dict] = []

    milestone_phases = [
        ["Discovery", "Design", "Build", "Test", "Deploy"],
        ["Requirements", "Architecture", "Dev Sprint 1", "Dev Sprint 2", "UAT", "Launch"],
        ["Analysis", "Prototyping", "Implementation", "Review", "Rollout"],
    ]
    risk_pool = [
        ("Resource shortage",    "H", "CTO"),
        ("Scope creep",          "M", "PM"),
        ("Integration failure",  "H", "Eng Lead"),
        ("Timeline slip",        "M", "Delivery Mgr"),
        ("Security gap",         "H", "CISO"),
        ("Vendor delay",         "L", "Procurement"),
        ("Regulatory change",    "M", "Legal"),
        ("Performance issue",    "L", "Arch Lead"),
    ]

    for ti, (team, project, status) in enumerate(chosen):
        # --- Inner table 1: Sprint History (2-tier header) ---
        n_sprints = int(rng.integers(3, 5))
        sprints: list[dict] = []
        sprint_rows: list[str] = []
        for si in range(n_sprints):
            velocity    = int(rng.integers(28, 60))
            done        = int(rng.integers(int(velocity * 0.65), velocity + 1))
            bugs        = int(rng.integers(0, 10))
            regressions = int(rng.integers(0, 4))
            sn          = f"S{si + 1}"
            ir = _rbg(si)
            sprint_rows.append(
                f"<tr>"
                f'<td style="{_IN_TD}">{_esc(sn)}</td>'
                f'<td style="{_IN_TD}">{velocity}</td>'
                f'<td style="{_IN_TD}">{done}</td>'
                f'<td style="{_IN_TD}">{bugs}</td>'
                f'<td style="{_IN_TD}">{regressions}</td>'
                f"</tr>"
            )
            sprints.append({"sprint": sn, "velocity": velocity, "done": done,
                            "bugs": bugs, "regressions": regressions})

        sprint_inner = (
            f"<table {_IN_TA}>"
            # Header row 1: Sprint(rs=2) | Delivery(cs=2) | Quality(cs=2)
            f"<tr>"
            f'<th rowspan="2" style="{_IN_TH}">Sprint</th>'
            f'<th colspan="2" style="{_IN_TH} background-color:#2e75b6;color:#fff;">Delivery</th>'
            f'<th colspan="2" style="{_IN_TH} background-color:#1f4e79;color:#fff;">Quality</th>'
            f"</tr>"
            # Header row 2: Velocity | Done | Bugs | Regressions
            f"<tr>"
            f'<th style="{_IN_TH}">Velocity</th><th style="{_IN_TH}">Done</th>'
            f'<th style="{_IN_TH}">Bugs</th><th style="{_IN_TH}">Regress.</th>'
            f"</tr>"
            + "".join(sprint_rows)
            + "</table>"
        )

        # --- Inner table 2: Milestone Plan (rowspan merging in data) ---
        phases = milestone_phases[ti % len(milestone_phases)]
        n_phases = int(rng.integers(3, len(phases) + 1))
        chosen_phases = phases[:n_phases]
        ms_rows: list[str] = []
        for mi, phase in enumerate(chosen_phases):
            pct   = int(rng.integers(0, 101))
            rag   = rng.choice(["G", "A", "R"])  # Green/Amber/Red
            rag_color = {"G": "#27ae60", "A": "#f39c12", "R": "#e74c3c"}[rag]
            # First two phases share a "Planning" rowspan label; rest share "Execution"
            phase_grp = "Planning" if mi < 2 else "Execution"
            if mi == 0:
                ms_rows.append(
                    f"<tr>"
                    f'<td rowspan="2" style="{_IN_TD} background-color:#dce6f1;font-weight:bold;">'
                    f"{_esc(phase_grp)}</td>"
                    f'<td style="{_IN_TD}">{_esc(phase)}</td>'
                    f'<td style="{_IN_TD}">{pct}%</td>'
                    f'<td style="{_IN_TD} color:{rag_color};font-weight:bold;">{rag}</td>'
                    f"</tr>"
                )
            elif mi == 2:
                ms_rows.append(
                    f"<tr>"
                    f'<td rowspan="{n_phases - 2}" style="{_IN_TD} background-color:#dce6f1;font-weight:bold;">'
                    f"{_esc(phase_grp)}</td>"
                    f'<td style="{_IN_TD}">{_esc(phase)}</td>'
                    f'<td style="{_IN_TD}">{pct}%</td>'
                    f'<td style="{_IN_TD} color:{rag_color};font-weight:bold;">{rag}</td>'
                    f"</tr>"
                )
            elif mi == 1:
                # Row 2 of "Planning" group — omit group cell (covered by rowspan)
                ms_rows.append(
                    f"<tr>"
                    f'<td style="{_IN_TD}">{_esc(phase)}</td>'
                    f'<td style="{_IN_TD}">{pct}%</td>'
                    f'<td style="{_IN_TD} color:{rag_color};font-weight:bold;">{rag}</td>'
                    f"</tr>"
                )
            else:
                ms_rows.append(
                    f"<tr>"
                    f'<td style="{_IN_TD}">{_esc(phase)}</td>'
                    f'<td style="{_IN_TD}">{pct}%</td>'
                    f'<td style="{_IN_TD} color:{rag_color};font-weight:bold;">{rag}</td>'
                    f"</tr>"
                )

        ms_inner = (
            f"<table {_IN_TA}>"
            f"<tr>"
            f'<th style="{_IN_TH}">Group</th>'
            f'<th style="{_IN_TH}">Phase</th>'
            f'<th style="{_IN_TH}">% Done</th>'
            f'<th style="{_IN_TH}">RAG</th>'
            f"</tr>"
            + "".join(ms_rows)
            + "</table>"
        )

        # --- Inner table 3: Risk Log (colour-coded severity) ---
        n_risks = int(rng.integers(2, 4))
        risk_rows: list[str] = []
        proj_risks: list[dict] = []
        for rki in range(n_risks):
            rdesc, rsev, rown = risk_pool[(ti * 3 + rki) % len(risk_pool)]
            sev_style = _RISK_H if rsev == "H" else (_RISK_M if rsev == "M" else _RISK_L)
            risk_rows.append(
                f"<tr>"
                f'<td style="{_IN_TD}">{_esc(rdesc)}</td>'
                f'<td style="{sev_style}">{rsev}</td>'
                f'<td style="{_IN_TD}">{_esc(rown)}</td>'
                f"</tr>"
            )
            proj_risks.append({"desc": rdesc, "severity": rsev, "owner": rown})

        risk_inner = (
            f"<table {_IN_TA}>"
            f"<tr>"
            f'<th style="{_IN_TH}">Risk</th>'
            f'<th style="{_IN_TH}">Sev</th>'
            f'<th style="{_IN_TH}">Owner</th>'
            f"</tr>"
            + "".join(risk_rows)
            + "</table>"
        )

        s  = _rbg(ti)
        ss = status_styles.get(status, s)
        rows_html.append(
            "<tr>"
            + _td(team, style=s)
            + _td(project, style=s)
            + _td_raw(sprint_inner, style=s)
            + _td_raw(ms_inner, style=s)
            + _td_raw(
                f'<span style="font-weight:bold;">{_esc(status)}</span>', style=ss
            )
            + _td_raw(risk_inner, style=s)
            + "</tr>"
        )
        logical.append({
            "team": team, "project": project, "status": status,
            "sprints": sprints, "milestones": chosen_phases, "risks": proj_risks,
            "avg_velocity": round(sum(sp["velocity"] for sp in sprints) / max(len(sprints), 1), 1),
        })

    caption = "Project Status Report — Sprint, Milestones, and Risk (Multi-Nested)"
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>"
        + outer_h1
        + "\n".join(rows_html)
        + "</table>"
    )
    gt = {
        "complex_type": "nested_table",
        "caption": caption,
        "description": (
            "Each outer row contains THREE nested inner tables: "
            "(1) Sprint History with a 2-tier header (Delivery/Quality groups); "
            "(2) Milestone Plan with rowspan-merged phase-group labels; "
            "(3) Risk Log with colour-coded severity cells. "
            "Extraction must parse all three nested structures and the outer table correctly."
        ),
        "logical_rows": logical,
        "structure": {
            "header_rows": 1, "data_rows": len(logical), "outer_cols": 6,
            "has_nested": True,
            "nested_cols": ["Sprint History", "Milestone Plan", "Risk Log"],
            "sprint_schema": ["Sprint", "Velocity", "Done", "Bugs", "Regressions"],
            "milestone_schema": ["Group", "Phase", "% Done", "RAG"],
            "risk_schema": ["Risk", "Severity", "Owner"],
        },
        "qa_hints": [
            "Which team has the highest average sprint velocity?",
            "How many regressions were reported in Sprint S2 for the Beta project?",
            "What percentage complete is the 'Build' phase for the Alpha project?",
            "How many high-severity risks does the Delta project have?",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# Type 6: Footnoted table — superscript markers + footnote block
# ---------------------------------------------------------------------------

def _gen_footnoted_table(rng: np.random.Generator, content_labels: dict | None = None) -> tuple[str, dict]:
    """Annual financial metrics with footnote markers + one footnote embeds a reconciliation mini-table.

    Some cells carry two footnote markers (double-annotation). One footnote entry
    replaces plain text with a mini reconciliation table showing the adjustment breakdown.
    """
    years = [2021, 2022, 2023, 2024]
    metrics_defs = [
        ("Revenue",              "M",  (3.0,  12.0), 0),   # 0 = not footnoted
        ("Gross Profit",         "M",  (1.0,   5.0), 0),
        ("EBITDA",               "M",  (0.5,   3.0), 1),   # 1 = single footnote
        ("Net Income",           "M",  (-0.5,  2.5), 2),   # 2 = double footnote (two markers)
        ("EPS",                  "$",  (0.1,   4.5), 0),
        ("R&D Spend",            "M",  (0.3,   2.0), 1),
        ("Capex",                "M",  (0.5,   4.0), 1),
        ("Headcount",            "",   (200,  2000), 0),
        ("Revenue per FTE",      "K",  (80,    250), 0),
        ("Effective Tax Rate",   "%",  (18.0,  32.0), 1),
    ]
    n_metrics = int(rng.integers(6, 10))
    chosen_metrics = metrics_defs[:n_metrics]

    footnote_texts = [
        "Restated to reflect adoption of IFRS 16 lease accounting standard.",
        "Excludes one-time restructuring charges and impairment write-downs.",
        "Adjusted to exclude acquisition-related intangible amortisation.",
        "Includes estimated tax benefit from R&D credit programme.",
        "Pro-forma figure; excludes contribution from divested business unit.",
        "Subject to external audit review; figures are preliminary.",
    ]
    # ── LLM content injection ─────────────────────────────────────────────
    if content_labels:
        _names  = _cl(content_labels, "metric_names", [])
        _ftexts = _cl(content_labels, "footnote_texts", [])
        if _names:
            metrics_defs = [
                (_names[i] if i < len(_names) else nm, unit, rng_tuple, fnt)
                for i, (nm, unit, rng_tuple, fnt) in enumerate(metrics_defs)
            ]
            chosen_metrics = metrics_defs[:n_metrics]
        if _ftexts:
            footnote_texts = list(_ftexts)

    # Assign footnote numbers; track which metrics get double-markers
    footnote_map: dict[str, list[int]] = {}  # metric_name → [fn_num, ...]
    fn_counter = 1
    reconciliation_fn: int | None = None  # which footnote gets the mini-table

    for name, _, _, fn_type in chosen_metrics:
        if fn_type == 0:
            continue
        if fn_type >= 1 and rng.random() > 0.25:
            nums = [fn_counter]
            fn_counter += 1
            if fn_type == 2 and rng.random() > 0.4:
                nums.append(fn_counter)
                fn_counter += 1
            footnote_map[name] = nums
            # Reserve the first footnote with two numbers for the mini-table footnote
            if reconciliation_fn is None and len(nums) >= 1:
                reconciliation_fn = nums[0]

    # Header: 3 rows — Metric (rs=2), Category (rs=2), then Year sub-headers with colspan=2 each
    # (Reported + Adjusted per year)
    n_cols = 2 + len(years) * 2   # Metric + Category + 2 cols per year
    hrow1 = (
        "<tr>"
        + _th("Metric", rs=2, w="18%")
        + _th("Category", rs=2, w="12%")
        + "".join(_th(str(y), cs=2, style=_TH2) for y in years)
        + "</tr>"
    )
    hrow2 = "<tr>" + (_th("Reported") + _th("Adjusted")) * len(years) + "</tr>"

    categories = {
        "Revenue": "P&L", "Gross Profit": "P&L", "EBITDA": "P&L",
        "Net Income": "P&L", "EPS": "P&L",
        "R&D Spend": "Investment", "Capex": "Investment",
        "Headcount": "Operations", "Revenue per FTE": "Operations",
        "Effective Tax Rate": "Tax",
    }

    rows_html: list[str] = []
    logical: list[dict] = []

    for mi, (name, unit, (lo, hi), _) in enumerate(chosen_metrics):
        fn_nums = footnote_map.get(name, [])
        values_rep: list[float] = []
        values_adj: list[float] = []
        for _ in years:
            rep = round(float(rng.uniform(lo, hi)), 2)
            adj = round(rep * float(rng.uniform(0.88, 1.12)), 2)
            values_rep.append(rep)
            values_adj.append(adj)

        s = _rbg(mi)
        cat = categories.get(name, "Other")

        # Build superscript markers
        markers = "".join(
            f"<sup style='font-size:6pt;vertical-align:super;'>{n}</sup>"
            for n in fn_nums
        )
        label_html = f"{_esc(name)}{markers}"
        cells = [_td_raw(label_html, style=s), _td(cat, style=s)]

        def _fmt(v: float, u: str) -> str:
            if u == "M":   return f"${v:.1f}M"
            if u == "K":   return f"${v:.0f}K"
            if u == "$":   return f"${v:.2f}"
            if u == "%":   return f"{v:.1f}%"
            return f"{int(v):,}"

        for rep, adj in zip(values_rep, values_adj):
            cells.append(_td_raw(_esc(_fmt(rep, unit)), style=s))
            cells.append(_td_raw(_esc(_fmt(adj, unit)), style=s))
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
        logical.append({
            "metric": name, "unit": unit, "category": cat,
            "reported": dict(zip(years, values_rep)),
            "adjusted": dict(zip(years, values_adj)),
            "footnotes": fn_nums,
        })

    # Build footnote block; one footnote (reconciliation_fn) gets a mini-table
    fn_lines: list[str] = []
    all_fns: dict[int, str] = {}
    for name, fn_nums in footnote_map.items():
        for i, fn_num in enumerate(fn_nums):
            all_fns[fn_num] = footnote_texts[(fn_num - 1) % len(footnote_texts)]

    for fn_num in sorted(all_fns):
        fn_text = all_fns[fn_num]
        sup = f"<sup style='font-size:6pt;vertical-align:super;'>{fn_num}</sup>"
        if fn_num == reconciliation_fn:
            # Embed a mini reconciliation table inside this footnote
            recon_items = [
                ("Reported GAAP figure",          "+$2.8M"),
                ("Less: restructuring charges",   "-$0.6M"),
                ("Less: intangible amortisation", "-$0.3M"),
                ("Add: tax benefit (R&D credits)", "+$0.2M"),
                ("Adjusted figure",               "+$2.1M"),
            ]
            rcon_rows = "".join(
                f"<tr>"
                f'<td style="{_IN_TD}">{_esc(item)}</td>'
                f'<td style="{_IN_TD} text-align:right;">{_esc(val)}</td>'
                f"</tr>"
                for item, val in recon_items
            )
            mini_tbl = (
                f"<table {_IN_TA}>"
                f"<tr>"
                f'<th style="{_IN_TH}" colspan="2">Reconciliation of Reported to Adjusted</th>'
                f"</tr>"
                + rcon_rows
                + "</table>"
            )
            fn_lines.append(f"{sup}&nbsp;{_esc(fn_text)} See reconciliation below:{mini_tbl}")
        else:
            fn_lines.append(f"{sup}&nbsp;{_esc(fn_text)}")

    footnote_block = ""
    if fn_lines:
        footnote_block = f"<p style=\"{_FN}\">" + "<br/>".join(fn_lines) + "</p>"

    caption = "Annual Financial & Operational Metrics — Reported vs. Adjusted"
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>"
        + hrow1 + hrow2
        + "\n".join(rows_html)
        + "</table>"
        + footnote_block
    )
    gt = {
        "complex_type": "footnoted_table",
        "caption": caption,
        "description": (
            "Two-tier column headers: year (colspan=2) over Reported/Adjusted columns. "
            "Some metric rows have single or double superscript footnote markers. "
            f"Footnote {reconciliation_fn} embeds a reconciliation mini-table in the footnote block."
        ),
        "schema": ["Metric", "Category"] + [f"{y} Rep." for y in years] + [f"{y} Adj." for y in years],
        "logical_rows": logical,
        "footnotes": {str(fn): txt for fn, txt in all_fns.items()},
        "reconciliation_footnote": reconciliation_fn,
        "structure": {
            "header_rows": 2, "data_rows": len(logical), "cols": n_cols,
            "has_footnotes": True, "n_footnotes": len(all_fns),
            "has_double_markers": any(len(v) > 1 for v in footnote_map.values()),
            "has_footnote_table": True,
        },
        "qa_hints": [
            "Which metric carries two footnote markers?",
            f"What does footnote {reconciliation_fn} explain, and what is in the reconciliation table?",
            "What is the adjusted EBITDA for 2023?",
            "Which metric has the largest gap between reported and adjusted values?",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# Type 7: Matrix table — symmetric correlation/heatmap
# ---------------------------------------------------------------------------

def _gen_matrix_table(rng: np.random.Generator) -> tuple[str, dict]:
    """Symmetric feature-correlation matrix with colour-coded cell backgrounds."""
    feature_pools = [
        ["Revenue Growth", "Customer Acquisition Cost", "Churn Rate", "NPS Score", "Headcount", "R&D Spend"],
        ["Inflation Rate", "Interest Rate", "GDP Growth", "Unemployment", "Consumer Confidence", "PMI Index"],
        ["CPU Utilisation", "Memory Usage", "Disk I/O", "Network Latency", "Error Rate", "Request Volume"],
    ]
    pool = feature_pools[int(rng.integers(0, len(feature_pools)))]
    n = int(rng.integers(4, 7))
    features = pool[:n]

    # Generate a symmetric correlation matrix
    raw = rng.uniform(-0.9, 0.9, (n, n))
    # Make symmetric
    corr = (raw + raw.T) / 2
    # Set diagonal to 1.0
    for i in range(n):
        corr[i, i] = 1.0
    # Clip to [-1, 1]
    corr = corr.clip(-1.0, 1.0)

    def _cell_color(r: float) -> tuple[str, str]:
        """Return (bg_hex, text_hex) based on correlation value."""
        a = abs(r)
        if r == 1.0:
            return "#1a5276", "#ffffff"
        elif r > 0.7:
            return "#1a5276", "#ffffff"
        elif r > 0.4:
            return "#5dade2", "#ffffff"
        elif r > 0.1:
            return "#d6eaf8", "#000000"
        elif r > -0.1:
            return "#ffffff", "#555555"
        elif r > -0.4:
            return "#fadbd8", "#000000"
        elif r > -0.7:
            return "#e74c3c", "#ffffff"
        else:
            return "#922b21", "#ffffff"

    # Blank-cell style for the top-left empty header cell
    _CORNER = _B + " background-color:#f0f0f0;"

    rows_html: list[str] = []
    matrix: list[list[float]] = []

    for i, feat_i in enumerate(features):
        cells = [_th(feat_i, style=_TH3)]
        row_vals: list[float] = []
        for j, _ in enumerate(features):
            r_val = round(float(corr[i, j]), 2)
            row_vals.append(r_val)
            bg, fg = _cell_color(r_val)
            cell_style = _B + f" background-color:{bg}; color:{fg}; text-align:center;"
            cells.append(_td_raw(f"{r_val:.2f}", style=cell_style))
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
        matrix.append(row_vals)

    # Top-left empty corner + feature headers
    header = (
        "<tr>"
        + _td_raw("", style=_CORNER)
        + "".join(_th(f, style=_TH3) for f in features)
        + "</tr>"
    )

    caption = "Feature Correlation Matrix"
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>"
        + header + "\n".join(rows_html)
        + "</table>"
        + f"<p style=\"{_FN}\">Colour scale: dark blue = strong positive, red = strong negative, white = no correlation.</p>"
    )
    gt = {
        "complex_type": "matrix_table",
        "caption": caption,
        "description": (
            "Symmetric correlation matrix. Diagonal = 1.00. Cell background colour encodes "
            "correlation strength (dark blue for high positive, red for strong negative). "
            "Both row and column headers are feature names."
        ),
        "features": features,
        "matrix": matrix,
        "structure": {
            "header_rows": 1, "data_rows": n, "cols": n + 1,
            "is_symmetric": True, "diagonal_value": 1.0,
        },
        "derived_answers": {
            "strongest_positive": max(
                ((i, j, matrix[i][j]) for i in range(n) for j in range(n) if i != j),
                key=lambda x: x[2]
            ),
            "strongest_negative": min(
                ((i, j, matrix[i][j]) for i in range(n) for j in range(n) if i != j),
                key=lambda x: x[2]
            ),
        },
        "qa_hints": [
            "Which pair of features shows the strongest positive correlation?",
            "Are any features negatively correlated with all others?",
            "What is the average off-diagonal correlation value?",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# Type 8: Ragged colspan — section-break rows spanning all columns mid-table
# ---------------------------------------------------------------------------

def _gen_ragged_colspan(rng: np.random.Generator, content_labels: dict | None = None) -> tuple[str, dict]:
    """Risk register with full-width section headers dividing risk categories."""
    risk_categories = {
        "Operational Risks": [
            ("Supply chain disruption", "Medium", "High", 12),
            ("Key personnel departure", "Low",    "High",  8),
            ("IT system outage",        "Medium", "High", 12),
            ("Process compliance failure","Low",  "Medium", 4),
            ("Data quality issues",     "Medium", "Medium", 9),
        ],
        "Financial Risks": [
            ("Foreign exchange exposure","High",  "High", 16),
            ("Credit counterparty default","Low", "Critical", 10),
            ("Budget overrun",          "Medium", "Medium",   9),
            ("Liquidity shortfall",     "Low",    "High",     8),
        ],
        "Strategic Risks": [
            ("Competitor product launch","High",  "High",    16),
            ("Regulatory change",       "Medium", "High",    12),
            ("Market demand shift",     "Medium", "Medium",   9),
            ("M&A integration failure", "Low",    "Critical",  10),
        ],
        "Technology Risks": [
            ("Cybersecurity breach",    "Medium", "Critical", 15),
            ("AI model drift",          "Low",    "High",      8),
            ("Cloud dependency failure","Medium", "High",     12),
            ("Legacy system failure",   "Medium", "Medium",    9),
        ],
    }
    # ── LLM content injection ─────────────────────────────────────────────
    if content_labels:
        _cnames = _cl(content_labels, "risk_category_names", [])
        _descs  = _cl(content_labels, "risk_descriptions", [])
        if _cnames or _descs:
            old_entries = list(risk_categories.items())
            risk_categories = {}
            desc_idx = 0
            for ci, (old_cat, old_risks) in enumerate(old_entries):
                new_cat = _cnames[ci] if ci < len(_cnames) else old_cat
                new_risks = []
                for desc, prob, impact, score in old_risks:
                    new_desc = _descs[desc_idx] if desc_idx < len(_descs) else desc
                    desc_idx += 1
                    new_risks.append((new_desc, prob, impact, score))
                risk_categories[new_cat] = new_risks
    n_cats = int(rng.integers(3, 5))
    cat_names = list(risk_categories.keys())[:n_cats]

    n_cols = 6
    header = (
        "<tr>"
        + _th("Risk ID") + _th("Description") + _th("Probability")
        + _th("Impact") + _th("Risk Score") + _th("Owner")
        + "</tr>"
    )

    owners = ["Risk Mgmt", "CTO", "CFO", "CISO", "COO", "Legal", "VP Eng"]
    rows_html: list[str] = []
    logical_cats: list[dict] = []
    risk_counter = 1

    for cat in cat_names:
        cat_short = cat.split()[0][:3].upper()
        rows_html.append(
            "<tr>" + _td(cat, cs=n_cols, style=_SEC) + "</tr>"
        )
        risks = risk_categories[cat]
        n_risks = int(rng.integers(2, min(4, len(risks) + 1)))
        cat_risks: list[dict] = []

        for ri in range(n_risks):
            desc, prob, impact, score_base = risks[ri]
            score = score_base + int(rng.integers(-2, 3))
            owner = owners[int(rng.integers(0, len(owners)))]
            risk_id = f"{cat_short}-{risk_counter:03d}"
            risk_counter += 1
            s = _rbg(ri)
            rows_html.append(
                "<tr>"
                + _td(risk_id, style=s) + _td(desc, style=s)
                + _td(prob, style=s) + _td(impact, style=s)
                + _td(score, style=s) + _td(owner, style=s)
                + "</tr>"
            )
            cat_risks.append({"id": risk_id, "description": desc, "probability": prob,
                               "impact": impact, "score": score, "owner": owner})

        logical_cats.append({"category": cat, "risks": cat_risks})

    caption = "Enterprise Risk Register"
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>" + header + "\n".join(rows_html) + "</table>"
    )
    gt = {
        "complex_type": "ragged_colspan",
        "caption": caption,
        "description": (
            "Risk register divided into categories by full-width section-header rows "
            "(colspan=6, dark background). Each section contains its own risk entries."
        ),
        "logical_categories": logical_cats,
        "structure": {
            "header_rows": 1, "cols": n_cols, "n_categories": len(cat_names),
            "has_section_headers": True, "colspan_full_width": n_cols,
        },
        "derived_answers": {
            "total_risks": sum(len(c["risks"]) for c in logical_cats),
            "avg_score": round(
                sum(r["score"] for c in logical_cats for r in c["risks"])
                / max(1, sum(len(c["risks"]) for c in logical_cats)), 1
            ),
            "max_score": max((r["score"] for c in logical_cats for r in c["risks"]), default=0),
        },
        "qa_hints": [
            "Which risk has the highest risk score and what category is it in?",
            "How many risks are rated as High or Critical impact?",
            "Which category has the most risks and what is their average score?",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# Type 9: Stacked comparison — original vs. revised with amendment markers
# ---------------------------------------------------------------------------

def _gen_stacked_comparison(rng: np.random.Generator, content_labels: dict | None = None) -> tuple[str, dict]:
    """3-version × 4-period stacked comparison with category sections and delta column group.

    Header structure (3 rows):
      Row 1: Line Item (rs=3) | Baseline (cs=4) | Q1-Revised (cs=4) | Q2-Revised (cs=4) | Actuals (cs=4) | Delta vs Baseline (cs=2, rs=2 for the sub-header)
      Row 2: Q1 Q2 Q3 Q4 repeated per version
      Row 3: $000 unit labels
    """
    cost_sections: dict[str, list[str]] = {
        "Revenue": [
            "Core Product Sales", "Cloud Subscriptions", "Services & Consulting", "Partner Channel",
        ],
        "Operating Costs": [
            "Engineering & R&D", "Sales & Marketing", "Customer Success", "G&A",
        ],
        "Capital Expenditure": [
            "Infrastructure", "Hardware Refresh", "Software Licensing",
        ],
    }
    # ── LLM content injection ─────────────────────────────────────────────
    if content_labels:
        _cats  = _cl(content_labels, "cost_categories", [])
        _items = _cl(content_labels, "cost_line_items", [])
        if _cats or _items:
            old_sections = list(cost_sections.items())
            cost_sections = {}
            item_idx = 0
            for si, (old_cat, old_lines) in enumerate(old_sections):
                new_cat = _cats[si] if si < len(_cats) else old_cat
                new_lines = [
                    _items[item_idx + li] if (item_idx + li) < len(_items) else old_lines[li]
                    for li in range(len(old_lines))
                ]
                item_idx += len(old_lines)
                cost_sections[new_cat] = new_lines
    n_sections = int(rng.integers(2, len(cost_sections) + 1))
    chosen_sections = list(cost_sections.keys())[:n_sections]

    versions = ["Baseline", "Q1-Revised", "Q2-Revised", "Actuals"]
    periods  = ["Q1", "Q2", "Q3", "Q4"]
    n_v = len(versions)
    n_p = len(periods)

    # Styles per version (background tones)
    _V_STYLES = [
        _B + " background-color:#fef9e7;",   # pale yellow  = Baseline
        _B + " background-color:#ebf5fb;",   # pale blue    = Q1-Revised
        _B + " background-color:#eafaf1;",   # pale green   = Q2-Revised
        _B + " background-color:#f5eef8;",   # pale purple  = Actuals
    ]
    _V_HDR_STYLES = [_TH2, _TH3, _TH2, _TH1]
    _CHG_POS = _B + " background-color:#eafaf1; color:#1e8449; font-weight:bold;"
    _CHG_NEG = _B + " background-color:#fdedec; color:#cb4335; font-weight:bold;"

    # Header row 1: Line Item (rs=3) | version groups (cs=4 each) | Delta (cs=2)
    hrow1 = (
        "<tr>"
        + _th("Line Item", rs=3, w="20%")
        + "".join(_th(v, cs=n_p, style=_V_HDR_STYLES[vi]) for vi, v in enumerate(versions))
        + _th("Delta vs Baseline", cs=2, style=_TH1)
        + "</tr>"
    )
    # Header row 2: Q1 Q2 Q3 Q4 × 4 versions + Net | % headers (rs=1)
    hrow2 = (
        "<tr>"
        + "".join(_th(p, style=_V_HDR_STYLES[vi]) for vi in range(n_v) for p in periods)
        + _th("Net ($000)") + _th("Chg %")
        + "</tr>"
    )
    # Header row 3: $000 labels
    hrow3 = (
        "<tr>"
        + (_th("$000") * (n_v * n_p))
        + _th("") + _th("")
        + "</tr>"
    )

    rows_html: list[str] = []
    logical_sections: list[dict] = []

    for section in chosen_sections:
        items = cost_sections[section]
        n_items = int(rng.integers(2, min(4, len(items) + 1)))
        chosen_items = items[:n_items]

        # Section break row
        n_data_cols = n_v * n_p + 2
        rows_html.append(
            f"<tr><td colspan=\"{1 + n_data_cols}\" style=\"{_SEC}\">{_esc(section)}</td></tr>"
        )

        sec_values: dict[str, dict] = {}  # item → {version: {period: value}}
        sec_logical: list[dict] = []

        for ii, item in enumerate(chosen_items):
            row_vals: dict[str, dict] = {}
            s = _rbg(ii)
            cells = [_td(item, style=s)]

            base_total = 0.0
            act_total  = 0.0

            for vi, ver in enumerate(versions):
                row_vals[ver] = {}
                for pi, prd in enumerate(periods):
                    # Values drift slightly from version to version
                    base = round(float(rng.uniform(50, 500)) * 1000, -3)
                    noise = float(rng.uniform(0.85, 1.18)) if ver != "Baseline" else 1.0
                    val = round(base * noise, -3)
                    row_vals[ver][prd] = val
                    cells.append(_td(f"{val/1000:.0f}", style=_V_STYLES[vi]))
                    if ver == "Baseline":
                        base_total += val
                    if ver == "Actuals":
                        act_total += val

            net_delta = act_total - base_total
            pct_delta = round(net_delta / base_total * 100, 1) if base_total else 0.0
            d_style   = _CHG_POS if net_delta >= 0 else _CHG_NEG
            cells.append(_td(f"{net_delta/1000:+.0f}", style=d_style))
            cells.append(_td(f"{pct_delta:+.1f}%", style=d_style))

            rows_html.append("<tr>" + "".join(cells) + "</tr>")
            sec_values[item] = row_vals
            sec_logical.append({"item": item, "versions": row_vals,
                                 "net_delta": net_delta, "pct_delta": pct_delta})

        # Section subtotal
        sub_cells = [_td(f"{section} Total", cs=1, style=_SUB)]
        for vi, ver in enumerate(versions):
            for prd in periods:
                sub_cells.append(_td(
                    f"{sum(sec_values[it][ver][prd] for it in chosen_items)/1000:.0f}",
                    style=_SUB
                ))
        sub_base = sum(sec_values[it]["Baseline"][p] for it in chosen_items for p in periods)
        sub_act  = sum(sec_values[it]["Actuals"][p]  for it in chosen_items for p in periods)
        sub_net  = sub_act - sub_base
        sub_pct  = round(sub_net / sub_base * 100, 1) if sub_base else 0.0
        sub_d    = _CHG_POS if sub_net >= 0 else _CHG_NEG
        sub_cells += [_td(f"{sub_net/1000:+.0f}", style=sub_d), _td(f"{sub_pct:+.1f}%", style=sub_d)]
        rows_html.append("<tr>" + "".join(sub_cells) + "</tr>")
        logical_sections.append({"section": section, "items": sec_logical})

    caption = "Multi-Version Budget Comparison (3 Revisions × 4 Quarters)"
    legend  = (
        "Yellow = Baseline | Blue = Q1-Revised | Green = Q2-Revised | Purple = Actuals. "
        "Values in $000. Delta columns compare Actuals vs. Baseline."
    )
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>"
        + hrow1 + hrow2 + hrow3
        + "\n".join(rows_html)
        + "</table>"
        + f"<p style=\"{_FN}\">{_esc(legend)}</p>"
    )
    gt = {
        "complex_type": "stacked_comparison",
        "caption": caption,
        "description": (
            "Three-tier column headers: version groups (Baseline/Q1-Revised/Q2-Revised/Actuals) "
            "spanning period sub-columns (Q1–Q4), plus a Delta vs Baseline column group. "
            "Colour-coded by version. Section-break rows divide cost categories."
        ),
        "logical_sections": logical_sections,
        "structure": {
            "header_rows": 3, "n_sections": len(chosen_sections),
            "cols": 1 + n_v * n_p + 2, "has_colspan": True, "has_rowspan": True,
            "versions": versions, "periods": periods,
        },
        "qa_hints": [
            "What is the Actuals Q2 value for 'Engineering & R&D'?",
            "Which line item has the largest percentage delta vs. Baseline?",
            "What is the Q1-Revised Q3 total for the 'Operating Costs' section?",
            "Which version shows the highest Q4 value across all items?",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# Type 10: Transposed metrics — metrics as rows, time periods as columns
# ---------------------------------------------------------------------------

def _gen_transposed_metrics(rng: np.random.Generator, content_labels: dict | None = None) -> tuple[str, dict]:
    """KPI dashboard pivoted so each row is a metric and columns are monthly periods."""
    metric_defs: list[tuple[str, str, float, float, str]] = [
        ("Monthly Revenue",       "$M",    0.8,  5.0,  "higher_is_better"),
        ("New Customers",         "",    150,  1200,   "higher_is_better"),
        ("Churn Rate",            "%",    1.0,   8.5,  "lower_is_better"),
        ("Avg. Contract Value",   "$K",   12.0,  85.0, "higher_is_better"),
        ("Support Tickets (Open)","",      30,   450,  "lower_is_better"),
        ("NPS Score",             "",      15,    72,  "higher_is_better"),
        ("CAC",                   "$",   180,   950,   "lower_is_better"),
        ("Gross Margin",          "%",    42,    78,   "higher_is_better"),
        ("Employee Satisfaction", "/10",  6.1,   9.2,  "higher_is_better"),
        ("Feature Release Velocity","",    2,    12,   "higher_is_better"),
    ]
    # ── LLM content injection ─────────────────────────────────────────────
    if content_labels:
        _names = _cl(content_labels, "kpi_names", [])
        if _names:
            metric_defs = [
                (_names[i] if i < len(_names) else nm, unit, lo, hi, direction)
                for i, (nm, unit, lo, hi, direction) in enumerate(metric_defs)
            ]
    n_metrics = int(rng.integers(6, 10))
    chosen_metrics = metric_defs[:n_metrics]

    # Generate months
    months_all = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    n_periods = int(rng.integers(4, 9))
    start_month = int(rng.integers(0, 12 - n_periods))
    periods = months_all[start_month: start_month + n_periods]

    # Header: Metric | Unit | P1 | P2 | ... | MoM | YoY
    header = (
        "<tr>"
        + _th("Metric") + _th("Unit")
        + "".join(_th(p) for p in periods)
        + _th("MoM Δ") + _th("Trend")
        + "</tr>"
    )

    trend_chars = {"up": "▲", "down": "▼", "flat": "►"}
    rows_html: list[str] = []
    logical: list[dict] = []

    for mi, (name, unit, lo, hi, direction) in enumerate(chosen_metrics):
        # Generate time series with mild trend
        trend_dir = rng.choice(["up", "down", "flat"])
        base = float(rng.uniform(lo, hi))
        values: list[float] = []
        cur = base
        for _ in periods:
            noise = float(rng.uniform(-0.05, 0.05))
            drift = 0.02 if trend_dir == "up" else (-0.02 if trend_dir == "down" else 0.0)
            cur = round(cur * (1 + drift + noise), 2)
            values.append(cur)

        mom = round((values[-1] - values[-2]) / values[-2] * 100, 1) if len(values) >= 2 else 0.0
        s = _rbg(mi)

        # Format values by unit
        def fmt_val(v: float) -> str:
            if unit == "$M":
                return f"${v:.2f}M"
            elif unit == "$K":
                return f"${v:.1f}K"
            elif unit == "$":
                return f"${v:,.0f}"
            elif unit == "%":
                return f"{v:.1f}%"
            elif unit == "/10":
                return f"{v:.1f}"
            else:
                return f"{v:,.0f}"

        mom_style = (
            _B + " color:#1e8449; background-color:#eafaf1;" if mom >= 0 and direction == "higher_is_better"
            else _B + " color:#cb4335; background-color:#fdedec;" if mom < 0 and direction == "higher_is_better"
            else s
        )
        t_char = trend_chars[str(trend_dir)]

        row = (
            "<tr>"
            + _td(name, style=s) + _td(unit or "—", style=s)
            + "".join(_td(fmt_val(v), style=s) for v in values)
            + _td_raw(f"{mom:+.1f}%", style=mom_style)
            + _td(t_char, style=s)
            + "</tr>"
        )
        rows_html.append(row)
        logical.append({
            "metric": name, "unit": unit, "direction": direction,
            "values": dict(zip(periods, values)),
            "mom_change_pct": mom, "trend": str(trend_dir),
        })

    caption = "Monthly KPI Dashboard — Transposed View"
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>" + header + "\n".join(rows_html) + "</table>"
    )
    gt = {
        "complex_type": "transposed_metrics",
        "caption": caption,
        "description": (
            "Pivoted KPI table: each row is a metric, each column is a monthly period. "
            "Additional columns show month-over-month change and trend direction."
        ),
        "schema": ["Metric", "Unit"] + periods + ["MoM Δ", "Trend"],
        "logical_rows": logical,
        "periods": periods,
        "structure": {
            "header_rows": 1, "data_rows": len(logical),
            "cols": 2 + len(periods) + 2, "is_transposed": True,
        },
        "qa_hints": [
            "Which metric shows the strongest month-over-month growth in the latest period?",
            "Which KPIs are trending in the wrong direction relative to their goal?",
            "What is the revenue growth rate from first to last period?",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# Type 11: Extreme merge — chaotic staircase of overlapping rowspan/colspan
# ---------------------------------------------------------------------------

def _gen_extreme_merge(rng: np.random.Generator, content_labels: dict | None = None) -> tuple[str, dict]:
    """Wild multi-level merge table: staircase rowspans, section rows, colspan in data cells.

    Structure:
    - 3-tier header: super-group (colspan=6+) / sub-group (colspan=2-3) / metrics
    - 3-4 section groups, each with 2 data sub-groups
    - Sub-group label: rowspan=4 to 6 in column 1
    - Detail label:    rowspan=2 in column 2
    - Some data cells use colspan=2 for merged totals
    - Section subtotals and grand total
    """
    # Domain: product / region cost tracking
    sections = [
        ("APAC", [
            ("Enterprise",  ["Cloud Rev", "On-Prem Rev", "SaaS Rev", "Consulting"]),
            ("SMB",         ["Subscriptions", "Seats", "Add-ons",   "Support"]),
            ("Government",  ["Licenses",      "Services", "Hardware", "Training"]),
        ]),
        ("EMEA", [
            ("Enterprise",  ["Cloud Rev", "On-Prem Rev", "SaaS Rev", "Consulting"]),
            ("Partners",    ["Reseller Margin", "Referral", "Co-Sell", "Marketplace"]),
        ]),
        ("Americas", [
            ("Enterprise",  ["Cloud Rev", "On-Prem Rev", "SaaS Rev", "Consulting"]),
            ("Growth",      ["New Logo", "Expansion", "Cross-sell", "Upsell"]),
            ("Renewal",     ["Auto-renew", "Negotiated", "Multi-year", "Co-term"]),
        ]),
        ("Global", [
            ("HQ Alloc",    ["IT Infra", "HR & Benefits", "Finance Ops", "Legal"]),
            ("R&D Centre",  ["Eng Salaries", "Tools & Infra", "Patents", "Labs"]),
        ]),
    ]
    # ── LLM content injection ─────────────────────────────────────────────
    if content_labels:
        _regions = _cl(content_labels, "region_names", [])
        if _regions:
            sections = [
                (_regions[i] if i < len(_regions) else reg, subs)
                for i, (reg, subs) in enumerate(sections)
            ]
    n_sections = int(rng.integers(2, len(sections) + 1))
    chosen_sections = sections[:n_sections]

    quarters = ["Q1", "Q2", "Q3", "Q4"]
    n_quarters = int(rng.integers(3, 5))
    chosen_qs = quarters[:n_quarters]

    # --- 3-tier header ---
    # Row 1: Region (rs=3) | Quarters pair-grouped (colspan=2 each) | YTD Total (rs=3)
    n_pairs = n_quarters // 2 + (n_quarters % 2)  # ceil(n_quarters/2) pairs
    # Actually simpler: group as H1 (first half) + H2 (second half)
    h1_qs = chosen_qs[:n_quarters // 2] or chosen_qs[:1]
    h2_qs = chosen_qs[n_quarters // 2:] or chosen_qs[1:]
    grp_names = ["First Half", "Second Half"]
    grp_qs    = [h1_qs, h2_qs]

    hrow1_cells = (
        _th("Region",   rs=3, w="10%")
        + _th("Sub-Group", rs=3, w="12%")
        + _th("Line Item",  rs=3, w="15%")
        + "".join(_th(g, cs=len(q), style=_TH1) for g, q in zip(grp_names, grp_qs) if q)
        + _th("YTD Total", rs=3, w="10%")
        + _th("vs Prior", rs=3, w="8%")
    )
    hrow1 = "<tr>" + hrow1_cells + "</tr>"

    hrow2_cells = "".join(
        _th(qname, cs=1, style=_TH2)
        for grp in grp_qs if grp for qname in grp
    )
    hrow2 = "<tr>" + hrow2_cells + "</tr>"

    hrow3_cells = "".join(
        _th("$K", style=_TH3)
        for _ in range(sum(len(g) for g in grp_qs if g))
    )
    hrow3 = "<tr>" + hrow3_cells + "</tr>"

    rows_html: list[str] = []
    logical_sections: list[dict] = []
    grand_totals: dict[str, float] = {q: 0.0 for q in chosen_qs}

    for si, (region, sub_groups) in enumerate(chosen_sections):
        n_subs = int(rng.integers(2, len(sub_groups) + 1))
        chosen_subs = sub_groups[:n_subs]

        # Count total rows in this region for outer rowspan
        total_rows = sum(len(items[:int(rng.integers(3, 5))]) for _, items in chosen_subs)
        region_rowspan = total_rows
        region_emitted = False
        sec_totals: dict[str, float] = {q: 0.0 for q in chosen_qs}

        for gi, (sub_name, line_items) in enumerate(chosen_subs):
            n_lines = int(rng.integers(3, min(5, len(line_items) + 1)))
            chosen_lines = line_items[:n_lines]
            sub_rowspan = n_lines
            sub_emitted = False

            sub_totals: dict[str, float] = {q: 0.0 for q in chosen_qs}

            for li, line in enumerate(chosen_lines):
                q_vals: dict[str, float] = {}
                for q in chosen_qs:
                    v = round(float(rng.uniform(80, 1200)) * 1000, -3)
                    q_vals[q] = v
                    sub_totals[q] += v
                    sec_totals[q] += v
                    grand_totals[q] += v

                ytd   = sum(q_vals.values())
                prior = round(ytd * float(rng.uniform(0.80, 1.25)), -3)
                delta = ytd - prior

                row_style = _rbg(li)
                cells: list[str] = []

                # Region cell: large rowspan, emitted only for first row
                if not region_emitted:
                    cells.append(_td(region, rs=region_rowspan, style=_MRG))
                    region_emitted = True

                # Sub-group cell: medium rowspan
                if not sub_emitted:
                    cells.append(_td(sub_name, rs=sub_rowspan, style=_MRG + " background-color:#dce6f1;"))
                    sub_emitted = True

                cells.append(_td(line, style=row_style))
                for q in chosen_qs:
                    cells.append(_td(f"${q_vals[q]/1000:,.0f}K", style=row_style))
                cells.append(_td(f"${ytd/1000:,.0f}K", style=row_style))
                delta_s = _B + " color:#1e8449;" if delta >= 0 else _B + " color:#cb4335;"
                cells.append(_td(f"{'+' if delta >= 0 else ''}{delta/1000:,.0f}K", style=delta_s))

                rows_html.append("<tr>" + "".join(cells) + "</tr>")

            # Sub-group subtotal (colspan=2 for sub+line label merged)
            sub_ytd = sum(sub_totals.values())
            sub_cells = (
                _td(f"{sub_name} Subtotal", cs=2, style=_SUB)
                + "".join(_td(f"${sub_totals[q]/1000:,.0f}K", style=_SUB) for q in chosen_qs)
                + _td(f"${sub_ytd/1000:,.0f}K", style=_SUB)
                + _td("—", style=_SUB)
            )
            rows_html.append("<tr>" + sub_cells + "</tr>")

        # Section total (colspan=3 for region+sub+line)
        sec_ytd = sum(sec_totals.values())
        sec_cells = (
            _td(f"{region} TOTAL", cs=3, style=_TOT)
            + "".join(_td(f"${sec_totals[q]/1000:,.0f}K", style=_TOT) for q in chosen_qs)
            + _td(f"${sec_ytd/1000:,.0f}K", style=_TOT)
            + _td("—", style=_TOT)
        )
        rows_html.append("<tr>" + sec_cells + "</tr>")
        logical_sections.append({
            "region": region,
            "sub_groups": [sg for sg, _ in chosen_subs],
            "totals": sec_totals,
        })

    # Grand total
    grand_ytd = sum(grand_totals.values())
    grand_cells = (
        _td("GRAND TOTAL", cs=3, style=_TOT)
        + "".join(_td(f"${grand_totals[q]/1000:,.0f}K", style=_TOT) for q in chosen_qs)
        + _td(f"${grand_ytd/1000:,.0f}K", style=_TOT)
        + _td("—", style=_TOT)
    )
    rows_html.append("<tr>" + grand_cells + "</tr>")

    n_cols = 3 + len(chosen_qs) + 2
    caption = "Revenue & Cost Detail — Extreme Merge Layout"
    html = (
        f"<p><strong>{_esc(caption)}</strong></p>"
        f"<table {_TA}>"
        + hrow1 + hrow2 + hrow3
        + "\n".join(rows_html)
        + "</table>"
        + f'<p style="{_FN}">Region column spans all sub-group rows. Sub-group column spans all '
          f'line-item rows within the group. Subtotal rows use colspan=2 (sub+line merged).</p>'
    )
    gt = {
        "complex_type": "extreme_merge",
        "caption": caption,
        "description": (
            "Three-tier headers (Region group / Quarter group / labels). "
            "Three simultaneous rowspan levels in data: Region (large), Sub-Group (medium), "
            "plus colspan=2 in subtotal rows and colspan=3 in section total rows. "
            "Grand total at the bottom."
        ),
        "logical_sections": logical_sections,
        "structure": {
            "header_rows": 3, "cols": n_cols,
            "has_rowspan": True, "has_colspan": True,
            "rowspan_levels": 3, "max_colspan": 3,
        },
        "derived_answers": {
            "grand_total_ytd": grand_ytd,
            "by_quarter": {q: grand_totals[q] for q in chosen_qs},
        },
        "qa_hints": [
            "How many rows does the APAC region label span in the extreme merge table?",
            "What is the EMEA Enterprise Subtotal for Q2?",
            "Which region has the highest YTD total?",
            "What does the colspan=3 in the section total row span?",
        ],
    }
    return html, gt


# ---------------------------------------------------------------------------
# LLM-driven jagged table: JSON → HTML renderer + QA extractor
# ---------------------------------------------------------------------------

def _render_jagged_html(data: dict, _depth: int = 0) -> str:
    """Render an LLM-generated merged-cell table JSON to an HTML <table>.

    Supports all standard rowspan/colspan merging plus:
    - ``is_section`` on a single-cell row → full-width dark section-break row
    - ``is_subtotal`` / ``is_total`` → styled summary cells
    - ``inner_table`` in a cell → recursively rendered nested table
    - ``footnote`` (int) in a cell → superscript marker appended to cell text
    - top-level ``footnotes`` list → explanatory block rendered below the table
      (each item may also carry a ``table`` key to embed a mini-table in the note)
    """
    headers   = data.get("headers", [])
    rows      = data.get("rows", [])
    caption   = data.get("caption", "")
    footnotes = data.get("footnotes", [])

    ta = _IN_TA if _depth > 0 else _TA
    lines = [f"<table {ta}>"]

    if caption and _depth == 0:
        cap_style = (
            "font-size:7.5pt; font-weight:bold; text-align:left; "
            "padding:4px 0; caption-side:top;"
        )
        lines.append(f'<caption style="{cap_style}">{_esc(caption)}</caption>')

    if headers:
        lines.append("<thead>")
        for r_idx, row in enumerate(headers):
            if not isinstance(row, list):
                continue
            lines.append("<tr>")
            for cell in row:
                if not isinstance(cell, dict):
                    continue
                text  = str(cell.get("text", ""))
                cs    = max(1, int(cell.get("colspan", 1)))
                rs    = max(1, int(cell.get("rowspan", 1)))
                style = _TH1 if r_idx == 0 else (_TH2 if r_idx == 1 else _TH3)
                fn    = cell.get("footnote")
                if fn:
                    content = f'{_esc(text)}<sup style="font-size:6pt;vertical-align:super;">{fn}</sup>'
                    lines.append(_th_raw(content, cs=cs, rs=rs, style=style))
                else:
                    lines.append(_th(text, cs=cs, rs=rs, style=style))
            lines.append("</tr>")
        lines.append("</thead>")

    if rows:
        lines.append("<tbody>")
        n_cols = _count_header_cols(headers)

        for r_idx, row in enumerate(rows):
            if not isinstance(row, list):
                continue
            # Section-break row: single-cell list with is_section=true
            if (len(row) == 1 and isinstance(row[0], dict)
                    and row[0].get("is_section")):
                text = str(row[0].get("text", ""))
                lines.append(
                    f'<tr><td colspan="{n_cols}" style="{_SEC}">{_esc(text)}</td></tr>'
                )
                continue

            lines.append("<tr>")
            for cell in row:
                if not isinstance(cell, dict):
                    continue
                text  = str(cell.get("text", ""))
                cs    = max(1, int(cell.get("colspan", 1)))
                rs    = max(1, int(cell.get("rowspan", 1)))
                fn    = cell.get("footnote")
                inner = cell.get("inner_table")

                if cell.get("is_total"):
                    style = _TOT
                elif cell.get("is_subtotal"):
                    style = _SUB
                elif rs > 1:
                    style = _MRG
                else:
                    style = _rbg(r_idx)

                if inner and isinstance(inner, dict):
                    # Recursively render the nested table
                    inner_html = _render_jagged_html(inner, _depth=_depth + 1)
                    if text:
                        label = (
                            f'<div style="font-size:7pt;font-weight:bold;'
                            f'padding-bottom:2px">{_esc(text)}</div>'
                        )
                        content = label + inner_html
                    else:
                        content = inner_html
                    lines.append(_td_raw(content, cs=cs, rs=rs, style=style))
                elif fn:
                    content = (
                        f'{_esc(text)}'
                        f'<sup style="font-size:6pt;vertical-align:super;">{fn}</sup>'
                    )
                    lines.append(_td_raw(content, cs=cs, rs=rs, style=style))
                else:
                    lines.append(_td(text, cs=cs, rs=rs, style=style))
            lines.append("</tr>")
        lines.append("</tbody>")

    lines.append("</table>")

    # Footnote block — only rendered at top level
    if footnotes and _depth == 0:
        parts: list[str] = []
        for item in footnotes:
            if not isinstance(item, dict):
                continue
            num  = item.get("number", "")
            txt  = str(item.get("text", ""))
            mini = item.get("table")
            sup  = f'<sup style="font-size:6pt;vertical-align:super;">{num}</sup>'
            if mini and isinstance(mini, dict):
                mini_html = _render_jagged_html(mini, _depth=1)
                parts.append(f"{sup}&nbsp;{_esc(txt)}{mini_html}")
            else:
                parts.append(f"{sup}&nbsp;{_esc(txt)}")
        if parts:
            lines.append(f'<p style="{_FN}">' + "<br/>".join(parts) + "</p>")

    return "\n".join(lines)


def _derive_qa_from_merge_facts(merge_facts: list) -> list:
    """Convert the LLM-supplied merge_facts into standard QA candidate dicts.

    Each fact must have ``question`` and ``answer`` keys.  The assertion is
    derived directly from the answer so it is always concrete and verifiable.
    """
    qa: list[dict] = []
    for fact in merge_facts:
        if not isinstance(fact, dict):
            continue
        q = str(fact.get("question", "")).strip()
        a = str(fact.get("answer", "")).strip()
        if not q or not a:
            continue
        qa.append({
            "question":   q,
            "answer":     a,
            "assertions": [{"text": f"The response must state '{a}'."}],
        })
    return qa


_LLM_ARCHETYPES = [
    "ultra_jagged",
    "deep_hierarchy",
    "nested_annotated",
    "multi_version_stacked",
    "transposed_heatmap",
    "cross_reference_grid",
]

_MF_RULE = (
    "IMPORTANT: Every question in merge_facts MUST begin with "
    "'In the table titled \"[your caption here]\", ...' — "
    "replace [your caption here] with the actual caption you chose. "
    "The question must be self-sufficient: a reader given only the full PDF "
    "(no other context) must be able to locate the table and find the answer."
)

_CELL_RULE = (
    "CELL OMISSION RULE (critical for correct HTML): "
    "When a cell has rowspan=N, omit that cell from the next N-1 rows for that column. "
    "When a cell has colspan=M, omit the next M-1 cells in that same row."
)


def _build_archetype_prompt(archetype: str, topic: Any) -> str:  # noqa: C901
    base = (
        f"You are designing a COMPLEX data table for a formal enterprise document.\n"
        f"Document topic: {topic.description}\n\n"
    )
    common_footer = (
        f"\n\n{_CELL_RULE}\n\n"
        "Make all data realistic and domain-specific. "
        f"{_MF_RULE}\n"
        "Return ONLY valid JSON — no markdown fences, no commentary."
    )

    if archetype == "ultra_jagged":
        return base + """
Design an ULTRA-JAGGED table: 3 header tiers, rowspan up to 6 in data, two levels of subtotals.

HEADER (3 rows):
- Row 1: 1 anchor label column (rowspan=3) + 2-3 SUPER-GROUP headers each using colspan=4-6.
  At least one super-group cell must ALSO use rowspan=2 (pins across rows 1-2).
- Row 2: SUB-GROUP headers within each super-group, using colspan=2-3.
- Row 3: LEAF metric columns (single cells, one per data column).

DATA (14-18 rows):
- Column 1 (category): rowspan=4 to 6 per category group (NOT just 2-3).
- Column 2 (sub-category): rowspan=2 to 3 per sub-group within each category.
- At least 3 active rowspan cells must coexist simultaneously in different columns.
- Some data cells use colspan=2 for a merged summary value within a row.

SECTION & SUMMARY ROWS:
- 2-3 full-width section-break rows: [{\"text\": \"Section Name\", \"is_section\": true}]
- After each sub-category group: subtotal row (label colspan=2, is_subtotal=true)
- After each section: group total row (label colspan=3, is_subtotal=true)
- Final row: grand total (label colspan=3, is_total=true)
- 9-13 total columns.

Return JSON shape:
{
  "caption": "...",
  "headers": [
    [{"text":"Anchor","rowspan":3}, {"text":"Super-A","colspan":5}, {"text":"Super-B","colspan":4}],
    [{"text":"Sub-A1","colspan":2}, {"text":"Sub-A2","colspan":3,"rowspan":1}, ...],
    [{"text":"M1"}, {"text":"M2"}, ...]
  ],
  "rows": [
    [{"text":"Section One", "is_section":true}],
    [{"text":"Cat-1","rowspan":5}, {"text":"Sub-X","rowspan":2}, {"text":"$1.2M"}, ...],
    [{"text":"$0.8M"}, ...],
    [{"text":"Sub-X Sub-Total","colspan":2,"is_subtotal":true}, ...],
    ...,
    [{"text":"Cat-1 Group Total","colspan":3,"is_subtotal":true}, ...],
    ...,
    [{"text":"GRAND TOTAL","colspan":3,"is_total":true}, ...]
  ],
  "merge_facts": [
    {"question":"In the table titled '...', which sub-columns fall under the '...' super-group?","answer":"..."},
    {"question":"In the table titled '...', how many rows does the '...' category label span?","answer":"..."},
    {"question":"In the table titled '...', what is [metric] for [sub-category] in the [super-group] column?","answer":"..."},
    {"question":"In the table titled '...', what is the grand total for [metric]?","answer":"..."},
    {"question":"In the table titled '...', what does the '[sub-category] Sub-Total' row show for [metric]?","answer":"..."}
  ]
}
""" + common_footer

    elif archetype == "deep_hierarchy":
        return base + """
Design a 3-LEVEL COLUMN HIERARCHY table.

HEADER (3 rows):
- Row 1: "Entity" label (rowspan=3, w=18%) + exactly 2 SUPER-GROUP headers (colspan=4 each).
- Row 2: exactly 2 GROUP headers within each super-group (colspan=2 each) → 4 group cells.
- Row 3: LEAF metric columns (2 per group: "Value" and "Chg %") → 8 leaf cells.
  → total columns: 1 + 2 super-groups × 2 groups × 2 leaves = 9.

DATA (8-12 rows): simple leaf values — no rowspan/colspan in data rows.
GRAND TOTAL: last row, is_total=true.

Return JSON shape:
{
  "caption": "...",
  "headers": [
    [{"text":"Entity","rowspan":3}, {"text":"Super-A","colspan":4}, {"text":"Super-B","colspan":4}],
    [{"text":"Group-A1","colspan":2}, {"text":"Group-A2","colspan":2}, {"text":"Group-B1","colspan":2}, {"text":"Group-B2","colspan":2}],
    [{"text":"Value"},{"text":"Chg%"}, {"text":"Value"},{"text":"Chg%"}, {"text":"Value"},{"text":"Chg%"}, {"text":"Value"},{"text":"Chg%"}]
  ],
  "rows": [
    [{"text":"Entity 1"}, {"text":"$1.2M"}, {"text":"+3.1%"}, ...],
    ...,
    [{"text":"TOTAL","is_total":true}, {"text":"$..."}, ...]
  ],
  "merge_facts": [
    {"question":"In the table titled '...', which groups fall under the '...' super-group?","answer":"..."},
    {"question":"In the table titled '...', how many leaf metric columns does the '...' super-group contain?","answer":"..."},
    {"question":"In the table titled '...', what is the '...' value for '...' under [super-group]/[group]?","answer":"..."},
    {"question":"In the table titled '...', what is the grand total '...' for '...' super-group?","answer":"..."},
    {"question":"In the table titled '...', which group shows the highest '...' change for '...'?","answer":"..."}
  ]
}
""" + common_footer

    elif archetype == "nested_annotated":
        return base + """
Design a table where MULTIPLE CELLS CONTAIN NESTED INNER TABLES and cells have footnote markers.

OUTER TABLE: 4-5 data rows, 5 columns.
OUTER HEADER (2 rows):
- Row 1: "Category" (rowspan=2) + 4 group labels (some with colspan=2).
- Row 2: sub-label columns.

FOR EACH OUTER DATA ROW: at least 2 cells must have "inner_table" (a nested table object).
INNER TABLE REQUIREMENTS:
- At least one inner_table MUST have 2-tier headers (colspan grouping in row 1).
- At least one inner_table MUST have rowspan merging in its data cells.
- Inner tables: 3-5 columns, 3-5 rows.

FOOTNOTES: at least 3 cells have a "footnote" integer marker. Include top-level "footnotes" list.

Return JSON shape:
{
  "caption": "...",
  "headers": [
    [{"text":"Category","rowspan":2}, {"text":"Overview"}, {"text":"Breakdown","colspan":2}, {"text":"Status"}],
    [{"text":"Primary"}, {"text":"Detail A"}, {"text":"Detail B"}, {"text":"Flag"}]
  ],
  "rows": [
    [
      {"text":"Alpha"},
      {"text":"Summary text", "footnote":1},
      {"inner_table": {
        "headers": [
          [{"text":"Period","rowspan":2}, {"text":"Financials","colspan":2}],
          [{"text":"Rev"}, {"text":"Cost"}]
        ],
        "rows": [
          [{"text":"Q1"}, {"text":"1.2"}, {"text":"0.8"}],
          [{"text":"Q2"}, {"text":"1.5"}, {"text":"0.9"}]
        ]
      }},
      {"inner_table": {
        "headers": [[{"text":"Phase","rowspan":2}, {"text":"Results","colspan":2}],
                    [{"text":"Score"}, {"text":"Rating"}]],
        "rows": [
          [{"text":"Phase 1","rowspan":2}, {"text":"88"}, {"text":"A"}],
          [{"text":"72"}, {"text":"B"}],
          [{"text":"Phase 2"}, {"text":"91"}, {"text":"A+"}]
        ]
      }},
      {"text":"On Track", "footnote":2}
    ]
  ],
  "footnotes": [
    {"number":1, "text":"Revenue includes partner channel allocations."},
    {"number":2, "text":"Status reflects end-of-sprint review."},
    {"number":3, "text":"Figures are preliminary pending audit sign-off."}
  ],
  "merge_facts": [
    {"question":"In the table titled '...', what sub-columns fall under '...' in the '...' row's inner table?","answer":"..."},
    {"question":"In the table titled '...', how many rows does the '...' label span in the inner breakdown table for '...'?","answer":"..."},
    {"question":"In the table titled '...', what does footnote 1 explain?","answer":"..."},
    {"question":"In the table titled '...', what is the Q2 revenue in the inner financial table for the '...' row?","answer":"..."},
    {"question":"In the table titled '...', what rating did Phase 1 receive in the '...' row's inner results table?","answer":"..."}
  ]
}
""" + common_footer

    elif archetype == "multi_version_stacked":
        return base + """
Design a MULTI-VERSION STACKED COMPARISON table: 2 versions × 4 periods + delta group.

HEADER (3 rows):
- Row 1: "Line Item" (rowspan=3, w=22%) + 2 version groups each using colspan=4:
  "Baseline" and "Actuals" (use distinct TH styles) +
  "Delta vs Baseline" group using colspan=2.
- Row 2: "Q1"/"Q2"/"Q3"/"Q4" repeated for each version + "Net ($K)" / "Chg %".
- Row 3: "$000" unit labels for each leaf column + "" for delta cols.

DATA:
- 3-4 cost-category section-break rows (is_section=true).
- Each section: 3-4 line items + subtotal row (is_subtotal=true, label colspan=1).
- Grand total row (is_total=true).
- 11 total columns: 1 + 2 versions × 4 periods + 2 delta = 11.

Return JSON shape:
{
  "caption": "...",
  "headers": [
    [{"text":"Line Item","rowspan":3}, {"text":"Baseline","colspan":4}, {"text":"Actuals","colspan":4},
     {"text":"Delta vs Baseline","colspan":2}],
    [{"text":"Q1"},{"text":"Q2"},{"text":"Q3"},{"text":"Q4"}, {"text":"Q1"},{"text":"Q2"},{"text":"Q3"},{"text":"Q4"}, {"text":"Net ($K)"}, {"text":"Chg %"}],
    [{"text":"$000"},{"text":"$000"},{"text":"$000"},{"text":"$000"}, {"text":"$000"},{"text":"$000"},{"text":"$000"},{"text":"$000"}, {"text":""}, {"text":""}]
  ],
  "rows": [
    [{"text":"Revenue", "is_section":true}],
    [{"text":"Core Product"}, {"text":"120"},{"text":"130"},{"text":"140"},{"text":"150"},
     {"text":"119"},{"text":"128"},{"text":"143"},{"text":"158"},
     {"text":"-3"}, {"text":"-1.9%"}],
    ...more line items...,
    [{"text":"Revenue Total","is_subtotal":true}, {"text":"..."}, ...],
    ...more sections...,
    [{"text":"GRAND TOTAL","is_total":true}, {"text":"..."}, ...]
  ],
  "merge_facts": [
    {"question":"In the table titled '...', how many period columns does each version group span?","answer":"..."},
    {"question":"In the table titled '...', what is the Actuals Q3 value for '...'?","answer":"..."},
    {"question":"In the table titled '...', what is the '...' section total for Baseline Q1?","answer":"..."},
    {"question":"In the table titled '...', which version shows the highest Q4 value for '...'?","answer":"..."},
    {"question":"In the table titled '...', what is the delta (Net $K) for '...'?","answer":"..."}
  ]
}
""" + common_footer

    elif archetype == "transposed_heatmap":
        return base + """
Design a TRANSPOSED HEATMAP PIVOT: metrics as rows, time periods as columns with Actual/Target/Attain% sub-columns.

HEADER (3 rows):
- Row 1: "Category" (rowspan=3, w=14%) + "Metric" (rowspan=3, w=22%) + "Dir" (rowspan=3, w=8%) +
  exactly 2 PERIOD groups (e.g. "H1" and "H2"), each using colspan=3.
- Row 2: "Actual" / "Target" / "Attain%" for each period (repeated 2x).
- Row 3: unit labels ("$M" / "$M" / "%") for each period (repeated 2x).

DATA (8-12 metric rows):
- 2-3 category section-break rows (is_section=true).
- Category label column uses rowspan=N (N = metrics in that category), style like _MRG.
- Attainment cells: values like "94%" or "108%".
- Direction col: "Higher" or "Lower" indicating if high is good.
- Subtotal per category (is_subtotal=true, label colspan=2).
- 9 total columns: 3 fixed + 2 periods × 3 sub-cols = 9.

Return JSON shape:
{
  "caption": "...",
  "headers": [
    [{"text":"Category","rowspan":3}, {"text":"Metric","rowspan":3}, {"text":"Dir","rowspan":3},
     {"text":"H1","colspan":3}, {"text":"H2","colspan":3}],
    [{"text":"Actual"},{"text":"Target"},{"text":"Attain%"}, {"text":"Actual"},{"text":"Target"},{"text":"Attain%"}],
    [{"text":"$M"},{"text":"$M"},{"text":"%"}, {"text":"$M"},{"text":"$M"},{"text":"%"}]
  ],
  "rows": [
    [{"text":"Revenue KPIs","is_section":true}],
    [{"text":"Revenue","rowspan":3}, {"text":"Monthly Rev"}, {"text":"Higher"},
     {"text":"$4.2M"},{"text":"$4.0M"},{"text":"105%"},
     {"text":"$4.8M"},{"text":"$4.5M"},{"text":"107%"}],
    ...,
    [{"text":"Revenue Sub-Total","colspan":2,"is_subtotal":true}, ...],
    ...
  ],
  "merge_facts": [
    {"question":"In the table titled '...', how many sub-columns does each period group span?","answer":"..."},
    {"question":"In the table titled '...', what is the H1 Attainment% for '...'?","answer":"..."},
    {"question":"In the table titled '...', how many metrics does the '...' category rowspan cover?","answer":"..."},
    {"question":"In the table titled '...', which metric has the lower H2 attainment in the '...' category?","answer":"..."},
    {"question":"In the table titled '...', what is the H2 target for '...'?","answer":"..."}
  ]
}
""" + common_footer

    else:  # cross_reference_grid
        return base + """
Design an ASYMMETRIC CROSS-REFERENCE ALIGNMENT GRID: strategic initiatives vs. objectives.

HEADER (3 rows):
- Row 1: blank corner (rowspan=3, colspan=2) + 3-4 objective CATEGORY groups (colspan=3-4 each) + "Total Score" (rowspan=3).
- Row 2: individual objectives within each category (3-4 per group).
- Row 3: "Wt" (weight multiplier) per objective column.

ROW LABELS:
- Column 1: initiative CATEGORY label (rowspan=N for N initiatives, uses merge style).
- Column 2: initiative name.

DATA CELLS: alignment rating "H" (High) / "M" (Med) / "L" (Low) / "—" (None).
TOTAL SCORE COLUMN: weighted sum for each initiative.
SUMMARY ROW: last row is_total=true — objective weighted total scores.
Section-break rows separate initiative categories (is_section=true).

Return JSON shape:
{
  "caption": "...",
  "headers": [
    [{"text":"","rowspan":3,"colspan":2}, {"text":"Customer Goals","colspan":3},
     {"text":"Operational Goals","colspan":3}, {"text":"Financial Goals","colspan":3}, {"text":"Score","rowspan":3}],
    [{"text":"Satisfaction"}, {"text":"Retention"}, {"text":"NPS"},
     {"text":"Efficiency"}, {"text":"Quality"}, {"text":"Cost"},
     {"text":"Revenue"}, {"text":"Margin"}, {"text":"Growth"}],
    [{"text":"x3"}, {"text":"x2"}, {"text":"x2"},
     {"text":"x2"}, {"text":"x3"}, {"text":"x1"},
     {"text":"x3"}, {"text":"x2"}, {"text":"x2"}]
  ],
  "rows": [
    [{"text":"Customer","is_section":true}],
    [{"text":"Retention Initiatives","rowspan":3}, {"text":"Loyalty Rewards"},
     {"text":"H"},{"text":"H"},{"text":"M"}, {"text":"L"},{"text":"M"},{"text":"—"},
     {"text":"M"},{"text":"L"},{"text":"H"}, {"text":"28"}],
    [{"text":"Personalisation"}, {"text":"M"},{"text":"H"},{"text":"H"}, ...],
    [{"text":"Feedback Loop"}, ...],
    ...more categories...,
    [{"text":"Objective Weighted Total","is_total":true,"colspan":2}, {"text":"42"}, ...]
  ],
  "merge_facts": [
    {"question":"In the table titled '...', which objective categories are the column groups?","answer":"..."},
    {"question":"In the table titled '...', how many initiatives fall under the '...' initiative category?","answer":"..."},
    {"question":"In the table titled '...', what is the alignment rating of '...' for the '...' objective?","answer":"..."},
    {"question":"In the table titled '...', what is the total score for the '...' initiative?","answer":"..."},
    {"question":"In the table titled '...', which objective category has the highest weighted total score?","answer":"..."}
  ]
}
""" + common_footer


def _llm_generate_complex_table(
    client: Any,
    topic: Any,
    rng: Any,
) -> tuple[str, dict]:
    """Pick a random structural archetype and generate an LLM complex table.

    Six archetypes cycle through on successive calls (seeded by rng):
    ultra_jagged, deep_hierarchy, nested_annotated,
    multi_version_stacked, transposed_heatmap, cross_reference_grid.
    """
    archetype = _LLM_ARCHETYPES[int(rng.integers(0, len(_LLM_ARCHETYPES)))]
    prompt    = _build_archetype_prompt(archetype, topic)

    raw = client.complete(prompt, json_mode=True, tier="hard",
                          _label=f"llm_cx_{archetype[:12]}")

    # Robust JSON extraction
    try:
        data = _json.loads(raw)
    except (_json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = _json.loads(m.group())
        else:
            raise ValueError(
                f"LLM returned non-JSON for complex table (archetype={archetype}): {raw[:300]}"
            )

    html          = _render_jagged_html(data)
    qa_candidates = _derive_qa_from_merge_facts(data.get("merge_facts", []))

    gt = {
        "complex_type":  f"llm_{archetype}",
        "archetype":     archetype,
        "caption":       data.get("caption", "Complex Table"),
        "description":   (
            f"LLM-generated {archetype.replace('_', ' ')} complex table. "
            "Uses extensive rowspan/colspan merging, multi-tier headers, and structured "
            "QA candidates testing extraction of merged cell values."
        ),
        "headers":       data.get("headers", []),
        "row_count":     len(data.get("rows", [])),
        "footnotes":     data.get("footnotes", []),
        "merge_facts":   data.get("merge_facts", []),
        "qa_candidates": qa_candidates,
    }
    return html, gt


# ---------------------------------------------------------------------------
# LLM content injection helpers
# ---------------------------------------------------------------------------

def _cl(labels: dict | None, key: str, default: Any) -> Any:
    """Return labels[key] if it's a non-empty value, else default."""
    val = (labels or {}).get(key)
    return val if val else default


def _llm_entity_labels_all(client: Any, topic: Any) -> dict:
    """One tier='easy' LLM call: returns domain-specific names for all static generators.

    Returns a dict with keys:
      department_names, product_names, metric_names, cost_categories,
      risk_categories, region_names, kpi_names, footnote_texts
    Falls back to {} on any error so generators use their hardcoded defaults.
    """
    prompt = (
        f"Generate realistic, domain-specific content labels for data tables in a "
        f"document about: {topic.description}\n\n"
        "Return a JSON object with EXACTLY these keys and formats:\n"
        '"department_names": list of 5 department names (2-4 words each)\n'
        '"sub_team_names": list of 8 sub-team / function names (2-4 words each)\n'
        '"product_names": list of 7 product or service names (2-5 words each)\n'
        '"metric_names": list of 10 financial or operational metric names (2-5 words each)\n'
        '"cost_categories": list of 4 budget cost-category names (1-3 words each)\n'
        '"cost_line_items": list of 12 budget line-item names (2-5 words each)\n'
        '"risk_category_names": list of 4 risk category names (1-3 words each)\n'
        '"risk_descriptions": list of 12 short risk description phrases (4-8 words each)\n'
        '"region_names": list of 4 geographic or business-unit names (1-4 words each)\n'
        '"kpi_names": list of 10 KPI / dashboard metric names (2-5 words each)\n'
        '"footnote_texts": list of 4 short accounting/legal disclaimer sentences\n'
        "\nMake all names realistic and specific to the domain. "
        "Return ONLY valid JSON — no markdown fences."
    )
    try:
        raw = client.complete(
            prompt, json_mode=True, tier="easy", _label="cx_entity_labels"
        )
        return _json.loads(raw)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "_llm_entity_labels_all failed (%s) — using hardcoded defaults", exc
        )
        return {}


def _gt_facts_summary(gt: dict) -> str:
    """Compact, human-readable summary of a complex table GT for QnA prompting."""
    lines: list[str] = []
    ctype   = gt.get("complex_type", "")
    caption = gt.get("caption", "")
    lines.append(f"Table: \"{caption}\"  type={ctype}")
    if "description" in gt:
        lines.append(f"Structure: {gt['description']}")

    def _rows(key: str, n: int = 6) -> None:
        for r in (gt.get(key) or [])[:n]:
            lines.append(f"  row: {r}")

    def _groups(key: str, n: int = 4) -> None:
        for g in (gt.get(key) or [])[:n]:
            lines.append(f"  group: {g}")

    if ctype in ("merged_cells",):
        schema = gt.get("schema", [])
        lines.append(f"Columns: {schema}")
        _rows("logical_rows")
        da = gt.get("derived_answers", {})
        if da:
            lines.append(f"Aggregates: {da}")

    elif ctype in ("column_groups",):
        sc = gt.get("schema", {})
        lines.append(f"Schema: {sc}")
        _rows("logical_rows", 4)

    elif ctype in ("multi_row_headers",):
        sc = gt.get("schema", {})
        lines.append(f"Schema: {sc}")
        _rows("logical_rows", 4)

    elif ctype in ("group_summary_rows",):
        _groups("logical_groups", 3)
        da = gt.get("derived_answers", {})
        if da:
            lines.append(f"Totals: {da}")

    elif ctype in ("nested_table",):
        st = gt.get("structure", {})
        lines.append(f"Outer cols: {st.get('outer_cols')}, nested cols: {st.get('nested_cols')}")
        _rows("logical_rows", 3)

    elif ctype in ("footnoted_table",):
        schema = gt.get("schema", [])
        lines.append(f"Schema: {schema}")
        for r in (gt.get("logical_rows") or [])[:6]:
            lines.append(f"  metric={r.get('metric')} fn={r.get('footnotes')} "
                         f"reported={list(r.get('reported', {}).values())[:2]}")
        fns = gt.get("footnotes", {})
        if fns:
            lines.append(f"Footnotes: {fns}")

    elif ctype in ("matrix_table",):
        features = gt.get("features", [])
        matrix   = gt.get("matrix", [])
        lines.append(f"Features: {features}")
        for i, row in enumerate(matrix[:4]):
            label = features[i] if i < len(features) else str(i)
            lines.append(f"  {label}: {row}")

    elif ctype in ("ragged_colspan",):
        for cat in (gt.get("logical_categories") or [])[:3]:
            lines.append(f"  cat={cat.get('category')} risks={[r['description'] for r in cat.get('risks', [])[:2]]}")

    elif ctype in ("stacked_comparison",):
        versions = gt.get("structure", {}).get("versions", [])
        periods  = gt.get("structure", {}).get("periods", [])
        lines.append(f"Versions: {versions}  Periods: {periods}")
        for sec in (gt.get("logical_sections") or [])[:3]:
            for item in (sec.get("items") or [])[:2]:
                lines.append(f"  item={item.get('item')} net_delta={item.get('net_delta')}")

    elif ctype in ("transposed_metrics",):
        periods = gt.get("periods", [])
        lines.append(f"Periods: {periods}")
        for r in (gt.get("logical_rows") or [])[:6]:
            lines.append(f"  metric={r.get('metric')} unit={r.get('unit')} "
                         f"trend={r.get('trend')} mom={r.get('mom_change_pct')}")

    elif ctype in ("extreme_merge",):
        for sec in (gt.get("logical_sections") or [])[:3]:
            lines.append(f"  region={sec.get('region')} subs={sec.get('sub_groups')} "
                         f"totals={sec.get('totals')}")
        da = gt.get("derived_answers", {})
        if da:
            lines.append(f"Grand totals: {da}")

    else:  # LLM archetypes
        headers = gt.get("headers", [])
        if headers:
            lines.append(f"Header tiers: {len(headers)} rows")
            lines.append(f"Header row 1 cells: {[c.get('text') for c in (headers[0] or [])[:6]]}")
        lines.append(f"Data rows: {gt.get('row_count', '?')}")
        for fact in (gt.get("merge_facts") or [])[:3]:
            lines.append(f"  fact: Q={fact.get('question','')[:80]}  A={fact.get('answer','')}")

    return "\n".join(lines)


def _llm_qa_for_table(
    client: Any,
    topic: Any,
    gt: dict,
    n: int = 5,
) -> list[dict]:
    """Generate n diverse, non-template QnA pairs from any complex table GT.

    Runs as a tier='easy' call AFTER the table is already built, so questions
    are grounded in the actual cell values, not in prompt-example templates.
    Falls back to [] so callers never crash.
    """
    caption = gt.get("caption", "the table")
    summary = _gt_facts_summary(gt)

    prompt = (
        f"You are writing evaluation Q&A for a complex data table in a document "
        f"about: {topic.description}\n\n"
        f"TABLE FACTS:\n{summary}\n\n"
        f"Generate exactly {n} question-answer pairs. Rules:\n"
        f"1. EVERY question MUST begin with: "
        f"'In the table titled \"{caption}\", ...'\n"
        f"2. Cover diverse question TYPES across the {n} questions:\n"
        f"   - Value lookup (what is X for entity Y?)\n"
        f"   - Structure (how many rows does label Z span? which columns fall under header W?)\n"
        f"   - Comparison or aggregation (which entity has the highest X?)\n"
        f"   - Footnote/annotation (what does footnote N explain?)\n"
        f"   - Cross-group (what is the subtotal for group G in column C?)\n"
        f"3. Each question must be answerable from the table facts listed above\n"
        f"4. Answers: short, 1-15 words, directly readable from the data\n"
        f"5. Do NOT use technical metadata terms: no 'GT', 'dict', 'array', 'key', 'field', "
        f"'logical_rows', 'rowspan', 'colspan'. Phrase as a reader examining the PDF.\n"
        f"6. Vary sentence structure — no two questions with the same opening phrase\n\n"
        "Return ONLY valid JSON:\n"
        '{{"qa": [{{"question": "...", "answer": "...", "assertions": [{{"text": "The response must state \'[answer]\'."}}]}}]}}'
    )
    try:
        raw   = client.complete(prompt, json_mode=True, tier="easy", _label="cx_qa")
        pairs = _json.loads(raw).get("qa", [])
        result: list[dict] = []
        for p in pairs[:n]:
            if not isinstance(p, dict) or not p.get("question"):
                continue
            assertions = [
                {"text": str(a["text"] if isinstance(a, dict) else a)}
                for a in p.get("assertions", [])
                if a
            ]
            result.append({
                "question":   str(p.get("question", "")),
                "answer":     str(p.get("answer",   "")),
                "assertions": assertions,
            })
        return result
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "_llm_qa_for_table failed (caption=%r): %s", caption, exc
        )
        return []


def _generate_llm_complex_variants(
    client: Any,
    topic: Any,
    rng: Any,
    count: int,
) -> list[Variant]:
    """Generate *count* complex table variants — structure AND content fully LLM-driven.

    Every table is generated by calling ``_llm_generate_complex_table()`` which:
      - Randomly selects one of 6 structural archetypes (ultra_jagged, deep_hierarchy,
        nested_annotated, multi_version_stacked, transposed_heatmap, cross_reference_grid)
      - Sends a detailed prompt to the LLM asking it to invent the full table JSON
        (headers with rowspan/colspan, data rows, footnotes, captions) from scratch
      - The LLM decides the structure: number of header tiers, which cells merge and by
        how much, the column hierarchy, section breaks — nothing is hardcoded
      - ``_render_jagged_html()`` converts the JSON to HTML faithfully

    After the table is built, ``_llm_qa_for_table()`` generates 5 non-template QnA
    pairs grounded in the actual cell values, appended to any QnA in the table GT.

    The static generators in ``_COMPLEX_GENERATORS`` are used ONLY as a no-client
    fallback (see ``generate_complex_table_variants``).
    """
    import concurrent.futures as _cf

    def _one(i: int) -> Variant:
        child_rng = fork_rng(rng, f"cx_table_{i}")

        try:
            html, gt_data = _llm_generate_complex_table(client, topic, child_rng)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            html = f"<p><em>LLM complex table failed: {_esc(str(exc))}</em></p>"
            gt_data = {
                "complex_type": "llm_jagged",
                "caption": f"Complex Table {i + 1}",
                "error": str(exc),
                "qa_candidates": [],
            }

        # Non-template LLM QnA grounded in the actual rendered table data
        llm_qa = _llm_qa_for_table(client, topic, gt_data, n=5)
        existing_qa = gt_data.get("qa_candidates") or []
        gt_data["qa_candidates"] = list(existing_qa) + llm_qa

        caption      = gt_data.get("caption", f"Complex Table {i + 1}")
        complex_type = gt_data.get("complex_type", "llm_jagged")

        return Variant(
            id=f"cx_table_{i:04d}_{_uid()}",
            kind="table",
            metadata={
                "complex_table": True,
                "complex_type":  complex_type,
                "caption":       caption,
                "row_count":     gt_data.get("row_count", 0),
                "col_count":     0,
                "perturbations": [],
                "topic_id":      getattr(topic, "id", ""),
            },
            render_payload={
                "markdown": html,
                "caption":  caption,
                "format":   "html",
            },
            ground_truth={
                **gt_data,
                "caption": caption,
            },
        )

    results: list[Variant | None] = [None] * count
    if count == 1:
        results[0] = _one(0)
    else:
        with _cf.ThreadPoolExecutor(max_workers=min(count, 4)) as pool:
            futures = {pool.submit(_one, i): i for i in range(count)}
            for fut in _cf.as_completed(futures):
                results[futures[fut]] = fut.result()
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

_COMPLEX_GENERATORS = [
    ("merged_cells",        _gen_merged_cells),
    ("multi_row_headers",   _gen_multi_row_headers),
    ("column_groups",       _gen_column_groups),
    ("group_summary_rows",  _gen_group_summary_rows),
    ("nested_table",        _gen_nested_table),
    ("footnoted_table",     _gen_footnoted_table),
    ("matrix_table",        _gen_matrix_table),
    ("ragged_colspan",      _gen_ragged_colspan),
    ("stacked_comparison",  _gen_stacked_comparison),
    ("transposed_metrics",  _gen_transposed_metrics),
    ("extreme_merge",       _gen_extreme_merge),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_complex_table_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Any" = None,
    client: "Any" = None,
    topic: "Any" = None,
) -> list[Variant]:
    """Generate *count* complex table variants.

    When *client* and *topic* are both provided, delegates to the LLM-driven
    path (``_generate_llm_complex_variants``) which produces truly jagged tables
    with rowspan/colspan structure and semantically-grounded QA.

    Falls back to cycling through the static generators when no LLM client is
    available.  Each variant has ``metadata["complex_type"]`` set to the
    generator name and ``metadata["complex_table"] = True``.
    """
    # --- LLM path ---
    if client is not None and topic is not None:
        return _generate_llm_complex_variants(client, topic, rng, count)

    # --- Static fallback path ---
    variants: list[Variant] = []

    for i in range(count):
        child_rng = fork_rng(rng, f"complex_table_{i}")
        gen_name, gen_fn = _COMPLEX_GENERATORS[i % len(_COMPLEX_GENERATORS)]

        try:
            html, gt_data = gen_fn(child_rng)
        except Exception as exc:
            # Never let a single generator failure crash the whole pool
            import traceback
            traceback.print_exc()
            # Fallback: a minimal placeholder table
            html = f"<p><em>Table generation failed: {_esc(str(exc))}</em></p>"
            gt_data = {"complex_type": gen_name, "error": str(exc)}

        caption = gt_data.get("caption", f"Complex Table {i + 1}: {gen_name.replace('_', ' ').title()}")

        variants.append(Variant(
            id=f"cx_table_{i:04d}_{_uid()}",
            kind="table",
            metadata={
                "complex_table": True,
                "complex_type": gen_name,
                "caption": caption,
                "row_count": gt_data.get("structure", {}).get("data_rows", 0),
                "col_count": gt_data.get("structure", {}).get("cols", 0),
                "perturbations": [],
            },
            render_payload={
                "markdown": html,
                "caption": caption,
                "format": "html",
            },
            ground_truth={
                **gt_data,
                "caption": caption,
                "qa_candidates": [],
            },
        ))

    return variants
