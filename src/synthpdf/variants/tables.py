# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Table variant generator.

Generates synthetic tables of five base types, each with a set of
perturbations applied probabilistically.  Every variant emits:
  - Markdown or HTML table string
  - Caption string
  - Structured ground truth (schema, cell values, derived answers)

Also provides generate_llm_table_variants() for topic-specific LLM-driven tables.
"""
from __future__ import annotations

import math
import time
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

import click
import numpy as np

from synthpdf.rng import fork_rng
from synthpdf.types import Variant

if TYPE_CHECKING:
    from synthpdf.llm.client import AzureOpenAIClient
    from synthpdf.llm.content_generator import TopicSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid() -> str:
    return str(uuid.uuid4())


def _fmt_currency(rng: np.random.Generator, value: float) -> str:
    currencies = ["$", "€", "£", "¥"]
    sym = currencies[int(rng.integers(0, len(currencies)))]
    return f"{sym}{value:,.2f}"


def _fmt_percent(value: float) -> str:
    return f"{value:.1f}%"


def _fmt_scientific(value: float) -> str:
    return f"{value:.3e}"


def _maybe_typo(rng: np.random.Generator, text: str, prob: float = 0.05) -> str:
    """With probability *prob* introduce a single-character swap typo."""
    if len(text) < 3 or rng.random() > prob:
        return text
    pos = int(rng.integers(1, len(text) - 1))
    chars = list(text)
    chars[pos], chars[pos - 1] = chars[pos - 1], chars[pos]
    return "".join(chars)


# ---------------------------------------------------------------------------
# Base table generators
# ---------------------------------------------------------------------------

def _financial_summary(rng: np.random.Generator) -> tuple[list[str], list[list[Any]]]:
    headers = ["Quarter", "Revenue", "Expenses", "Net Profit", "Margin %"]
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    rows = []
    for q in quarters:
        rev = float(rng.uniform(500_000, 5_000_000))
        exp = float(rng.uniform(300_000, rev * 0.9))
        profit = rev - exp
        margin = (profit / rev) * 100
        rows.append([q, rev, exp, profit, margin])
    return headers, rows


def _inventory_list(rng: np.random.Generator) -> tuple[list[str], list[list[Any]]]:
    products = [
        "Widget A", "Gadget B", "Part C", "Module D", "Component E",
        "Device F", "Unit G", "Item H",
    ]
    headers = ["SKU", "Product", "Qty In Stock", "Unit Price", "Reorder Level"]
    rows = []
    for i in range(int(rng.integers(5, 10))):
        sku = f"SKU-{1000 + i}"
        product = products[i % len(products)]
        qty = int(rng.integers(0, 500))
        price = float(rng.uniform(1.99, 999.99))
        reorder = int(rng.integers(10, 100))
        rows.append([sku, product, qty, price, reorder])
    return headers, rows


def _experiment_results(rng: np.random.Generator) -> tuple[list[str], list[list[Any]]]:
    headers = ["Run ID", "Algorithm", "Accuracy", "F1 Score", "Latency (ms)"]
    algos = ["Baseline", "ModelA", "ModelB", "ModelC", "Ensemble"]
    rows = []
    for i, algo in enumerate(algos):
        run_id = f"run_{i+1:03d}"
        acc = float(rng.uniform(0.70, 0.99))
        f1 = float(rng.uniform(acc - 0.05, min(acc + 0.02, 1.0)))
        latency = float(rng.uniform(10.0, 500.0))
        rows.append([run_id, algo, acc, f1, latency])
    return headers, rows


def _hr_directory(rng: np.random.Generator) -> tuple[list[str], list[list[Any]]]:
    first_names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank"]
    last_names = ["Smith", "Jones", "Williams", "Brown", "Taylor", "Wilson"]
    departments = ["Engineering", "Marketing", "HR", "Finance", "Sales", "Legal"]
    headers = ["Employee ID", "Name", "Department", "Title", "Start Date"]
    rows = []
    for i in range(int(rng.integers(6, 12))):
        emp_id = f"EMP-{2000 + i}"
        name = (
            first_names[int(rng.integers(0, len(first_names)))]
            + " "
            + last_names[int(rng.integers(0, len(last_names)))]
        )
        dept = departments[int(rng.integers(0, len(departments)))]
        titles = {"Engineering": "Engineer", "Marketing": "Analyst", "HR": "Specialist",
                  "Finance": "Accountant", "Sales": "Rep", "Legal": "Counsel"}
        title = titles.get(dept, "Associate")
        year = int(rng.integers(2010, 2024))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        rows.append([emp_id, name, dept, title, date_str])
    return headers, rows


def _action_items(rng: np.random.Generator) -> tuple[list[str], list[list[Any]]]:
    owners = ["Alice", "Bob", "Carol", "David", "Eve"]
    statuses = ["Open", "In Progress", "Blocked", "Done"]
    priorities = ["High", "Medium", "Low"]
    headers = ["ID", "Task", "Owner", "Priority", "Status", "Due Date"]
    tasks = [
        "Finalize report", "Review PR", "Update docs", "Deploy service",
        "Fix bug", "Write tests", "Schedule meeting", "Send invoice",
    ]
    rows = []
    for i in range(int(rng.integers(5, 9))):
        task_id = f"ACT-{i+1:03d}"
        task = tasks[i % len(tasks)]
        owner = owners[int(rng.integers(0, len(owners)))]
        priority = priorities[int(rng.integers(0, len(priorities)))]
        status = statuses[int(rng.integers(0, len(statuses)))]
        year = 2024
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        due = f"{year:04d}-{month:02d}-{day:02d}"
        rows.append([task_id, task, owner, priority, status, due])
    return headers, rows


def _purchase_orders(rng: np.random.Generator, n_rows: int | None = None) -> tuple[list[str], list[list[Any]]]:
    """Large purchase order table for multipage use."""
    n = n_rows if n_rows is not None else int(rng.integers(8, 15))
    vendors = ["Acme Corp", "GlobalSupply", "TechParts Ltd", "Industrial Co", "FastShip Inc",
               "Prime Vendor", "QualityGoods", "BulkMart", "SpecialtyOps", "EcoSource"]
    items = ["Office Chair", "Laptop Stand", "Cable Bundle", "Safety Gear", "Printer Paper",
             "Server Rack", "Network Switch", "Monitor Arm", "Keyboard Kit", "USB Hub",
             "Power Strip", "Label Maker", "Filing Cabinet", "Headset", "Webcam",
             "Dock Station", "Battery Pack", "Cleaning Kit", "Badge Holder", "Desk Pad"]
    statuses = ["Pending", "Approved", "Shipped", "Delivered", "Cancelled", "On Hold"]
    headers = ["PO Number", "Vendor", "Item", "Qty", "Unit Price", "Line Total", "Status"]
    rows = []
    for i in range(n):
        po = f"PO-{10000 + i}"
        vendor = vendors[int(rng.integers(0, len(vendors)))]
        item = items[int(rng.integers(0, len(items)))]
        qty = int(rng.integers(1, 100))
        unit_price = round(float(rng.uniform(5.0, 500.0)), 2)
        line_total = round(qty * unit_price, 2)
        status = statuses[int(rng.integers(0, len(statuses)))]
        rows.append([po, vendor, item, qty, unit_price, line_total, status])
    return headers, rows


def _time_series_metrics(rng: np.random.Generator, n_rows: int | None = None) -> tuple[list[str], list[list[Any]]]:
    """Large time-series metrics table for multipage use."""
    n = n_rows if n_rows is not None else int(rng.integers(8, 15))
    regions = ["North", "South", "East", "West", "Central", "Pacific", "Atlantic"]
    metrics = ["Page Views", "Active Users", "Conversion Rate", "Revenue", "Latency (ms)",
               "Error Rate", "Throughput", "Uptime %", "CPU Usage", "Memory Usage"]
    headers = ["Date", "Metric", "Value", "Target", "Delta", "Region"]
    rows = []
    base_date_offset = int(rng.integers(0, 365))
    for i in range(n):
        day_offset = base_date_offset + i
        year = 2024 + day_offset // 365
        doy = day_offset % 365
        month = min(12, doy // 30 + 1)
        day = min(28, doy % 30 + 1)
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        metric = metrics[int(rng.integers(0, len(metrics)))]
        value = round(float(rng.uniform(10.0, 10000.0)), 2)
        target = round(float(rng.uniform(value * 0.8, value * 1.2)), 2)
        delta = round(value - target, 2)
        region = regions[int(rng.integers(0, len(regions)))]
        rows.append([date_str, metric, value, target, delta, region])
    return headers, rows


_BASE_GENERATORS = [
    ("financial_summary", _financial_summary),
    ("inventory_list", _inventory_list),
    ("experiment_results", _experiment_results),
    ("hr_directory", _hr_directory),
    ("action_items", _action_items),
    ("purchase_orders", _purchase_orders),
    ("time_series_metrics", _time_series_metrics),
]


# ---------------------------------------------------------------------------
# Complex multipage base generators (grouped data for subtotal/column-group)
# ---------------------------------------------------------------------------

def _grouped_ledger(
    rng: np.random.Generator,
    n_rows: int = 60,
) -> tuple[list[str], list[list[Any]], list[tuple[str, int]]]:
    """Department budget ledger: rows grouped by department, quarters as columns.

    Returns (headers, rows, column_groups) where column_groups drives a 2-tier header.
    """
    departments = [
        "Operations", "Engineering", "Sales & Marketing",
        "Finance", "Information Technology", "Human Resources", "Legal & Compliance",
    ]
    cost_types = [
        "Personnel", "Equipment", "Consulting",
        "Travel & Entertainment", "Software Licenses", "Facilities", "Other",
    ]
    headers = [
        "Department", "Cost Center", "Category",
        "Q1 ($)", "Q2 ($)", "Q3 ($)", "Q4 ($)", "Annual Total ($)",
    ]
    column_groups: list[tuple[str, int]] = [
        ("Identification", 3),
        ("Quarterly Budget", 4),
        ("Summary", 1),
    ]

    n_depts = min(len(departments), max(3, math.ceil(n_rows / 8)))
    rows_per_dept = max(3, math.ceil(n_rows / n_depts))
    rows: list[list[Any]] = []

    for dept in departments[:n_depts]:
        for j in range(rows_per_dept):
            if len(rows) >= n_rows:
                break
            cost_center = f"{dept[:3].upper()}-CC{100 + j:03d}"
            category    = cost_types[j % len(cost_types)]
            q1 = round(float(rng.uniform(5_000, 200_000)), 2)
            q2 = round(float(rng.uniform(5_000, 200_000)), 2)
            q3 = round(float(rng.uniform(5_000, 200_000)), 2)
            q4 = round(float(rng.uniform(5_000, 200_000)), 2)
            rows.append([dept, cost_center, category, q1, q2, q3, q4, round(q1+q2+q3+q4, 2)])

    return headers, rows[:n_rows], column_groups


def _project_tracker(
    rng: np.random.Generator,
    n_rows: int = 60,
) -> tuple[list[str], list[list[Any]], list[tuple[str, int]]]:
    """Multi-phase project tracker grouped by project ID.

    Returns (headers, rows, column_groups).
    """
    import datetime as _dt
    projects = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
    phases   = ["Planning", "Design", "Development", "Testing", "Deployment", "Closure"]
    owners   = [f"Owner-{i}" for i in range(1, 9)]
    statuses = ["On Track", "At Risk", "Delayed", "Complete"]

    headers = [
        "Project", "Phase", "Milestone",
        "Start Date", "End Date", "Owner",
        "% Done", "Budget ($)", "Spent ($)", "Variance ($)",
    ]
    column_groups: list[tuple[str, int]] = [
        ("Project Info", 3),
        ("Schedule", 3),
        ("Financials", 4),
    ]

    n_proj = min(len(projects), max(3, math.ceil(n_rows / 6)))
    rows_per_proj = max(3, math.ceil(n_rows / n_proj))
    rows: list[list[Any]] = []
    base = _dt.date(2024, 1, 1)

    for pi, proj in enumerate(projects[:n_proj]):
        for j in range(rows_per_proj):
            if len(rows) >= n_rows:
                break
            phase     = phases[j % len(phases)]
            milestone = f"{phase[:3]}-{j+1:02d}"
            start     = base + _dt.timedelta(days=j * 14 + pi * 60)
            end       = start + _dt.timedelta(days=int(rng.integers(7, 90)))
            owner     = owners[int(rng.integers(0, len(owners)))]
            pct       = round(float(rng.uniform(0, 100)), 1)
            budget    = round(float(rng.uniform(10_000, 500_000)), 2)
            spent     = round(float(rng.uniform(0, budget * 1.1)), 2)
            variance  = round(budget - spent, 2)
            rows.append([proj, phase, milestone, str(start), str(end), owner,
                         f"{pct}%", budget, spent, variance])

    return headers, rows[:n_rows], column_groups


# Large-format multipage generators (no column_groups; use existing paginate_table)
_MP_LARGE_GENERATORS = [
    ("purchase_orders",    _purchase_orders),
    ("time_series_metrics", _time_series_metrics),
]

# Grouped multipage generators (use paginate_grouped_table)
_MP_GROUPED_GENERATORS = [
    ("grouped_ledger",   _grouped_ledger),
    ("project_tracker",  _project_tracker),
]


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------

def _apply_formatting_perturbation(
    rng: np.random.Generator,
    headers: list[str],
    rows: list[list[Any]],
    table_type: str,
) -> tuple[list[str], list[list[Any]]]:
    """Apply currency / % / scientific formatting to numeric columns."""
    formatted_rows = []
    for row in rows:
        new_row = []
        for i, cell in enumerate(row):
            if isinstance(cell, float):
                choice = int(rng.integers(0, 3))
                if choice == 0:
                    cell = _fmt_currency(rng, cell)
                elif choice == 1:
                    cell = _fmt_percent(cell) if 0 <= cell <= 100 else _fmt_scientific(cell)
                else:
                    cell = f"{cell:.2f}"
            new_row.append(cell)
        formatted_rows.append(new_row)
    return headers, formatted_rows


def _apply_structural_perturbation(
    rng: np.random.Generator,
    headers: list[str],
    rows: list[list[Any]],
) -> tuple[list[str], list[list[Any]]]:
    """Introduce missing cells in some rows."""
    new_rows = []
    for row in rows:
        new_row = list(row)
        if rng.random() < 0.2 and len(new_row) > 2:
            col = int(rng.integers(1, len(new_row)))
            new_row[col] = ""
        new_rows.append(new_row)
    return headers, new_rows


def _apply_noise_perturbation(
    rng: np.random.Generator,
    headers: list[str],
    rows: list[list[Any]],
) -> tuple[list[str], list[list[Any]]]:
    """Add typos to string cells."""
    new_rows = []
    for row in rows:
        new_row = [
            _maybe_typo(rng, str(cell), prob=0.08) if isinstance(cell, str) else cell
            for cell in row
        ]
        new_rows.append(new_row)
    return headers, new_rows


def _apply_semantic_perturbation(
    rng: np.random.Generator,
    headers: list[str],
    rows: list[list[Any]],
) -> tuple[list[str], list[list[Any]]]:
    """Reorder columns randomly."""
    if len(headers) <= 2:
        return headers, rows
    # Keep first column fixed, shuffle the rest
    perm = list(range(1, len(headers)))
    rng.shuffle(perm)
    perm = [0] + perm
    new_headers = [headers[i] for i in perm]
    new_rows = [[row[i] for i in perm] for row in rows]
    return new_headers, new_rows


# ---------------------------------------------------------------------------
# HTML table rendering
# ---------------------------------------------------------------------------

def _fmt_cell(v: Any) -> str:
    """Format a cell value for display — rounds bare floats to 2 d.p."""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _esc(text: str) -> str:
    """Minimal HTML escaping for cell content."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _parse_numeric(v: Any) -> float | None:
    """Try to parse a cell value as a float; return None on failure."""
    try:
        return float(str(v).replace(",", "").lstrip("$€£¥").rstrip("%"))
    except (ValueError, TypeError, AttributeError):
        return None


def _detect_numeric_columns(
    headers: list[str],
    rows: list[list[Any]],
    min_fraction: float = 0.8,
) -> list[int]:
    """Return column indices where ≥min_fraction of non-empty values parse as float.

    Always skips col 0 (treated as ID/label column).
    """
    if not rows:
        return []
    numeric_cols = []
    for col_idx in range(1, len(headers)):
        vals = [row[col_idx] for row in rows if col_idx < len(row) and str(row[col_idx]).strip() != ""]
        if not vals:
            continue
        n_numeric = sum(1 for v in vals if _parse_numeric(v) is not None)
        if n_numeric / len(vals) >= min_fraction:
            numeric_cols.append(col_idx)
    return numeric_cols


def _smart_column_widths(headers: list[str], rows: list[list[Any]]) -> list[int]:
    """Compute content-aware percentage column widths.

    Uses header name heuristics and actual cell content length to assign wider
    allocations to text-heavy columns (descriptions, notes) and narrower ones
    to short-data columns (dates, IDs, status, numeric amounts).

    Returns a list of integers that sum to exactly 100.
    """
    n = len(headers)
    if n == 0:
        return []
    if n == 1:
        return [100]

    weights: list[float] = []
    for h in headers:
        hl = h.lower().strip()
        # Very narrow: sequence / ID / reference columns
        if any(k in hl for k in ("#", "id", "no.", " no", "seq", "ref", "num")):
            weights.append(0.65)
        # Narrow: dates and time columns
        elif any(k in hl for k in ("date", "day", "month", "year", "time", "period", "quarter")):
            weights.append(0.85)
        # Narrow: short categorical / numeric columns
        elif any(k in hl for k in (
            "status", "type", "code", "flag", "tier", "level", "class",
            "priority", "version", "phase", "stage",
        )):
            weights.append(1.0)
        elif any(k in hl for k in (
            "amount", "total", "value", "price", "cost", "budget",
            "qty", "quantity", "count", "units", "rate", "pct",
            "percent", "%", "score", "size", "balance", "fee",
        )):
            weights.append(1.0)
        # Medium-wide: names, owners, departments, vendors
        elif any(k in hl for k in (
            "name", "title", "owner", "assigned", "dept", "department",
            "region", "area", "unit", "team", "contact", "vendor",
            "category", "product", "by", "from", "to",
        )):
            weights.append(1.7)
        # Wide: free-text descriptive columns
        elif any(k in hl for k in (
            "description", "detail", "notes", "note", "comment",
            "summary", "item", "purpose", "objective", "action",
            "measure", "finding", "recommendation", "observation",
            "risk", "mitigation", "impact", "treatment", "rationale",
            "justification", "remarks",
        )):
            weights.append(2.8)
        else:
            weights.append(1.4)

    # Boost columns whose actual content is long if currently tagged narrow
    if rows:
        for i in range(n):
            max_len = max(
                (len(str(row[i])) if i < len(row) else 0 for row in rows),
                default=0,
            )
            max_len = max(max_len, len(headers[i]))
            if max_len > 30 and weights[i] < 1.5:
                weights[i] = max(weights[i], 1.5)

    # Normalize to integers, enforce 8% minimum per column
    total = sum(weights)
    pcts = [max(8, round(w / total * 100)) for w in weights]

    # Clamp to exactly 100
    while sum(pcts) > 100:
        idx = max(range(len(pcts)), key=lambda i: pcts[i] if pcts[i] > 8 else -1)
        pcts[idx] -= 1
    while sum(pcts) < 100:
        idx = max(range(len(pcts)), key=lambda i: pcts[i])
        pcts[idx] += 1

    return pcts


def _to_html_table(
    headers: list[str],
    rows: list[list[Any]],
    footer_rows: list[list[Any]] | None = None,
    continued: bool = False,
    row_range: tuple[int, int] | None = None,
) -> str:
    """Render an HTML table with content-aware per-column widths.

    Uses HTML width= attributes (most reliable in xhtml2pdf/reportlab) with
    smart column widths from _smart_column_widths(), which gives text-heavy
    columns (description, notes) proportionally more space than narrow columns
    (date, status, ID, numeric amounts), preventing cell overflow.

    Args:
        headers: Column headers.
        rows: Data rows.
        footer_rows: Optional footer rows rendered with bold blue footer style.
        continued: If True, inserts a "(continued)" sub-row after the header.
        row_range: If given, appends a "Rows N-M" annotation row before footer.
    """
    n = len(headers)
    widths = _smart_column_widths(headers, rows)

    # Shared inline styles — word-wrap aids renderers that respect it
    BASE = (
        "font-size:7.5pt; padding:4px 6px; vertical-align:top; "
        "border:1px solid #cccccc; text-align:left; "
        "word-wrap:break-word; overflow-wrap:break-word;"
    )
    TH_STYLE = (
        "font-size:7.5pt; padding:5px 6px; vertical-align:middle; "
        "border:1px solid #aaaaaa; text-align:left; font-weight:bold; "
        "background-color:#dce6f1; word-wrap:break-word;"
    )
    EVEN_BG = "background-color:#f4f8ff;"
    ODD_BG  = "background-color:#ffffff;"
    FOOTER_STYLE = (
        "font-size:7.5pt; padding:4px 6px; vertical-align:top; "
        "border:1px solid #aaaaaa; text-align:left; font-weight:bold; "
        "background-color:#dce6f1; border-top:2px solid #555555; "
        "word-wrap:break-word;"
    )
    ANNOT_STYLE = (
        "font-size:7pt; color:#666666; padding:2px 6px; "
        "border:1px solid #cccccc; text-align:right; background-color:#f8f8f8;"
    )

    lines: list[str] = []
    lines.append(
        '<table border="0" cellspacing="0" cellpadding="0" '
        'width="100%" style="border-collapse:collapse; table-layout:fixed; '
        'font-size:7.5pt; word-wrap:break-word;">'
    )

    # Colgroup: explicitly declare each column width so table-layout:fixed works
    lines.append("<colgroup>")
    for w in widths:
        lines.append(f'  <col width="{w}%" />')
    lines.append("</colgroup>")

    # Header row
    lines.append("<tr>")
    for h, w in zip(headers, widths):
        lines.append(f'<th width="{w}%" style="{TH_STYLE}">{_esc(h)}</th>')
    lines.append("</tr>")

    # (continued) sub-row
    if continued:
        lines.append("<tr>")
        lines.append(
            f'<td colspan="{n}" style="{ANNOT_STYLE} text-align:center;">'
            f"(continued)</td>"
        )
        lines.append("</tr>")

    # Data rows
    for i, row in enumerate(rows):
        row_bg = EVEN_BG if i % 2 == 0 else ODD_BG
        lines.append("<tr>")
        for j in range(n):
            c = row[j] if j < len(row) else ""
            w = widths[j]
            lines.append(
                f'<td width="{w}%" style="{BASE} {row_bg}">'
                f"{_esc(_fmt_cell(c))}</td>"
            )
        lines.append("</tr>")

    # Row range annotation
    if row_range is not None:
        lines.append("<tr>")
        lines.append(
            f'<td colspan="{n}" style="{ANNOT_STYLE}">'
            f"Rows {row_range[0]}\u2013{row_range[1]}</td>"
        )
        lines.append("</tr>")

    # Footer rows
    if footer_rows:
        for frow in footer_rows:
            lines.append("<tr>")
            for j in range(n):
                c = frow[j] if j < len(frow) else ""
                w = widths[j]
                lines.append(
                    f'<td width="{w}%" style="{FOOTER_STYLE}">'
                    f"{_esc(_fmt_cell(c))}</td>"
                )
            lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


def _insert_group_subtotals(
    headers: list[str],
    rows: list[list[Any]],
    group_col: int = 0,
) -> tuple[list[list[Any]], set[int]]:
    """Insert a subtotal row after every group (rows sharing the same group_col value).

    Returns (enriched_rows, subtotal_indices) where subtotal_indices is the set of
    row indices (in enriched_rows) that are subtotal rows.
    """
    if len(rows) < 4:
        return rows, set()
    group_vals = [str(r[group_col]) if group_col < len(r) else "" for r in rows]
    unique_groups = list(dict.fromkeys(group_vals))
    if not (2 <= len(unique_groups) <= 12):
        return rows, set()
    numeric_cols = _detect_numeric_columns(headers, rows)
    if not numeric_cols:
        return rows, set()

    result: list[list[Any]] = []
    subtotal_indices: set[int] = set()
    current_group: list[list[Any]] = []
    current_val = group_vals[0]

    def _flush_group(gval: str, grows: list[list[Any]]) -> None:
        result.extend(grows)
        sub: list[Any] = [""] * len(headers)
        sub[group_col] = f"{gval} \u2014 Subtotal"
        for ci in numeric_cols:
            sub[ci] = round(sum(_parse_numeric(r[ci]) or 0.0 for r in grows if ci < len(r)), 2)
        subtotal_indices.add(len(result))
        result.append(sub)

    for row, val in zip(rows, group_vals):
        if val != current_val and current_group:
            _flush_group(current_val, current_group)
            current_group = [row]
            current_val = val
        else:
            current_group.append(row)
    if current_group:
        _flush_group(current_val, current_group)

    return result, subtotal_indices


def _to_html_table_grouped(
    headers: list[str],
    rows: list[list[Any]],
    column_groups: list[tuple[str, int]],
    subtotal_row_indices: set[int],
    footer_rows: list[list[Any]] | None = None,
    continued: bool = False,
    row_range: tuple[int, int] | None = None,
) -> str:
    """Render an HTML table with a 2-tier grouped header and styled subtotal rows."""
    n = len(headers)
    widths = _smart_column_widths(headers, rows)

    BASE          = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; "
                     "border:1px solid #cccccc; text-align:left; "
                     "word-wrap:break-word; overflow-wrap:break-word;")
    TH_STYLE      = ("font-size:7.5pt; padding:5px 6px; vertical-align:middle; "
                     "border:1px solid #aaaaaa; text-align:left; font-weight:bold; "
                     "background-color:#dce6f1; word-wrap:break-word;")
    GROUP_TH      = ("font-size:7.5pt; padding:5px 6px; vertical-align:middle; "
                     "border:1px solid #8899bb; text-align:center; font-weight:bold; "
                     "background-color:#b8cce4; word-wrap:break-word;")
    SUBTOTAL_STYLE= ("font-size:7.5pt; padding:4px 6px; vertical-align:top; "
                     "border:1px solid #aaaaaa; text-align:left; font-weight:bold; "
                     "background-color:#e8f0e8; border-top:1px solid #666666; "
                     "word-wrap:break-word;")
    EVEN_BG       = "background-color:#f4f8ff;"
    ODD_BG        = "background-color:#ffffff;"
    FOOTER_STYLE  = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; "
                     "border:1px solid #aaaaaa; text-align:left; font-weight:bold; "
                     "background-color:#dce6f1; border-top:2px solid #555555; "
                     "word-wrap:break-word;")
    ANNOT_STYLE   = ("font-size:7pt; color:#666666; padding:2px 6px; "
                     "border:1px solid #cccccc; text-align:right; background-color:#f8f8f8;")

    lines: list[str] = []
    lines.append(
        '<table border="0" cellspacing="0" cellpadding="0" '
        'width="100%" style="border-collapse:collapse; table-layout:fixed; '
        'font-size:7.5pt; word-wrap:break-word;">'
    )
    lines.append("<colgroup>")
    for w in widths:
        lines.append(f'  <col width="{w}%" />')
    lines.append("</colgroup>")

    # Row 1: column group headers (spans)
    if column_groups:
        lines.append("<tr>")
        for group_name, span in column_groups:
            lines.append(f'<th colspan="{span}" style="{GROUP_TH}">{_esc(group_name)}</th>')
        lines.append("</tr>")

    # Row 2: individual column headers
    lines.append("<tr>")
    for h, w in zip(headers, widths):
        lines.append(f'<th width="{w}%" style="{TH_STYLE}">{_esc(h)}</th>')
    lines.append("</tr>")

    if continued:
        lines.append("<tr>")
        lines.append(f'<td colspan="{n}" style="{ANNOT_STYLE} text-align:center;">(continued)</td>')
        lines.append("</tr>")

    # Data rows — subtotals get special styling
    data_row_idx = 0  # index among non-subtotal rows for alternating colour
    for i, row in enumerate(rows):
        if i in subtotal_row_indices:
            lines.append("<tr>")
            for j in range(n):
                c = row[j] if j < len(row) else ""
                lines.append(f'<td width="{widths[j]}%" style="{SUBTOTAL_STYLE}">{_esc(_fmt_cell(c))}</td>')
            lines.append("</tr>")
        else:
            row_bg = EVEN_BG if data_row_idx % 2 == 0 else ODD_BG
            data_row_idx += 1
            lines.append("<tr>")
            for j in range(n):
                c = row[j] if j < len(row) else ""
                lines.append(f'<td width="{widths[j]}%" style="{BASE} {row_bg}">{_esc(_fmt_cell(c))}</td>')
            lines.append("</tr>")

    if row_range is not None:
        lines.append("<tr>")
        lines.append(f'<td colspan="{n}" style="{ANNOT_STYLE}">Rows {row_range[0]}\u2013{row_range[1]}</td>')
        lines.append("</tr>")

    if footer_rows:
        for frow in footer_rows:
            lines.append("<tr>")
            for j in range(n):
                c = frow[j] if j < len(frow) else ""
                lines.append(f'<td width="{widths[j]}%" style="{FOOTER_STYLE}">{_esc(_fmt_cell(c))}</td>')
            lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


def paginate_grouped_table(
    headers: list[str],
    rows: list[list[Any]],
    rows_per_page: int = 20,
    column_groups: list[tuple[str, int]] | None = None,
    group_col: int = 0,
) -> str:
    """Paginate a grouped table with subtotal rows and optional 2-tier column headers.

    Inserts a subtotal row after every group (rows sharing the same value in group_col),
    renders a two-row header (group spans + column names) when column_groups is given,
    and adds per-page and grand-total footer rows.  Page breaks are separated by
    ``<div class="synthpdf-page-break"></div>``.
    """
    if not rows:
        return _to_html_table(headers, rows)

    enriched_rows, subtotal_indices = _insert_group_subtotals(headers, rows, group_col)
    numeric_cols = _detect_numeric_columns(headers, rows)
    grand_totals: dict[int, float] = {ci: 0.0 for ci in numeric_cols}
    for row in rows:
        for ci in numeric_cols:
            v = _parse_numeric(row[ci]) if ci < len(row) else None
            if v is not None:
                grand_totals[ci] += v

    n_pages = max(1, math.ceil(len(enriched_rows) / rows_per_page))
    page_htmls: list[str] = []

    for page_idx in range(n_pages):
        start     = page_idx * rows_per_page
        end       = min(start + rows_per_page, len(enriched_rows))
        page_rows = enriched_rows[start:end]
        is_last   = page_idx == n_pages - 1
        page_sub  = {i - start for i in subtotal_indices if start <= i < end}

        footer_rows: list[list[Any]] = []
        if numeric_cols:
            non_sub = [r for i, r in enumerate(page_rows) if i not in page_sub]
            if non_sub:
                pg_sub_row: list[Any] = ["Page Subtotal"] + [""] * (len(headers) - 1)
                for ci in numeric_cols:
                    pg_sub_row[ci] = round(sum(_parse_numeric(r[ci]) or 0.0 for r in non_sub if ci < len(r)), 2)
                footer_rows.append(pg_sub_row)
            if is_last:
                grand_row: list[Any] = ["GRAND TOTAL"] + [""] * (len(headers) - 1)
                for ci in numeric_cols:
                    grand_row[ci] = round(grand_totals[ci], 2)
                footer_rows.append(grand_row)

        page_html = _to_html_table_grouped(
            headers, page_rows,
            column_groups=column_groups or [],
            subtotal_row_indices=page_sub,
            footer_rows=footer_rows or None,
            continued=(page_idx > 0),
            row_range=(start + 1, end),
        )
        page_htmls.append(page_html)

    return '\n<div class="synthpdf-page-break"></div>\n'.join(page_htmls)


def paginate_table(
    headers: list[str],
    rows: list[list[Any]],
    rows_per_page: int = 20,
) -> str:
    """Split a large table into multiple pages joined by page-break divs.

    Each page gets its own <table> with header, optional (continued) marker,
    per-page subtotals for numeric columns, and grand totals on the last page.

    Returns:
        HTML string with <div class="synthpdf-page-break"></div> between pages.
    """
    if not rows:
        return _to_html_table(headers, rows)

    numeric_cols = _detect_numeric_columns(headers, rows)

    # Pre-compute grand totals for numeric columns
    grand_totals: dict[int, float] = {ci: 0.0 for ci in numeric_cols}
    for row in rows:
        for ci in numeric_cols:
            v = _parse_numeric(row[ci]) if ci < len(row) else None
            if v is not None:
                grand_totals[ci] += v

    # Chunk rows into pages
    n_pages = max(1, math.ceil(len(rows) / rows_per_page))
    page_htmls: list[str] = []

    for page_idx in range(n_pages):
        start = page_idx * rows_per_page
        end = min(start + rows_per_page, len(rows))
        page_rows = rows[start:end]
        is_last = page_idx == n_pages - 1

        # Per-page subtotals
        footer_rows: list[list[Any]] = []
        if numeric_cols:
            subtotal_row: list[Any] = ["Subtotal" if not is_last else "Subtotal"] + [""] * (len(headers) - 1)
            for ci in numeric_cols:
                page_sum = sum(
                    _parse_numeric(r[ci]) or 0.0
                    for r in page_rows if ci < len(r)
                )
                subtotal_row[ci] = round(page_sum, 2)
            footer_rows.append(subtotal_row)

            # Grand total on last page only
            if is_last:
                grand_row: list[Any] = ["GRAND TOTAL"] + [""] * (len(headers) - 1)
                for ci in numeric_cols:
                    grand_row[ci] = round(grand_totals[ci], 2)
                footer_rows.append(grand_row)

        page_html = _to_html_table(
            headers,
            page_rows,
            footer_rows=footer_rows if footer_rows else None,
            continued=(page_idx > 0),
            row_range=(start + 1, end),
        )
        page_htmls.append(page_html)

    return '\n<div class="synthpdf-page-break"></div>\n'.join(page_htmls)


# ---------------------------------------------------------------------------
# Complex table HTML renderers
# ---------------------------------------------------------------------------

def _to_html_merged_cells(
    group_col_label: str,
    col_headers: list[str],
    row_groups: list[tuple[str, list[list[Any]]]],
) -> str:
    """P&L / category-breakdown style: group label cell uses rowspan over its rows."""
    all_flat = [r for _, rows in row_groups for r in rows]
    all_headers = [group_col_label] + col_headers
    widths = _smart_column_widths(all_headers, all_flat)

    TH   = ("font-size:7.5pt; padding:5px 6px; vertical-align:middle; border:1px solid #aaaaaa; "
            "text-align:left; font-weight:bold; background-color:#dce6f1; word-wrap:break-word;")
    TD   = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; border:1px solid #cccccc; "
            "text-align:left; word-wrap:break-word; overflow-wrap:break-word;")
    GRP  = ("font-size:7.5pt; padding:4px 6px; vertical-align:middle; border:1px solid #8899bb; "
            "text-align:center; font-weight:bold; background-color:#dce6f1;")
    EVEN = "background-color:#f4f8ff;"
    ODD  = "background-color:#ffffff;"

    lines: list[str] = []
    lines.append('<table border="0" cellspacing="0" cellpadding="0" width="100%" '
                 'style="border-collapse:collapse; table-layout:fixed; font-size:7.5pt;">')
    lines.append("<colgroup>")
    for w in widths:
        lines.append(f'  <col width="{w}%" />')
    lines.append("</colgroup>")

    lines.append("<thead><tr>")
    for h in all_headers:
        lines.append(f'<th style="{TH}">{_esc(h)}</th>')
    lines.append("</tr></thead><tbody>")

    flat_idx = 0
    for gname, grows in row_groups:
        n_g = len(grows)
        for j, row in enumerate(grows):
            bg = EVEN if flat_idx % 2 == 0 else ODD
            flat_idx += 1
            lines.append("<tr>")
            if j == 0:
                lines.append(f'<td rowspan="{n_g}" style="{GRP}">{_esc(gname)}</td>')
            for cell in row:
                lines.append(f'<td style="{TD} {bg}">{_esc(_fmt_cell(cell))}</td>')
            lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _to_html_nested_table(
    outer_headers: list[str],
    outer_rows: list[list[Any]],
    nested_data: list[tuple[list[str], list[list[Any]]]],
    nested_col: int,
) -> str:
    """Outer table with a mini nested sub-table embedded in one column per row."""
    widths = _smart_column_widths(outer_headers, outer_rows)
    TH       = ("font-size:7.5pt; padding:5px 6px; vertical-align:middle; border:1px solid #aaaaaa; "
                "font-weight:bold; background-color:#dce6f1; word-wrap:break-word;")
    TD       = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; "
                "border:1px solid #cccccc; word-wrap:break-word;")
    INNER_TH = ("font-size:6.5pt; padding:2px 4px; border:1px solid #99aacc; "
                "font-weight:bold; background-color:#e0e8f4; text-align:left;")
    INNER_TD = "font-size:6.5pt; padding:2px 4px; border:1px solid #cccccc; text-align:left;"
    EVEN     = "background-color:#f4f8ff;"
    ODD      = "background-color:#ffffff;"

    lines: list[str] = []
    lines.append('<table border="0" cellspacing="0" cellpadding="0" width="100%" '
                 'style="border-collapse:collapse; table-layout:fixed; font-size:7.5pt;">')
    lines.append("<colgroup>")
    for w in widths:
        lines.append(f'  <col width="{w}%" />')
    lines.append("</colgroup>")
    lines.append("<thead><tr>")
    for h in outer_headers:
        lines.append(f'<th style="{TH}">{_esc(h)}</th>')
    lines.append("</tr></thead><tbody>")

    for i, (row, (n_hdrs, n_rows_inner)) in enumerate(zip(outer_rows, nested_data)):
        bg = EVEN if i % 2 == 0 else ODD
        lines.append("<tr>")
        for ci, cell in enumerate(row):
            if ci == nested_col:
                inner = [
                    '<table border="0" cellspacing="0" cellpadding="0" '
                    'style="border-collapse:collapse; width:100%; font-size:6.5pt;">',
                    "<tr>",
                ]
                for nh in n_hdrs:
                    inner.append(f'<th style="{INNER_TH}">{_esc(nh)}</th>')
                inner.append("</tr>")
                for nr in n_rows_inner:
                    inner.append("<tr>")
                    for nc in nr:
                        inner.append(f'<td style="{INNER_TD}">{_esc(str(nc))}</td>')
                    inner.append("</tr>")
                inner.append("</table>")
                lines.append(f'<td style="{TD} {bg}">{"".join(inner)}</td>')
            else:
                lines.append(f'<td style="{TD} {bg}">{_esc(_fmt_cell(cell))}</td>')
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _to_html_footnoted(
    headers: list[str],
    rows: list[list[Any]],
    footnotes: list[str],
    footnote_map: dict[tuple[int, int], int],
) -> str:
    """Table with superscript footnote markers in cells and a footnote section below."""
    widths = _smart_column_widths(headers, rows)
    TH   = ("font-size:7.5pt; padding:5px 6px; vertical-align:middle; border:1px solid #aaaaaa; "
            "font-weight:bold; background-color:#dce6f1; word-wrap:break-word;")
    TD   = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; "
            "border:1px solid #cccccc; word-wrap:break-word;")
    SUP  = "font-size:6pt; vertical-align:super; color:#555555;"
    EVEN = "background-color:#f4f8ff;"
    ODD  = "background-color:#ffffff;"

    lines: list[str] = []
    lines.append('<table border="0" cellspacing="0" cellpadding="0" width="100%" '
                 'style="border-collapse:collapse; table-layout:fixed; font-size:7.5pt;">')
    lines.append("<colgroup>")
    for w in widths:
        lines.append(f'  <col width="{w}%" />')
    lines.append("</colgroup>")
    lines.append("<thead><tr>")
    for h in headers:
        lines.append(f'<th style="{TH}">{_esc(h)}</th>')
    lines.append("</tr></thead><tbody>")

    for i, row in enumerate(rows):
        bg = EVEN if i % 2 == 0 else ODD
        lines.append("<tr>")
        for j, cell in enumerate(row):
            marker = footnote_map.get((i, j))
            marker_html = f'<sup style="{SUP}">{marker}</sup>' if marker else ""
            lines.append(f'<td style="{TD} {bg}">{_esc(_fmt_cell(cell))}{marker_html}</td>')
        lines.append("</tr>")

    lines.append("</tbody></table>")
    if footnotes:
        lines.append(
            '<div style="font-size:6.5pt; color:#444444; margin-top:4px; '
            'padding:3px 6px; border-top:1px solid #bbbbbb;">'
        )
        for k, note in enumerate(footnotes, 1):
            lines.append(f'<span style="{SUP}">{k}</span>&nbsp;{_esc(note)}<br/>')
        lines.append("</div>")
    return "\n".join(lines)


def _to_html_matrix(
    corner_label: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[Any]],
) -> str:
    """Cross-reference matrix (RACI / competency / heat-map) with color-coded cells."""
    _COLORS: dict[str, str] = {
        "Expert": "#1a7a1a", "Proficient": "#4a8e4a", "Developing": "#b88020",
        "Beginner": "#b85020", "N/A": "#888888",
        "R": "#1a5276", "A": "#154360", "C": "#2471a3", "I": "#5dade2",
        "High": "#b03030", "Medium": "#b88020", "Low": "#1a7a1a",
        "Critical": "#7b0000",
    }
    n_cols = len(col_labels)
    col_w = max(7, 75 // max(1, n_cols))
    label_w = 100 - col_w * n_cols
    widths = [max(10, label_w)] + [col_w] * n_cols

    TH      = ("font-size:7pt; padding:4px 5px; vertical-align:middle; border:1px solid #aaaaaa; "
               "font-weight:bold; background-color:#dce6f1; text-align:center; word-wrap:break-word;")
    TH_LEFT = ("font-size:7pt; padding:4px 5px; vertical-align:middle; border:1px solid #aaaaaa; "
               "font-weight:bold; background-color:#dce6f1; text-align:left;")
    TD_ROW  = ("font-size:7pt; padding:4px 5px; vertical-align:top; border:1px solid #aaaaaa; "
               "font-weight:bold; background-color:#eef3fa;")
    TD_BASE = "font-size:7pt; padding:4px 5px; border:1px solid #cccccc; text-align:center;"

    lines: list[str] = []
    lines.append('<table border="0" cellspacing="0" cellpadding="0" width="100%" '
                 'style="border-collapse:collapse; table-layout:fixed; font-size:7pt;">')
    lines.append("<colgroup>")
    for w in widths:
        lines.append(f'  <col width="{w}%" />')
    lines.append("</colgroup>")
    lines.append("<thead><tr>")
    lines.append(f'<th style="{TH_LEFT}">{_esc(corner_label)}</th>')
    for cl in col_labels:
        lines.append(f'<th style="{TH}">{_esc(cl)}</th>')
    lines.append("</tr></thead><tbody>")

    for rl, row_vals in zip(row_labels, values):
        lines.append("<tr>")
        lines.append(f'<td style="{TD_ROW}">{_esc(rl)}</td>')
        for val in row_vals:
            val_str = str(val)
            color = _COLORS.get(val_str)
            style = f"{TD_BASE} color:{color}; font-weight:bold;" if color else TD_BASE
            lines.append(f'<td style="{style}">{_esc(val_str)}</td>')
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _to_html_ragged_colspan(
    header_groups: list[tuple[str, int]],
    sub_headers: list[str],
    rows: list[list[Any]],
) -> str:
    """Two-tier header where each top-level group spans an irregular number of columns."""
    widths = _smart_column_widths(sub_headers, rows)
    n = len(sub_headers)
    GROUP_TH = ("font-size:7.5pt; padding:4px 6px; vertical-align:middle; border:1px solid #8899bb; "
                "text-align:center; font-weight:bold; background-color:#b8cce4;")
    TH       = ("font-size:7.5pt; padding:5px 6px; vertical-align:middle; border:1px solid #aaaaaa; "
                "text-align:left; font-weight:bold; background-color:#dce6f1;")
    TD       = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; "
                "border:1px solid #cccccc; word-wrap:break-word;")
    EVEN     = "background-color:#f4f8ff;"
    ODD      = "background-color:#ffffff;"

    lines: list[str] = []
    lines.append('<table border="0" cellspacing="0" cellpadding="0" width="100%" '
                 'style="border-collapse:collapse; table-layout:fixed; font-size:7.5pt;">')
    lines.append("<colgroup>")
    for w in widths:
        lines.append(f'  <col width="{w}%" />')
    lines.append("</colgroup>")
    lines.append("<thead><tr>")
    for gname, gspan in header_groups:
        lines.append(f'<th colspan="{gspan}" style="{GROUP_TH}">{_esc(gname)}</th>')
    lines.append("</tr><tr>")
    for sh in sub_headers:
        lines.append(f'<th style="{TH}">{_esc(sh)}</th>')
    lines.append("</tr></thead><tbody>")

    for i, row in enumerate(rows):
        bg = EVEN if i % 2 == 0 else ODD
        lines.append("<tr>")
        for j in range(n):
            c = row[j] if j < len(row) else ""
            lines.append(f'<td style="{TD} {bg}">{_esc(_fmt_cell(c))}</td>')
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _to_html_stacked_comparison(
    blocks: list[dict],
    shared_headers: list[str],
) -> str:
    """Multiple comparison sections stacked in a single table with bold section title rows."""
    all_rows_flat = [r for b in blocks for r in b.get("rows", [])]
    widths = _smart_column_widths(shared_headers, all_rows_flat)
    n = len(shared_headers)
    TH         = ("font-size:7.5pt; padding:5px 6px; vertical-align:middle; border:1px solid #aaaaaa; "
                  "font-weight:bold; background-color:#dce6f1;")
    SECTION_TD = ("font-size:7.5pt; padding:5px 6px; font-weight:bold; "
                  "background-color:#2c4770; color:#ffffff; border:1px solid #1a2e4a;")
    TD         = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; "
                  "border:1px solid #cccccc; word-wrap:break-word;")
    TOTAL_TD   = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; font-weight:bold; "
                  "border:1px solid #aaaaaa; background-color:#dce6f1;")
    EVEN       = "background-color:#f4f8ff;"
    ODD        = "background-color:#ffffff;"

    lines: list[str] = []
    lines.append('<table border="0" cellspacing="0" cellpadding="0" width="100%" '
                 'style="border-collapse:collapse; table-layout:fixed; font-size:7.5pt;">')
    lines.append("<colgroup>")
    for w in widths:
        lines.append(f'  <col width="{w}%" />')
    lines.append("</colgroup>")
    lines.append("<thead><tr>")
    for h in shared_headers:
        lines.append(f'<th style="{TH}">{_esc(h)}</th>')
    lines.append("</tr></thead><tbody>")

    row_counter = 0
    for block in blocks:
        lines.append("<tr>")
        lines.append(f'<td colspan="{n}" style="{SECTION_TD}">{_esc(block["title"])}</td>')
        lines.append("</tr>")
        for row in block.get("rows", []):
            is_total = str(row[0]).upper().startswith("TOTAL") if row else False
            bg = EVEN if row_counter % 2 == 0 else ODD
            row_counter += 1
            lines.append("<tr>")
            for j in range(n):
                c = row[j] if j < len(row) else ""
                s = TOTAL_TD if is_total else f"{TD} {bg}"
                lines.append(f'<td style="{s}">{_esc(_fmt_cell(c))}</td>')
            lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _to_html_transposed(
    row_header: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[Any]],
) -> str:
    """Transposed metrics table: metrics as rows, time periods as columns."""
    all_headers = [row_header] + col_labels
    all_rows = [[rl] + list(vals) for rl, vals in zip(row_labels, values)]
    widths = _smart_column_widths(all_headers, all_rows)

    TH_LEFT   = ("font-size:7.5pt; padding:5px 6px; vertical-align:middle; border:1px solid #aaaaaa; "
                 "font-weight:bold; background-color:#dce6f1; text-align:left;")
    TH_CENTER = ("font-size:7.5pt; padding:5px 6px; vertical-align:middle; border:1px solid #aaaaaa; "
                 "font-weight:bold; background-color:#dce6f1; text-align:center;")
    TH_TOTAL  = ("font-size:7.5pt; padding:5px 6px; vertical-align:middle; border:1px solid #8899bb; "
                 "font-weight:bold; background-color:#b8cce4; text-align:center;")
    TD_METRIC = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; border:1px solid #aaaaaa; "
                 "font-weight:bold; background-color:#eef3fa;")
    TD        = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; border:1px solid #cccccc; "
                 "text-align:right; word-wrap:break-word;")
    TD_TOTAL  = ("font-size:7.5pt; padding:4px 6px; vertical-align:top; font-weight:bold; "
                 "border:1px solid #8899bb; background-color:#dce6f1; text-align:right;")
    EVEN      = "background-color:#f4f8ff;"
    ODD       = "background-color:#ffffff;"

    _TOTAL_LABELS = {"total", "fy total", "annual", "ytd", "full year", "grand total"}

    lines: list[str] = []
    lines.append('<table border="0" cellspacing="0" cellpadding="0" width="100%" '
                 'style="border-collapse:collapse; table-layout:fixed; font-size:7.5pt;">')
    lines.append("<colgroup>")
    for w in widths:
        lines.append(f'  <col width="{w}%" />')
    lines.append("</colgroup>")
    lines.append("<thead><tr>")
    lines.append(f'<th style="{TH_LEFT}">{_esc(row_header)}</th>')
    for cl in col_labels:
        is_total = cl.lower() in _TOTAL_LABELS
        lines.append(f'<th style="{TH_TOTAL if is_total else TH_CENTER}">{_esc(cl)}</th>')
    lines.append("</tr></thead><tbody>")

    for i, (rl, vals) in enumerate(zip(row_labels, values)):
        bg = EVEN if i % 2 == 0 else ODD
        lines.append("<tr>")
        lines.append(f'<td style="{TD_METRIC}">{_esc(rl)}</td>')
        for ci, val in enumerate(vals):
            col = col_labels[ci] if ci < len(col_labels) else ""
            is_total = col.lower() in _TOTAL_LABELS
            s = TD_TOTAL if is_total else f"{TD} {bg}"
            lines.append(f'<td style="{s}">{_esc(_fmt_cell(val))}</td>')
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Complex table data generators
# Each returns (html_str, flat_headers, flat_rows, type_name)
# ---------------------------------------------------------------------------

def _gen_pl_statement(
    rng: np.random.Generator,
) -> tuple[str, list[str], list[list[Any]], str]:
    """P&L with rowspan-merged category cells (merged_cells variant)."""
    groups_def = [
        ("Revenue",           ["Product Sales", "Service Revenue", "Licensing Fees", "Other Revenue"]),
        ("Cost of Revenue",   ["Materials & Components", "Direct Labor", "Mfg. Overhead"]),
        ("Operating Expenses",["Research & Development", "Sales & Marketing", "G&A"]),
        ("Non-Operating",     ["Interest Income", "Interest Expense", "Other Income/(Expense)"]),
    ]
    n_groups = int(rng.integers(3, len(groups_def) + 1))
    col_headers = ["Line Item", "Q1 ($K)", "Q2 ($K)", "Q3 ($K)", "Q4 ($K)", "Annual ($K)"]

    row_groups: list[tuple[str, list[list[Any]]]] = []
    flat_rows: list[list[Any]] = []
    for gname, items in groups_def[:n_groups]:
        n_items = int(rng.integers(2, len(items) + 1))
        grows: list[list[Any]] = []
        for item in items[:n_items]:
            q = [round(float(rng.uniform(500, 50_000)), 1) for _ in range(4)]
            grows.append([item] + q + [round(sum(q), 1)])
            flat_rows.append(grows[-1])
        row_groups.append((gname, grows))

    html = _to_html_merged_cells("Category", col_headers, row_groups)
    return html, ["Category"] + col_headers, flat_rows, "pl_statement"


def _gen_dept_contacts(
    rng: np.random.Generator,
) -> tuple[str, list[str], list[list[Any]], str]:
    """Department roster with nested key-contacts sub-table (nested_table variant)."""
    depts    = ["Engineering", "Product", "Finance", "Legal", "Operations",
                "Marketing", "HR", "Customer Success"]
    firsts   = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Iris"]
    lasts    = ["Smith", "Jones", "Lee", "Park", "Brown", "Garcia", "Wilson", "Nguyen"]
    roles    = ["Tech Lead", "Manager", "Director", "Analyst", "Engineer", "Specialist"]
    cities   = ["New York", "London", "San Francisco", "Singapore", "Austin", "Seattle"]

    outer_headers = ["Department", "Head", "Headcount", "Location", "Key Contacts"]
    n_depts = int(rng.integers(4, 7))
    outer_rows: list[list[Any]] = []
    nested_data: list[tuple[list[str], list[list[Any]]]] = []
    flat_rows: list[list[Any]] = []

    for i in range(n_depts):
        dept = depts[i % len(depts)]
        head = firsts[int(rng.integers(0, len(firsts)))] + " " + lasts[int(rng.integers(0, len(lasts)))]
        hc   = int(rng.integers(5, 120))
        city = cities[int(rng.integers(0, len(cities)))]
        outer_rows.append([dept, head, hc, city, ""])  # "" is the nested col placeholder
        flat_rows.append([dept, head, hc, city])

        n_inner = int(rng.integers(2, 4))
        inner_rows: list[list[Any]] = []
        for _ in range(n_inner):
            name = firsts[int(rng.integers(0, len(firsts)))] + " " + lasts[int(rng.integers(0, len(lasts)))]
            role = roles[int(rng.integers(0, len(roles)))]
            ext  = f"x{int(rng.integers(1000, 9999))}"
            inner_rows.append([name, role, ext])
        nested_data.append((["Name", "Role", "Ext."], inner_rows))

    html = _to_html_nested_table(outer_headers, outer_rows, nested_data, nested_col=4)
    return html, outer_headers[:-1], flat_rows, "dept_contacts"


def _gen_compliance_footnoted(
    rng: np.random.Generator,
) -> tuple[str, list[str], list[list[Any]], str]:
    """Compliance control table with footnote markers (footnoted_table variant)."""
    controls   = ["Access Control Policy", "Data Encryption at Rest", "Network Firewall Rules",
                  "Vulnerability Scanning", "Incident Response Plan", "Business Continuity Plan",
                  "Third-Party Risk Assessment", "Employee Security Training",
                  "Audit Log Retention", "Patch Management Process"]
    statuses   = ["Compliant", "Partial", "Non-Compliant", "Under Review", "Exempt"]
    frameworks = ["ISO 27001", "SOC 2 Type II", "GDPR Art. 32", "HIPAA", "PCI-DSS", "NIST CSF"]
    evidences  = ["Policy Doc", "System Report", "Audit Trail", "Certificate", "Assessment Report"]
    footnote_texts = [
        "Reviewed and approved by the Chief Information Security Officer.",
        "Pending renewal; current exemption valid through Q2 2025.",
        "Third-party audit scheduled for next quarter.",
        "Automated controls verified via continuous monitoring platform.",
        "Remediation in progress; target completion date Q3 2025.",
    ]
    headers    = ["Control", "Framework", "Status", "Evidence", "Last Review", "Owner"]
    n_controls = int(rng.integers(6, 10))
    rows: list[list[Any]] = []
    footnote_map: dict[tuple[int, int], int] = {}
    used: set[int] = set()

    for i in range(n_controls):
        status = statuses[int(rng.integers(0, len(statuses)))]
        year   = 2024
        month  = int(rng.integers(1, 13))
        rows.append([
            controls[i % len(controls)],
            frameworks[int(rng.integers(0, len(frameworks)))],
            status,
            evidences[int(rng.integers(0, len(evidences)))],
            f"{year}-{month:02d}-01",
            f"Owner-{int(rng.integers(1, 8))}",
        ])
        if rng.random() < 0.40:
            fn_idx = int(rng.integers(1, len(footnote_texts) + 1))
            footnote_map[(i, 2)] = fn_idx  # on Status column
            used.add(fn_idx)

    # Renumber footnotes to be sequential
    old_to_new = {old: new for new, old in enumerate(sorted(used), 1)}
    footnote_map = {k: old_to_new[v] for k, v in footnote_map.items() if v in old_to_new}
    active = [footnote_texts[k - 1] for k in sorted(used)]

    html = _to_html_footnoted(headers, rows, active, footnote_map)
    return html, headers, rows, "compliance_footnoted"


def _gen_skill_matrix(
    rng: np.random.Generator,
) -> tuple[str, list[str], list[list[Any]], str]:
    """Team competency matrix: members x skills (matrix_table variant)."""
    skill_pools = [
        ["Python", "SQL", "Cloud (AWS)", "Docker", "CI/CD", "Data Analysis", "ML/AI", "Spark"],
        ["Project Mgmt", "Budgeting", "Risk Analysis", "Stakeholder Comms", "Agile/Scrum", "Change Mgmt"],
        ["Sales CRM", "Lead Generation", "Negotiation", "Market Research", "Forecasting", "Presentation"],
    ]
    ratings  = ["Expert", "Proficient", "Developing", "Beginner", "N/A"]
    weights  = [0.15, 0.35, 0.30, 0.15, 0.05]
    members  = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"]

    pool     = skill_pools[int(rng.integers(0, len(skill_pools)))]
    n_skills = int(rng.integers(4, min(8, len(pool)) + 1))
    col_lbls = pool[:n_skills]
    n_mem    = int(rng.integers(4, 8))
    row_lbls = [members[i % len(members)] for i in range(n_mem)]

    values: list[list[Any]] = []
    flat_rows: list[list[Any]] = []
    for rl in row_lbls:
        rv = [ratings[int(rng.choice(len(ratings), p=weights))] for _ in col_lbls]
        values.append(rv)
        flat_rows.append([rl] + rv)

    html    = _to_html_matrix("Team Member", row_lbls, col_lbls, values)
    headers = ["Team Member"] + col_lbls
    return html, headers, flat_rows, "skill_matrix"


def _gen_raci_matrix(
    rng: np.random.Generator,
) -> tuple[str, list[str], list[list[Any]], str]:
    """RACI responsibility assignment matrix (matrix_table variant)."""
    processes = ["Requirements Gathering", "Architecture Design", "Development Sprint",
                 "Testing & QA", "Deployment", "Production Support",
                 "Change Management", "Security Review", "Budget Approval", "Vendor Selection"]
    roles     = ["Product Owner", "Tech Lead", "Dev Team", "QA Team", "DevOps", "CISO", "CFO", "Procurement"]
    ratings   = ["R", "A", "C", "I", ""]
    w         = [0.22, 0.13, 0.27, 0.27, 0.11]

    n_proc = int(rng.integers(5, 9))
    n_role = int(rng.integers(4, 7))
    row_lbls = processes[:n_proc]
    col_lbls = roles[:n_role]

    values: list[list[Any]] = []
    flat_rows: list[list[Any]] = []
    for proc in row_lbls:
        rv = [ratings[int(rng.choice(len(ratings), p=w))] for _ in col_lbls]
        # Ensure every row has at least one R and one A
        rv[int(rng.integers(0, len(rv)))] = "R"
        rv[int(rng.integers(0, len(rv)))] = "A"
        values.append(rv)
        flat_rows.append([proc] + rv)

    html    = _to_html_matrix("Process / Activity", row_lbls, col_lbls, values)
    headers = ["Process / Activity"] + col_lbls
    return html, headers, flat_rows, "raci_matrix"


def _gen_budget_ragged(
    rng: np.random.Generator,
) -> tuple[str, list[str], list[list[Any]], str]:
    """Budget allocation with irregular (ragged) colspan header groupings."""
    cost_centers = ["Corp-IT", "Corp-HR", "Sales-US", "Sales-EU", "Eng-Core",
                    "Eng-Infra", "Ops-Logistics", "Ops-Facilities", "R&D-Platform", "Mktg-Digital"]
    header_groups = [
        ("Cost Center", 1),
        ("Q1 Actuals ($K)", 3),
        ("Q2 Actuals ($K)", 3),
        ("Variance", 2),
    ]
    sub_headers = [
        "Center",
        "Personnel", "OpEx", "CapEx",
        "Personnel", "OpEx", "CapEx",
        "Abs. ($K)", "Rel. (%)",
    ]
    n = int(rng.integers(5, 10))
    rows: list[list[Any]] = []
    for i in range(n):
        cc  = cost_centers[i % len(cost_centers)]
        q1p = round(float(rng.uniform(50, 500)), 1)
        q1o = round(float(rng.uniform(20, 200)), 1)
        q1c = round(float(rng.uniform(0, 100)), 1)
        q2p = round(float(rng.uniform(50, 500)), 1)
        q2o = round(float(rng.uniform(20, 200)), 1)
        q2c = round(float(rng.uniform(0, 100)), 1)
        abs_v = round((q1p + q1o + q1c) - (q2p + q2o + q2c), 1)
        rel_v = round(abs_v / max(1, q1p + q1o + q1c) * 100, 1)
        rows.append([cc, q1p, q1o, q1c, q2p, q2o, q2c, abs_v, f"{rel_v}%"])

    html = _to_html_ragged_colspan(header_groups, sub_headers, rows)
    return html, sub_headers, rows, "budget_ragged"


def _gen_annual_comparison(
    rng: np.random.Generator,
) -> tuple[str, list[str], list[list[Any]], str]:
    """Year-over-year stacked comparison table (stacked_comparison variant)."""
    departments   = ["Engineering", "Sales", "Marketing", "Operations", "Finance", "HR", "Legal", "Product"]
    shared_headers = ["Department", "Headcount", "Revenue ($M)", "Expenses ($M)", "Net ($M)", "Var vs. Target"]
    years         = [2023, 2024]
    blocks: list[dict] = []
    all_flat: list[list[Any]] = []

    for year in years:
        n_d = int(rng.integers(4, 7))
        rows: list[list[Any]] = []
        for i in range(n_d):
            dept  = departments[i % len(departments)]
            hc    = int(rng.integers(10, 250))
            rev   = round(float(rng.uniform(1.0, 50.0)), 2)
            exp   = round(float(rng.uniform(0.5, rev * 0.9)), 2)
            net   = round(rev - exp, 2)
            pct   = round(float(rng.uniform(-20.0, 30.0)), 1)
            rows.append([dept, hc, rev, exp, net, f"{pct}%"])
            all_flat.append(rows[-1])
        # Add total row
        rows.append(["TOTAL",
                     sum(r[1] for r in rows),
                     round(sum(r[2] for r in rows), 2),
                     round(sum(r[3] for r in rows), 2),
                     round(sum(r[4] for r in rows), 2),
                     "—"])
        blocks.append({"title": f"Fiscal Year {year} — Actuals", "rows": rows})

    html = _to_html_stacked_comparison(blocks, shared_headers)
    return html, shared_headers, all_flat, "annual_comparison"


def _gen_kpi_scorecard(
    rng: np.random.Generator,
) -> tuple[str, list[str], list[list[Any]], str]:
    """KPI scorecard: metrics as rows, quarters + FY total as columns (transposed_metrics)."""
    kpi_pools = [
        [("Revenue ($M)", 5, 100), ("Gross Margin (%)", 25, 70), ("EBITDA ($M)", 1, 30),
         ("Net Income ($M)", 0.5, 15), ("Headcount", 50, 600), ("Customer NPS", 15, 80),
         ("Churn Rate (%)", 0.5, 12)],
        [("Page Views (M)", 1, 50), ("Active Users (K)", 10, 800), ("Conversion Rate (%)", 1, 18),
         ("Avg. Session (min)", 2, 25), ("CAC ($)", 15, 600), ("CLV ($)", 200, 5000),
         ("Bounce Rate (%)", 20, 70)],
        [("Projects Completed", 2, 25), ("On-Time Rate (%)", 55, 100), ("Budget Variance (%)", -20, 20),
         ("Defect Rate (%)", 0.1, 8), ("Cycle Time (days)", 2, 45), ("Team Velocity", 8, 120),
         ("Utilization (%)", 60, 95)],
    ]
    pool    = kpi_pools[int(rng.integers(0, len(kpi_pools)))]
    n_met   = int(rng.integers(4, 7))
    metrics = pool[:n_met]
    col_lbls = ["Q1", "Q2", "Q3", "Q4", "FY Total"]
    row_lbls = [m[0] for m in metrics]

    values: list[list[Any]] = []
    flat_rows: list[list[Any]] = []
    for name, lo, hi in metrics:
        q = [round(float(rng.uniform(lo, hi)), 1) for _ in range(4)]
        total = round(sum(q) / 4, 1) if "%" in name or "Rate" in name else round(sum(q), 1)
        values.append(q + [total])
        flat_rows.append([name] + q + [total])

    html    = _to_html_transposed("KPI Metric", row_lbls, col_lbls, values)
    headers = ["KPI Metric"] + col_lbls
    return html, headers, flat_rows, "kpi_scorecard"


def _gen_product_matrix(
    rng: np.random.Generator,
) -> tuple[str, list[str], list[list[Any]], str]:
    """Product feature availability matrix across SKUs (matrix_table variant)."""
    features = ["Single Sign-On", "API Access", "Custom Branding", "SLA 99.9%",
                "Dedicated Support", "Audit Logs", "SAML/OIDC", "Data Export",
                "Webhooks", "IP Allowlisting", "Advanced Analytics", "Multi-Region"]
    skus     = ["Starter", "Professional", "Business", "Enterprise", "Enterprise+"]
    ratings  = ["Included", "Add-on", "Partial", "—"]
    weights  = [0.40, 0.20, 0.15, 0.25]

    n_feat = int(rng.integers(5, 10))
    n_sku  = int(rng.integers(3, 5))
    row_lbls = features[:n_feat]
    col_lbls = skus[:n_sku]

    values: list[list[Any]] = []
    flat_rows: list[list[Any]] = []
    for fi, feat in enumerate(row_lbls):
        # Higher-tier SKUs more likely to have Included
        rv: list[str] = []
        for si in range(len(col_lbls)):
            tier_boost = si / max(1, len(col_lbls) - 1)  # 0..1
            if rng.random() < 0.3 + 0.5 * tier_boost:
                rv.append("Included")
            else:
                rv.append(ratings[int(rng.choice(len(ratings), p=weights))])
        values.append(rv)
        flat_rows.append([feat] + rv)

    html    = _to_html_matrix("Feature", row_lbls, col_lbls, values)
    headers = ["Feature"] + col_lbls
    return html, headers, flat_rows, "product_matrix"


def _gen_regional_comparison(
    rng: np.random.Generator,
) -> tuple[str, list[str], list[list[Any]], str]:
    """Regional performance stacked comparison (stacked_comparison variant)."""
    regions  = ["North America", "EMEA", "APAC", "LATAM"]
    segments = ["Enterprise", "Mid-Market", "SMB", "Self-Serve"]
    shared_headers = ["Segment", "Accounts", "ARR ($M)", "NRR (%)", "Win Rate (%)", "Avg. Deal Size ($K)"]

    n_regions = int(rng.integers(2, len(regions) + 1))
    blocks: list[dict] = []
    all_flat: list[list[Any]] = []

    for region in regions[:n_regions]:
        n_seg = int(rng.integers(3, len(segments) + 1))
        rows: list[list[Any]] = []
        for seg in segments[:n_seg]:
            accts   = int(rng.integers(10, 2000))
            arr     = round(float(rng.uniform(0.5, 80.0)), 2)
            nrr     = round(float(rng.uniform(85.0, 135.0)), 1)
            win     = round(float(rng.uniform(15.0, 65.0)), 1)
            avg_deal = round(float(rng.uniform(5.0, 500.0)), 1)
            rows.append([seg, accts, arr, f"{nrr}%", f"{win}%", avg_deal])
            all_flat.append(rows[-1])
        blocks.append({"title": region, "rows": rows})

    html = _to_html_stacked_comparison(blocks, shared_headers)
    return html, shared_headers, all_flat, "regional_comparison"


# Registry of all complex table generators
# Each callable: (rng) -> (html_str, flat_headers, flat_rows, type_name)
_COMPLEX_GENERATORS: list[tuple[str, Any]] = [
    ("pl_statement",         _gen_pl_statement),
    ("dept_contacts",        _gen_dept_contacts),
    ("compliance_footnoted", _gen_compliance_footnoted),
    ("skill_matrix",         _gen_skill_matrix),
    ("raci_matrix",          _gen_raci_matrix),
    ("budget_ragged",        _gen_budget_ragged),
    ("annual_comparison",    _gen_annual_comparison),
    ("kpi_scorecard",        _gen_kpi_scorecard),
    ("product_matrix",       _gen_product_matrix),
    ("regional_comparison",  _gen_regional_comparison),
]


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def _static_table_qa(
    caption: str,
    headers: list[str],
    rows: list[list[Any]],
) -> list[dict[str, Any]]:
    """Return simple, deterministic QA anchored by table caption.

    Used by the random/static table path so every table variant — not just
    the LLM path — carries testable QA candidates in its ground truth.
    Questions are self-sufficient: each embeds the table caption so the
    evaluator knows exactly which table is being referenced.
    """
    qa: list[dict[str, Any]] = []
    if not headers:
        return qa
    header_str = ", ".join(headers)
    qa.append({
        "question": f"In the table titled '{caption}', how many data rows are shown?",
        "answer": str(len(rows)),
        "assertions": [{"text": f"The response must state {len(rows)}."}],
    })
    qa.append({
        "question": f"In the table titled '{caption}', what are the column headers?",
        "answer": header_str,
        "assertions": [{"text": f"The response must state the column headers, including '{headers[0]}'."}],
    })
    if rows:
        val = str(rows[0][0])
        qa.append({
            "question": f"In the table titled '{caption}', what value appears in the '{headers[0]}' column for the first data row?",
            "answer": val,
            "assertions": [{"text": f"The response must state '{val}'."}],
        })
    return qa


def _derived_answers(headers: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    """Compute basic derived answers: totals, max, min for numeric columns."""
    answers: dict[str, Any] = {}
    for col_idx, header in enumerate(headers):
        numeric_vals = []
        for row in rows:
            v = row[col_idx] if col_idx < len(row) else None
            parsed = _parse_numeric(v)
            if parsed is not None:
                numeric_vals.append(parsed)
        if numeric_vals:
            answers[f"sum_{header}"] = sum(numeric_vals)
            answers[f"max_{header}"] = max(numeric_vals)
            answers[f"min_{header}"] = min(numeric_vals)
    return answers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_llm_table_variants(
    client: "AzureOpenAIClient",
    topic: "TopicSpec",
    rng: np.random.Generator,
    count: int,
    max_workers: int = 1,
    table_briefs: "list[dict] | None" = None,
) -> list[Variant]:
    """Generate *count* table variants using LLM-generated data for the given topic.

    Falls back to random table generation if LLM returns unparseable data.
    When max_workers > 1 all tables are generated concurrently.

    table_briefs (optional): list of brief dicts from generate_document_brief().
      Each dict may contain 'caption', 'purpose', 'key_data'.
      When provided, each table's purpose and required cell values come from the brief,
      ensuring coherence across all document artifacts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from synthpdf.llm.content_generator import ContentGenerator

    _TABLE_ROLES = [
        "summary statistics",
        "comparison matrix",
        "timeline",
        "budget breakdown",
        "performance metrics",
        "risk register",
        "inventory listing",
        "action item tracker",
    ]

    generator = ContentGenerator(client, topic)
    perturbations_list = ["formatting", "structural", "noise", "semantic", "none"]

    def _generate_one(i: int) -> Variant:
        child_rng = fork_rng(rng, f"llm_table_{i}")
        n_rows = int(child_rng.integers(5, 12))

        # Use brief slot if available, else fall back to random role
        brief_slot: dict[str, Any] | None = table_briefs[i] if (table_briefs and i < len(table_briefs)) else None
        if brief_slot:
            role = brief_slot.get("purpose", "summary statistics")[:60]
            purpose = brief_slot["purpose"] if brief_slot.get("purpose") else f"summary statistics for {topic.description[:80]}"
            brief_caption = brief_slot.get("caption", "")
        else:
            role = _TABLE_ROLES[int(child_rng.integers(0, len(_TABLE_ROLES)))]
            purpose = f"{role} for a document about: {topic.description[:100]}"
            brief_caption = ""

        headers: list[str] = []
        rows: list[list[Any]] = []

        t0 = time.perf_counter()
        try:
            data = generator.generate_table(purpose, n_rows, brief_slot=brief_slot)
            if data and "headers" in data and "rows" in data:
                headers = [str(h) for h in data["headers"]]
                rows = [[str(c) for c in row] for row in data["rows"]]
        except Exception:
            pass
        data_time = time.perf_counter() - t0

        if not headers or not rows:
            if child_rng.random() < 0.25:
                # Use a complex generator as fallback ~25% of the time
                _, complex_fn = _COMPLEX_GENERATORS[i % len(_COMPLEX_GENERATORS)]
                try:
                    _, headers, rows, base_name = complex_fn(child_rng)
                except Exception:
                    base_name, base_fn = _BASE_GENERATORS[i % len(_BASE_GENERATORS)]
                    headers, rows = base_fn(child_rng)
            else:
                base_name, base_fn = _BASE_GENERATORS[i % len(_BASE_GENERATORS)]
                headers, rows = base_fn(child_rng)
        else:
            base_name = role.replace(" ", "_")

        n_perturb = int(child_rng.integers(0, 3))
        chosen = list(child_rng.choice(len(perturbations_list), size=n_perturb, replace=False))
        applied = []
        for p_idx in chosen:
            p = perturbations_list[p_idx]
            if p == "formatting":
                headers, rows = _apply_formatting_perturbation(child_rng, headers, rows, base_name)
                applied.append("formatting")
            elif p == "structural":
                headers, rows = _apply_structural_perturbation(child_rng, headers, rows)
                applied.append("structural")
            elif p == "noise":
                headers, rows = _apply_noise_perturbation(child_rng, headers, rows)
                applied.append("noise")
            elif p == "semantic":
                headers, rows = _apply_semantic_perturbation(child_rng, headers, rows)
                applied.append("semantic")

        html_table = _to_html_table(headers, rows)
        caption = (
            brief_caption
            if brief_caption
            else f"Table {i+1}: {role.title()} — {topic.id.replace('_', ' ').title()}"
        )

        header_line = " | ".join(headers)
        row_lines = "\n".join(" | ".join(str(c) for c in row) for row in rows[:10])
        table_content = f"Caption: {caption}\nColumns: {header_line}\nData:\n{row_lines}"

        t0 = time.perf_counter()
        qa_candidates = generator.generate_qa_candidates(table_content, content_type="table", n=3)
        qa_time = time.perf_counter() - t0

        click.echo(
            f"      table [{i+1}/{count}] {role!r}: "
            f"data({n_rows}r)->{data_time:.0f}s | QA({len(qa_candidates)})->{qa_time:.0f}s"
        )
        try:
            from synthpdf import stats_sink as _ss
            _ss.log_artifact(
                type="table", index=i + 1, total=count,
                caption=caption[:80], rows=n_rows,
                data_time_s=round(data_time, 1),
                qa_count=len(qa_candidates), qa_time_s=round(qa_time, 1),
            )
        except Exception:
            pass
        return Variant(
            id=f"table_{i:04d}_{_uid()[:8]}",
            kind="table",
            metadata={
                "base_type": base_name,
                "perturbations": applied,
                "row_count": len(rows),
                "col_count": len(headers),
                "caption": caption,
                "topic_id": topic.id,
            },
            render_payload={
                "markdown": html_table,
                "caption": caption,
                "format": "html",
            },
            ground_truth={
                "schema": headers,
                "cells": [[str(c) for c in row] for row in rows],
                "derived_answers": _derived_answers(headers, rows),
                "caption": caption,
                "qa_candidates": qa_candidates,
            },
        )

    if count == 0:
        return []
    results: list[Variant | None] = [None] * count
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, count)) as executor:
            future_to_idx = {executor.submit(_generate_one, i): i for i in range(count)}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
    else:
        for i in range(count):
            results[i] = _generate_one(i)
    return results  # type: ignore[return-value]


def generate_multipage_table_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,
) -> list[Variant]:
    """Generate *count* multipage table variants, alternating between simple and grouped layouts."""
    variants: list[Variant] = []
    rows_per_page = 20

    for i in range(count):
        child_rng = fork_rng(rng, f"mp_table_{i}")
        n_rows = int(child_rng.integers(40, 80))

        use_grouped = bool(child_rng.random() < 0.5)
        column_groups: list[tuple[str, int]] | None = None

        if use_grouped:
            base_name, base_fn = _MP_GROUPED_GENERATORS[i % len(_MP_GROUPED_GENERATORS)]
            headers, rows, column_groups = base_fn(child_rng, n_rows=n_rows)
            html = paginate_grouped_table(headers, rows, rows_per_page=rows_per_page, column_groups=column_groups)
        else:
            base_name, base_fn = _MP_LARGE_GENERATORS[i % len(_MP_LARGE_GENERATORS)]
            headers, rows = base_fn(child_rng, n_rows=n_rows)
            html = paginate_table(headers, rows, rows_per_page=rows_per_page)

        n_pages = math.ceil(len(rows) / rows_per_page)
        caption = f"Table {i+1}: {base_name.replace('_', ' ').title()}"

        # Sample 15 rows for QA context: first 3, 3 random middle, last 3, 6 random
        n = len(rows)
        sample_indices: list[int] = list(range(min(3, n)))
        if n > 6:
            mid_start, mid_end = 3, n - 3
            mid_count = min(3, mid_end - mid_start)
            if mid_count > 0:
                mid_sample = list(child_rng.choice(mid_end - mid_start, size=mid_count, replace=False) + mid_start)
                sample_indices.extend(mid_sample)
            sample_indices.extend(range(max(0, n - 3), n))
            remaining = [j for j in range(n) if j not in sample_indices]
            extra = min(6, len(remaining))
            if extra > 0:
                extra_sample = list(child_rng.choice(len(remaining), size=extra, replace=False))
                sample_indices.extend([remaining[k] for k in extra_sample])
        sample_rows = [rows[j] for j in sorted(set(sample_indices)) if j < n]

        derived = _derived_answers(headers, rows)
        qa_context = (
            f"Caption: {caption}\nColumns: {' | '.join(headers)}\n"
            f"Sampled rows (from {n} total):\n"
            + "\n".join(" | ".join(str(c) for c in r) for r in sample_rows)
            + f"\nAggregates: {derived}"
        )

        variants.append(Variant(
            id=f"mp_table_{i:04d}_{_uid()[:8]}",
            kind="table",
            metadata={
                "base_type": base_name,
                "multipage": True,
                "grouped": use_grouped,
                "pages": n_pages,
                "row_count": len(rows),
                "col_count": len(headers),
                "caption": caption,
            },
            render_payload={
                "markdown": html,
                "caption": caption,
                "format": "html",
                "rows_per_page": rows_per_page,
            },
            ground_truth={
                "schema": headers,
                "cells": [[str(c) for c in row] for row in rows],
                "total_rows": len(rows),
                "pages": n_pages,
                "rows_per_page": rows_per_page,
                "derived_answers": derived,
                "caption": caption,
                "qa_candidates": _static_table_qa(caption, headers, rows),
            },
        ))

    return variants


def generate_llm_multipage_table_variants(
    rng: np.random.Generator,
    count: int,
    generator: Any,
    assets_dir: "Path | None" = None,
) -> list[Variant]:
    """Generate *count* multipage table variants using LLM-generated data.

    Falls back to random large generators if LLM returns insufficient rows.
    """
    from synthpdf.llm.content_generator import ContentGenerator

    _TABLE_ROLES = [
        "purchase order register",
        "time series performance metrics",
        "budget expenditure ledger",
        "project milestone tracker",
        "inventory audit log",
        "employee activity report",
    ]
    variants: list[Variant] = []
    rows_per_page = 20
    # All fallback generators (both simple and grouped); cycle through them
    _all_fallback = _MP_LARGE_GENERATORS + _MP_GROUPED_GENERATORS

    for i in range(count):
        child_rng = fork_rng(rng, f"llm_mp_table_{i}")
        role = _TABLE_ROLES[int(child_rng.integers(0, len(_TABLE_ROLES)))]
        n_rows = int(child_rng.integers(50, 80))

        headers: list[str] = []
        rows: list[list[Any]] = []
        base_name = role.replace(" ", "_")
        column_groups: list[tuple[str, int]] | None = None
        used_grouped = False

        try:
            topic_desc = generator._topic.description[:100] if hasattr(generator, "_topic") else "general business"
            purpose = f"{role} for a document about: {topic_desc}"
            data = generator.generate_large_table(purpose, n_rows=n_rows)
            if data and "headers" in data and "rows" in data and len(data["rows"]) >= n_rows * 0.75:
                headers = [str(h) for h in data["headers"]]
                rows = [[str(c) for c in row] for row in data["rows"]]
        except Exception:
            pass

        if not headers or not rows:
            # Cycle through all fallback generators (simple + grouped)
            fallback_name, fallback_fn = _all_fallback[i % len(_all_fallback)]
            base_name = fallback_name
            result = fallback_fn(child_rng, n_rows=n_rows)
            if len(result) == 3:
                headers, rows, column_groups = result
                used_grouped = True
            else:
                headers, rows = result

        # Use grouped paginator when we have column_groups or LLM data (adds group subtotals)
        if used_grouped:
            html = paginate_grouped_table(headers, rows, rows_per_page=rows_per_page, column_groups=column_groups)
        elif bool(child_rng.random() < 0.4):
            # 40% chance: apply group subtotals to LLM data even without column groups
            html = paginate_grouped_table(headers, rows, rows_per_page=rows_per_page)
        else:
            html = paginate_table(headers, rows, rows_per_page=rows_per_page)
        n_pages = math.ceil(len(rows) / rows_per_page)
        caption = f"Table {i+1}: {role.title()}"

        # Sample rows for QA context
        n = len(rows)
        sample_indices: list[int] = list(range(min(3, n)))
        if n > 6:
            mid_start, mid_end = 3, n - 3
            mid_count = min(3, mid_end - mid_start)
            if mid_count > 0:
                mid_sample = list(child_rng.choice(mid_end - mid_start, size=mid_count, replace=False) + mid_start)
                sample_indices.extend(mid_sample)
            sample_indices.extend(range(max(0, n - 3), n))
            remaining = [j for j in range(n) if j not in sample_indices]
            extra = min(6, len(remaining))
            if extra > 0:
                extra_sample = list(child_rng.choice(len(remaining), size=extra, replace=False))
                sample_indices.extend([remaining[k] for k in extra_sample])
        sample_rows = [rows[j] for j in sorted(set(sample_indices)) if j < n]

        derived = _derived_answers(headers, rows)
        qa_context = (
            f"Caption: {caption}\nColumns: {' | '.join(headers)}\n"
            f"Sampled rows (from {n} total):\n"
            + "\n".join(" | ".join(str(c) for c in r) for r in sample_rows)
            + f"\nAggregates: {derived}"
        )

        qa_candidates: list[Any] = []
        try:
            qa_candidates = generator.generate_qa_candidates(qa_context, content_type="table", n=3)
        except Exception:
            pass

        variants.append(Variant(
            id=f"llm_mp_table_{i:04d}_{_uid()[:8]}",
            kind="table",
            metadata={
                "base_type": base_name,
                "multipage": True,
                "grouped": used_grouped,
                "pages": n_pages,
                "row_count": len(rows),
                "col_count": len(headers),
                "caption": caption,
            },
            render_payload={
                "markdown": html,
                "caption": caption,
                "format": "html",
                "rows_per_page": rows_per_page,
            },
            ground_truth={
                "schema": headers,
                "cells": [[str(c) for c in row] for row in rows],
                "total_rows": len(rows),
                "pages": n_pages,
                "rows_per_page": rows_per_page,
                "derived_answers": derived,
                "caption": caption,
                "qa_candidates": qa_candidates,
            },
        ))

    return variants


def generate_table_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,  # noqa: F821
) -> list[Variant]:
    """Generate *count* table variants.

    ~35% of tables use a complex renderer (merged cells, nested table, footnoted,
    matrix, ragged colspan, stacked comparison, transposed metrics, etc.).
    The remaining 65% use the original base generators with random perturbations.
    """
    variants: list[Variant] = []
    perturbations = ["formatting", "structural", "noise", "semantic", "none"]

    for i in range(count):
        child_rng = fork_rng(rng, f"table_{i}")
        use_complex = bool(child_rng.random() < 0.35)
        applied: list[str] = []

        if use_complex:
            _, complex_fn = _COMPLEX_GENERATORS[i % len(_COMPLEX_GENERATORS)]
            try:
                html_table, headers, rows, base_name = complex_fn(child_rng)
            except Exception:
                # Fall through to base generator on any error
                use_complex = False

        if not use_complex:
            base_name, base_fn = _BASE_GENERATORS[i % len(_BASE_GENERATORS)]
            headers, rows = base_fn(child_rng)

            n_perturb = int(child_rng.integers(0, 3))
            chosen = list(child_rng.choice(len(perturbations), size=n_perturb, replace=False))
            for p_idx in chosen:
                p = perturbations[p_idx]
                if p == "formatting":
                    headers, rows = _apply_formatting_perturbation(child_rng, headers, rows, base_name)
                    applied.append("formatting")
                elif p == "structural":
                    headers, rows = _apply_structural_perturbation(child_rng, headers, rows)
                    applied.append("structural")
                elif p == "noise":
                    headers, rows = _apply_noise_perturbation(child_rng, headers, rows)
                    applied.append("noise")
                elif p == "semantic":
                    headers, rows = _apply_semantic_perturbation(child_rng, headers, rows)
                    applied.append("semantic")

            html_table = _to_html_table(headers, rows)

        caption = f"Table {i+1}: {base_name.replace('_', ' ').title()}"

        variants.append(Variant(
            id=f"table_{i:04d}_{_uid()[:8]}",
            kind="table",
            metadata={
                "base_type": base_name,
                "complex": use_complex,
                "perturbations": applied,
                "row_count": len(rows),
                "col_count": len(headers),
                "caption": caption,
            },
            render_payload={
                "markdown": html_table,
                "caption": caption,
                "format": "html",
            },
            ground_truth={
                "schema": headers,
                "cells": [[str(c) for c in row] for row in rows],
                "derived_answers": _derived_answers(headers, rows),
                "caption": caption,
                "qa_candidates": _static_table_qa(caption, headers, rows),
            },
        ))

    return variants
