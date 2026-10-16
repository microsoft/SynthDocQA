# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Rich annotation variant generator for synthpdf.

Generates seven annotation variant types organised into three rendering families:

  Path A — Visual (HTML/CSS baked into PDF pixels during HTML→PDF step):
    1. highlight_block      Inline <mark> colour highlights on key phrases
    2. callout_block        Styled callout / admonition box (warning/info/danger/success/tip/note)
    3. label_block          Status / classification banner (CONFIDENTIAL, DRAFT, HIGH RISK …)

  Path B — True PDF annotations (stored in render_payload["pdf_annotations"];
            applied post-render by render/pdf_annotate.py via PyMuPDF):
    4. sticky_note_block    Search-and-place PDF sticky-note comments
    5. freetext_box_block   PDF free-text callout boxes with border + fill colour

  Hybrid (Path A + Path B simultaneously):
    6. review_block         Visual <mark> highlights AND matching PDF reviewer comments
    7. annotated_report     Dense multi-type section: banner + callout + marks + PDF notes

Public pool function:
    generate_annotation_pool(rng, count, assets_dir) -> list[Variant]
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from synthpdf.llm.client import AzureOpenAIClient
    from synthpdf.llm.content_generator import TopicSpec

from synthpdf.rng import fork_rng
from synthpdf.types import Variant


def _uid() -> str:
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# Highlight <mark> background colours  (hex, name)
_MARK_COLORS: list[tuple[str, str]] = [
    ("#fff176", "yellow"),
    ("#bfdbfe", "blue"),
    ("#bbf7d0", "green"),
    ("#fed7aa", "orange"),
    ("#fce7f3", "pink"),
    ("#ddd6fe", "purple"),
    ("#fecaca", "red"),
    ("#cffafe", "cyan"),
]

# Callout / admonition styles  {key: (border_color, bg_color, icon, label)}
_CALLOUT_STYLES: dict[str, tuple[str, str, str, str]] = {
    "warning": ("#f59e0b", "#fffbeb", "⚠️",  "Warning"),
    "info":    ("#3b82f6", "#eff6ff", "ℹ️",  "Note"),
    "danger":  ("#ef4444", "#fef2f2", "🚨",  "Critical"),
    "success": ("#10b981", "#f0fdf4", "✅",  "Approved"),
    "tip":     ("#8b5cf6", "#f5f3ff", "💡",  "Tip"),
    "note":    ("#6b7280", "#f9fafb", "📝",  "Editorial Note"),
}

# Label / badge styles  {label: (bg_color, text_color)}
_LABEL_STYLES: dict[str, tuple[str, str]] = {
    "CONFIDENTIAL":         ("#dc2626", "#ffffff"),
    "DRAFT":                ("#d97706", "#ffffff"),
    "APPROVED":             ("#059669", "#ffffff"),
    "HIGH RISK":            ("#b91c1c", "#ffffff"),
    "LOW RISK":             ("#16a34a", "#ffffff"),
    "PENDING REVIEW":       ("#7c3aed", "#ffffff"),
    "CRITICAL":             ("#991b1b", "#ffffff"),
    "ARCHIVED":             ("#374151", "#ffffff"),
    "FOR INTERNAL USE ONLY": ("#1e40af", "#ffffff"),
    "SUBJECT TO CHANGE":    ("#0891b2", "#ffffff"),
}

# Cell background colour palette for colored_table variants: (light_bg, dark_fg)
_CELL_BG_COLORS: list[tuple[str, str]] = [
    ("#dcfce7", "#15803d"),   # green
    ("#dbeafe", "#1e40af"),   # blue
    ("#fef9c3", "#854d0e"),   # yellow
    ("#fee2e2", "#b91c1c"),   # red
    ("#f3e8ff", "#7e22ce"),   # purple
    ("#e0f2fe", "#0369a1"),   # cyan
    ("#fff7ed", "#c2410c"),   # orange
    ("#f1f5f9", "#334155"),   # slate
]
_HEADER_BG = "#1e3a5f"  # dark navy header row

# Structured table datasets for colored_table variants
_TABLE_DATA: list[dict[str, Any]] = [
    {
        "title": "Financial Performance Summary",
        "headers": ["Metric", "Q3 2024", "Q4 2024", "YoY Change"],
        "rows": [
            {"cells": ["Revenue",           "$4.2M",  "$5.1M",  "+21%"],   "trend": "positive", "highlight_col": 2},
            {"cells": ["Gross Margin",      "61.4%",  "63.2%",  "+1.8pp"], "trend": "positive", "highlight_col": 2},
            {"cells": ["Operating Expenses","$2.1M",  "$2.4M",  "+14%"],   "trend": "negative", "highlight_col": None},
            {"cells": ["Net Income",        "$890K",  "$1.1M",  "+24%"],   "trend": "positive", "highlight_col": 2},
            {"cells": ["EBITDA",            "$1.3M",  "$1.6M",  "+23%"],   "trend": "positive", "highlight_col": None},
        ],
    },
    {
        "title": "Risk Register Overview",
        "headers": ["Risk Item", "Likelihood", "Impact", "Status"],
        "rows": [
            {"cells": ["Data Breach",        "Medium", "High",   "OPEN"],       "trend": "negative", "highlight_col": 3},
            {"cells": ["Vendor Delay",       "High",   "High",   "CRITICAL"],   "trend": "negative", "highlight_col": 1},
            {"cells": ["Budget Overrun",     "Low",    "Medium", "MONITORING"], "trend": "neutral",  "highlight_col": None},
            {"cells": ["Staff Turnover",     "Medium", "Low",    "MITIGATED"],  "trend": "positive", "highlight_col": 3},
            {"cells": ["Regulatory Change",  "Low",    "High",   "PENDING"],    "trend": "neutral",  "highlight_col": None},
        ],
    },
    {
        "title": "Department Budget Allocation",
        "headers": ["Department", "Allocated", "Spent", "Remaining"],
        "rows": [
            {"cells": ["Engineering", "$1.2M",  "$0.9M",  "$0.3M"],  "trend": "positive", "highlight_col": 3},
            {"cells": ["Marketing",   "$0.8M",  "$0.85M", "-$50K"],  "trend": "negative", "highlight_col": 2},
            {"cells": ["Operations",  "$0.5M",  "$0.48M", "$20K"],   "trend": "positive", "highlight_col": None},
            {"cells": ["Legal",       "$0.3M",  "$0.22M", "$80K"],   "trend": "positive", "highlight_col": None},
            {"cells": ["HR",          "$0.25M", "$0.25M", "$0"],     "trend": "neutral",  "highlight_col": None},
        ],
    },
    {
        "title": "Key Performance Indicators",
        "headers": ["KPI", "Target", "Actual", "Variance"],
        "rows": [
            {"cells": ["Customer Satisfaction", "90%",   "92%",   "+2%"],  "trend": "positive", "highlight_col": 2},
            {"cells": ["On-Time Delivery",       "95%",   "91%",   "-4%"],  "trend": "negative", "highlight_col": 3},
            {"cells": ["Employee Retention",     "85%",   "88%",   "+3%"],  "trend": "positive", "highlight_col": 2},
            {"cells": ["Revenue per User",       "$150",  "$162",  "+8%"],  "trend": "positive", "highlight_col": 2},
            {"cells": ["Defect Rate",            "<2%",   "1.8%",  "Pass"], "trend": "positive", "highlight_col": None},
        ],
    },
]


# PyMuPDF RGB colour tuples (0–1) for PDF annotation strokes
_PDF_COLORS: dict[str, tuple[float, float, float]] = {
    "yellow": (1.0, 1.0,  0.0),
    "blue":   (0.5, 0.75, 1.0),
    "green":  (0.4, 0.85, 0.5),
    "orange": (1.0, 0.65, 0.2),
    "red":    (1.0, 0.3,  0.3),
    "purple": (0.7, 0.5,  1.0),
}


# ---------------------------------------------------------------------------
# Document paragraph corpora
# ---------------------------------------------------------------------------

_FINANCIAL_PARAGRAPHS: list[dict[str, Any]] = [
    {
        "text": (
            "Total revenue for Q3 reached $4.2M, representing a 23% year-over-year increase. "
            "Gross margin improved to 61.4% driven by a favourable product-mix shift. "
            "Operating expenses rose to $2.1M, primarily due to headcount additions in engineering. "
            "Net income after tax was $890K, with EBITDA of $1.3M."
        ),
        "highlights": ["$4.2M", "23% year-over-year increase", "61.4%", "Net income after tax"],
        "notes": [
            "Verify against audited financials",
            "Reconcile with CFO report",
            "Check margin calculation",
            "Confirm after-tax figure",
        ],
    },
    {
        "text": (
            "Capital expenditure for the fiscal year totalled $6.8M, up from $4.1M in the prior year. "
            "The majority of spend — $4.2M — was allocated to data-centre infrastructure upgrades. "
            "Depreciation and amortisation charges were $1.9M. "
            "Free cash flow remained positive at $2.3M despite the elevated capex cycle."
        ),
        "highlights": ["$6.8M", "$4.2M", "data-centre infrastructure upgrades", "Free cash flow"],
        "notes": [
            "Capex approval documented?",
            "Align with board resolution",
            "D&A schedule attached?",
            "Positive FCF confirmed",
        ],
    },
    {
        "text": (
            "Accounts receivable days outstanding increased to 47 days from 38 days in Q2. "
            "Bad debt provisions were raised by $180K following a review of aged debtors. "
            "The top five customers account for 62% of total receivables. "
            "Management has initiated a credit-control programme targeting overdue balances above $50K."
        ),
        "highlights": ["47 days", "Bad debt provisions", "62% of total receivables", "$50K"],
        "notes": [
            "DSO trend concerning",
            "Provision methodology?",
            "Concentration risk — escalate",
            "Credit policy reference needed",
        ],
    },
    {
        "text": (
            "Revenue from recurring subscriptions grew to $3.1M, representing 74% of total revenue. "
            "Annual contract value per enterprise customer averaged $185K, up 12% from prior year. "
            "Customer acquisition cost for the quarter was $8,400, while lifetime value reached $92K. "
            "Net revenue retention stood at 118%, indicating strong upsell momentum."
        ),
        "highlights": ["$3.1M", "74% of total revenue", "$185K", "Net revenue retention stood at 118%"],
        "notes": [
            "Recurring revenue definition consistent?",
            "ACV growth driver?",
            "CAC vs prior quarter?",
            "NRR calculation methodology",
        ],
    },
]

_RISK_PARAGRAPHS: list[dict[str, Any]] = [
    {
        "text": (
            "The programme faces a HIGH likelihood of schedule overrun due to unresolved third-party "
            "API integration dependencies. Mitigation: escalate to vendor within 5 business days. "
            "Data privacy risk is rated MEDIUM — the current retention policy does not align with "
            "GDPR Article 17 obligations. Legal review is required before go-live."
        ),
        "highlights": [
            "HIGH likelihood of schedule overrun",
            "5 business days",
            "MEDIUM",
            "GDPR Article 17",
        ],
        "notes": [
            "Owner: PM Alice Johnson",
            "Escalation logged?",
            "Legal sign-off needed",
            "DPO review required",
        ],
    },
    {
        "text": (
            "Cyber-security assessment identifies the legacy authentication module as a critical "
            "vulnerability with a CVSS score of 8.7. Immediate patching is recommended. "
            "Business continuity risk remains LOW following the completion of DR testing on 2024-11-15. "
            "Supplier concentration risk is MEDIUM — 78% of critical components sourced from a single vendor."
        ),
        "highlights": ["CVSS score of 8.7", "Immediate patching", "78% of critical components", "single vendor"],
        "notes": [
            "Patch ticket raised?",
            "DR test report attached",
            "Supplier diversification plan?",
            "Procurement to review",
        ],
    },
    {
        "text": (
            "Liquidity risk is assessed as LOW following the successful $20M revolving credit facility "
            "renewal in October. Interest rate sensitivity analysis indicates a 100bp increase would "
            "reduce annual net income by approximately $340K. Foreign exchange exposure is MEDIUM — "
            "38% of revenue is denominated in EUR, with no active hedging programme in place."
        ),
        "highlights": ["$20M revolving credit facility", "100bp increase", "$340K", "38% of revenue"],
        "notes": [
            "Credit facility terms reviewed?",
            "Sensitivity model assumptions?",
            "FX hedging policy needed",
            "Treasurer to advise",
        ],
    },
]

_PROJECT_PARAGRAPHS: list[dict[str, Any]] = [
    {
        "text": (
            "Sprint 14 achieved 87% of planned story points, with 3 items carried over to Sprint 15. "
            "The mobile onboarding flow — the highest-priority deliverable — was completed and released "
            "to staging on 2024-10-28. Performance testing revealed a 2.4-second load time for the "
            "dashboard view, exceeding the 2.0-second SLA threshold. A hotfix is in progress."
        ),
        "highlights": [
            "87% of planned story points",
            "mobile onboarding flow",
            "2.4-second load time",
            "2.0-second SLA threshold",
        ],
        "notes": [
            "Carry-over impact?",
            "Staging sign-off required",
            "Performance regression — P1",
            "Hotfix ETA?",
        ],
    },
    {
        "text": (
            "Phase 2 kick-off is scheduled for 2025-01-06, contingent on Phase 1 sign-off by the "
            "Architecture Review Board. The estimated budget for Phase 2 is $1.45M over 9 months. "
            "Key risks include resource availability in Q1 due to concurrent projects and the "
            "dependency on the Data Platform team delivering the new ETL pipeline by 2025-02-14."
        ),
        "highlights": ["2025-01-06", "$1.45M over 9 months", "resource availability in Q1", "2025-02-14"],
        "notes": [
            "ARB session scheduled?",
            "Budget approved by Finance?",
            "Resource plan needed",
            "ETL dependency tracked in JIRA",
        ],
    },
]

_COMPLIANCE_PARAGRAPHS: list[dict[str, Any]] = [
    {
        "text": (
            "The internal audit found three material control deficiencies in the procurement process: "
            "(1) purchase orders exceeding $25,000 lacked dual approval in 14% of cases; "
            "(2) vendor onboarding checklists were incomplete for 6 of 22 new suppliers; "
            "(3) segregation of duties was not enforced for payment approvals below $5,000. "
            "Management has committed to remediation by 2025-03-31."
        ),
        "highlights": [
            "three material control deficiencies",
            "$25,000",
            "6 of 22 new suppliers",
            "2025-03-31",
        ],
        "notes": [
            "Escalate to Audit Committee",
            "Dual-approval policy updated?",
            "Vendor risk register?",
            "Remediation owner assigned?",
        ],
    },
    {
        "text": (
            "The organisation must achieve ISO 27001 certification by 2025-06-30 as required under "
            "the enterprise customer contract signed on 2024-08-15. Gap analysis identified 34 controls "
            "requiring remediation, of which 12 are rated critical. External assessor fees are "
            "estimated at $85,000. Failure to certify triggers a contract penalty of $500,000."
        ),
        "highlights": [
            "ISO 27001 certification",
            "34 controls requiring remediation",
            "12 are rated critical",
            "$500,000",
        ],
        "notes": [
            "Contract clause reference",
            "Gap analysis owner?",
            "Budget allocated?",
            "Penalty clause confirmed by Legal",
        ],
    },
]

_MEETING_PARAGRAPHS: list[dict[str, Any]] = [
    {
        "text": (
            "The Q4 planning meeting agreed that the product roadmap will be frozen after 2024-11-01 "
            "to allow engineering to focus on delivery. Three features were deprioritised: advanced "
            "reporting, multi-tenant SSO, and the AI recommendation engine. The CTO requested a "
            "technical feasibility review for the AI engine before the next board presentation on 2024-12-10."
        ),
        "highlights": [
            "frozen after 2024-11-01",
            "advanced reporting, multi-tenant SSO",
            "AI recommendation engine",
            "2024-12-10",
        ],
        "notes": [
            "Roadmap version v3.2 locked",
            "Deprioritisation rationale documented?",
            "Feasibility owner: Bob Smith",
            "Board deck to include AI section",
        ],
    },
    {
        "text": (
            "Action items from the executive review: (1) CFO to present revised Q4 forecast by 2024-11-08; "
            "(2) Head of Engineering to provide headcount plan for 2025 by 2024-11-15; "
            "(3) Legal to complete review of vendor contract amendments by 2024-11-22. "
            "Next steering committee is scheduled for 2024-12-03 at 10:00 AM."
        ),
        "highlights": ["2024-11-08", "headcount plan for 2025", "vendor contract amendments", "2024-12-03"],
        "notes": [
            "CFO forecast in Confluence?",
            "HC plan approved by Board?",
            "Which vendor contracts?",
            "Calendar invite sent?",
        ],
    },
]

# Pool of all paragraph categories
_ALL_PARAGRAPH_POOLS: list[tuple[str, list[dict[str, Any]]]] = [
    ("financial",   _FINANCIAL_PARAGRAPHS),
    ("risk",        _RISK_PARAGRAPHS),
    ("project",     _PROJECT_PARAGRAPHS),
    ("compliance",  _COMPLIANCE_PARAGRAPHS),
    ("meeting",     _MEETING_PARAGRAPHS),
]

# Callout messages by style
_CALLOUT_MESSAGES: dict[str, list[str]] = {
    "warning": [
        "This figure excludes Q4 adjustments pending board approval. Do not distribute externally.",
        "The deadline shown is subject to regulatory approval. Allow 10–15 business days buffer.",
        "Vendor pricing is indicative only and valid for 30 days from the date of this document.",
        "Exchange rates used are as of 2024-10-01. Actual figures will vary with currency movements.",
        "This section contains forward-looking statements that involve material risks and uncertainties.",
        "Preliminary data only — do not use for investment decisions before final audit sign-off.",
    ],
    "info": [
        "All monetary values are expressed in USD unless otherwise stated.",
        "This document is prepared on the basis of information available as of the report date.",
        "Figures shown are unaudited management accounts and subject to revision.",
        "For full disclosure, refer to the Notes to Financial Statements in Appendix B.",
        "Headcount figures include contractors and part-time employees on a full-time-equivalent basis.",
        "Version history and change log are maintained in the document management system.",
    ],
    "danger": [
        "CRITICAL: This system contains personal data subject to GDPR. Unauthorised access is a criminal offence.",
        "DO NOT PROCEED without written sign-off from the Chief Risk Officer and Legal Counsel.",
        "This configuration change will cause a service outage of up to 4 hours. Approve during maintenance window only.",
        "Sensitive financial data is included. This document must not leave the secure document management system.",
        "Alert: Contract penalty clause of $500K activates if milestone is missed. Escalate immediately.",
        "Regulatory reporting deadline: failure to file by 2024-12-31 incurs statutory fines.",
    ],
    "success": [
        "Approved by Finance Committee on 2024-10-15. Reference: FC-2024-047.",
        "All required regulatory approvals have been obtained. Project may proceed to Phase 2.",
        "Security assessment passed. Penetration test completed with no critical findings.",
        "Audit complete. No material misstatements identified. Report signed off by external auditors.",
        "Change request CR-2024-112 has been approved and is scheduled for the next deployment window.",
        "Vendor qualification complete. Supplier added to approved vendor list effective 2024-11-01.",
    ],
    "tip": [
        "Use the filter panel on the right to narrow results by date range, department, or cost centre.",
        "Keyboard shortcut: Press Ctrl+Shift+E to export the current view to Excel.",
        "Best practice: Always attach the supporting evidence document when raising a control exception.",
        "Tip: Save a draft version before submitting. Draft submissions can be edited up to 48 hours.",
        "You can bulk-upload supplier invoices using the CSV template in the Accounts Payable portal.",
        "Set a reminder in the task tracker 5 days before each milestone to review status.",
    ],
    "note": [
        "Editorial: This section was revised following feedback from the Risk Committee on 2024-10-20.",
        "Author's note: The methodology used here differs from the approach in the 2023 Annual Report.",
        "Reviewer comment: Recommend expanding this section with the Q4 variance analysis before final publication.",
        "Cross-reference: See Exhibit 4.2 for the full segmentation breakdown referenced in this paragraph.",
        "Status: Preliminary draft — figures subject to change after CFO sign-off.",
        "Scope clarification: This analysis covers EMEA only; APAC data is reported separately.",
    ],
}

# Generic reviewer comments for PDF sticky-note annotations
_REVIEWER_COMMENTS: list[str] = [
    "Verify against Q4 audited financials before publishing.",
    "Legal review required — potential regulatory implications.",
    "Check with Procurement: vendor contract number?",
    "Reconcile this figure with the board pack version (v2.1).",
    "Needs clarification — definition of 'critical' differs from Risk Policy.",
    "Owner: confirm this assumption is still valid post-restructure.",
    "TODO: add source reference (dataset, date, methodology).",
    "Flagged for Audit Committee attention.",
    "Sensitivity: this paragraph contains non-public information.",
    "Update required: data is as of Q3; Q4 data now available.",
    "Confirm approval documented in the change log.",
    "Escalate to CTO — technical assumption not independently validated.",
    "Add confidence interval — point estimate is misleading without context.",
    "Please re-state in percentage terms for the executive summary.",
    "Cross-reference Section 4.3 which contradicts this statement.",
    "Source data quality: survey response rate was only 34%. Add caveat.",
    "Recheck arithmetic — column total does not reconcile.",
    "Governance: was this decision within delegated authority limits?",
]


# ---------------------------------------------------------------------------
# HTML rendering helpers  (all use inline styles — no external CSS needed)
# ---------------------------------------------------------------------------

def _mark_html(text: str, phrase: str, color: str) -> str:
    """Wrap the first occurrence of *phrase* in an inline-styled <mark>."""
    return text.replace(
        phrase,
        f'<mark style="background-color: {color}; padding: 1px 3px; '
        f'border-radius: 2px;">{phrase}</mark>',
        1,
    )


def _callout_html(style: str, body: str) -> str:
    """Render an inline-styled callout <div>."""
    border, bg, icon, label = _CALLOUT_STYLES[style]
    return (
        f'<div style="border-left: 4px solid {border}; background: {bg}; '
        f'padding: 10px 14px; margin: 14px 0; border-radius: 0 4px 4px 0; '
        f'font-size: 0.93em; line-height: 1.55;">'
        f'<strong>{icon}&nbsp;{label}:</strong>&nbsp;{body}'
        f'</div>'
    )


def _label_banner_html(label_key: str) -> str:
    """Render a full-width classification banner."""
    bg, fg = _LABEL_STYLES[label_key]
    return (
        f'<div style="background: {bg}; color: {fg}; padding: 5px 14px; '
        f'font-weight: bold; font-size: 0.80em; border-radius: 3px; '
        f'letter-spacing: 0.07em; margin-bottom: 12px; font-family: monospace;">'
        f'&#x258C;&nbsp;{label_key}'
        f'</div>'
    )


def _inline_label_html(label_key: str) -> str:
    """Render an inline badge/tag <span>."""
    bg, fg = _LABEL_STYLES[label_key]
    return (
        f'<span style="background: {bg}; color: {fg}; padding: 2px 7px; '
        f'font-weight: bold; font-size: 0.75em; border-radius: 3px; '
        f'letter-spacing: 0.05em; font-family: monospace;">{label_key}</span>'
    )


# ---------------------------------------------------------------------------
# LLM QA helper — anchored, diverse QnA via ContentGenerator
# ---------------------------------------------------------------------------

def _gen_element_qa(
    client: Any,
    topic: Any,
    element_type: str,
    ground_truth: dict[str, Any],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return LLM-generated QnA if client is available, else return anchored *fallback*."""
    if client and topic:
        try:
            from synthpdf.llm.content_generator import ContentGenerator
            qa = ContentGenerator(client, topic).generate_element_qa(
                element_type,
                ground_truth,
                [q["question"] for q in fallback],
                count=len(fallback),
            )
            if qa:
                return qa
        except Exception:
            pass
    return fallback


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _pick_paragraph(rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    """Pick a random (category, paragraph) pair from the corpus."""
    cat_idx = int(rng.integers(0, len(_ALL_PARAGRAPH_POOLS)))
    cat_name, pool = _ALL_PARAGRAPH_POOLS[cat_idx]
    para = pool[int(rng.integers(0, len(pool)))]
    return cat_name, para


def _pick_notes(para: dict[str, Any], idx: int, rng: np.random.Generator) -> str:
    notes = para.get("notes", [])
    if idx < len(notes):
        return notes[idx]
    return _REVIEWER_COMMENTS[int(rng.integers(0, len(_REVIEWER_COMMENTS)))]


# ---------------------------------------------------------------------------
# Variant generators
# ---------------------------------------------------------------------------

# ── 1. Highlight block ──────────────────────────────────────────────────────

def _generate_highlight_block(
    rng: np.random.Generator, i: int, client: Any = None, topic: Any = None
) -> Variant:
    cat_name, para = _pick_paragraph(rng)
    text = para["text"]
    highlights = para["highlights"]

    n_marks = int(rng.integers(2, min(5, len(highlights) + 1)))
    chosen_idx = list(rng.choice(len(highlights), size=n_marks, replace=False))

    color_pool = list(rng.choice(len(_MARK_COLORS), size=n_marks, replace=False))
    html_text = text
    applied: list[dict[str, str]] = []
    for pos, ci in zip(chosen_idx, color_pool):
        phrase = highlights[pos]
        hex_c, color_name = _MARK_COLORS[ci]
        if phrase in html_text:
            html_text = _mark_html(html_text, phrase, hex_c)
            applied.append({"phrase": phrase, "color": hex_c, "color_name": color_name})

    markdown = f"<p>{html_text}</p>\n"

    first_phrase = applied[0]["phrase"] if applied else ""
    gt_data = {
        "annotation_type": "highlight", "category": cat_name,
        "highlighted_phrases": applied, "highlight_count": len(applied),
    }
    fallback = [
        {
            "question": f"In the '{cat_name}' highlighted-text section, how many phrases are marked?",
            "answer": str(len(applied)),
            "assertions": [{"text": f"The response must state {len(applied)}."}],
        },
        {
            "question": f"In the '{cat_name}' highlighted-text section, what is the first highlighted phrase?",
            "answer": first_phrase,
            "assertions": [{"text": f"The response must mention '{first_phrase}'."}],
        },
    ]
    qa = _gen_element_qa(client, topic, "text highlight", gt_data, fallback)
    return Variant(
        id=f"ann_hl_{i:04d}_{_uid()}",
        kind="annotation",
        metadata={
            "annotation_family": "visual",
            "annotation_type": "highlight",
            "category": cat_name,
            "highlight_count": len(applied),
        },
        render_payload={"markdown": markdown},
        ground_truth={
            "annotation_type": "highlight",
            "category": cat_name,
            "base_text": text,
            "highlighted_phrases": applied,
            "qa_candidates": qa,
        },
    )


# ── 2. Callout block ────────────────────────────────────────────────────────

def _generate_callout_block(
    rng: np.random.Generator, i: int, client: Any = None, topic: Any = None
) -> Variant:
    style_keys = list(_CALLOUT_STYLES)
    style = style_keys[int(rng.integers(0, len(style_keys)))]
    messages = _CALLOUT_MESSAGES[style]
    msg = messages[int(rng.integers(0, len(messages)))]

    cat_name, para = _pick_paragraph(rng)
    text = para["text"]

    # Random placement: callout before or after the paragraph
    if rng.random() < 0.5:
        markdown = _callout_html(style, msg) + f"\n\n{text}\n"
        placement = "before"
    else:
        markdown = f"{text}\n\n" + _callout_html(style, msg) + "\n"
        placement = "after"

    _, _, _, label = _CALLOUT_STYLES[style]
    gt_data = {
        "annotation_type": "callout", "category": cat_name,
        "callout_label": label, "callout_text": msg, "callout_style": style,
    }
    fallback = [
        {
            "question": f"In the '{cat_name}' section, what type of callout box is shown?",
            "answer": label,
            "assertions": [{"text": f"The response must state '{label}'."}],
        },
        {
            "question": f"What does the {label} callout box in the '{cat_name}' section state?",
            "answer": msg[:100],
            "assertions": [{"text": f"The response must mention '{msg[:60]}'."}],
        },
    ]
    qa = _gen_element_qa(client, topic, "callout box", gt_data, fallback)
    return Variant(
        id=f"ann_co_{i:04d}_{_uid()}",
        kind="annotation",
        metadata={
            "annotation_family": "visual",
            "annotation_type": "callout",
            "callout_style": style,
            "placement": placement,
            "category": cat_name,
        },
        render_payload={"markdown": markdown},
        ground_truth={
            "annotation_type": "callout",
            "callout_style": style,
            "callout_label": label,
            "callout_text": msg,
            "category": cat_name,
            "placement": placement,
            "qa_candidates": qa,
        },
    )


# ── 3. Label block ──────────────────────────────────────────────────────────

def _generate_label_block(
    rng: np.random.Generator, i: int, client: Any = None, topic: Any = None
) -> Variant:
    label_keys = list(_LABEL_STYLES)
    banner_key = label_keys[int(rng.integers(0, len(label_keys)))]
    # Pick a distinct inline status label
    inline_key = label_keys[int(rng.integers(0, len(label_keys)))]
    if inline_key == banner_key:
        inline_key = label_keys[(label_keys.index(banner_key) + 1) % len(label_keys)]

    cat_name, para = _pick_paragraph(rng)
    text = para["text"]

    markdown = (
        _label_banner_html(banner_key)
        + "\n"
        + text
        + f"\n\n*Status: {_inline_label_html(inline_key)}*\n"
    )

    gt_data = {
        "annotation_type": "label", "category": cat_name,
        "banner_label": banner_key, "inline_label": inline_key,
    }
    fallback = [
        {
            "question": f"In the section with a '{banner_key}' classification banner, what inline status label appears at the bottom?",
            "answer": inline_key,
            "assertions": [{"text": f"The response must state '{inline_key}'."}],
        },
        {
            "question": f"What classification banner is shown in the '{cat_name}' section that has a '{inline_key}' status label?",
            "answer": banner_key,
            "assertions": [{"text": f"The response must state '{banner_key}'."}],
        },
    ]
    qa = _gen_element_qa(client, topic, "classification label", gt_data, fallback)
    return Variant(
        id=f"ann_lb_{i:04d}_{_uid()}",
        kind="annotation",
        metadata={
            "annotation_family": "visual",
            "annotation_type": "label",
            "banner_label": banner_key,
            "inline_label": inline_key,
            "category": cat_name,
        },
        render_payload={"markdown": markdown},
        ground_truth={
            "annotation_type": "label",
            "banner_label": banner_key,
            "inline_label": inline_key,
            "category": cat_name,
            "qa_candidates": qa,
        },
    )


# ── 4. Sticky note block (PDF-level) ────────────────────────────────────────

def _generate_sticky_note_block(
    rng: np.random.Generator, i: int, client: Any = None, topic: Any = None
) -> Variant:
    cat_name, para = _pick_paragraph(rng)
    text = para["text"]
    highlights = para["highlights"]

    n_notes = int(rng.integers(1, min(4, len(highlights) + 1)))
    chosen_idx = list(rng.choice(len(highlights), size=n_notes, replace=False))

    pdf_annotations: list[dict[str, Any]] = []
    for j in chosen_idx:
        phrase = highlights[j]
        note_text = _pick_notes(para, j, rng)
        pdf_annotations.append({
            "type": "sticky_note",
            "target_text": phrase,
            "note": note_text,
            "color": list(_PDF_COLORS["yellow"]),
        })

    first_phrase = highlights[chosen_idx[0]]
    gt_data = {
        "annotation_type": "sticky_note", "category": cat_name,
        "note_count": n_notes, "first_annotated_phrase": first_phrase,
    }
    fallback = [
        {
            "question": f"In the '{cat_name}' section with sticky-note annotations, how many reviewer comments are attached?",
            "answer": str(n_notes),
            "assertions": [{"text": f"The response must state {n_notes}."}],
        },
        {
            "question": f"In the '{cat_name}' sticky-note section, what phrase has the first annotation?",
            "answer": first_phrase,
            "assertions": [{"text": f"The response must mention '{first_phrase}'."}],
        },
    ]
    qa = _gen_element_qa(client, topic, "sticky note annotation", gt_data, fallback)
    return Variant(
        id=f"ann_sn_{i:04d}_{_uid()}",
        kind="annotation",
        metadata={
            "annotation_family": "pdf",
            "annotation_type": "sticky_note",
            "note_count": n_notes,
            "category": cat_name,
        },
        render_payload={
            "markdown": text + "\n",
            "pdf_annotations": pdf_annotations,
        },
        ground_truth={
            "annotation_type": "sticky_note",
            "category": cat_name,
            "pdf_annotations": pdf_annotations,
            "qa_candidates": qa,
        },
    )


# ── 5. Freetext box block (PDF-level) ────────────────────────────────────────

# Visual styles for freetext boxes: {(fill_rgb, border_rgb, text_rgb)}
_FREETEXT_STYLES: list[dict[str, list[float]]] = [
    {"fill": [1.0, 1.0, 0.88], "border": [0.80, 0.50, 0.00], "text_color": [0.50, 0.10, 0.00]},
    {"fill": [0.88, 0.94, 1.0], "border": [0.20, 0.40, 0.80], "text_color": [0.10, 0.20, 0.60]},
    {"fill": [0.90, 1.00, 0.90], "border": [0.10, 0.60, 0.20], "text_color": [0.05, 0.40, 0.10]},
    {"fill": [1.00, 0.90, 0.90], "border": [0.80, 0.20, 0.20], "text_color": [0.60, 0.00, 0.00]},
    {"fill": [0.95, 0.90, 1.00], "border": [0.50, 0.20, 0.80], "text_color": [0.35, 0.05, 0.55]},
]


def _generate_freetext_box_block(
    rng: np.random.Generator, i: int, client: Any = None, topic: Any = None
) -> Variant:
    cat_name, para = _pick_paragraph(rng)
    text = para["text"]
    highlights = para["highlights"]

    n_boxes = int(rng.integers(1, 3))
    chosen_idx = list(rng.choice(len(highlights), size=min(n_boxes, len(highlights)), replace=False))

    pdf_annotations: list[dict[str, Any]] = []
    for k, j in enumerate(chosen_idx):
        phrase = highlights[j]
        comment = _pick_notes(para, j, rng)
        style = _FREETEXT_STYLES[k % len(_FREETEXT_STYLES)]
        pdf_annotations.append({
            "type": "freetext",
            "target_text": phrase,
            "text": f"⚠ {comment}",
            "fill_color": style["fill"],
            "border_color": style["border"],
            "text_color": style["text_color"],
            "fontsize": 8,
        })

    first_phrase = highlights[chosen_idx[0]] if chosen_idx else ""
    gt_data = {
        "annotation_type": "freetext", "category": cat_name,
        "box_count": len(pdf_annotations), "first_annotated_phrase": first_phrase,
    }
    fallback = [
        {
            "question": f"In the '{cat_name}' section with freetext boxes, how many annotation boxes are attached?",
            "answer": str(len(pdf_annotations)),
            "assertions": [{"text": f"The response must state {len(pdf_annotations)}."}],
        },
        {
            "question": f"In the '{cat_name}' freetext-annotated section, what phrase triggers the first annotation?",
            "answer": first_phrase,
            "assertions": [{"text": f"The response must mention '{first_phrase}'."}],
        },
    ]
    qa = _gen_element_qa(client, topic, "freetext annotation", gt_data, fallback)
    return Variant(
        id=f"ann_ft_{i:04d}_{_uid()}",
        kind="annotation",
        metadata={
            "annotation_family": "pdf",
            "annotation_type": "freetext",
            "box_count": len(pdf_annotations),
            "category": cat_name,
        },
        render_payload={
            "markdown": text + "\n",
            "pdf_annotations": pdf_annotations,
        },
        ground_truth={
            "annotation_type": "freetext",
            "category": cat_name,
            "pdf_annotations": pdf_annotations,
            "qa_candidates": qa,
        },
    )


# ── 6. Review block (hybrid: visual marks + PDF sticky notes) ───────────────

def _generate_review_block(
    rng: np.random.Generator, i: int, client: Any = None, topic: Any = None
) -> Variant:
    cat_name, para = _pick_paragraph(rng)
    text = para["text"]
    highlights = para["highlights"]

    n_visual = int(rng.integers(2, min(4, len(highlights) + 1)))
    visual_idx = list(rng.choice(len(highlights), size=n_visual, replace=False))
    color_pool = list(rng.choice(len(_MARK_COLORS), size=n_visual, replace=False))

    html_text = text
    applied_marks: list[dict[str, str]] = []
    for j, ci in zip(visual_idx, color_pool):
        phrase = highlights[j]
        hex_c, color_name = _MARK_COLORS[ci]
        if phrase in html_text:
            html_text = _mark_html(html_text, phrase, hex_c)
            applied_marks.append({"phrase": phrase, "color": hex_c})

    # PDF sticky notes on the same phrases
    pdf_annotations: list[dict[str, Any]] = []
    for j in visual_idx:
        phrase = highlights[j]
        note = _pick_notes(para, j, rng)
        pdf_annotations.append({
            "type": "sticky_note",
            "target_text": phrase,
            "note": note,
            "color": list(_PDF_COLORS["yellow"]),
        })

    markdown = f"<p>{html_text}</p>\n"

    has_notes = bool(pdf_annotations)
    first_note = pdf_annotations[0]["note"] if pdf_annotations else ""
    gt_data = {
        "annotation_type": "review", "category": cat_name,
        "highlight_count": len(applied_marks),
        "has_pdf_notes": has_notes, "first_note": first_note,
    }
    fallback = [
        {
            "question": f"In the '{cat_name}' review section, how many phrases are visually highlighted?",
            "answer": str(len(applied_marks)),
            "assertions": [{"text": f"The response must state {len(applied_marks)}."}],
        },
        {
            "question": f"Does the '{cat_name}' review section include PDF reviewer sticky-note comments?",
            "answer": "Yes" if has_notes else "No",
            "assertions": [{"text": f"The response must state '{'Yes' if has_notes else 'No'}'."}],
        },
        {
            "question": f"In the '{cat_name}' review section, what does the first reviewer comment say?",
            "answer": first_note,
            "assertions": [{"text": f"The response must mention '{first_note[:60]}'."}],
        },
    ]
    qa = _gen_element_qa(client, topic, "review annotation", gt_data, fallback)
    return Variant(
        id=f"ann_rv_{i:04d}_{_uid()}",
        kind="annotation",
        metadata={
            "annotation_family": "hybrid",
            "annotation_type": "review",
            "highlight_count": len(applied_marks),
            "pdf_note_count": len(pdf_annotations),
            "category": cat_name,
        },
        render_payload={
            "markdown": markdown,
            "pdf_annotations": pdf_annotations,
        },
        ground_truth={
            "annotation_type": "review",
            "category": cat_name,
            "highlighted_phrases": applied_marks,
            "pdf_annotations": pdf_annotations,
            "qa_candidates": qa,
        },
    )


# ── 8. Colored table block ──────────────────────────────────────────────────

def _colored_table_html(
    table_data: dict[str, Any],
    rng: np.random.Generator,
) -> str:
    """Build an HTML table with per-cell background colours and <mark> highlights.

    - Header row: dark navy background with white text.
    - Data rows: green/red/neutral cell bg matching the row's trend.
    - highlight_col cells wrap their value in <mark style="background-color: X">.
    """
    headers = table_data["headers"]
    rows = table_data["rows"]
    title = table_data["title"]

    lines: list[str] = [f"<p><strong>{title}</strong></p>"]
    lines.append('<table style="border-collapse: collapse; width: 100%;">')

    # Header row
    lines.append("<tr>")
    for h in headers:
        lines.append(
            f'<th bgcolor="{_HEADER_BG}" '
            f'style="background-color: {_HEADER_BG}; color: #ffffff; '
            f'padding: 6px 10px; font-weight: bold; border: 1px solid #334155;">'
            f"{h}</th>"
        )
    lines.append("</tr>")

    # Data rows
    for row_data in rows:
        cells = row_data["cells"]
        trend = row_data.get("trend", "neutral")
        highlight_col = row_data.get("highlight_col")

        if trend == "positive":
            row_bg, row_fg = "#dcfce7", "#15803d"
        elif trend == "negative":
            row_bg, row_fg = "#fee2e2", "#b91c1c"
        else:
            # Pick a neutral colour from the non-red/green end of the palette
            idx = int(rng.integers(4, len(_CELL_BG_COLORS)))
            row_bg, row_fg = _CELL_BG_COLORS[idx]

        lines.append("<tr>")
        for col_idx, cell_val in enumerate(cells):
            border = 'border: 1px solid #ccc;'
            if col_idx == 0:
                # Label column: light grey background
                lines.append(
                    f'<td style="padding: 5px 10px; {border} '
                    f'background-color: #f8fafc; font-weight: 500;">'
                    f"{cell_val}</td>"
                )
            elif highlight_col is not None and col_idx == highlight_col:
                # Highlighted cell: apply <mark> with a random colour from the palette
                mark_hex, _ = _MARK_COLORS[int(rng.integers(0, len(_MARK_COLORS)))]
                lines.append(
                    f'<td bgcolor="{row_bg}" '
                    f'style="background-color: {row_bg}; padding: 5px 10px; {border}">'
                    f'<mark style="background-color: {mark_hex}; '
                    f'padding: 1px 3px; border-radius: 2px;">'
                    f"{cell_val}</mark></td>"
                )
            else:
                # Regular data cell
                lines.append(
                    f'<td bgcolor="{row_bg}" '
                    f'style="background-color: {row_bg}; color: {row_fg}; '
                    f'padding: 5px 10px; {border}">'
                    f"{cell_val}</td>"
                )
        lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


def _generate_colored_table_block(
    rng: np.random.Generator, i: int, client: Any = None, topic: Any = None
) -> Variant:
    """Annotation variant: table with per-cell background colours and marked values."""
    table_data = _TABLE_DATA[int(rng.integers(0, len(_TABLE_DATA)))]
    html = _colored_table_html(table_data, rng)

    headers = table_data["headers"]
    rows = table_data["rows"]
    positive = sum(1 for r in rows if r.get("trend") == "positive")
    negative = sum(1 for r in rows if r.get("trend") == "negative")
    table_title = table_data["title"]

    gt_data = {
        "annotation_type": "colored_table", "table_title": table_title,
        "headers": headers, "row_count": len(rows),
        "positive_rows": positive, "negative_rows": negative,
    }
    fallback = [
        {
            "question": f"How many columns does the '{table_title}' colored table have?",
            "answer": str(len(headers)),
            "assertions": [{"text": f"The response must state {len(headers)}."}],
        },
        {
            "question": f"In the '{table_title}' table, how many rows show a positive trend (highlighted green)?",
            "answer": str(positive),
            "assertions": [{"text": f"The response must state {positive}."}],
        },
        {
            "question": f"In the '{table_title}' table, how many rows show a negative trend (highlighted red)?",
            "answer": str(negative),
            "assertions": [{"text": f"The response must state {negative}."}],
        },
    ]
    qa = _gen_element_qa(client, topic, "colored table", gt_data, fallback)
    return Variant(
        id=f"ann_ct_{i:04d}_{_uid()}",
        kind="annotation",
        metadata={
            "annotation_family": "visual",
            "annotation_type": "colored_table",
            "table_title": table_title,
            "n_rows": len(rows),
            "n_cols": len(headers),
            "positive_rows": positive,
            "negative_rows": negative,
        },
        render_payload={"markdown": html},
        ground_truth={
            "annotation_type": "colored_table",
            "table_title": table_title,
            "headers": headers,
            "row_count": len(rows),
            "positive_rows": positive,
            "negative_rows": negative,
            "qa_candidates": qa,
        },
    )


# ── 7. Annotated report section (dense multi-type) ──────────────────────────

def _generate_annotated_report(
    rng: np.random.Generator, i: int, client: Any = None, topic: Any = None
) -> Variant:
    """Dense section: classification banner + callout box + marked text + PDF notes."""
    label_keys = list(_LABEL_STYLES)
    banner = label_keys[int(rng.integers(0, len(label_keys)))]

    style_keys = list(_CALLOUT_STYLES)
    style = style_keys[int(rng.integers(0, len(style_keys)))]
    callout_msg = _CALLOUT_MESSAGES[style][int(rng.integers(0, len(_CALLOUT_MESSAGES[style])))]

    cat_name, para = _pick_paragraph(rng)
    text = para["text"]
    highlights = para["highlights"]

    # Apply 2 visual marks
    n_marks = min(2, len(highlights))
    mark_idx = list(rng.choice(len(highlights), size=n_marks, replace=False))
    color_pool = list(rng.choice(len(_MARK_COLORS), size=n_marks, replace=False))

    html_text = text
    applied_marks: list[dict[str, str]] = []
    for j, ci in zip(mark_idx, color_pool):
        phrase = highlights[j]
        hex_c, _ = _MARK_COLORS[ci]
        if phrase in html_text:
            html_text = _mark_html(html_text, phrase, hex_c)
            applied_marks.append({"phrase": phrase, "color": hex_c})

    markdown = (
        _label_banner_html(banner)
        + "\n"
        + _callout_html(style, callout_msg)
        + "\n\n"
        + f"<p>{html_text}</p>\n"
    )

    # PDF sticky notes (one per marked phrase)
    pdf_annotations: list[dict[str, Any]] = []
    for j in mark_idx:
        phrase = highlights[j]
        note = _pick_notes(para, j, rng)
        pdf_annotations.append({
            "type": "sticky_note",
            "target_text": phrase,
            "note": note,
            "color": list(_PDF_COLORS["yellow"]),
        })

    _, _, _, callout_label = _CALLOUT_STYLES[style]
    has_notes = bool(pdf_annotations)
    gt_data = {
        "annotation_type": "annotated_report", "category": cat_name,
        "banner_label": banner, "callout_label": callout_label,
        "highlight_count": len(applied_marks), "has_pdf_notes": has_notes,
    }
    fallback = [
        {
            "question": f"In the '{cat_name}' section with a '{banner}' banner, what type of callout box is present?",
            "answer": callout_label,
            "assertions": [{"text": f"The response must state '{callout_label}'."}],
        },
        {
            "question": f"In the '{cat_name}' section classified as '{banner}', how many phrases are highlighted?",
            "answer": str(len(applied_marks)),
            "assertions": [{"text": f"The response must state {len(applied_marks)}."}],
        },
        {
            "question": f"Does the '{cat_name}' annotated section with a '{banner}' banner have PDF reviewer comments attached?",
            "answer": "Yes" if has_notes else "No",
            "assertions": [{"text": f"The response must state '{'Yes' if has_notes else 'No'}'."}],
        },
    ]
    qa = _gen_element_qa(client, topic, "annotated report section", gt_data, fallback)
    return Variant(
        id=f"ann_ar_{i:04d}_{_uid()}",
        kind="annotation",
        metadata={
            "annotation_family": "hybrid",
            "annotation_type": "annotated_report",
            "banner_label": banner,
            "callout_style": style,
            "highlight_count": len(applied_marks),
            "pdf_note_count": len(pdf_annotations),
            "category": cat_name,
        },
        render_payload={
            "markdown": markdown,
            "pdf_annotations": pdf_annotations,
        },
        ground_truth={
            "annotation_type": "annotated_report",
            "category": cat_name,
            "banner_label": banner,
            "callout_style": style,
            "callout_text": callout_msg,
            "highlighted_phrases": applied_marks,
            "pdf_annotations": pdf_annotations,
            "qa_candidates": qa,
        },
    )


# ---------------------------------------------------------------------------
# Variant-type dispatch table
# ---------------------------------------------------------------------------

_VARIANT_GENERATORS: list[tuple[str, Any]] = [
    ("highlight",        _generate_highlight_block),
    ("callout",          _generate_callout_block),
    ("label",            _generate_label_block),
    ("sticky_note",      _generate_sticky_note_block),
    ("freetext",         _generate_freetext_box_block),
    ("review",           _generate_review_block),
    ("annotated_report", _generate_annotated_report),
    ("colored_table",    _generate_colored_table_block),
]


# ---------------------------------------------------------------------------
# Public per-type generator functions
# ---------------------------------------------------------------------------

def generate_highlight_variants(
    rng: np.random.Generator, count: int, assets_dir: "Path | None" = None,
    client: Any = None, topic: Any = None,
) -> list[Variant]:
    return [_generate_highlight_block(fork_rng(rng, f"hl_{i}"), i, client=client, topic=topic) for i in range(count)]


def generate_callout_variants(
    rng: np.random.Generator, count: int, assets_dir: "Path | None" = None,
    client: Any = None, topic: Any = None,
) -> list[Variant]:
    return [_generate_callout_block(fork_rng(rng, f"co_{i}"), i, client=client, topic=topic) for i in range(count)]


def generate_label_variants(
    rng: np.random.Generator, count: int, assets_dir: "Path | None" = None,
    client: Any = None, topic: Any = None,
) -> list[Variant]:
    return [_generate_label_block(fork_rng(rng, f"lb_{i}"), i, client=client, topic=topic) for i in range(count)]


def generate_sticky_note_variants(
    rng: np.random.Generator, count: int, assets_dir: "Path | None" = None,
    client: Any = None, topic: Any = None,
) -> list[Variant]:
    return [_generate_sticky_note_block(fork_rng(rng, f"sn_{i}"), i, client=client, topic=topic) for i in range(count)]


def generate_freetext_variants(
    rng: np.random.Generator, count: int, assets_dir: "Path | None" = None,
    client: Any = None, topic: Any = None,
) -> list[Variant]:
    return [_generate_freetext_box_block(fork_rng(rng, f"ft_{i}"), i, client=client, topic=topic) for i in range(count)]


def generate_review_variants(
    rng: np.random.Generator, count: int, assets_dir: "Path | None" = None,
    client: Any = None, topic: Any = None,
) -> list[Variant]:
    return [_generate_review_block(fork_rng(rng, f"rv_{i}"), i, client=client, topic=topic) for i in range(count)]


def generate_annotated_report_variants(
    rng: np.random.Generator, count: int, assets_dir: "Path | None" = None,
    client: Any = None, topic: Any = None,
) -> list[Variant]:
    return [_generate_annotated_report(fork_rng(rng, f"ar_{i}"), i, client=client, topic=topic) for i in range(count)]


def generate_colored_table_variants(
    rng: np.random.Generator, count: int, assets_dir: "Path | None" = None,
    client: Any = None, topic: Any = None,
) -> list[Variant]:
    return [_generate_colored_table_block(fork_rng(rng, f"ct_{i}"), i, client=client, topic=topic) for i in range(count)]


# ---------------------------------------------------------------------------
# Public pool function
# ---------------------------------------------------------------------------

def generate_annotation_pool(
    rng: np.random.Generator,
    count: int,
    assets_dir: Path | None = None,
) -> list[Variant]:
    """Generate a balanced mix of all 7 rich annotation variant types.

    Splits *count* evenly across all 7 types (round-robin for any remainder).
    All returned variants have ``kind="annotation"`` so they route into the
    ``"annotations"`` pool in the document builder and are eligible for the
    PDF annotation post-process pass.
    """
    n_types = len(_VARIANT_GENERATORS)
    base = count // n_types
    extras = count % n_types

    variants: list[Variant] = []
    for k, (type_name, gen_fn) in enumerate(_VARIANT_GENERATORS):
        type_count = base + (1 if k < extras else 0)
        for j in range(type_count):
            child_rng = fork_rng(rng, f"ann_{type_name}_{j}")
            variants.append(gen_fn(child_rng, j))
    return variants
