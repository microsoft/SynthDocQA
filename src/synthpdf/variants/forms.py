# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Form-type variant generators.

Generates HTML representations of four interactive/structured element types:

  - **Checklists** – task, inspection, deployment, or onboarding checklists
    with checked/unchecked items.
  - **Buttons** – action-button groups (approve/reject, submit/cancel, etc.)
    styled as inline HTML spans.
  - **Forms** – multi-section visual forms (contact, expense, feedback) with
    labelled field types rendered as a two-column HTML table.
  - **Fillable forms** – complex, realistic document forms (insurance claim,
    job application) with section headings, field types, and an office-use box.

Each variant emits:
  - An HTML string in ``render_payload["markdown"]`` — passed unchanged through
    the Markdown → PDF pipeline (xhtml2pdf/WeasyPrint respect HTML blocks).
  - ``kind = "text"`` so existing ``markdown_builder`` routing works without
    modification.
  - ``metadata["subtype"]`` to distinguish form subtypes in manifests / SEVAL.
  - A structured ``ground_truth`` dict with field counts, completion rates, etc.

Public API (mirrors tables.py / text_annotations.py):
    generate_checklist_variants(rng, count, assets_dir) -> list[Variant]
    generate_button_variants(rng, count, assets_dir)    -> list[Variant]
    generate_form_variants(rng, count, assets_dir)      -> list[Variant]
    generate_fillable_form_variants(rng, count, assets_dir) -> list[Variant]
    generate_form_pool(rng, count, assets_dir)          -> list[Variant]
"""
from __future__ import annotations

import json as _json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from synthpdf.rng import fork_rng
from synthpdf.types import Variant


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Corpus data
# ---------------------------------------------------------------------------

_CHECKLIST_TOPICS: list[tuple[str, list[str]]] = [
    (
        "Software Deployment Checklist",
        [
            "Build artefacts compiled and verified",
            "Unit tests passing (coverage ≥ 80%)",
            "Integration tests passing",
            "Security vulnerability scan completed",
            "Database migrations reviewed and approved",
            "Environment configuration files updated",
            "Rollback plan documented and tested",
            "Monitoring and alerting configured",
            "API documentation updated",
            "Stakeholder sign-off obtained",
            "Load testing completed",
            "DNS and SSL certificate verified",
        ],
    ),
    (
        "Workplace Safety Inspection",
        [
            "Fire extinguishers inspected and tagged",
            "Emergency exits clear and unobstructed",
            "First aid kits stocked and in-date",
            "Electrical panels accessible and labelled",
            "Safety signs visible and legible",
            "Hazardous materials properly stored and labelled",
            "Personal protective equipment available",
            "Incident log reviewed and up to date",
            "Emergency contact list posted",
            "Spill kits available and serviceable",
        ],
    ),
    (
        "Project Review Checklist",
        [
            "Requirements documented and stakeholder-approved",
            "Design reviewed by all stakeholders",
            "Technical debt itemised and tracked",
            "Performance benchmarks achieved",
            "Accessibility guidelines followed (WCAG 2.1)",
            "Data privacy impact assessment completed",
            "Third-party vendor dependencies confirmed",
            "Budget reviewed and approved",
            "Timeline milestones verified",
            "Risk register updated",
            "Change management plan in place",
        ],
    ),
    (
        "Employee Onboarding Checklist",
        [
            "Welcome email and orientation materials sent",
            "IT equipment provisioned and tested",
            "System access credentials issued",
            "HR paperwork (tax forms, contracts) completed",
            "Benefits enrollment submitted",
            "Team introductions arranged",
            "First-week schedule shared with employee",
            "Mentor or buddy assigned",
            "30/60/90-day goals set with manager",
            "Training plan created and acknowledged",
            "Office access badge issued",
        ],
    ),
    (
        "Quality Assurance Release Checklist",
        [
            "Test plan reviewed and approved",
            "Test environment configured and stable",
            "Smoke tests executed — all pass",
            "Full regression suite completed",
            "Performance / load tests executed",
            "Security penetration testing completed",
            "User acceptance testing signed off",
            "All critical and high defects resolved",
            "Release notes prepared and distributed",
            "Go/No-Go decision recorded",
            "Post-release monitoring plan activated",
        ],
    ),
    (
        "Financial Close Checklist",
        [
            "All journal entries posted",
            "Bank reconciliations completed",
            "Accounts receivable ageing reviewed",
            "Accounts payable invoices approved",
            "Fixed asset register updated",
            "Intercompany eliminations posted",
            "Trial balance reviewed by controller",
            "Tax provisions calculated",
            "Management accounts distributed",
            "Variance analysis completed",
        ],
    ),
    (
        "Product Launch Checklist",
        [
            "Marketing collateral approved and ready",
            "Press release drafted and scheduled",
            "Social media campaign scheduled",
            "Sales training materials distributed",
            "Pricing finalised and communicated",
            "Support team briefed on known issues",
            "Analytics tracking verified",
            "Legal review of advertising copy completed",
            "Inventory levels confirmed sufficient",
            "Executive communications approved",
        ],
    ),
]

_BUTTON_GROUPS: list[dict[str, Any]] = [
    {
        "title": "Document Actions",
        "buttons": [
            {"label": "Approve", "style": "success"},
            {"label": "Reject", "style": "danger"},
            {"label": "Request Changes", "style": "warning"},
            {"label": "Save Draft", "style": "secondary"},
        ],
    },
    {
        "title": "Form Controls",
        "buttons": [
            {"label": "Submit", "style": "primary"},
            {"label": "Cancel", "style": "secondary"},
            {"label": "Reset", "style": "warning"},
        ],
    },
    {
        "title": "Step Navigation",
        "buttons": [
            {"label": "&#x2190; Previous", "style": "secondary"},
            {"label": "Next &#x2192;", "style": "primary"},
            {"label": "Save &amp; Continue", "style": "success"},
        ],
    },
    {
        "title": "Data Management",
        "buttons": [
            {"label": "Export CSV", "style": "secondary"},
            {"label": "Export PDF", "style": "secondary"},
            {"label": "Import Data", "style": "primary"},
            {"label": "Delete Selected", "style": "danger"},
        ],
    },
    {
        "title": "Workflow Controls",
        "buttons": [
            {"label": "Start", "style": "success"},
            {"label": "Pause", "style": "warning"},
            {"label": "Stop", "style": "danger"},
            {"label": "Restart", "style": "primary"},
        ],
    },
    {
        "title": "Review Actions",
        "buttons": [
            {"label": "Accept All", "style": "success"},
            {"label": "Accept Selected", "style": "primary"},
            {"label": "Reject All", "style": "danger"},
            {"label": "Defer", "style": "warning"},
        ],
    },
    {
        "title": "Report Options",
        "buttons": [
            {"label": "Generate Report", "style": "primary"},
            {"label": "Schedule", "style": "secondary"},
            {"label": "Preview", "style": "secondary"},
            {"label": "Download", "style": "success"},
        ],
    },
]

_FORM_TEMPLATES: list[dict[str, Any]] = [
    {
        "title": "Contact Information Form",
        "description": "Please complete all required fields marked with an asterisk (*).",
        "sections": [
            {
                "name": "Personal Details",
                "fields": [
                    {"label": "First Name", "type": "text", "required": True, "value": ""},
                    {"label": "Last Name", "type": "text", "required": True, "value": ""},
                    {"label": "Email Address", "type": "email", "required": True, "value": ""},
                    {"label": "Phone Number", "type": "text", "required": False, "value": ""},
                    {"label": "Date of Birth", "type": "date", "required": False, "value": ""},
                ],
            },
            {
                "name": "Address",
                "fields": [
                    {"label": "Street Address", "type": "text", "required": True, "value": ""},
                    {"label": "City", "type": "text", "required": True, "value": ""},
                    {
                        "label": "State / Province",
                        "type": "dropdown",
                        "required": False,
                        "options": ["Select…", "California", "New York", "Texas", "Florida", "Other"],
                    },
                    {"label": "Postal Code", "type": "text", "required": True, "value": ""},
                    {
                        "label": "Country",
                        "type": "dropdown",
                        "required": True,
                        "options": ["Select…", "United States", "United Kingdom", "Canada", "Australia", "Other"],
                    },
                ],
            },
        ],
    },
    {
        "title": "Employee Expense Report",
        "description": "Submit within 30 days of incurring the expense. Attach all receipts.",
        "sections": [
            {
                "name": "Employee Information",
                "fields": [
                    {"label": "Employee Name", "type": "text", "required": True, "value": ""},
                    {"label": "Employee ID", "type": "text", "required": True, "value": ""},
                    {
                        "label": "Department",
                        "type": "dropdown",
                        "required": True,
                        "options": ["Engineering", "Marketing", "Finance", "HR", "Operations"],
                    },
                    {"label": "Report Period", "type": "text", "required": True, "value": ""},
                    {"label": "Manager Name", "type": "text", "required": True, "value": ""},
                ],
            },
            {
                "name": "Expense Details",
                "fields": [
                    {
                        "label": "Expense Category",
                        "type": "dropdown",
                        "required": True,
                        "options": ["Travel", "Meals", "Accommodation", "Equipment", "Other"],
                    },
                    {"label": "Date of Expense", "type": "date", "required": True, "value": ""},
                    {"label": "Amount", "type": "number", "required": True, "value": ""},
                    {
                        "label": "Currency",
                        "type": "dropdown",
                        "required": False,
                        "options": ["USD", "EUR", "GBP", "CAD", "AUD"],
                    },
                    {"label": "Business Purpose", "type": "textarea", "required": True, "value": ""},
                    {"label": "Receipt Attached", "type": "checkbox", "required": False, "checked": False},
                ],
            },
        ],
    },
    {
        "title": "Customer Feedback Form",
        "description": "Your feedback helps us improve our services. All responses are confidential.",
        "sections": [
            {
                "name": "Service Experience",
                "fields": [
                    {
                        "label": "Overall Satisfaction",
                        "type": "radio",
                        "required": True,
                        "options": ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"],
                        "value": "",
                    },
                    {"label": "Service Received", "type": "text", "required": True, "value": ""},
                    {"label": "Service Date", "type": "date", "required": True, "value": ""},
                    {"label": "Staff Member (if known)", "type": "text", "required": False, "value": ""},
                ],
            },
            {
                "name": "Additional Comments",
                "fields": [
                    {"label": "What went well?", "type": "textarea", "required": False, "value": ""},
                    {"label": "What could be improved?", "type": "textarea", "required": False, "value": ""},
                    {
                        "label": "Would you recommend us?",
                        "type": "radio",
                        "required": False,
                        "options": ["Yes, definitely", "Probably yes", "Probably not", "Definitely not"],
                        "value": "",
                    },
                    {"label": "May we contact you for follow-up?", "type": "checkbox", "required": False, "checked": False},
                    {"label": "Contact Email (if yes)", "type": "email", "required": False, "value": ""},
                ],
            },
        ],
    },
    {
        "title": "IT Support Request Form",
        "description": "Complete this form to log a support ticket. Response within 1 business day.",
        "sections": [
            {
                "name": "Requester Details",
                "fields": [
                    {"label": "Full Name", "type": "text", "required": True, "value": ""},
                    {"label": "Email", "type": "email", "required": True, "value": ""},
                    {"label": "Phone Extension", "type": "text", "required": False, "value": ""},
                    {
                        "label": "Department",
                        "type": "dropdown",
                        "required": True,
                        "options": ["Engineering", "Sales", "Finance", "Legal", "HR", "Other"],
                    },
                    {"label": "Location / Office", "type": "text", "required": False, "value": ""},
                ],
            },
            {
                "name": "Issue Details",
                "fields": [
                    {
                        "label": "Issue Category",
                        "type": "dropdown",
                        "required": True,
                        "options": ["Hardware", "Software", "Network", "Account Access", "Email", "Printing", "Other"],
                    },
                    {
                        "label": "Priority",
                        "type": "radio",
                        "required": True,
                        "options": ["Critical", "High", "Medium", "Low"],
                        "value": "",
                    },
                    {"label": "Subject / Short Description", "type": "text", "required": True, "value": ""},
                    {"label": "Detailed Description", "type": "textarea", "required": True, "value": ""},
                    {"label": "Date / Time Issue Occurred", "type": "date", "required": False, "value": ""},
                    {"label": "Attach screenshot or log file?", "type": "checkbox", "required": False, "checked": False},
                ],
            },
        ],
    },
]

_FILLABLE_FORM_TEMPLATES: list[dict[str, Any]] = [
    {
        "title": "Insurance Claim Form",
        "form_number": "INS-CLM-2024-001",
        "instructions": (
            "Complete all sections in BLOCK CAPITALS. "
            "Incomplete or illegible forms will be returned. "
            "Attach supporting documentation and receipts."
        ),
        "sections": [
            {
                "name": "Section A — Claimant Information",
                "fields": [
                    {"label": "Policy Number", "type": "text", "name": "policy_number", "required": True, "value": ""},
                    {"label": "Claimant Full Name", "type": "text", "name": "claimant_name", "required": True, "value": ""},
                    {"label": "Date of Birth", "type": "date", "name": "dob", "required": True, "value": ""},
                    {"label": "National ID / Passport No.", "type": "text", "name": "national_id", "required": True, "value": ""},
                    {"label": "Contact Phone", "type": "text", "name": "phone", "required": True, "value": ""},
                    {"label": "Contact Email", "type": "email", "name": "email", "required": False, "value": ""},
                    {"label": "Mailing Address", "type": "textarea", "name": "address", "required": True, "value": ""},
                ],
            },
            {
                "name": "Section B — Incident Details",
                "fields": [
                    {"label": "Date of Incident", "type": "date", "name": "incident_date", "required": True, "value": ""},
                    {"label": "Time of Incident", "type": "text", "name": "incident_time", "required": False, "value": ""},
                    {"label": "Location of Incident", "type": "text", "name": "location", "required": True, "value": ""},
                    {
                        "label": "Claim Type",
                        "type": "dropdown",
                        "name": "claim_type",
                        "required": True,
                        "options": ["Property Damage", "Personal Injury", "Theft", "Natural Disaster", "Medical", "Other"],
                        "value": "",
                    },
                    {"label": "Estimated Loss Amount (USD)", "type": "number", "name": "loss_amount", "required": True, "value": ""},
                    {"label": "Description of Incident", "type": "textarea", "name": "description", "required": True, "value": ""},
                    {"label": "Were police / authorities notified?", "type": "checkbox", "name": "police_notified", "checked": False},
                    {"label": "Police Report Number (if applicable)", "type": "text", "name": "police_report", "required": False, "value": ""},
                ],
            },
            {
                "name": "Section C — Supporting Documents",
                "fields": [
                    {"label": "Photographs attached", "type": "checkbox", "name": "photos", "checked": False},
                    {"label": "Receipts / invoices attached", "type": "checkbox", "name": "receipts", "checked": False},
                    {"label": "Witness statements attached", "type": "checkbox", "name": "witness", "checked": False},
                    {"label": "Medical reports attached (if applicable)", "type": "checkbox", "name": "medical", "checked": False},
                    {"label": "Number of pages attached", "type": "number", "name": "page_count", "required": False, "value": ""},
                ],
            },
            {
                "name": "Section D — Declaration",
                "fields": [
                    {"label": "I certify the information above is true and correct", "type": "checkbox", "name": "certification", "checked": False},
                    {"label": "I consent to information sharing for claim processing", "type": "checkbox", "name": "consent", "checked": False},
                    {"label": "Signature Date", "type": "date", "name": "sig_date", "required": True, "value": ""},
                ],
            },
        ],
    },
    {
        "title": "Job Application Form",
        "form_number": "HR-APP-2024",
        "instructions": (
            "Read all instructions before completing. "
            "Use BLOCK CAPITALS throughout. "
            "Incomplete applications will not be considered."
        ),
        "sections": [
            {
                "name": "Section 1 — Personal Details",
                "fields": [
                    {"label": "Surname", "type": "text", "name": "surname", "required": True, "value": ""},
                    {"label": "First Name(s)", "type": "text", "name": "firstname", "required": True, "value": ""},
                    {"label": "National ID / Passport No.", "type": "text", "name": "id_number", "required": True, "value": ""},
                    {"label": "Nationality", "type": "text", "name": "nationality", "required": True, "value": ""},
                    {"label": "Date of Birth", "type": "date", "name": "dob", "required": True, "value": ""},
                    {
                        "label": "Gender",
                        "type": "radio",
                        "name": "gender",
                        "required": False,
                        "options": ["Male", "Female", "Non-binary", "Prefer not to say"],
                        "value": "",
                    },
                    {"label": "Current Address", "type": "textarea", "name": "address", "required": True, "value": ""},
                    {"label": "Contact Phone", "type": "text", "name": "phone", "required": True, "value": ""},
                    {"label": "Email Address", "type": "email", "name": "email", "required": True, "value": ""},
                ],
            },
            {
                "name": "Section 2 — Position Applied For",
                "fields": [
                    {"label": "Position Title", "type": "text", "name": "position", "required": True, "value": ""},
                    {
                        "label": "Department",
                        "type": "dropdown",
                        "name": "department",
                        "required": True,
                        "options": ["Engineering", "Sales", "Marketing", "Operations", "Finance", "HR"],
                        "value": "",
                    },
                    {
                        "label": "Employment Type",
                        "type": "radio",
                        "name": "emp_type",
                        "required": True,
                        "options": ["Full-time", "Part-time", "Contract", "Internship"],
                        "value": "",
                    },
                    {"label": "Available Start Date", "type": "date", "name": "start_date", "required": True, "value": ""},
                    {"label": "Expected Annual Salary (USD)", "type": "number", "name": "salary", "required": False, "value": ""},
                ],
            },
            {
                "name": "Section 3 — Education &amp; Experience",
                "fields": [
                    {
                        "label": "Highest Qualification",
                        "type": "dropdown",
                        "name": "qualification",
                        "required": True,
                        "options": [
                            "High School",
                            "Associate Degree",
                            "Bachelor's Degree",
                            "Master's Degree",
                            "PhD",
                            "Professional Certification",
                        ],
                        "value": "",
                    },
                    {
                        "label": "Years of Relevant Experience",
                        "type": "dropdown",
                        "name": "experience",
                        "required": True,
                        "options": ["0-1 years", "2-3 years", "4-5 years", "6-10 years", "10+ years"],
                        "value": "",
                    },
                    {"label": "Key Skills / Technologies", "type": "textarea", "name": "skills", "required": True, "value": ""},
                    {
                        "label": "How did you hear about this role?",
                        "type": "dropdown",
                        "name": "referral",
                        "required": False,
                        "options": ["Company Website", "LinkedIn", "Job Board", "Employee Referral", "Recruiter", "Other"],
                        "value": "",
                    },
                ],
            },
            {
                "name": "Section 4 — Declaration",
                "fields": [
                    {"label": "All information provided is accurate and complete", "type": "checkbox", "name": "accuracy", "checked": False},
                    {"label": "I consent to background verification checks", "type": "checkbox", "name": "bg_check", "checked": False},
                    {"label": "I have the right to work in the advertised country", "type": "checkbox", "name": "work_right", "checked": False},
                    {"label": "Signature Date", "type": "date", "name": "sig_date", "required": True, "value": ""},
                ],
            },
        ],
    },
    {
        "title": "Vendor Registration Form",
        "form_number": "PROC-VND-001",
        "instructions": (
            "Complete all mandatory fields. Attach company registration certificate "
            "and bank confirmation letter. Approval takes 5–10 business days."
        ),
        "sections": [
            {
                "name": "Company Details",
                "fields": [
                    {"label": "Legal Company Name", "type": "text", "name": "company_name", "required": True, "value": ""},
                    {"label": "Trading Name (if different)", "type": "text", "name": "trading_name", "required": False, "value": ""},
                    {"label": "Company Registration No.", "type": "text", "name": "reg_number", "required": True, "value": ""},
                    {"label": "VAT / Tax ID", "type": "text", "name": "tax_id", "required": True, "value": ""},
                    {"label": "Date of Incorporation", "type": "date", "name": "inc_date", "required": False, "value": ""},
                    {
                        "label": "Business Type",
                        "type": "dropdown",
                        "name": "biz_type",
                        "required": True,
                        "options": ["Sole Trader", "Partnership", "Private Limited", "Public Limited", "NGO / Non-Profit", "Other"],
                        "value": "",
                    },
                ],
            },
            {
                "name": "Primary Contact",
                "fields": [
                    {"label": "Contact Full Name", "type": "text", "name": "contact_name", "required": True, "value": ""},
                    {"label": "Job Title", "type": "text", "name": "contact_title", "required": True, "value": ""},
                    {"label": "Direct Phone", "type": "text", "name": "contact_phone", "required": True, "value": ""},
                    {"label": "Email Address", "type": "email", "name": "contact_email", "required": True, "value": ""},
                ],
            },
            {
                "name": "Banking Details",
                "fields": [
                    {"label": "Bank Name", "type": "text", "name": "bank_name", "required": True, "value": ""},
                    {"label": "Account Name", "type": "text", "name": "account_name", "required": True, "value": ""},
                    {"label": "Account Number", "type": "text", "name": "account_number", "required": True, "value": ""},
                    {"label": "Sort Code / Routing No.", "type": "text", "name": "sort_code", "required": True, "value": ""},
                    {
                        "label": "Currency",
                        "type": "dropdown",
                        "name": "currency",
                        "required": True,
                        "options": ["USD", "EUR", "GBP", "CAD", "AUD", "Other"],
                        "value": "",
                    },
                ],
            },
            {
                "name": "Declaration",
                "fields": [
                    {"label": "Information provided is accurate and complete", "type": "checkbox", "name": "accuracy", "checked": False},
                    {"label": "Company registration certificate attached", "type": "checkbox", "name": "cert_attached", "checked": False},
                    {"label": "Bank confirmation letter attached", "type": "checkbox", "name": "bank_letter", "checked": False},
                    {"label": "Authorised signatory name", "type": "text", "name": "signatory", "required": True, "value": ""},
                    {"label": "Date", "type": "date", "name": "sig_date", "required": True, "value": ""},
                ],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

_BUTTON_COLORS: dict[str, tuple[str, str]] = {
    "primary":   ("#1a3a5c", "#ffffff"),
    "secondary": ("#6c757d", "#ffffff"),
    "success":   ("#28a745", "#ffffff"),
    "danger":    ("#dc3545", "#ffffff"),
    "warning":   ("#ffc107", "#333333"),
}


def _render_checklist_html(title: str, items: list[dict[str, Any]]) -> str:
    """Render a checklist as a two-column HTML table (symbol | item text)."""
    rows: list[str] = []
    for item in items:
        checked = item.get("checked", False)
        text = str(item.get("text", ""))
        symbol = "&#x2611;" if checked else "&#x2610;"
        text_style = (
            "color:#555555; text-decoration:line-through;"
            if checked
            else "color:#222222;"
        )
        rows.append(
            f"<tr>"
            f'<td width="8%" style="font-size:13pt; text-align:center; padding:3px 6px; '
            f'border:1px solid #cccccc; background:#f8f8f8; vertical-align:middle;">'
            f"{symbol}</td>"
            f'<td width="92%" style="font-size:9pt; padding:4px 8px; '
            f'border:1px solid #cccccc; {text_style} vertical-align:middle;">'
            f"{text}</td>"
            f"</tr>"
        )
    rows_html = "\n".join(rows)
    return (
        f'<table border="0" cellspacing="0" cellpadding="0" width="92%" '
        f'style="border-collapse:collapse; margin:6px 0 10px 0;">\n'
        f'<tr><th colspan="2" style="background:#1a3a5c; color:#ffffff; '
        f'font-size:11pt; font-weight:bold; padding:6px 10px; text-align:left; '
        f'border:1px solid #1a3a5c;">{title}</th></tr>\n'
        f"{rows_html}\n"
        f"</table>"
    )


def _render_buttons_html(title: str, buttons: list[dict[str, Any]]) -> str:
    """Render a button group as an HTML inline table with styled spans."""
    cells: list[str] = []
    for btn in buttons:
        label = btn.get("label", "Action")
        style_key = btn.get("style", "primary")
        bg, fg = _BUTTON_COLORS.get(style_key, ("#1a3a5c", "#ffffff"))
        cells.append(
            f'<td style="padding:3px 5px;">'
            f'<span style="display:inline-block; background:{bg}; color:{fg}; '
            f"padding:5px 14px; font-size:9pt; font-family:Helvetica,Arial,sans-serif; "
            f'border-radius:3px; border:1px solid {bg};">{label}</span>'
            f"</td>"
        )
    cells_html = "\n".join(cells)
    return (
        f'<p style="font-size:10pt; font-weight:bold; margin:4px 0 4px 0;">{title}</p>\n'
        f'<table border="0" cellspacing="0" cellpadding="0" '
        f'style="margin:0 0 8px 0;">\n'
        f"<tr>{cells_html}</tr>\n"
        f"</table>"
    )


def _render_field_row(field: dict[str, Any]) -> str:
    """Render a single form field as an HTML two-column table row."""
    ftype = field.get("type", "text")
    label = field.get("label", "Field")
    value = str(field.get("value", ""))
    options: list[str] = field.get("options", [])
    checked = field.get("checked", False)
    required = field.get("required", False)

    req_mark = ' <span style="color:#cc0000;">*</span>' if required else ""
    label_cell = (
        f'<td width="42%" style="padding:4px 6px; border:1px solid #dddddd; '
        f'background:#f5f5f5; font-size:9pt; vertical-align:middle;">'
        f"<b>{label}</b>{req_mark}</td>"
    )

    if ftype == "text":
        widget = f"<u>{value if value else '_' * 30}</u>"
    elif ftype == "email":
        widget = (
            f'<u style="color:#1a3a5c;">'
            f"{value if value else '____@__________.__'}"
            f"</u>"
        )
    elif ftype == "date":
        widget = f"<u>{value if value else '__ / __ / ____'}</u>"
    elif ftype == "number":
        widget = (
            f'<u><span style="font-family:Courier,monospace;">'
            f"{value if value else '____________'}"
            f"</span></u>"
        )
    elif ftype == "dropdown":
        selected = value if value else (options[0] if options else "Select…")
        widget = f"[&#x25BC;&nbsp;{selected}]"
    elif ftype == "radio":
        parts: list[str] = []
        for opt in options:
            sym = "&#x25CF;" if opt == value else "&#x25CB;"
            parts.append(f"({sym})&nbsp;{opt}")
        widget = "&nbsp;&nbsp;&nbsp;".join(parts) if parts else "(&#x25CB;)&nbsp;Yes&nbsp;&nbsp;&nbsp;(&#x25CB;)&nbsp;No"
    elif ftype == "checkbox":
        sym = "&#x2611;" if checked else "&#x2610;"
        widget = f'<span style="font-size:12pt;">{sym}</span>'
    elif ftype == "textarea":
        content = value if value else ("_" * 44 + "<br/>" + "_" * 44 + "<br/>" + "_" * 44)
        widget = f'<span style="font-size:9pt;">{content}</span>'
    else:
        widget = f"<u>{value if value else '_' * 30}</u>"

    value_cell = (
        f'<td width="58%" style="padding:4px 8px; border:1px solid #dddddd; '
        f"font-size:9pt; vertical-align:middle;\">{widget}</td>"
    )
    return f"<tr>{label_cell}{value_cell}</tr>"


def _render_form_html(title: str, description: str, sections: list[dict[str, Any]]) -> str:
    """Render a multi-section visual form as HTML."""
    parts: list[str] = [
        f'<table border="0" cellspacing="0" cellpadding="0" width="100%" '
        f'style="border-collapse:collapse; margin-bottom:6px;">\n'
        f'<tr><td style="background:#1a3a5c; color:#ffffff; font-size:12pt; '
        f'font-weight:bold; padding:8px 10px; border:1px solid #1a3a5c;">{title}</td></tr>\n'
        f'<tr><td style="font-size:8pt; color:#666666; padding:4px 10px; '
        f'border:1px solid #dddddd; background:#f9f9f9;">{description}</td></tr>\n'
        f"</table>"
    ]
    for section in sections:
        sec_name = section.get("name", "")
        fields = section.get("fields", [])
        field_rows = "\n".join(_render_field_row(f) for f in fields)
        parts.append(
            f'<p style="font-size:10pt; font-weight:bold; color:#2e6da4; '
            f'margin:6px 0 2px 0;">{sec_name}</p>\n'
            f'<table border="0" cellspacing="0" cellpadding="0" width="100%" '
            f'style="border-collapse:collapse; margin-bottom:6px;">\n'
            f"{field_rows}\n"
            f"</table>"
        )
    return "\n".join(parts)


def _render_fillable_form_html(
    title: str,
    form_number: str,
    instructions: str,
    sections: list[dict[str, Any]],
) -> str:
    """Render a detailed fillable-form as HTML with header banner, sections, and signature box."""
    header = (
        f'<table border="0" cellspacing="0" cellpadding="0" width="100%" '
        f'style="border-collapse:collapse; margin-bottom:4px;">\n'
        f"<tr>\n"
        f'<td width="72%" style="background:#1a3a5c; color:#ffffff; font-size:13pt; '
        f'font-weight:bold; padding:8px 12px; border:1px solid #1a3a5c;">{title}</td>\n'
        f'<td width="28%" style="background:#1a3a5c; color:#cce0f0; font-size:8pt; '
        f'text-align:right; padding:8px 12px; border:1px solid #1a3a5c;">'
        f"Form No:&nbsp;{form_number}</td>\n"
        f"</tr>\n"
        f'<tr><td colspan="2" style="background:#fff8e0; color:#664400; font-size:8pt; '
        f'padding:5px 12px; border:1px solid #dddddd;">'
        f"<b>Instructions:</b>&nbsp;{instructions}</td></tr>\n"
        f"</table>"
    )
    parts: list[str] = [header]

    for section in sections:
        sec_name = section.get("name", "")
        fields = section.get("fields", [])
        field_rows = "\n".join(_render_field_row(f) for f in fields)
        sec_header = (
            f'<table border="0" cellspacing="0" cellpadding="0" width="100%" '
            f'style="border-collapse:collapse; margin:5px 0 1px 0;">\n'
            f'<tr><td style="background:#dce6f1; color:#1a3a5c; font-size:10pt; '
            f'font-weight:bold; padding:4px 8px; border:1px solid #a0b4cc;">'
            f"{sec_name}</td></tr>\n"
            f"</table>"
        )
        parts.append(sec_header)
        parts.append(
            f'<table border="0" cellspacing="0" cellpadding="0" width="100%" '
            f'style="border-collapse:collapse; margin-bottom:4px;">\n'
            f"{field_rows}\n"
            f"</table>"
        )

    # Signature / office-use row
    sig_row = (
        f'<table border="0" cellspacing="0" cellpadding="0" width="100%" '
        f'style="border-collapse:collapse; margin-top:6px;">\n'
        f"<tr>\n"
        f'<td width="50%" style="border:1px solid #aaaaaa; padding:5px 8px; '
        f'font-size:9pt; height:32px;">'
        f"<b>Authorised Signature:</b>&nbsp;_______________________</td>\n"
        f'<td width="25%" style="border:1px solid #aaaaaa; padding:5px 8px; '
        f'font-size:9pt;">'
        f"<b>Date:</b>&nbsp;__ / __ / ____</td>\n"
        f'<td width="25%" style="border:1px solid #aaaaaa; padding:5px 8px; '
        f'font-size:8pt; background:#f0f4ff;">'
        f"<b>FOR OFFICE USE ONLY</b><br/>Ref:&nbsp;________<br/>"
        f"Processed by:&nbsp;________</td>\n"
        f"</tr>\n"
        f"</table>"
    )
    parts.append(sig_row)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Ground-truth builders
# ---------------------------------------------------------------------------

def _checklist_ground_truth(title: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    checked = [it for it in items if it.get("checked")]
    unchecked = [it for it in items if not it.get("checked")]
    return {
        "title": title,
        "total_items": len(items),
        "checked_count": len(checked),
        "unchecked_count": len(unchecked),
        "completion_rate_pct": round(len(checked) / max(len(items), 1) * 100, 1),
        "checked_items": [it["text"] for it in checked],
        "unchecked_items": [it["text"] for it in unchecked],
    }


def _form_ground_truth(title: str, sections: list[dict[str, Any]]) -> dict[str, Any]:
    all_fields: list[dict[str, Any]] = []
    for sec in sections:
        for f in sec.get("fields", []):
            all_fields.append(
                {
                    "label": f.get("label", ""),
                    "type": f.get("type", "text"),
                    "section": sec.get("name", ""),
                    "required": f.get("required", False),
                }
            )
    required = [f for f in all_fields if f["required"]]
    field_type_counts: dict[str, int] = {}
    for f in all_fields:
        field_type_counts[f["type"]] = field_type_counts.get(f["type"], 0) + 1
    return {
        "title": title,
        "section_count": len(sections),
        "field_count": len(all_fields),
        "required_field_count": len(required),
        "field_type_counts": field_type_counts,
        "sections": [s.get("name", "") for s in sections],
        "required_fields": [f["label"] for f in required],
    }


# ---------------------------------------------------------------------------
# Form subtype constant (used by manifest and markdown_builder)
# ---------------------------------------------------------------------------

FORM_SUBTYPES: frozenset[str] = frozenset((
    "checklist", "button", "form", "fillable_form",
    "filled_form", "filled_fillable",
))


# ---------------------------------------------------------------------------
# LLM QA helper — generates diverse, anchored QnA via ContentGenerator
# ---------------------------------------------------------------------------

def _gen_element_qa(
    client: Any,
    topic: Any,
    element_type: str,
    ground_truth: dict[str, Any],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return LLM-generated QnA if client is available, otherwise return *fallback*.

    Uses existing question strings in *fallback* as few-shot phrasing inspiration
    only — the LLM generates varied formulations rather than copying verbatim.
    """
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
# QA candidate builders — anchored to element title; LLM path when available
# ---------------------------------------------------------------------------

def _checklist_qa_candidates(
    title: str, items: list[dict[str, Any]], gt: dict[str, Any],
    client: Any = None, topic: Any = None,
) -> list[dict[str, Any]]:
    total = gt.get("total_items", len(items))
    checked = gt.get("checked_count", 0)
    fallback: list[dict[str, Any]] = [
        {
            "question": f"How many total items are on the '{title}' checklist?",
            "answer": str(total),
            "assertions": [{"text": f"The response must state {total}."}],
        },
        {
            "question": f"How many items are marked as completed on the '{title}' checklist?",
            "answer": str(checked),
            "assertions": [{"text": f"The response must state {checked}."}],
        },
    ]
    for first_checked in gt.get("checked_items", [])[:1]:
        fallback.append({
            "question": f"Name one item that is marked as checked on the '{title}' checklist.",
            "answer": first_checked,
            "assertions": [{"text": f"The response must mention '{first_checked}'."}],
        })
    for first_unchecked in gt.get("unchecked_items", [])[:1]:
        fallback.append({
            "question": f"Name one item that is not yet checked on the '{title}' checklist.",
            "answer": first_unchecked,
            "assertions": [{"text": f"The response must mention '{first_unchecked}'."}],
        })
    return _gen_element_qa(client, topic, "checklist", gt, fallback)


def _button_qa_candidates(
    title: str, context: str, buttons: list[dict[str, Any]], gt: dict[str, Any],
    client: Any = None, topic: Any = None,
) -> list[dict[str, Any]]:
    count = gt.get("button_count", len(buttons))
    btn_list = gt.get("buttons", [])
    fallback: list[dict[str, Any]] = [
        {
            "question": f"How many buttons are shown in the '{title}' button group?",
            "answer": str(count),
            "assertions": [{"text": f"The response must state {count}."}],
        },
    ]
    if btn_list:
        first_label = btn_list[0].get("label", "")
        if first_label:
            fallback.append({
                "question": f"What is the label of the first button in the '{title}' group?",
                "answer": first_label,
                "assertions": [{"text": f"The response must state '{first_label}'."}],
            })
        last_label = btn_list[-1].get("label", "")
        if last_label and last_label != first_label:
            fallback.append({
                "question": f"What is the label of the last button in the '{title}' group?",
                "answer": last_label,
                "assertions": [{"text": f"The response must state '{last_label}'."}],
            })
    if context:
        short_ctx = context[:80].rstrip(" ,.")
        fallback.append({
            "question": f"What is the purpose or context of the '{title}' button group?",
            "answer": short_ctx,
            "assertions": [{"text": f"The response must mention '{short_ctx[:60]}'."}],
        })
    return _gen_element_qa(client, topic, "button group", gt, fallback)


def _form_qa_candidates(
    title: str, description: str, sections: list[dict[str, Any]], gt: dict[str, Any],
    client: Any = None, topic: Any = None,
) -> list[dict[str, Any]]:
    sec_count = gt.get("section_count", len(sections))
    sec_names = gt.get("sections", [s.get("name", "") for s in sections])
    req_count = gt.get("required_field_count", 0)
    fallback: list[dict[str, Any]] = []
    if description:
        short_desc = description[:80].rstrip(" ,.")
        fallback.append({
            "question": f"What does the description text on the '{title}' form say?",
            "answer": short_desc,
            "assertions": [{"text": f"The response must mention '{short_desc}'."}],
        })
    fallback.append({
        "question": f"How many sections does the '{title}' form contain?",
        "answer": str(sec_count),
        "assertions": [{"text": f"The response must state {sec_count}."}],
    })
    if sec_names:
        fallback.append({
            "question": f"What is the name of the first section on the '{title}' form?",
            "answer": sec_names[0],
            "assertions": [{"text": f"The response must state '{sec_names[0]}'."}],
        })
    fallback.append({
        "question": f"How many fields on the '{title}' form are required?",
        "answer": str(req_count),
        "assertions": [{"text": f"The response must state {req_count}."}],
    })
    return _gen_element_qa(client, topic, "form", gt, fallback)


def _fillable_qa_candidates(
    title: str,
    form_number: str,
    instructions: str,
    sections: list[dict[str, Any]],
    gt: dict[str, Any],
    client: Any = None,
    topic: Any = None,
) -> list[dict[str, Any]]:
    sec_count = gt.get("section_count", len(sections))
    sec_names = gt.get("sections", [s.get("name", "") for s in sections])
    req_count = gt.get("required_field_count", 0)
    fallback: list[dict[str, Any]] = [
        {
            "question": f"What form number is shown on the '{title}' form?",
            "answer": form_number,
            "assertions": [{"text": f"The response must state '{form_number}'."}],
        },
    ]
    if instructions:
        first_sent = (instructions.split(".")[0].strip() + ".") if "." in instructions else instructions[:80]
        fallback.append({
            "question": f"What do the instructions on the '{title}' form say?",
            "answer": first_sent,
            "assertions": [{"text": f"The response must mention '{first_sent[:60]}'."}],
        })
    fallback.append({
        "question": f"How many sections does the '{title}' form have?",
        "answer": str(sec_count),
        "assertions": [{"text": f"The response must state {sec_count}."}],
    })
    if sec_names:
        fallback.append({
            "question": f"What is the name of the first section on the '{title}' form?",
            "answer": sec_names[0],
            "assertions": [{"text": f"The response must state '{sec_names[0]}'."}],
        })
    fallback.append({
        "question": f"How many fields on the '{title}' form are required?",
        "answer": str(req_count),
        "assertions": [{"text": f"The response must state {req_count}."}],
    })
    return _gen_element_qa(client, topic, "fillable form", gt, fallback)


# ---------------------------------------------------------------------------
# LLM content generators — topic-aware; fall back to static on any failure
# ---------------------------------------------------------------------------

def _llm_generate_checklist(client: Any, topic: Any) -> "dict[str, Any] | None":
    """Generate a topic-aware checklist via LLM.

    Returns {"title": str, "items": [{"text": str, "checked": bool}]} or None.
    """
    prompt = (
        f"Generate a realistic checklist for a document about: {topic.description}\n\n"
        "Return valid JSON only (no markdown fences):\n"
        '{"title": "...", "items": [{"text": "...", "checked": true}, ...]}\n\n'
        "Requirements:\n"
        "- title: a specific named checklist (e.g. 'Q4 Audit Readiness Checklist')\n"
        "- 8 to 12 items, about 40-60% marked checked: true\n"
        "- Every item text must be specific to the document topic — no generic placeholders\n"
        "- Checked items = completed steps; unchecked = pending\n"
        "- Items are action-oriented (start with a verb)"
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="easy", _label="form_checklist")
        data = _json.loads(raw)
        items = [
            {"text": str(it.get("text", "")).strip(), "checked": bool(it.get("checked", False))}
            for it in data.get("items", [])
            if str(it.get("text", "")).strip()
        ]
        if data.get("title") and items:
            return {"title": str(data["title"]).strip(), "items": items}
    except Exception:
        pass
    return None


def _llm_generate_button_group(client: Any, topic: Any) -> "dict[str, Any] | None":
    """Generate a topic-aware button group via LLM.

    Returns {"title": str, "context": str, "buttons": [{"label": str, "style": str}]}
    or None.
    """
    _valid_styles = {"primary", "secondary", "success", "danger", "warning"}
    prompt = (
        f"Generate a button group for a document workflow about: {topic.description}\n\n"
        "Return valid JSON only (no markdown fences):\n"
        '{"title": "...", "context": "one sentence describing when this button group appears", '
        '"buttons": [{"label": "...", "style": "primary|secondary|success|danger|warning"}, ...]}\n\n'
        "Requirements:\n"
        "- title: short action group name (e.g. 'Approval Controls')\n"
        "- context: one sentence describing the workflow step\n"
        "- 3 to 5 buttons forming a logical action set for the topic\n"
        "- Button labels specific to the topic — not generic Submit/Cancel\n"
        "- style must be one of: primary, secondary, success, danger, warning"
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="easy", _label="form_button")
        data = _json.loads(raw)
        buttons = [
            {
                "label": str(btn.get("label", "Action")).strip(),
                "style": btn.get("style") if btn.get("style") in _valid_styles else "primary",
            }
            for btn in data.get("buttons", [])
            if str(btn.get("label", "")).strip()
        ]
        if data.get("title") and buttons:
            return {
                "title": str(data["title"]).strip(),
                "context": str(data.get("context", "")).strip(),
                "buttons": buttons,
            }
    except Exception:
        pass
    return None


def _llm_generate_form(client: Any, topic: Any) -> "dict[str, Any] | None":
    """Generate a topic-aware multi-section form via LLM.

    Returns {"title": str, "description": str, "sections": [...]} or None.
    """
    _valid_types = {"text", "email", "date", "number", "dropdown", "radio", "checkbox", "textarea"}
    prompt = (
        f"Generate a realistic multi-section form for a document about: {topic.description}\n\n"
        "Return valid JSON only (no markdown fences):\n"
        '{"title": "...", "description": "...", "sections": [\n'
        '  {"name": "...", "fields": [\n'
        '    {"label": "...", "type": "text|email|date|number|dropdown|radio|checkbox|textarea", '
        '"required": true, "value": ""},\n'
        '    {"label": "...", "type": "dropdown", "required": false, '
        '"options": ["Opt1", "Opt2", "Opt3"], "value": ""},\n'
        "    ...\n"
        "  ]}\n"
        "]}\n\n"
        "Requirements:\n"
        "- title and description specific to the topic\n"
        "- description: one sentence instructing how to complete the form\n"
        "- 2 to 3 sections, 4 to 7 fields each\n"
        "- At least one dropdown (with options), one radio or checkbox, one textarea\n"
        "- Field labels specific to the topic — no generic 'Field 1'\n"
        "- About half the fields marked required: true\n"
        "- dropdown and radio fields must include an 'options' array of 3-5 realistic values"
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="medium", _label="form_standard")
        data = _json.loads(raw)
        cleaned_sections = []
        for sec in data.get("sections", []):
            fields = []
            for f in sec.get("fields", []):
                if not f.get("label"):
                    continue
                ftype = f.get("type", "text")
                if ftype not in _valid_types:
                    ftype = "text"
                clean = {"label": str(f["label"]).strip(), "type": ftype,
                         "required": bool(f.get("required", False)), "value": ""}
                if ftype in ("dropdown", "radio"):
                    opts = [str(o) for o in f.get("options", []) if str(o).strip()]
                    clean["options"] = opts if opts else ["Option A", "Option B"]
                if ftype == "checkbox":
                    clean["checked"] = bool(f.get("checked", False))
                fields.append(clean)
            if fields:
                cleaned_sections.append({"name": str(sec.get("name", "Section")), "fields": fields})
        if data.get("title") and cleaned_sections:
            return {
                "title": str(data["title"]).strip(),
                "description": str(data.get("description",
                    "Please complete all required fields marked with an asterisk (*).")).strip(),
                "sections": cleaned_sections,
            }
    except Exception:
        pass
    return None


def _llm_generate_fillable_form(client: Any, topic: Any) -> "dict[str, Any] | None":
    """Generate a topic-aware fillable form via LLM.

    Returns {"title": str, "form_number": str, "instructions": str, "sections": [...]}
    or None.
    """
    _valid_types = {"text", "email", "date", "number", "dropdown", "checkbox", "textarea"}
    prompt = (
        f"Generate a realistic fillable form for a document about: {topic.description}\n\n"
        "Return valid JSON only (no markdown fences):\n"
        '{"title": "...", "form_number": "...", "instructions": "...", "sections": [\n'
        '  {"name": "...", "fields": [\n'
        '    {"label": "...", "type": "text|email|date|number|dropdown|checkbox|textarea", '
        '"name": "field_key", "required": true, "value": ""},\n'
        '    {"label": "...", "type": "dropdown", "name": "key2", "required": false, '
        '"options": ["Opt1", "Opt2"], "value": ""},\n'
        "    ...\n"
        "  ]}\n"
        "]}\n\n"
        "Requirements:\n"
        "- title: a formal document name (e.g. 'Capital Expenditure Approval Form')\n"
        "- form_number: a realistic reference (e.g. 'FIN-CAPEX-2024', 'HR-007')\n"
        "- instructions: 1-2 sentences on how to complete and submit the form\n"
        "- 3 to 4 sections, 4 to 6 fields each\n"
        "- Include at least one dropdown (with options) and at least one checkbox\n"
        "- About half the fields marked required: true\n"
        "- All content specific to the document topic"
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="medium", _label="form_fillable")
        data = _json.loads(raw)
        cleaned_sections = []
        for sec in data.get("sections", []):
            fields = []
            for f in sec.get("fields", []):
                if not f.get("label"):
                    continue
                ftype = f.get("type", "text")
                if ftype not in _valid_types:
                    ftype = "text"
                clean = {"label": str(f["label"]).strip(), "type": ftype,
                         "name": str(f.get("name", "field")).strip(),
                         "required": bool(f.get("required", False)), "value": ""}
                if ftype == "dropdown":
                    opts = [str(o) for o in f.get("options", []) if str(o).strip()]
                    clean["options"] = opts if opts else ["Option A", "Option B"]
                if ftype == "checkbox":
                    clean["checked"] = bool(f.get("checked", False))
                fields.append(clean)
            if fields:
                cleaned_sections.append({"name": str(sec.get("name", "Section")), "fields": fields})
        if data.get("title") and data.get("form_number") and cleaned_sections:
            return {
                "title": str(data["title"]).strip(),
                "form_number": str(data["form_number"]).strip(),
                "instructions": str(data.get("instructions",
                    "Complete all required fields. Submit to the relevant department for approval.")).strip(),
                "sections": cleaned_sections,
            }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public variant generators
# ---------------------------------------------------------------------------

def generate_checklist_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,
    client: Any = None,
    topic: Any = None,
) -> list[Variant]:
    """Generate *count* checklist variants with mixed check states.

    When *client* and *topic* are provided, generates unique topic-aware content
    via LLM.  Falls back to the static corpus on any failure.
    ``kind="text"``, ``metadata["subtype"] = "checklist"``.
    """
    variants: list[Variant] = []
    for i in range(count):
        child_rng = fork_rng(rng, f"checklist_{i}")

        # LLM path: generate unique, topic-aware checklist
        llm_data = None
        if client is not None and topic is not None:
            llm_data = _llm_generate_checklist(client, topic)

        if llm_data is not None:
            title = llm_data["title"]
            items: list[dict[str, Any]] = llm_data["items"]
        else:
            # Static fallback
            topic_idx = int(child_rng.integers(0, len(_CHECKLIST_TOPICS)))
            title, all_items = _CHECKLIST_TOPICS[topic_idx]
            n_items = int(child_rng.integers(5, len(all_items) + 1))
            idxs = list(child_rng.choice(len(all_items), size=min(n_items, len(all_items)), replace=False))
            items = [
                {"text": all_items[int(j)], "checked": bool(child_rng.random() < 0.55)}
                for j in idxs
            ]

        gt = _checklist_ground_truth(title, items)
        gt["qa_candidates"] = _checklist_qa_candidates(title, items, gt, client=client, topic=topic)
        caption = f"Checklist: {title}"

        variants.append(
            Variant(
                id=f"checklist_{i:04d}_{_uid()[:8]}",
                kind="text",
                metadata={
                    "subtype": "checklist",
                    "title": title,
                    "item_count": len(items),
                    "checked_count": gt["checked_count"],
                    "completion_rate_pct": gt["completion_rate_pct"],
                    "caption": caption,
                    "llm_generated": llm_data is not None,
                },
                render_payload={
                    "markdown": _render_checklist_html(title, items),
                    "caption": caption,
                    "format": "html",
                },
                ground_truth=gt,
            )
        )
    return variants


def generate_button_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,
    client: Any = None,
    topic: Any = None,
) -> list[Variant]:
    """Generate *count* button-group variants.

    When *client* and *topic* are provided, generates unique topic-aware button
    groups via LLM.  Falls back to the static corpus on failure.
    ``kind="text"``, ``metadata["subtype"] = "button"``.
    """
    variants: list[Variant] = []
    for i in range(count):
        child_rng = fork_rng(rng, f"button_{i}")

        llm_data = None
        if client is not None and topic is not None:
            llm_data = _llm_generate_button_group(client, topic)

        if llm_data is not None:
            title = llm_data["title"]
            context = llm_data.get("context", "")
            buttons: list[dict[str, Any]] = llm_data["buttons"]
        else:
            group_idx = int(child_rng.integers(0, len(_BUTTON_GROUPS)))
            group = _BUTTON_GROUPS[group_idx]
            title = group["title"]
            context = ""
            buttons = group["buttons"]

        gt: dict[str, Any] = {
            "title": title,
            "context": context,
            "button_count": len(buttons),
            "buttons": [{"label": b["label"], "style": b.get("style", "primary")} for b in buttons],
        }
        gt["qa_candidates"] = _button_qa_candidates(title, context, buttons, gt, client=client, topic=topic)
        caption = f"Button Group: {title}"

        variants.append(
            Variant(
                id=f"button_{i:04d}_{_uid()[:8]}",
                kind="text",
                metadata={
                    "subtype": "button",
                    "title": title,
                    "button_count": len(buttons),
                    "button_labels": [b["label"] for b in buttons],
                    "caption": caption,
                    "llm_generated": llm_data is not None,
                },
                render_payload={
                    "markdown": _render_buttons_html(title, buttons),
                    "caption": caption,
                    "format": "html",
                },
                ground_truth=gt,
            )
        )
    return variants


def generate_form_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,
    client: Any = None,
    topic: Any = None,
) -> list[Variant]:
    """Generate *count* visual form variants (multi-section, labelled fields).

    When *client* and *topic* are provided, generates unique topic-aware form
    content via LLM.  Falls back to static templates on failure.
    ``kind="text"``, ``metadata["subtype"] = "form"``.
    """
    variants: list[Variant] = []
    for i in range(count):
        child_rng = fork_rng(rng, f"form_{i}")

        llm_data = None
        if client is not None and topic is not None:
            llm_data = _llm_generate_form(client, topic)

        if llm_data is not None:
            title = llm_data["title"]
            description = llm_data.get("description", "")
            sections = llm_data["sections"]
        else:
            tmpl_idx = int(child_rng.integers(0, len(_FORM_TEMPLATES)))
            tmpl = _FORM_TEMPLATES[tmpl_idx]
            title = tmpl["title"]
            description = tmpl.get("description", "")
            sections = tmpl["sections"]

        gt = _form_ground_truth(title, sections)
        gt["description"] = description
        gt["qa_candidates"] = _form_qa_candidates(title, description, sections, gt, client=client, topic=topic)
        caption = f"Form: {title}"

        variants.append(
            Variant(
                id=f"form_{i:04d}_{_uid()[:8]}",
                kind="text",
                metadata={
                    "subtype": "form",
                    "title": title,
                    "section_count": len(sections),
                    "field_count": gt["field_count"],
                    "caption": caption,
                    "llm_generated": llm_data is not None,
                },
                render_payload={
                    "markdown": _render_form_html(title, description, sections),
                    "caption": caption,
                    "format": "html",
                },
                ground_truth=gt,
            )
        )
    return variants


def generate_fillable_form_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,
    client: Any = None,
    topic: Any = None,
) -> list[Variant]:
    """Generate *count* fillable-form variants (complex, multi-section, with signature box).

    When *client* and *topic* are provided, generates unique topic-aware fillable
    forms via LLM.  Falls back to static templates on failure.
    ``kind="text"``, ``metadata["subtype"] = "fillable_form"``.
    """
    variants: list[Variant] = []
    for i in range(count):
        child_rng = fork_rng(rng, f"fillable_{i}")

        llm_data = None
        if client is not None and topic is not None:
            llm_data = _llm_generate_fillable_form(client, topic)

        if llm_data is not None:
            title = llm_data["title"]
            form_number = llm_data["form_number"]
            instructions = llm_data.get("instructions", "Complete all required fields.")
            sections = llm_data["sections"]
        else:
            tmpl_idx = int(child_rng.integers(0, len(_FILLABLE_FORM_TEMPLATES)))
            tmpl = _FILLABLE_FORM_TEMPLATES[tmpl_idx]
            title = tmpl["title"]
            form_number = tmpl.get("form_number", f"FORM-{i:04d}")
            instructions = tmpl.get("instructions", "Complete all required fields.")
            sections = tmpl["sections"]

        gt = _form_ground_truth(title, sections)
        gt["form_number"] = form_number
        gt["form_type"] = "fillable"
        gt["instructions"] = instructions
        gt["qa_candidates"] = _fillable_qa_candidates(title, form_number, instructions, sections, gt, client=client, topic=topic)
        caption = f"Fillable Form: {title}"

        variants.append(
            Variant(
                id=f"fillable_{i:04d}_{_uid()[:8]}",
                kind="text",
                metadata={
                    "subtype": "fillable_form",
                    "title": title,
                    "form_number": form_number,
                    "section_count": len(sections),
                    "field_count": gt["field_count"],
                    "caption": caption,
                    "llm_generated": llm_data is not None,
                },
                render_payload={
                    "markdown": _render_fillable_form_html(title, form_number, instructions, sections),
                    "caption": caption,
                    "format": "html",
                },
                ground_truth=gt,
            )
        )
    return variants


# ---------------------------------------------------------------------------
# Filled form helpers — style constants, value filler, renderer, QA builder
# ---------------------------------------------------------------------------

import re as _re

# Handwriting TTF fonts are packaged alongside the other local resources.
_FILL_FONTS_DIR = Path(__file__).parent.parent / "resources" / "fonts"

# Candidate fill fonts: (css_font_family, ttf_filename_or_None, allowed_pt_sizes)
# css_font_family MUST start with the exact ReportLab-registered font name.
# ttf_filename_or_None = None for fonts built-in to ReportLab (Helvetica etc.).
_FILL_FONT_CANDIDATES: list[tuple[str, "str | None", list[float]]] = [
    # --- Typed / document styles ---
    ("Helvetica, sans-serif",             None,                        [9.0, 9.5, 10.0, 10.5]),
    ("Times-Roman, serif",                None,                        [9.0, 9.5, 10.0, 10.5]),
    ("Courier, monospace",                None,                        [9.0, 9.5, 10.0]),
    # --- Handwriting / cursive styles (TTF required) ---
    ("KalamRegular, cursive",             "Kalam-Regular.ttf",         [10.0, 10.5, 11.0, 11.5]),
    ("PatrickHand, cursive",              "PatrickHand-Regular.ttf",   [10.0, 10.5, 11.0, 11.5]),
    ("IndieFlower, cursive",              "IndieFlower-Regular.ttf",   [10.5, 11.0, 11.5, 12.0]),
    ("CaveatRegular, cursive",            "Caveat-Regular.ttf",        [10.0, 10.5, 11.0]),
    ("DancingScript, cursive",            "DancingScript-Regular.ttf", [10.5, 11.0, 11.5, 12.0]),
    ("GrandHotel, cursive",               "GrandHotel.ttf",            [11.0, 11.5, 12.0, 12.5]),
    ("GochiHand, cursive",                "Gochi-Hand.ttf",            [10.0, 10.5, 11.0, 11.5]),
    ("ShadowsIntoLight, cursive",         "ShadowsIntoLight.ttf",      [10.5, 11.0, 11.5]),
]


def _build_filled_fonts() -> "list[tuple[str, list[float]]]":
    """Register handwriting TTF fonts with ReportLab and return usable (css, sizes) pairs.

    Only includes fonts that are either ReportLab built-ins or whose TTF file was
    successfully registered.  Always includes the 3 built-in typed fonts so the
    list is never empty even if the fonts/ directory is missing.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont as _TTFont
    except ImportError:
        return [(css, sizes) for css, ttf, sizes in _FILL_FONT_CANDIDATES if ttf is None]

    result: list[tuple[str, list[float]]] = []
    for css, ttf, sizes in _FILL_FONT_CANDIDATES:
        if ttf is None:
            result.append((css, sizes))
        else:
            rl_name = css.split(",")[0].strip()   # e.g. "KalamRegular"
            font_path = _FILL_FONTS_DIR / ttf
            if font_path.exists():
                try:
                    # Skip registration if already done (idempotent across reloads)
                    try:
                        pdfmetrics.getFont(rl_name)
                    except KeyError:
                        pdfmetrics.registerFont(_TTFont(rl_name, str(font_path)))
                    result.append((css, sizes))
                except Exception:
                    pass  # unregisterable font — silently skip
    return result or [("Helvetica, sans-serif", [9.0, 9.5, 10.0, 10.5])]


# Built once at module import; (css_family_str, [allowed_pt_sizes]) per entry.
_FILLED_FONTS: list[tuple[str, list[float]]] = _build_filled_fonts()

# Ink colours that simulate pen or typewriter text (never pure #000000)
_FILLED_COLORS: list[str] = [
    "#1c2db5",   # dark ballpoint blue
    "#0d4d0d",   # dark green pen
    "#222222",   # near-black
    "#b51c1c",   # dark red
    "#6b1c9e",   # dark purple
    "#004d66",   # teal-blue
]


def _llm_fill_form_values(
    client: Any, topic: Any, title: str, sections: list[dict[str, Any]]
) -> "dict[str, str] | None":
    """Ask the LLM to generate realistic filled values for every form field.

    Returns a dict mapping field label -> filled string value, or None on
    failure (caller falls back to blank values).
    """
    all_fields: list[dict[str, Any]] = []
    for sec in sections:
        for f in sec.get("fields", []):
            entry: dict[str, Any] = {"label": f["label"], "type": f.get("type", "text")}
            if f.get("options"):
                entry["options"] = f["options"]
            all_fields.append(entry)
    if not all_fields:
        return None

    prompt = (
        f"You are filling out the form '{title}' for a document about:\n"
        f"{topic.description}\n\n"
        "Generate a realistic, completed set of field values for this form.\n"
        "Return valid JSON only (no markdown fences):\n"
        '{"values": {"Field Label": "filled value", ...}}\n\n'
        f"Fields to fill:\n{_json.dumps(all_fields, indent=2)}\n\n"
        "Requirements per field type:\n"
        "- text: a specific name, ID, phrase, or short value\n"
        "- email: realistic address (name@domain.com)\n"
        "- date: realistic date in YYYY-MM-DD format\n"
        "- number: realistic numeric value (no units in the string)\n"
        "- textarea: 1-2 complete sentences relevant to the topic\n"
        "- dropdown: choose EXACTLY one option from the provided list\n"
        "- radio: choose EXACTLY one option from the provided list\n"
        "- checkbox: the word 'true' or 'false'\n"
        "All values must be internally consistent and specific to the topic."
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="easy", _label="form_fill_values")
        data = _json.loads(raw)
        values = data.get("values", {})
        if isinstance(values, dict) and values:
            return {str(k): str(v) for k, v in values.items()}
    except Exception:
        pass
    return None


def _apply_filled_sections(
    sections: list[dict[str, Any]],
    values: dict[str, str],
    font_family: str,
    ink_color: str,
    font_size_pt: float,
) -> list[dict[str, Any]]:
    """Return a deep copy of *sections* with field values populated and styled.

    Text-like values are wrapped in a ``<span>`` carrying the chosen font,
    ink colour, and size so the rendered PDF looks like a typed or handwritten
    entry.  Dropdown/radio selections are matched to the options list;
    checkbox fields use a bool derived from the LLM string value.
    """
    import copy
    filled = copy.deepcopy(sections)
    for sec in filled:
        for field in sec.get("fields", []):
            label = field.get("label", "")
            raw_val = values.get(label, "").strip()
            ftype = field.get("type", "text")

            if not raw_val:
                continue

            if ftype == "checkbox":
                field["checked"] = raw_val.lower() in ("true", "yes", "1", "x", "checked")

            elif ftype in ("dropdown", "radio"):
                opts = field.get("options", [])
                if raw_val in opts:
                    field["value"] = raw_val
                elif opts:
                    # Try case-insensitive match, then fall back to first option
                    lower_map = {o.lower(): o for o in opts}
                    field["value"] = lower_map.get(raw_val.lower(), opts[0])

            else:
                # Wrap in styled span — the existing renderer will embed it verbatim
                field["value"] = (
                    f'<span style="'
                    f"font-family:{font_family}; "
                    f"color:{ink_color}; "
                    f"font-size:{font_size_pt}pt;"
                    f'">{raw_val}</span>'
                )
    return filled


def _filled_form_qa_candidates(
    title: str,
    filled_sections: list[dict[str, Any]],
    gt: dict[str, Any],
    client: Any = None,
    topic: Any = None,
) -> list[dict[str, Any]]:
    """QA pairs for filled forms — anchored to title; LLM-diverse when client available.

    Tests up to 4 non-checkbox fields that actually have a value.
    Strips HTML styling tags so the expected answer is plain text.
    """
    fallback: list[dict[str, Any]] = []
    tested = 0
    for sec in filled_sections:
        for field in sec.get("fields", []):
            ftype = field.get("type", "text")
            if ftype == "checkbox":
                continue
            raw_val = field.get("value", "")
            if not raw_val:
                continue
            plain = _re.sub(r"<[^>]+>", "", str(raw_val)).strip()
            if not plain or plain.startswith("_"):
                continue
            label = field.get("label", "")
            display = plain[:80].rstrip(" ,.") + ("…" if len(plain) > 80 else "")
            fallback.append({
                "question": f"In the '{title}' form, what value is entered in the '{label}' field?",
                "answer": display,
                "assertions": [{"text": f"The response must state '{display[:60]}'."}],
            })
            tested += 1
            if tested >= 4:
                break
        if tested >= 4:
            break
    if not fallback:
        return []
    return _gen_element_qa(client, topic, "filled form", gt, fallback)


# ---------------------------------------------------------------------------
# Filled-form public generators
# ---------------------------------------------------------------------------

def generate_filled_form_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,
    client: Any = None,
    topic: Any = None,
) -> list[Variant]:
    """Generate *count* standard forms with all fields filled in.

    Flow:
      1. LLM (or static fallback) generates the form structure.
      2. LLM fills realistic values for every field.
      3. Values are rendered with varied font / ink colour so the form looks
         like a typed or handwritten completed document.

    ``kind="text"``, ``metadata["subtype"] = "filled_form"``.
    """
    variants: list[Variant] = []
    for i in range(count):
        child_rng = fork_rng(rng, f"filled_form_{i}")

        # Step 1 — form structure
        llm_struct = None
        if client is not None and topic is not None:
            llm_struct = _llm_generate_form(client, topic)

        if llm_struct is not None:
            title       = llm_struct["title"]
            description = llm_struct.get("description", "")
            sections    = llm_struct["sections"]
        else:
            tmpl_idx    = int(child_rng.integers(0, len(_FORM_TEMPLATES)))
            tmpl        = _FORM_TEMPLATES[tmpl_idx]
            title       = tmpl["title"]
            description = tmpl.get("description", "")
            sections    = tmpl["sections"]

        # Step 2 — fill values
        filled_values: dict[str, str] = {}
        if client is not None and topic is not None:
            filled_values = _llm_fill_form_values(client, topic, title, sections) or {}

        # Step 3 — visual style
        _font_entry = _FILLED_FONTS[int(child_rng.integers(0, len(_FILLED_FONTS)))]
        font_family, _size_opts = _font_entry
        ink_color   = _FILLED_COLORS[int(child_rng.integers(0, len(_FILLED_COLORS)))]
        font_size   = float(child_rng.choice(_size_opts))

        filled_sections = _apply_filled_sections(sections, filled_values, font_family, ink_color, font_size)

        gt = _form_ground_truth(title, filled_sections)
        gt["description"]  = description
        gt["filled"]       = True
        gt["fill_style"]   = {"font_family": font_family, "ink_color": ink_color, "font_size_pt": font_size}
        gt["qa_candidates"] = _filled_form_qa_candidates(title, filled_sections, gt, client=client, topic=topic)
        caption = f"Filled Form: {title}"

        variants.append(Variant(
            id=f"filled_form_{i:04d}_{_uid()[:8]}",
            kind="text",
            metadata={
                "subtype":       "filled_form",
                "title":         title,
                "section_count": len(filled_sections),
                "field_count":   gt["field_count"],
                "caption":       caption,
                "filled":        True,
                "llm_generated": llm_struct is not None,
            },
            render_payload={
                "markdown": _render_form_html(title, description, filled_sections),
                "caption":  caption,
                "format":   "html",
            },
            ground_truth=gt,
        ))
    return variants


def generate_filled_fillable_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,
    client: Any = None,
    topic: Any = None,
) -> list[Variant]:
    """Generate *count* fillable forms with all fields filled in.

    Same flow as ``generate_filled_form_variants`` but uses the fillable-form
    structure (form number, instruction banner, signature row).

    ``kind="text"``, ``metadata["subtype"] = "filled_fillable"``.
    """
    variants: list[Variant] = []
    for i in range(count):
        child_rng = fork_rng(rng, f"filled_fillable_{i}")

        # Step 1 — form structure
        llm_struct = None
        if client is not None and topic is not None:
            llm_struct = _llm_generate_fillable_form(client, topic)

        if llm_struct is not None:
            title        = llm_struct["title"]
            form_number  = llm_struct["form_number"]
            instructions = llm_struct.get("instructions", "Complete all required fields.")
            sections     = llm_struct["sections"]
        else:
            tmpl_idx     = int(child_rng.integers(0, len(_FILLABLE_FORM_TEMPLATES)))
            tmpl         = _FILLABLE_FORM_TEMPLATES[tmpl_idx]
            title        = tmpl["title"]
            form_number  = tmpl.get("form_number", f"FORM-{i:04d}")
            instructions = tmpl.get("instructions", "Complete all required fields.")
            sections     = tmpl["sections"]

        # Step 2 — fill values
        filled_values: dict[str, str] = {}
        if client is not None and topic is not None:
            filled_values = _llm_fill_form_values(client, topic, title, sections) or {}

        # Step 3 — visual style
        _font_entry = _FILLED_FONTS[int(child_rng.integers(0, len(_FILLED_FONTS)))]
        font_family, _size_opts = _font_entry
        ink_color   = _FILLED_COLORS[int(child_rng.integers(0, len(_FILLED_COLORS)))]
        font_size   = float(child_rng.choice(_size_opts))

        filled_sections = _apply_filled_sections(sections, filled_values, font_family, ink_color, font_size)

        gt = _form_ground_truth(title, filled_sections)
        gt["form_number"]  = form_number
        gt["form_type"]    = "filled_fillable"
        gt["instructions"] = instructions
        gt["filled"]       = True
        gt["fill_style"]   = {"font_family": font_family, "ink_color": ink_color, "font_size_pt": font_size}
        gt["qa_candidates"] = _filled_form_qa_candidates(title, filled_sections, gt, client=client, topic=topic)
        # Prepend form-number QA (anchored) if LLM didn't already cover it
        if not any(form_number in q.get("answer", "") for q in gt["qa_candidates"]):
            gt["qa_candidates"].insert(0, {
                "question": f"What form number is shown on the '{title}' form?",
                "answer": form_number,
                "assertions": [{"text": f"The response must state '{form_number}'."}],
            })
        caption = f"Filled Fillable Form: {title}"

        variants.append(Variant(
            id=f"filled_fillable_{i:04d}_{_uid()[:8]}",
            kind="text",
            metadata={
                "subtype":       "filled_fillable",
                "title":         title,
                "form_number":   form_number,
                "section_count": len(filled_sections),
                "field_count":   gt["field_count"],
                "caption":       caption,
                "filled":        True,
                "llm_generated": llm_struct is not None,
            },
            render_payload={
                "markdown": _render_fillable_form_html(title, form_number, instructions, filled_sections),
                "caption":  caption,
                "format":   "html",
            },
            ground_truth=gt,
        ))
    return variants


def generate_form_pool(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,
) -> list[Variant]:
    """Generate a mixed pool of all four form-type variants.

    The pool is split approximately equally among checklists, buttons,
    standard forms, and fillable forms (any rounding goes to checklists first).

    Args:
        rng:        Seeded numpy random generator.
        count:      Total number of form variants to generate.
        assets_dir: Output directory for any asset files (unused by forms,
                    kept for API consistency with other pool generators).

    Returns:
        A flat list of ``Variant`` objects in the order:
        checklists → buttons → forms → fillable forms.
    """
    rng_cl = fork_rng(rng, "pool_checklists")
    rng_bt = fork_rng(rng, "pool_buttons")
    rng_fm = fork_rng(rng, "pool_forms")
    rng_ff = fork_rng(rng, "pool_fillable")

    base = count // 4
    extras = count % 4  # distribute remainder to first `extras` subtypes

    n_cl = base + (1 if extras > 0 else 0)
    n_bt = base + (1 if extras > 1 else 0)
    n_fm = base + (1 if extras > 2 else 0)
    n_ff = base

    return (
        generate_checklist_variants(rng_cl, n_cl, assets_dir)
        + generate_button_variants(rng_bt, n_bt, assets_dir)
        + generate_form_variants(rng_fm, n_fm, assets_dir)
        + generate_fillable_form_variants(rng_ff, n_ff, assets_dir)
    )
