# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Scanned-document and handwriting variant generators.

Synthetic scanned-document and handwriting generation.
into the standard variant pattern with significantly improved realism:

Rendering improvements over the initial version
------------------------------------------------
* **150 DPI A4 page layout** (1240 × 1754 px) — text looks like genuine
  print output, not a low-res screen render.  Letterhead, margins, footer,
  and optional diagonal watermark ("DRAFT", "CONFIDENTIAL") are all included.
* **Drawn table grid** — proper PIL cell-border lines with header shading and
  right-aligned numeric columns; no ASCII art.
* **Advanced scan artifacts** on top of the base ``add_scan_artifacts`` pass:
  - Uneven scanner illumination (one-corner brightness gradient)
  - Fold / crease lines (1–3 thin dark diagonals or horizontals)
  - Edge shadow (scanner lid not fully closed)
  - Bleed-through (faint mirrored ghost from back of page, aged/poor only)
  - Slight perspective warp (misaligned feed / handheld scan)
* **Ruled notebook paper** background for handwriting variants (blue lines,
  red margin, optional hole-punch circles) + light scan pass over the result.
* **Realistic rubber stamps** — rotated ±20°, ink bleed via channel-shifted
  Gaussian blur, semi-transparent paste ON TOP of the scanned page; multiple
  ink colours (red approve, blue certify, purple confidential).

Public API (unchanged from v1):
    generate_scanned_text_variants(rng, count, assets_dir)  -> list[Variant]
    generate_scanned_table_variants(rng, count, assets_dir) -> list[Variant]
    generate_handwriting_variants(rng, count, assets_dir)   -> list[Variant]
    generate_stamp_mark_variants(rng, count, assets_dir)    -> list[Variant]
    generate_scan_pool(rng, count, assets_dir)              -> list[Variant]
"""
from __future__ import annotations

import io
import json
import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from synthpdf.rng import fork_rng
from synthpdf.types import Variant


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Scan quality parameter tiers
# ---------------------------------------------------------------------------

_SCAN_QUALITY_POOL: list[str] = ["pristine", "high", "medium", "aged", "poor"]

_SCAN_QUALITY_PARAMS: dict[str, dict[str, Any]] = {
    "pristine": {
        "noise_level": 0.01, "skew_angle": 0.05, "contrast_factor": 0.98,
        "age": "modern",   "jpeg_quality": 95, "dust_level": 0.00,
    },
    "high": {
        "noise_level": 0.03, "skew_angle": 0.20, "contrast_factor": 0.93,
        "age": "modern",   "jpeg_quality": 88, "dust_level": 0.00,
    },
    "medium": {
        "noise_level": 0.07, "skew_angle": 0.60, "contrast_factor": 0.86,
        "age": "modern",   "jpeg_quality": 82, "dust_level": 0.0002,
    },
    "aged": {
        "noise_level": 0.11, "skew_angle": 1.00, "contrast_factor": 0.80,
        "age": "aged_10y", "jpeg_quality": 76, "dust_level": 0.0005,
    },
    "poor": {
        "noise_level": 0.18, "skew_angle": 2.00, "contrast_factor": 0.70,
        "age": "aged_30y", "jpeg_quality": 65, "dust_level": 0.0012,
    },
}

# Paper background tint by age (used when building page from scratch)
_BG_TINT: dict[str, tuple[int, int, int]] = {
    "aged_30y": (248, 244, 230),
    "aged_10y": (250, 249, 242),
    "modern":   (252, 252, 250),
}

# ---------------------------------------------------------------------------
# A4 page geometry at 150 DPI
# ---------------------------------------------------------------------------

_PAGE_W = 1240      # A4 width  at 150 DPI
_PAGE_H = 1754      # A4 height at 150 DPI
_ML = 148           # left margin   (25 mm = 148 px @ 150 DPI)
_MR = 1092          # right margin x-coord
_MT = 118           # top margin    (20 mm = 118 px)
_MB = 1606          # bottom margin y-coord
_CW = _MR - _ML    # content width = 944 px

# Font sizes at 150 DPI  (1 pt ≈ 2.08 px @ 150 DPI)
_FS_BODY    = 17    # ~8 pt body
_FS_SUB     = 20    # ~10 pt secondary
_FS_SECTION = 24    # ~11.5 pt section heading
_FS_TITLE   = 30    # ~14 pt document title
_FS_SMALL   = 14    # ~7 pt footer

_LH_BODY    = 26    # body line height
_LH_SECTION = 38    # section heading line height
_LH_TITLE   = 48    # title line height

# ---------------------------------------------------------------------------
# Handwriting style / colour pools
# ---------------------------------------------------------------------------

_HW_STYLE_POOL: list[str] = [
    "normal", "bold", "neat", "rushed", "sans",
    "cursive", "gochi", "shadows", "caveat_bold",
]
_HW_COLOR_POOL: list[tuple[int, int, int]] = [
    (10, 10, 130),    # deep blue
    (15, 55, 15),     # dark green
    (0, 0, 0),        # black
    (100, 20, 20),    # dark red
    (50, 50, 100),    # slate blue
]
_HW_BACKGROUND_POOL: list[str] = ["white", "cream", "yellow", "aged", "pink"]

# ---------------------------------------------------------------------------
# Sample text content
# ---------------------------------------------------------------------------

_SCAN_TEXT_BLOCKS: list[str] = [
    (
        "MERIDIAN CAPITAL GROUP — CONFIDENTIAL\n"
        "Quarterly Performance Report  |  Q3 FY 2024\n\n"
        "Revenue for the quarter reached $4.82M, representing a 12.4% increase\n"
        "year-over-year against a target of $4.50M. Operating expenses remained\n"
        "within budget at $3.15M. Net profit margin improved to 34.6%.\n\n"
        "Key variances:\n"
        "  • APAC region exceeded target by 18%\n"
        "  • EMEA underperformed by 6% — recovery plan in progress\n"
        "  • Digital product revenue now 41% of total (up from 33%)\n\n"
        "Action items:\n"
        "  1. CFO to review variance analysis before 30 Oct\n"
        "  2. EMEA recovery plan due to steering committee by 15 Nov\n"
        "  3. Updated forecast to be circulated to board members\n\n"
        "Approval required from CFO before distribution."
    ),
    (
        "APEX MEDICAL CENTER — CLINICAL STUDY REPORT\n"
        "Protocol: APC-2024-031  |  Phase III  |  Interim Analysis\n\n"
        "Primary endpoint: 28-day all-cause mortality.\n"
        "Enrolment: 847 subjects (ITT population: 831).\n"
        "Treatment arm: 418 subjects; Control arm: 413 subjects.\n\n"
        "Interim analysis (database lock 15-Sep-2024):\n"
        "  • Primary endpoint achieved (p=0.0031, HR 0.76 [95% CI: 0.62–0.93])\n"
        "  • SAE rate 4.3% (treatment) vs 6.1% (control)\n"
        "  • No unexpected safety signals identified\n\n"
        "Section 3 — Recommendations\n"
        "The Data Safety Monitoring Board recommends continuation of the trial\n"
        "with no protocol modifications. Full analysis anticipated Q1 2025.\n\n"
        "Sponsor review pending. Do not distribute."
    ),
    (
        "MORRISON & PARTNERS LLP — ATTORNEY-CLIENT PRIVILEGED\n"
        "Matter: Restructuring — Apex Holdings Inc.\n"
        "File Reference: MP-2024-A-0471  |  Date: 10 October 2024\n\n"
        "This memorandum summarises key terms agreed in the term sheet session.\n\n"
        "Section 1 — Commercial Terms\n"
        "1. Governing Law: New York\n"
        "2. Arbitration: JAMS — expedited rules\n"
        "3. Liability Cap: Aggregate fees, preceding 12 months\n"
        "4. Indemnification: Mutual, excluding gross negligence\n"
        "5. Confidentiality: 5 years post-termination\n\n"
        "Section 2 — Outstanding Issues\n"
        "The following items remain subject to negotiation:\n"
        "  • Definition of 'Material Adverse Change'\n"
        "  • Scope of representations and warranties\n"
        "  • Termination fee structure\n\n"
        "Partner review required. Not for client circulation."
    ),
    (
        "NEXGEN SYSTEMS INC — TECHNICAL SPECIFICATION\n"
        "Document ID: NGS-SPEC-2024-0092  Rev. 3  |  Status: DRAFT\n\n"
        "Service: Authentication Microservice (Auth-SVC)\n"
        "SLA Uptime: 99.95%  |  RTO: 15 minutes  |  RPO: 5 minutes\n\n"
        "Functional Requirements:\n"
        "  FR-01  OAuth 2.0 / OIDC compliant token issuance\n"
        "  FR-02  MFA enforcement for privileged accounts\n"
        "  FR-03  Session invalidation within 30 seconds\n"
        "  FR-04  Rate limiting: 1000 req/min per client\n\n"
        "Non-Functional Requirements:\n"
        "  NFR-01  P99 latency < 120 ms under 10k concurrent users\n"
        "  NFR-02  Horizontal auto-scaling on CPU > 70%\n"
        "  NFR-03  All secrets stored in HashiCorp Vault\n\n"
        "Pending security review by InfoSec team."
    ),
    (
        "DEPARTMENT OF COMMERCE — INTERNAL POLICY\n"
        "Policy Reference: DOC-POL-2024-0189\n"
        "Effective Date: 01 January 2025  |  Review Date: 01 January 2026\n\n"
        "SUBJECT: Remote Work Authorisation Framework\n\n"
        "Section 1 — Eligibility\n"
        "Employees at GS-7 and above may request up to 3 days per week\n"
        "of remote work, subject to supervisor approval and operational\n"
        "requirements as defined in Section 4.\n\n"
        "Section 2 — Equipment Provision\n"
        "Approved remote workers will be issued a government-furnished\n"
        "laptop with PIV card reader. Personal devices are not permitted.\n\n"
        "Section 3 — Security Requirements\n"
        "All remote sessions must use agency-approved VPN. Data must not\n"
        "be stored on personal devices or unapproved cloud services.\n\n"
        "Approved by: Director, Human Capital Division"
    ),
    (
        "PACIFIC INDUSTRIAL CORP — QUALITY CONTROL REPORT\n"
        "NCR Reference: NCR-2024-2847  |  Batch: PIC-B-20241003\n"
        "Issued by: Quality Assurance Department\n\n"
        "Non-Conformance Description:\n"
        "Tensile strength measurement (Batch PIC-B-20241003) returned\n"
        "values of 312 MPa against a specification minimum of 340 MPa.\n"
        "Out-of-specification rate: 7 of 50 samples tested (14%).\n\n"
        "Root Cause (preliminary):\n"
        "  • Heat treatment cycle duration reduced by 4 minutes\n"
        "    due to thermocouple calibration drift (Cal. ID TC-0047).\n"
        "  • Calibration schedule was last updated 18 months ago\n"
        "    — overdue by 6 months per SOP-QA-012.\n\n"
        "Disposition: HOLD — rework pending engineering sign-off.\n"
        "Estimated resolution: 5 business days."
    ),
    (
        "GLOBAL STAFFING SOLUTIONS — CONFIDENTIAL\n"
        "Performance Improvement Plan (PIP)\n"
        "Employee ID: GS-00847  |  Review Date: 28 Oct 2024\n"
        "Line Manager: Sarah T. Williams  |  HR Partner: R. Okafor\n\n"
        "Area of Concern: Client deliverable quality and timeliness\n\n"
        "Performance Expectations (Q4 2024):\n"
        "  1. Achieve ≥ 95% client satisfaction score\n"
        "  2. Complete 3 mandatory training modules by 15 Nov 2024\n"
        "  3. Submit weekly progress reports every Friday by 5 PM\n"
        "  4. Zero missed deadlines for Tier-1 client deliverables\n\n"
        "Support Provided:\n"
        "  • Weekly 1:1 coaching with Line Manager\n"
        "  • Access to Learning Management System (LMS)\n"
        "  • Mentoring paired with senior consultant\n\n"
        "Acknowledgement required. Copy to HR file."
    ),
    (
        "CONTINENTAL FREIGHT LTD — BILL OF LADING\n"
        "BL Number: CFL-2024-00938  |  Date of Issue: 04 Nov 2024\n"
        "Place of Issue: Los Angeles, CA\n\n"
        "Shipper:    Pacific Industrial Corp, Los Angeles, CA 90071\n"
        "Consignee:  Apex Distribution GmbH, Hamburg 20095, DE\n"
        "Notify Party: Same as Consignee\n\n"
        "Cargo Description:\n"
        "  14 pallets — Machine Components (HS 8479.89)\n"
        "  Gross Weight: 4,820 kg  |  Volume: 18.6 CBM\n"
        "  Marks & Numbers: PIC/APX/NOV-24/001-014\n\n"
        "Freight: Prepaid  |  Port of Loading: Los Angeles\n"
        "Port of Discharge: Hamburg  |  Vessel: MSC Astrid\n"
        "Voyage: 2024W44N  |  ETD: 07 Nov 2024  |  ETA: 24 Nov 2024\n\n"
        "ORIGINAL — 1 of 3  —  Issued subject to Hague-Visby Rules"
    ),
    (
        "MIT RESEARCH GROUP — WORKING PAPER\n"
        "Title: Adaptive Inference in Sparse Neural Architectures\n"
        "Authors: [Redacted for blind review]  |  Draft v0.4\n\n"
        "Abstract:\n"
        "We propose a dynamic pruning strategy that reduces inference\n"
        "FLOPs by 43% while retaining 98.7% of baseline accuracy on\n"
        "ImageNet-1k. Our method achieves Pareto-optimal trade-offs on\n"
        "seven standard benchmarks and requires no task-specific fine-tuning.\n\n"
        "Key contributions:\n"
        "  1. Closed-form channel importance score (Section 3.2)\n"
        "  2. Gradient-free calibration procedure (Section 3.4)\n"
        "  3. Open-source implementation: github.com/[redacted]\n\n"
        "Experimental setup:\n"
        "All experiments were conducted on 8× NVIDIA A100 GPUs. Models were\n"
        "trained for 90 epochs using cosine annealing with warm restarts.\n\n"
        "Under review — do not cite without permission."
    ),
    (
        "HARBOR PROPERTIES LLC — PROPERTY VALUATION REPORT\n"
        "Property: 412 Lakeview Drive, Austin TX 78701\n"
        "Valuation Date: 01 November 2024  |  Appraiser: TX-45291\n\n"
        "Market Approach — Comparable Sales:\n"
        "  Comp 1: 408 Lakeview Dr  — $1,240,000  (Sep 2024, 2,180 sqft)\n"
        "  Comp 2: 419 Riverside Ct — $1,185,000  (Aug 2024, 2,050 sqft)\n"
        "  Comp 3: 396 Lakeview Dr  — $1,310,000  (Oct 2024, 2,290 sqft)\n\n"
        "Adjusted Market Value: $1,270,000 (±2.5%)\n\n"
        "Income Approach:\n"
        "  Estimated Annual NOI: $68,400\n"
        "  Capitalisation Rate: 5.35%\n"
        "  Indicated Value (Income): $1,278,000\n\n"
        "Reconciled Value: $1,272,000\n\n"
        "Certified Appraiser: ______________  License: TX-45291"
    ),
]

# ---------------------------------------------------------------------------
# Sample table data
# ---------------------------------------------------------------------------

_SCAN_TABLE_SAMPLES: list[dict[str, Any]] = [
    {
        "caption": "Quarterly Revenue Summary",
        "headers": ["Quarter", "Revenue ($)", "Expenses ($)", "Net Profit ($)", "Margin %"],
        "rows": [
            ["Q1 2024", "1,240,500", "890,200",  "350,300", "28.2"],
            ["Q2 2024", "1,380,700", "945,600",  "435,100", "31.5"],
            ["Q3 2024", "1,520,300", "1,010,800","509,500", "33.5"],
            ["Q4 2024", "1,680,000", "1,089,400","590,600", "35.2"],
            ["TOTAL",   "5,821,500", "3,936,000","1,885,500","32.4"],
        ],
    },
    {
        "caption": "Inventory Stock Report",
        "headers": ["SKU", "Product Name", "Qty On Hand", "Unit Cost ($)", "Total Value ($)"],
        "rows": [
            ["SKU-1001", "Widget Alpha",    "342",  "12.50",  "4,275.00"],
            ["SKU-1002", "Gadget Beta",     "198",  "45.00",  "8,910.00"],
            ["SKU-1003", "Part Gamma",      "1,024","3.80",   "3,891.20"],
            ["SKU-1004", "Module Delta",    "57",   "220.00", "12,540.00"],
            ["SKU-1005", "Component Eps.", "480",   "8.75",   "4,200.00"],
            ["SKU-1006", "Device Zeta",     "15",   "890.00", "13,350.00"],
        ],
    },
    {
        "caption": "Clinical Trial Endpoint Summary",
        "headers": ["Endpoint", "Treatment (n=418)", "Control (n=413)", "p-value", "HR [95% CI]"],
        "rows": [
            ["Primary: 28d mortality", "34 (8.1%)",  "52 (12.6%)", "0.0031", "0.76 [0.62–0.93]"],
            ["Secondary: ICU LOS",     "6.2 days",   "7.8 days",   "0.0140", "—"],
            ["Secondary: Re-admission","18 (4.3%)",  "25 (6.1%)",  "0.0890", "0.71 [0.48–1.05]"],
            ["SAE rate",               "18 (4.3%)",  "25 (6.1%)",  "0.0890", "—"],
            ["Discontinuation",        "9 (2.2%)",   "11 (2.7%)",  "0.5620", "—"],
        ],
    },
    {
        "caption": "HR Headcount by Department",
        "headers": ["Department", "Headcount", "Open Positions", "Avg Tenure (yrs)", "Turnover %"],
        "rows": [
            ["Engineering", "142", "8",  "3.4", "9.2"],
            ["Sales",        "87",  "12", "2.1", "18.7"],
            ["Operations",   "63",  "3",  "5.7", "6.4"],
            ["Marketing",    "41",  "5",  "2.9", "12.3"],
            ["Finance",      "29",  "1",  "6.8", "4.1"],
            ["HR",           "18",  "2",  "4.2", "7.5"],
            ["TOTAL / AVG",  "380", "31", "4.2", "10.5"],
        ],
    },
    {
        "caption": "Freight Shipment Manifest",
        "headers": ["BL No.", "Origin", "Destination", "Weight (kg)", "Status"],
        "rows": [
            ["CFL-938", "Los Angeles", "Hamburg",    "4,820", "In Transit"],
            ["CFL-939", "Shanghai",    "Rotterdam",  "8,340", "Customs Hold"],
            ["CFL-940", "New York",    "Le Havre",   "2,150", "Delivered"],
            ["CFL-941", "Santos",      "Antwerp",    "6,600", "Loading"],
            ["CFL-942", "Cape Town",   "Felixstowe", "3,290", "In Transit"],
        ],
    },
    {
        "caption": "Performance Benchmark Results",
        "headers": ["Run ID", "Algorithm", "Accuracy", "F1 Score", "Latency (ms)"],
        "rows": [
            ["run_001", "Baseline", "0.834", "0.821", "42"],
            ["run_002", "Model A",  "0.901", "0.897", "68"],
            ["run_003", "Model B",  "0.923", "0.918", "124"],
            ["run_004", "Model C",  "0.911", "0.908", "95"],
            ["run_005", "Ensemble", "0.937", "0.934", "210"],
        ],
    },
]

# ---------------------------------------------------------------------------
# Handwriting content pools
# ---------------------------------------------------------------------------

_HW_ANNOTATION_POOL: list[str] = [
    "APPROVED",
    "REVIEWED ✓",
    "HOLD — escalate",
    "See attached note",
    "Verified by: __________",
    "Action required by COB",
    "Confirm with manager",
    "→ Refer to Appendix B",
    "!! Discrepancy — recheck",
    "Cross-ref page 12 para. 3",
    "Flag for audit review",
    "Awaiting sign-off",
    "Non-standard clause — legal review",
    "OK to proceed after corrections",
    "Date reviewed: __________",
    "Note: threshold exceeded — escalate",
    "Confirmed via email 11/5/24",
    "Revised per steering committee feedback",
    "Pending final approval from Director",
    "→ See variance analysis in Appendix",
    "Initials: ____  Date: __________",
    "Refer to corresponding footnote",
    "Do not circulate without sign-off",
    "Budget code confirmed: ____-____",
    "Compliance check passed ✓",
]

_HW_SIGNATURE_LINES: list[str] = [
    "Signed: ___________________________",
    "Approved by: _______________________",
    "Reviewed by: _______________________",
    "Authorised signatory: ______________",
    "Date: ____ / ____ / ________",
    "Witness: __________________________",
    "Print Name: _______________________",
    "Title / Position: __________________",
]

_HW_FORM_FILL_POOL: list[str] = [
    "Full Name: John A. Martinez",
    "Employee ID: EMP-00847",
    "Date: 15 Nov 2024",
    "Department: Operations",
    "Amount: $4,280.50",
    "Policy No: INS-2024-00391",
    "Reference: QC-2847",
    "Location: Building 3, Floor 2",
    "Approval Code: AUTH-2024-0041",
    "Notes: See attached documentation",
]

# ---------------------------------------------------------------------------
# Stamp content pools
# ---------------------------------------------------------------------------

_STAMP_POOL: list[str] = [
    "APPROVED",
    "REVIEWED",
    "CERTIFIED",
    "EXECUTED",
    "QC PASS",
    "HOLD",
    "REJECTED",
    "CONFIDENTIAL",
    "DRAFT",
    "FINAL",
    "FOR INTERNAL USE ONLY",
    "PRIVILEGED & CONFIDENTIAL",
    "RECEIVED",
    "PROCESSED",
    "VERIFIED",
]

# Ink colour per stamp category (R, G, B)
_STAMP_INK: dict[str, tuple[int, int, int]] = {
    "APPROVED":               (160, 20,  20),
    "REVIEWED":               (160, 20,  20),
    "CERTIFIED":              (20,  60,  160),
    "EXECUTED":               (20,  60,  160),
    "QC PASS":                (20,  120, 40),
    "HOLD":                   (180, 100, 0),
    "REJECTED":               (160, 20,  20),
    "CONFIDENTIAL":           (80,  20,  140),
    "DRAFT":                  (80,  80,  80),
    "FINAL":                  (20,  60,  160),
    "FOR INTERNAL USE ONLY":  (80,  20,  140),
    "PRIVILEGED & CONFIDENTIAL": (80, 20, 140),
    "RECEIVED":               (20,  60,  160),
    "PROCESSED":              (20,  120, 40),
    "VERIFIED":               (20,  60,  160),
}

_INSPECTOR_NAMES: list[str] = [
    "J. Smith", "A. Jones", "M. Lee",
    "C. Patel", "R. Kumar", "S. Chen",
    "D. Okonkwo", "F. Müller",
]

_VERIFICATION_ITEMS: list[str] = [
    "Documentation complete",
    "Signatures verified",
    "Reference numbers validated",
    "Budget code confirmed",
    "Compliance check passed",
    "Legal review completed",
    "Finance sign-off received",
    "Audit trail confirmed",
    "Final approval pending",
    "Data integrity verified",
]

_WATERMARK_TEXTS: list[str] = [
    "DRAFT", "CONFIDENTIAL", "COPY", "VOID", "SUPERSEDED",
    "NOT FOR DISTRIBUTION", "PRELIMINARY",
]

# ---------------------------------------------------------------------------
# Low-level rendering helpers
# ---------------------------------------------------------------------------

def _get_font(size: int = 17, bold: bool = False) -> Any:
    """Return a PIL TrueType font at *size* px, falling back gracefully."""
    from PIL import ImageFont

    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/ArialBd.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _bg_rgb(quality: str) -> tuple[int, int, int]:
    """Return paper background colour for a given scan quality tier."""
    params = _SCAN_QUALITY_PARAMS.get(quality, _SCAN_QUALITY_PARAMS["medium"])
    return _BG_TINT.get(params["age"], (252, 252, 250))


def _paper_background(
    width: int,
    height: int,
    bg_color: tuple[int, int, int],
    rng: np.random.Generator,
) -> Any:
    """Create a realistic paper-textured PIL image using multi-scale fiber noise.

    Three octaves of Gaussian-smoothed random noise simulate:
    - Coarse: paper grain / fiber macro-structure
    - Medium: surface texture (calendering / coating)
    - Fine:   micro-roughness visible under scanner illumination
    """
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import zoom as _zoom

    arr = np.full((height, width, 3), bg_color, dtype=np.float32)

    for (scale_div, sigma, weight) in [(8, 1.8, 0.50), (4, 0.9, 0.30), (1, 0.4, 0.20)]:
        sh, sw = max(1, height // scale_div), max(1, width // scale_div)
        layer = rng.standard_normal((sh, sw)).astype(np.float32)
        if sigma > 0:
            layer = gaussian_filter(layer, sigma=sigma)
        if scale_div > 1:
            layer = _zoom(layer, (height / sh, width / sw), order=1)[:height, :width]
        layer /= (layer.std() + 1e-6)
        arr += layer[:, :, np.newaxis] * (2.5 * weight)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Word-wrap a single string to at most *max_chars* per line."""
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _add_watermark(
    img: Any,
    text: str,
    rng: np.random.Generator,
) -> Any:
    """Overlay a faint diagonal watermark onto *img*."""
    from PIL import Image, ImageDraw

    w, h = img.size
    font_size = max(60, w // 8)
    font = _get_font(font_size, bold=True)

    # Render text to RGBA tile
    try:
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw = font_size * len(text)
        th = font_size

    tile = Image.new("RGBA", (tw + 20, th + 20), (0, 0, 0, 0))
    opacity = int(rng.integers(18, 40))
    ImageDraw.Draw(tile).text((10, 10), text, font=font, fill=(160, 20, 20, opacity))

    # Rotate and centre
    angle = int(rng.integers(25, 45))
    tile_rot = tile.rotate(angle, expand=True, resample=Image.BICUBIC)
    rx, ry = tile_rot.size
    paste_x = (w - rx) // 2
    paste_y = (h - ry) // 2

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay.paste(tile_rot, (paste_x, paste_y), tile_rot)
    base = img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def _find_perspective_coeffs(
    src_pts: list[tuple[float, float]],
    dst_pts: list[tuple[float, float]],
) -> list[float]:
    """Solve for PIL PERSPECTIVE transform coefficients.

    PIL maps output pixel (x,y) → input pixel (X,Y) via:
      X = (a*x + b*y + c) / (g*x + h*y + 1)
      Y = (d*x + e*y + f) / (g*x + h*y + 1)

    Args:
        src_pts: 4 input (source) corners the output corners are sampled from.
        dst_pts: 4 output rectangle corners  (usually (0,0),(w,0),(w,h),(0,h)).
    """
    matrix: list[list[float]] = []
    b_vec: list[float] = []
    for (X, Y), (x, y) in zip(src_pts, dst_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        b_vec.append(X)
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        b_vec.append(Y)
    A = np.array(matrix, dtype=np.float64)
    b = np.array(b_vec, dtype=np.float64)
    return list(np.linalg.solve(A, b))


# ---------------------------------------------------------------------------
# Full-page renderer  (A4 @ 150 DPI)
# ---------------------------------------------------------------------------

def _render_full_page(
    text_block: str,
    rng: np.random.Generator,
    quality: str = "medium",
) -> Any:
    """Render a complete A4 page from *text_block* at 150 DPI.

    Produces a ``_PAGE_W × _PAGE_H`` PIL image with:
    - Paper texture background
    - Letterhead (first line) + double horizontal rule
    - Subtitle / reference line
    - Body paragraphs with heading detection, bullet indent, word-wrap
    - Optional diagonal watermark (20 % probability)
    - Footer with page number and thin rule
    """
    from PIL import Image, ImageDraw

    bg = _bg_rgb(quality)
    img = _paper_background(_PAGE_W, _PAGE_H, bg, rng)
    draw = ImageDraw.Draw(img)

    font_body    = _get_font(_FS_BODY)
    font_bold    = _get_font(_FS_BODY, bold=True)
    font_sub     = _get_font(_FS_SUB)
    font_section = _get_font(_FS_SECTION, bold=True)
    font_title   = _get_font(_FS_TITLE,   bold=True)
    font_small   = _get_font(_FS_SMALL)

    max_chars_body    = _CW // 10   # ~94 chars @ 10 px/char (font size 17)
    max_chars_section = _CW // 14

    lines = text_block.split("\n")
    y = _MT

    # ---- Letterhead block (first line) ----
    if lines:
        header_text = lines[0].strip()
        draw.text((_ML, y), header_text, font=font_title, fill=(20, 20, 20))
        y += _LH_TITLE + 4
        # Double rule
        draw.line([(_ML, y), (_MR, y)], fill=(60, 60, 60), width=3)
        draw.line([(_ML, y + 5), (_MR, y + 5)], fill=(140, 140, 140), width=1)
        y += 20

    # ---- Remaining lines ----
    for idx, raw_line in enumerate(lines[1:], start=1):
        if y + _LH_BODY > _MB - 60:
            break

        stripped = raw_line.strip()

        # Blank line → half-line gap
        if not stripped:
            y += _LH_BODY // 2
            continue

        # Second non-empty group = subtitle / reference line
        if idx <= 2:
            for wl in _wrap_text(stripped, max_chars_body):
                draw.text((_ML, y), wl, font=font_sub, fill=(70, 70, 70))
                y += _LH_BODY
            y += 4
            continue

        # Section heading detection: starts with "Section", all-caps short, FR-/NFR-
        is_section = (
            stripped.startswith(("Section ", "SECTION ", "FR-", "NFR-"))
            or (stripped.isupper() and 4 < len(stripped) < 50)
        )

        # Bullet / numbered indent
        indent = 0
        if stripped[:1] in ("•", "*", "-", "→", "!") or (
            len(stripped) >= 3 and stripped[0].isdigit() and stripped[1] in (".", ")")
        ):
            indent = 32

        if is_section:
            for wl in _wrap_text(stripped, max_chars_section):
                draw.text((_ML, y), wl, font=font_section, fill=(20, 20, 20))
                y += _LH_SECTION
            continue

        font_use = font_bold if stripped.endswith(":") else font_body
        fill = (30, 30, 30) if font_use is font_bold else (45, 45, 45)
        x_base = _ML + indent
        for wl in _wrap_text(stripped, max_chars_body - (indent // 10)):
            draw.text((x_base, y), wl, font=font_use, fill=fill)
            y += _LH_BODY

    # ---- Footer ----
    fy = _MB + 20
    draw.line([(_ML, fy), (_MR, fy)], fill=(160, 160, 160), width=1)
    draw.text((_MR - 100, fy + 10), "Page 1", font=font_small, fill=(130, 130, 130))

    # ---- Optional watermark (20 % chance) ----
    if rng.random() < 0.20:
        wm_text = _WATERMARK_TEXTS[int(rng.integers(0, len(_WATERMARK_TEXTS)))]
        img = _add_watermark(img, wm_text, rng)

    return img


# ---------------------------------------------------------------------------
# Table-page renderer  (A4 @ 150 DPI, drawn grid)
# ---------------------------------------------------------------------------

def _render_table_page(
    sample: dict[str, Any],
    rng: np.random.Generator,
    quality: str = "medium",
) -> Any:
    """Render a table with proper drawn cell borders on a full A4 page."""
    from PIL import Image, ImageDraw

    bg = _bg_rgb(quality)
    img = _paper_background(_PAGE_W, _PAGE_H, bg, rng)
    draw = ImageDraw.Draw(img)

    headers: list[str] = sample["headers"]
    rows: list[list[str]] = sample["rows"]
    caption: str = sample["caption"]
    n_cols = len(headers)

    font_body  = _get_font(_FS_BODY)
    font_bold  = _get_font(_FS_BODY, bold=True)
    font_title = _get_font(_FS_TITLE, bold=True)
    font_small = _get_font(_FS_SMALL)

    # ---- Title ----
    draw.text((_ML, _MT), caption.upper(), font=font_title, fill=(20, 20, 20))
    draw.line([(_ML, _MT + _LH_TITLE), (_MR, _MT + _LH_TITLE)],
              fill=(60, 60, 60), width=3)
    draw.line([(_ML, _MT + _LH_TITLE + 5), (_MR, _MT + _LH_TITLE + 5)],
              fill=(140, 140, 140), width=1)

    # ---- Column widths — proportional to max cell length ----
    all_rows_for_width = [headers] + rows
    col_max = [
        max(len(str(r[c])) for r in all_rows_for_width if c < len(r))
        for c in range(n_cols)
    ]
    total = sum(col_max)
    col_widths = [int(_CW * m / total) for m in col_max]
    col_widths[-1] = _CW - sum(col_widths[:-1])   # fill remainder

    ROW_H    = 42
    HDR_H    = 52
    PAD_X    = 10
    PAD_Y    = 10

    tbl_x = _ML
    tbl_y = _MT + _LH_TITLE + 30

    max_visible = min(len(rows), (_MB - tbl_y - HDR_H) // ROW_H)
    tbl_h = HDR_H + max_visible * ROW_H

    # ---- Header row background ----
    draw.rectangle(
        [tbl_x, tbl_y, tbl_x + _CW, tbl_y + HDR_H],
        fill=(210, 220, 235),
    )

    # ---- Data row alternating shading ----
    for i in range(max_visible):
        if i % 2 == 1:
            ry = tbl_y + HDR_H + i * ROW_H
            draw.rectangle(
                [tbl_x, ry, tbl_x + _CW, ry + ROW_H],
                fill=(245, 246, 250),
            )

    # ---- Cell text ----
    # Header
    x = tbl_x
    for hdr, cw in zip(headers, col_widths):
        draw.text((x + PAD_X, tbl_y + PAD_Y), str(hdr), font=font_bold, fill=(15, 15, 15))
        x += cw

    # Data
    for i, row in enumerate(rows[:max_visible]):
        ry = tbl_y + HDR_H + i * ROW_H
        x = tbl_x
        for j, (cell, cw) in enumerate(zip(row, col_widths)):
            cell_str = str(cell).strip()
            # Right-align numeric cells in non-first columns
            is_num = (j > 0 and any(c.isdigit() for c in cell_str.replace(",", "").replace(".", "").replace("–", "").replace("-", "").replace("%", "")))
            if is_num:
                try:
                    bbox = font_body.getbbox(cell_str)
                    tw = bbox[2] - bbox[0]
                except AttributeError:
                    tw = len(cell_str) * 9
                tx = x + cw - tw - PAD_X
            else:
                tx = x + PAD_X
            draw.text((tx, ry + PAD_Y), cell_str, font=font_body, fill=(35, 35, 35))
            x += cw

    # ---- Grid lines ----
    # Outer border
    draw.rectangle([tbl_x, tbl_y, tbl_x + _CW, tbl_y + tbl_h], outline=(50, 50, 50), width=2)
    # Header bottom separator
    draw.line([(tbl_x, tbl_y + HDR_H), (tbl_x + _CW, tbl_y + HDR_H)],
              fill=(50, 50, 50), width=2)
    # Data row separators
    for i in range(1, max_visible):
        ry = tbl_y + HDR_H + i * ROW_H
        draw.line([(tbl_x, ry), (tbl_x + _CW, ry)], fill=(140, 140, 140), width=1)
    # Vertical column separators
    x = tbl_x
    for cw in col_widths[:-1]:
        x += cw
        draw.line([(x, tbl_y), (x, tbl_y + tbl_h)], fill=(120, 120, 120), width=1)

    # ---- Footer ----
    fy = _MB + 20
    draw.line([(_ML, fy), (_MR, fy)], fill=(160, 160, 160), width=1)
    draw.text((_MR - 100, fy + 10), "Page 1", font=font_small, fill=(130, 130, 130))

    return img


# ---------------------------------------------------------------------------
# Realistic scan-artifact pipeline  (PIL + scipy + skimage)
# ---------------------------------------------------------------------------

def _scan_print_dot_gain(
    arr: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate laser-printer dot spread / optical diffusion on paper fibers.

    Applies a mild isotropic Gaussian blur — stronger for low-quality scans
    where the underlying printout was also lower quality.
    """
    from scipy.ndimage import gaussian_filter
    sigma = float(rng.uniform(0.35, 0.70 + noise_level * 2.8))
    return gaussian_filter(arr, sigma=[sigma, sigma, 0])


def _scan_toner_clump(
    arr: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate laser toner density variation: local patches brighter/darker.

    Uses a coarse random field multiplied against the image, so bright areas
    (paper) show more variation and dark (ink) areas are relatively stable.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import zoom as _zoom

    h, w = arr.shape[:2]
    # Coarse toner field at 1/16 resolution
    sh, sw = max(1, h // 16), max(1, w // 16)
    field = rng.standard_normal((sh, sw)).astype(np.float32)
    field = gaussian_filter(field, sigma=1.5)
    field = _zoom(field, (h / sh, w / sw), order=1)[:h, :w]
    field /= (field.std() + 1e-6)
    # Strength: ±1–2% for medium quality, ±3–5% for poor
    strength = noise_level * 0.18
    field = 1.0 + field * strength
    return np.clip(arr * field[:, :, np.newaxis], 0.0, 1.0)


def _scan_paper_texture(
    arr: np.ndarray,
    age: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Overlay multi-scale paper fiber texture onto the image array [0..1].

    The texture acts as a mild multiplicative perturbation, with strength
    proportional to paper age — older paper shows more fiber structure.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import zoom as _zoom

    h, w = arr.shape[:2]
    strength = {"modern": 0.016, "aged_10y": 0.030, "aged_30y": 0.055}.get(age, 0.020)

    texture = np.zeros((h, w), dtype=np.float32)
    for (scale_div, sigma, weight) in [(8, 1.8, 0.50), (4, 0.9, 0.30), (1, 0.4, 0.20)]:
        sh, sw = max(1, h // scale_div), max(1, w // scale_div)
        layer = rng.standard_normal((sh, sw)).astype(np.float32)
        if sigma > 0:
            layer = gaussian_filter(layer, sigma=sigma)
        if scale_div > 1:
            layer = _zoom(layer, (h / sh, w / sw), order=1)[:h, :w]
        layer /= (layer.std() + 1e-6)
        texture += layer * weight

    texture /= (texture.std() + 1e-6)
    texture = 1.0 + texture * strength
    return np.clip(arr * texture[:, :, np.newaxis], 0.0, 1.0)


def _scan_color_cast(
    arr: np.ndarray,
    age: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply paper yellowing + scanner warm-lamp color cast.

    Increases red/green channels and reduces blue to simulate:
    - Optical brightener degradation in aged paper
    - Fluorescent scanner lamp warm spectral output
    """
    intensity = float(rng.uniform(0.65, 1.0))
    warmth = {"modern": 0.007, "aged_10y": 0.020, "aged_30y": 0.048}.get(age, 0.010)
    r_add = intensity * warmth
    b_sub = intensity * warmth * 0.55
    out = arr.copy()
    # Multiplicative on background (paper), additive would over-boost ink
    paper_mask = arr.mean(axis=2, keepdims=True)   # ~1.0 = paper, ~0.0 = ink
    out[:, :, 0] = np.clip(arr[:, :, 0] + r_add * paper_mask[:, :, 0], 0.0, 1.0)
    out[:, :, 2] = np.clip(arr[:, :, 2] - b_sub * paper_mask[:, :, 0], 0.0, 1.0)
    return out


def _scan_tonal_curve(
    arr: np.ndarray,
    quality: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a mild gamma/contrast curve to simulate scanner ADC non-linearity.

    Real scanners have slight gamma errors and contrast compression that
    make scans look slightly different in midtones from the original print.
    """
    gamma = float(rng.uniform(*{
        "pristine": (0.97, 1.03),
        "high":     (0.94, 1.06),
        "medium":   (0.90, 1.10),
        "aged":     (0.86, 1.12),
        "poor":     (0.82, 1.18),
    }.get(quality, (0.90, 1.10))))
    # Avoid division by zero; power of 1.0 = no change
    return np.clip(np.power(np.clip(arr, 1e-6, 1.0), gamma), 0.0, 1.0)


def _scan_illumination(
    arr: np.ndarray,
    quality: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate scanner lamp non-uniformity: one-corner brightness falloff.

    Real flatbed scanners have fluorescent tubes running along one axis;
    the ends of the tube produce slightly less light, creating a gradient.
    """
    h, w = arr.shape[:2]
    strengths = {"pristine": 0.025, "high": 0.055, "medium": 0.10,
                 "aged": 0.16, "poor": 0.26}
    strength = strengths.get(quality, 0.10) * float(rng.uniform(0.7, 1.3))

    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :]

    corner = int(rng.integers(0, 4))
    if   corner == 0: raw = (1 - xx) * (1 - yy)   # brightest top-left
    elif corner == 1: raw = xx * (1 - yy)           # brightest top-right
    elif corner == 2: raw = (1 - xx) * yy           # brightest bottom-left
    else:             raw = xx * yy                  # brightest bottom-right

    # Normalise to [0,1] then build falloff mask
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    illum = 1.0 - strength * (1.0 - raw)            # brighter where raw≈1
    return np.clip(arr * illum[:, :, np.newaxis], 0.0, 1.0)


def _scan_sensor_noise(
    arr: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add scanner CCD sensor noise: Gaussian random + horizontal row banding.

    Row banding (slight row-to-row gain variation) is characteristic of
    CCD line sensors and is missing from simple Gaussian-only models.
    """
    h, _w = arr.shape[:2]

    # Random noise (Gaussian)
    sigma = noise_level * 0.40
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)

    # Row-level brightness variation (scanner gain drift per scan line)
    row_var = noise_level * 0.055
    row_offset = rng.normal(0, row_var, (h, 1, 1)).astype(np.float32)

    return np.clip(arr + noise + row_offset, 0.0, 1.0)


def _scan_jpeg_roundtrip(pil_img: Any, quality: int) -> Any:
    """Force JPEG block artifacts by encoding and decoding through PIL.

    At lower quality settings this produces the characteristic 8×8 pixel
    blocking pattern that all JPEG-compressed scans exhibit.
    """
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=1)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf).copy()


def _scan_chromatic_aberration(
    arr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Lateral chromatic aberration: R channel shifted right, B channel left.

    Real scanner optics have residual chromatic aberration (CA) where the
    red, green and blue focus planes are slightly offset.  The typical
    manifestation is coloured fringing on high-contrast edges.
    """
    shift = int(rng.integers(1, 3))   # 1 or 2 px
    out = arr.copy()
    # R → right
    out[:, shift:,  0] = arr[:, :-shift,  0]
    out[:, :shift,  0] = arr[:, 0:1,      0]
    # B → left
    out[:, :-shift, 2] = arr[:, shift:,   2]
    out[:, -shift:, 2] = arr[:, -1:,      2]
    return out


def _scan_skew_rotate(
    pil_img: Any,
    quality: str,
    rng: np.random.Generator,
) -> Any:
    """Slight rotation to simulate document misalignment on the scanner glass."""
    from PIL import Image
    max_angle = _SCAN_QUALITY_PARAMS[quality]["skew_angle"]
    if max_angle < 0.05:
        return pil_img
    angle = float(rng.uniform(-max_angle, max_angle))
    bg = _bg_rgb(quality)
    return pil_img.rotate(angle, expand=False, resample=Image.BICUBIC, fillcolor=bg)


def _scan_perspective(
    pil_img: Any,
    quality: str,
    rng: np.random.Generator,
) -> Any:
    """Slight perspective distortion (handheld or misaligned feed scan)."""
    from PIL import Image
    probs = {"pristine": 0.0, "high": 0.10, "medium": 0.28, "aged": 0.45, "poor": 0.65}
    if rng.random() >= probs.get(quality, 0.25):
        return pil_img

    w, h = pil_img.size
    bg = _bg_rgb(quality)
    max_off = {"pristine": 0, "high": 8, "medium": 18, "aged": 28, "poor": 44}.get(quality, 18)
    offsets = [int(rng.integers(0, max_off + 1)) for _ in range(8)]
    dx0, dy0, dx1, dy1, dx2, dy2, dx3, dy3 = offsets
    src = [(dx0, dy0), (w - dx1, dy1), (w - dx2, h - dy2), (dx3, h - dy3)]
    dst = [(0, 0), (w, 0), (w, h), (0, h)]
    try:
        coeffs = _find_perspective_coeffs(src, dst)
        return pil_img.transform(
            (w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=bg
        )
    except np.linalg.LinAlgError:
        return pil_img


def _scan_edge_shadow(
    arr: np.ndarray,
    quality: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Darken one edge to simulate scanner lid not fully closed."""
    probs = {"pristine": 0.0, "high": 0.15, "medium": 0.40, "aged": 0.65, "poor": 0.88}
    if rng.random() >= probs.get(quality, 0.40):
        return arr

    h, w = arr.shape[:2]
    edge = int(rng.integers(0, 4))
    ew = int(rng.integers(30, 120))
    strength = float(rng.uniform(0.08, 0.32))

    fade = np.linspace(1.0 - strength, 1.0, ew, dtype=np.float32)
    out = arr.copy()
    if   edge == 0: out[:, :ew, :]   *= fade[np.newaxis, :, np.newaxis]
    elif edge == 1: out[:, -ew:, :]  *= fade[::-1][np.newaxis, :, np.newaxis]
    elif edge == 2: out[:ew, :, :]   *= fade[:, np.newaxis, np.newaxis]
    else:           out[-ew:, :, :]  *= fade[::-1, np.newaxis, np.newaxis]
    return np.clip(out, 0.0, 1.0)


def _scan_fold_creases(
    arr: np.ndarray,
    quality: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add fold/crease lines as thin dark diagonals with soft edges."""
    from scipy.ndimage import gaussian_filter

    probs = {"pristine": 0.0, "high": 0.08, "medium": 0.28, "aged": 0.58, "poor": 0.82}
    if rng.random() >= probs.get(quality, 0.28):
        return arr

    h, w = arr.shape[:2]
    n_creases = int(rng.integers(1, 3))
    out = arr.copy()

    for _ in range(n_creases):
        cy = int(rng.integers(h // 6, 5 * h // 6))
        angle_deg = float(rng.uniform(-10.0, 10.0))
        half_w = max(1, int(rng.integers(1, 3)))
        strength = float(rng.uniform(0.06, 0.20))
        tan_a = math.tan(math.radians(angle_deg))

        mask = np.zeros((h, w), dtype=np.float32)
        xs = np.arange(w)
        for dy in range(-half_w, half_w + 1):
            rows = (cy + (xs * tan_a).astype(int) + dy)
            valid = (rows >= 0) & (rows < h)
            fade = 1.0 - abs(dy) / (half_w + 1)
            mask[rows[valid], xs[valid]] = strength * fade

        mask = gaussian_filter(mask, sigma=0.8)
        out -= mask[:, :, np.newaxis]

    return np.clip(out, 0.0, 1.0)


def _scan_bleed_through(
    arr: np.ndarray,
    quality: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Faint ghost text from the reverse side of the page (aged/poor only)."""
    from scipy.ndimage import gaussian_filter

    probs = {"pristine": 0.0, "high": 0.0, "medium": 0.04, "aged": 0.22, "poor": 0.50}
    if rng.random() >= probs.get(quality, 0.05):
        return arr

    strength = float(rng.uniform(0.03, 0.08))
    ghost = arr[:, ::-1, :].copy()                          # mirror horizontally
    ghost = gaussian_filter(ghost, sigma=[1.8, 1.8, 0])    # blur the ghost text
    return np.clip(arr * (1.0 - strength) + ghost * strength, 0.0, 1.0)


def _scan_coffee_stain(
    arr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add an elliptical coffee or water stain ring (aged/poor documents)."""
    h, w = arr.shape[:2]
    cx  = int(rng.integers(w // 5, 4 * w // 5))
    cy  = int(rng.integers(h // 5, 4 * h // 5))
    rx  = int(rng.integers(40, 120))
    ry  = int(rng.integers(30, 90))
    rot = float(rng.uniform(0, math.pi))

    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = (xx - cx).astype(np.float32), (yy - cy).astype(np.float32)
    # Rotated ellipse distance
    xr =  dx * math.cos(rot) + dy * math.sin(rot)
    yr = -dx * math.sin(rot) + dy * math.cos(rot)
    dist = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)

    # Ring: Gaussian centred on dist=1.0, width≈0.12
    ring = np.exp(-((dist - 1.0) / 0.12) ** 2).astype(np.float32)

    strength = float(rng.uniform(0.04, 0.13))
    # Brownish tint: darken more in G and B → warm brown
    out = arr.copy()
    out[:, :, 0] -= ring * strength * 0.30
    out[:, :, 1] -= ring * strength * 0.55
    out[:, :, 2] -= ring * strength * 0.75
    return np.clip(out, 0.0, 1.0)


def _apply_advanced_scan(
    img: Any,
    quality: str,
    rng: np.random.Generator,
) -> Any:
    """Full realistic scan simulation pipeline.

    Replaces the old ``add_scan_artifacts`` call with a layered physical model:

    1.  Print dot-gain        — laser-printer focal blur (0.4–1.5 px)
    2.  Toner clumping        — local density variation in the printed output
    3.  Paper fiber texture   — multi-scale multiplicative noise
    4.  Color cast            — paper yellowing + warm scanner lamp
    5.  Tonal curve           — scanner ADC gamma non-linearity
    6.  Scanner illumination  — corner/edge brightness falloff
    7.  Sensor noise          — Gaussian + horizontal row banding
    8.  Document skew         — slight rotation (misalignment)
    9.  JPEG compression      — actual PIL encode/decode cycle (8×8 blocking)
    10. Chromatic aberration  — R/B channel lateral shift (1–2 px)
    11. Perspective distortion — quadrilateral warp for handheld/feed scans
    12. Edge shadow           — one darkened edge (scanner lid not closed)
    13. Fold / crease lines   — thin dark diagonal lines
    14. Bleed-through         — faint ghost text from back of page (aged/poor)
    15. Coffee stain          — brown elliptical ring (aged/poor, 25 % chance)
    """
    from PIL import Image

    params = _SCAN_QUALITY_PARAMS.get(quality, _SCAN_QUALITY_PARAMS["medium"])
    noise_level  = params["noise_level"]
    jpeg_quality = params["jpeg_quality"]
    age          = params["age"]

    # Work in float32 [0..1] for all numpy operations
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    # Steps 1–7: physical simulation of print + paper + scanner optics
    arr = _scan_print_dot_gain(arr, noise_level, rng)
    arr = _scan_toner_clump(arr, noise_level, rng)
    arr = _scan_paper_texture(arr, age, rng)
    arr = _scan_color_cast(arr, age, rng)
    arr = _scan_tonal_curve(arr, quality, rng)
    arr = _scan_illumination(arr, quality, rng)
    arr = _scan_sensor_noise(arr, noise_level, rng)

    # Steps 8–10: require PIL intermediate (rotation, JPEG, CA)
    pil_step = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))
    pil_step = _scan_skew_rotate(pil_step, quality, rng)
    pil_step = _scan_jpeg_roundtrip(pil_step, jpeg_quality)

    arr = np.array(pil_step, dtype=np.float32) / 255.0
    if quality in ("medium", "aged", "poor"):
        arr = _scan_chromatic_aberration(arr, rng)

    # Step 11: perspective distortion (PIL-based)
    pil_step = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))
    pil_step = _scan_perspective(pil_step, quality, rng)
    arr = np.array(pil_step, dtype=np.float32) / 255.0

    # Steps 12–15: paper and environmental damage
    arr = _scan_edge_shadow(arr, quality, rng)
    arr = _scan_fold_creases(arr, quality, rng)
    arr = _scan_bleed_through(arr, quality, rng)
    if quality in ("aged", "poor") and rng.random() < 0.25:
        arr = _scan_coffee_stain(arr, rng)

    return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Ruled-paper background (for handwriting variants)
# ---------------------------------------------------------------------------

def _render_ruled_paper(
    width: int,
    height: int,
    rng: np.random.Generator,
    background: tuple[int, int, int] = (255, 255, 248),
    ruled: bool = True,
    has_margin: bool = True,
    has_holes: bool = False,
) -> Any:
    """Create a ruled notebook-paper background image."""
    from PIL import Image, ImageDraw

    img = _paper_background(width, height, background, rng)
    draw = ImageDraw.Draw(img)

    if ruled:
        spacing = 28
        line_color = (170, 195, 220)
        for y in range(spacing, height, spacing):
            draw.line([(0, y), (width, y)], fill=line_color, width=1)

    if has_margin:
        mx = 58
        draw.line([(mx, 0), (mx, height)], fill=(210, 90, 90), width=2)

    if has_holes:
        hx, hr = 22, 11
        for ypct in (0.15, 0.50, 0.85):
            hy = int(height * ypct)
            draw.ellipse(
                [hx - hr, hy - hr, hx + hr, hy + hr],
                fill=(195, 195, 195), outline=(155, 155, 155), width=1,
            )

    return img


# ---------------------------------------------------------------------------
# Stamp renderer
# ---------------------------------------------------------------------------

def _render_stamp(
    stamp_text: str,
    ink_color: tuple[int, int, int],
    rng: np.random.Generator,
    angle: float = 0.0,
) -> Any:
    """Render a single rubber-stamp impression as an RGBA PIL image.

    Features:
    - Bold text centred in the stamp
    - Oval or rectangular border (chosen randomly)
    - Ink bleed: slight Gaussian blur + per-channel colour spread
    - Random opacity (80–95 %) to simulate partial ink transfer
    - Rotated to *angle* degrees
    """
    from PIL import Image, ImageDraw, ImageFilter

    # Font sized to fit stamp width (~300 px wide stamp)
    base_w, base_h = 340, 100
    # Scale font to fill ~80 % of stamp width
    font_size = max(20, min(50, base_w * 4 // (max(len(stamp_text), 4) + 2)))
    font = _get_font(font_size, bold=True)

    # Measure text
    try:
        bbox = font.getbbox(stamp_text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw = font_size * len(stamp_text) // 2
        th = font_size
    w = max(tw + 40, base_w)
    h = max(th + 30, base_h)

    stamp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stamp)

    # Ink colour with slight variation
    r0, g0, b0 = ink_color
    dr = int(rng.integers(-15, 15))
    dg = int(rng.integers(-10, 10))
    db = int(rng.integers(-10, 10))
    ink = (
        max(0, min(255, r0 + dr)),
        max(0, min(255, g0 + dg)),
        max(0, min(255, b0 + db)),
    )
    opacity = int(rng.integers(200, 242))

    # Border style
    border_style = rng.choice(["oval", "rect", "double_oval"])
    bw = 3
    pad = 4
    if border_style == "oval":
        draw.ellipse([pad, pad, w - pad - 1, h - pad - 1],
                     outline=(*ink, opacity), width=bw)
    elif border_style == "rect":
        draw.rectangle([pad, pad, w - pad - 1, h - pad - 1],
                        outline=(*ink, opacity), width=bw)
    else:
        draw.ellipse([pad, pad, w - pad - 1, h - pad - 1],
                     outline=(*ink, opacity), width=bw)
        draw.ellipse([pad + 4, pad + 4, w - pad - 5, h - pad - 5],
                     outline=(*ink, opacity // 2), width=1)

    # Centred text
    tx = (w - tw) // 2
    ty = (h - th) // 2
    draw.text((tx, ty), stamp_text, font=font, fill=(*ink, opacity))

    # Ink bleed — blur then slightly shift channels to simulate ink spread
    stamp = stamp.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.6, 1.4))))

    # Rotate
    if abs(angle) > 0.5:
        stamp = stamp.rotate(angle, expand=True, resample=Image.BICUBIC)

    return stamp


# ---------------------------------------------------------------------------
# QA helpers for scanned variants
# ---------------------------------------------------------------------------

def _scanned_text_qa(caption: str, quality: str, first_line: str) -> "list[dict]":
    """Deterministic QA anchored by the scanned document caption."""
    return [
        {
            "question": f"What is the first line of text in the scanned document titled '{caption}'?",
            "answer": first_line,
            "assertions": [{"text": f"The response must state '{first_line}'."}],
        },
        {
            "question": f"What scan quality level is indicated for the document titled '{caption}'?",
            "answer": quality,
            "assertions": [{"text": f"The response must state '{quality}'."}],
        },
    ]


def _scanned_table_qa(
    caption: str,
    quality: str,
    table_caption: str,
    headers: "list[str]",
) -> "list[dict]":
    """Deterministic QA anchored by the scanned table document caption."""
    header_str = ", ".join(headers) if headers else ""
    return [
        {
            "question": f"In the scanned table document titled '{caption}', what is the data table's caption?",
            "answer": table_caption,
            "assertions": [{"text": f"The response must state '{table_caption}'."}],
        },
        {
            "question": f"In the scanned table document titled '{caption}', what are the column headers?",
            "answer": header_str,
            "assertions": [{"text": f"The response must state the column headers including '{headers[0] if headers else ''}'."}],
        },
        {
            "question": f"What scan quality level is indicated for the document titled '{caption}'?",
            "answer": quality,
            "assertions": [{"text": f"The response must state '{quality}'."}],
        },
    ]


def _handwriting_qa(caption: str, hw_style: str, has_signature: bool) -> "list[dict]":
    """Deterministic QA anchored by the handwriting sample caption."""
    sig_answer = "Yes" if has_signature else "No"
    return [
        {
            "question": f"What handwriting style is used in the sample titled '{caption}'?",
            "answer": hw_style,
            "assertions": [{"text": f"The response must state '{hw_style}'."}],
        },
        {
            "question": f"Does the handwriting sample titled '{caption}' include a signature?",
            "answer": sig_answer,
            "assertions": [{"text": f"The response must state '{sig_answer}'."}],
        },
    ]


def _stamp_qa(
    caption: str,
    stamp_marks: "list[str]",
    inspector: str,
    reference: str,
) -> "list[dict]":
    """Deterministic QA anchored by the stamp document caption."""
    marks_str = ", ".join(stamp_marks) if stamp_marks else ""
    return [
        {
            "question": f"What approval stamp mark(s) are shown on the document titled '{caption}'?",
            "answer": marks_str,
            "assertions": [{"text": f"The response must state '{marks_str}'."}],
        },
        {
            "question": f"Who is listed as the inspector on the document titled '{caption}'?",
            "answer": inspector,
            "assertions": [{"text": f"The response must state '{inspector}'."}],
        },
        {
            "question": f"What is the reference number on the document titled '{caption}'?",
            "answer": reference,
            "assertions": [{"text": f"The response must state '{reference}'."}],
        },
    ]


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------

def _save_pil(pil_img: Any, prefix: str, assets_dir: Path) -> Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    out = assets_dir / f"{prefix}_{_uid()[:8]}.png"
    pil_img.save(str(out))
    return out


# ---------------------------------------------------------------------------
# Public variant generators
# ---------------------------------------------------------------------------

def generate_scanned_text_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Path | None = None,
) -> list[Variant]:
    """Generate *count* scanned machine-printed text variants.

    Each variant renders a professional text block as a full A4 page at
    150 DPI then applies advanced scan artefacts (noise, skew, contrast,
    uneven illumination, fold creases, edge shadow, perspective warp).

    """
    out_dir = (assets_dir or Path(".")).resolve()
    variants: list[Variant] = []

    for i in range(count):
        child_rng = fork_rng(rng, f"scan_text_{i}")

        # Quality: all tiers, weighted toward medium/high for printed text
        weights = [0.10, 0.25, 0.35, 0.20, 0.10]
        cumw = [sum(weights[:k+1]) for k in range(len(weights))]
        r = float(child_rng.random())
        sq_idx = next(j for j, c in enumerate(cumw) if r < c)
        quality = _SCAN_QUALITY_POOL[sq_idx]

        text_idx = int(child_rng.integers(0, len(_SCAN_TEXT_BLOCKS)))
        text_block = _SCAN_TEXT_BLOCKS[text_idx]
        lines = text_block.split("\n")
        word_count = sum(len(ln.split()) for ln in lines)

        pil_img = _render_full_page(text_block, child_rng, quality)
        scanned  = _apply_advanced_scan(pil_img, quality, child_rng)

        img_path = _save_pil(scanned, "scan_text", out_dir)
        caption  = f"Scanned document — {quality.title()} quality"

        variants.append(Variant(
            id=f"scan_text_{i:04d}_{_uid()[:8]}",
            kind="image",
            metadata={
                "subtype":      "scanned_text",
                "scan_quality": quality,
                "line_count":   len(lines),
                "word_count":   word_count,
                "caption":      caption,
            },
            render_payload={
                "image_path": img_path.resolve().as_posix(),
                "alt_text":   caption,
                "caption":    caption,
                "format":     "png",
            },
            ground_truth={
                "scan_quality": quality,
                "first_line":   lines[0].strip() if lines else "",
                "line_count":   len(lines),
                "word_count":   word_count,
                "full_text":    text_block,
                "caption":      caption,
                "qa_candidates": _scanned_text_qa(caption, quality, lines[0].strip() if lines else ""),
            },
        ))
    return variants


def generate_scanned_table_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Path | None = None,
) -> list[Variant]:
    """Generate *count* scanned tabular data variants.

    Each variant renders a table with proper drawn cell borders on a full A4
    page then applies heavier scan artefacts (medium/aged/poor quality) to
    simulate the challenging OCR case of columnar data on scanned paper.

    """
    out_dir = (assets_dir or Path(".")).resolve()
    variants: list[Variant] = []

    for i in range(count):
        child_rng = fork_rng(rng, f"scan_table_{i}")

        # Heavier quality range for tables
        sq_idx = int(child_rng.integers(1, 5))  # high → poor
        quality = _SCAN_QUALITY_POOL[sq_idx]

        sample_idx = int(child_rng.integers(0, len(_SCAN_TABLE_SAMPLES)))
        sample = _SCAN_TABLE_SAMPLES[sample_idx]
        tbl_caption = sample["caption"]
        headers = sample["headers"]
        rows = sample["rows"]

        pil_img = _render_table_page(sample, child_rng, quality)
        scanned  = _apply_advanced_scan(pil_img, quality, child_rng)

        img_path = _save_pil(scanned, "scan_table", out_dir)
        caption  = f"Scanned table — {tbl_caption} ({quality} quality)"

        variants.append(Variant(
            id=f"scan_table_{i:04d}_{_uid()[:8]}",
            kind="image",
            metadata={
                "subtype":       "scanned_table",
                "table_caption": tbl_caption,
                "scan_quality":  quality,
                "row_count":     len(rows),
                "col_count":     len(headers),
                "caption":       caption,
            },
            render_payload={
                "image_path": img_path.resolve().as_posix(),
                "alt_text":   caption,
                "caption":    caption,
                "format":     "png",
            },
            ground_truth={
                "scan_quality":   quality,
                "table_caption":  tbl_caption,
                "headers":        headers,
                "row_count":      len(rows),
                "first_data_row": rows[0] if rows else [],
                "caption":        caption,
                "qa_candidates":  _scanned_table_qa(caption, quality, tbl_caption, headers),
            },
        ))
    return variants


def generate_handwriting_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Path | None = None,
) -> list[Variant]:
    """Generate *count* handwriting annotation variants.

    Each variant renders handwritten content on a **ruled-paper background**
    (blue horizontal lines, red margin line, optional hole-punch circles)
    then applies a light scan pass to make it look like a photographed or
    scanned handwritten page.

    Optionally includes a signature block at the bottom.

    """
    from synthpdf.shared.handwriting import simulate_handwriting
    from PIL import Image

    out_dir = (assets_dir or Path(".")).resolve()
    variants: list[Variant] = []

    hw_layout_pool = ["annotation", "form_fill", "full_page"]
    sig_style_pool = ["cursive", "formal", "bold", "caveat_bold"]

    for i in range(count):
        child_rng = fork_rng(rng, f"hw_{i}")

        hw_style  = _HW_STYLE_POOL[int(child_rng.integers(0, len(_HW_STYLE_POOL)))]
        hw_color  = _HW_COLOR_POOL[int(child_rng.integers(0, len(_HW_COLOR_POOL)))]
        hw_bg     = _HW_BACKGROUND_POOL[int(child_rng.integers(0, len(_HW_BACKGROUND_POOL)))]
        layout    = hw_layout_pool[int(child_rng.integers(0, len(hw_layout_pool)))]
        hw_seed   = int(child_rng.integers(0, 2**31))

        # Pick annotation lines
        n_lines = int(child_rng.integers(3, 8))
        idxs = list(child_rng.choice(
            len(_HW_ANNOTATION_POOL),
            size=min(n_lines, len(_HW_ANNOTATION_POOL)),
            replace=False,
        ))
        hw_lines = [_HW_ANNOTATION_POOL[int(j)] for j in idxs]

        if layout == "form_fill":
            n_ff = int(child_rng.integers(2, 5))
            ff_idxs = list(child_rng.choice(
                len(_HW_FORM_FILL_POOL),
                size=min(n_ff, len(_HW_FORM_FILL_POOL)),
                replace=False,
            ))
            hw_lines = [_HW_FORM_FILL_POOL[int(j)] for j in ff_idxs] + hw_lines

        bg_map = {
            "white":  (255, 255, 255),
            "cream":  (255, 255, 240),
            "yellow": (255, 255, 200),
            "aged":   (248, 244, 228),
            "pink":   (255, 240, 240),
        }
        bg_rgb = bg_map.get(hw_bg, (255, 255, 255))

        dim_map = {
            "annotation": (620, 180),
            "form_fill":  (620, 90),
            "full_page":  (700, 480),
        }
        w, h = dim_map.get(layout, (620, 180))

        has_margin = bool(child_rng.random() < 0.65)
        has_holes  = bool(child_rng.random() < 0.30)

        # Render handwriting first — height may grow beyond the hint
        hw_img = simulate_handwriting(
            hw_lines, width=w, height=h,
            color=hw_color, background_color=bg_rgb,
            seed=hw_seed, style=hw_style,
        )
        actual_w, actual_h = hw_img.size

        # Build ruled-paper background at the actual rendered dimensions
        bg_img = _render_ruled_paper(
            actual_w, actual_h, child_rng,
            background=bg_rgb,
            ruled=(layout != "form_fill"),
            has_margin=has_margin,
            has_holes=has_holes,
        )

        # Composite: blend ruled background (30 %) with handwriting (70 %)
        composite_arr = (
            np.array(bg_img, dtype=np.float32) * 0.30
            + np.array(hw_img, dtype=np.float32) * 0.70
        )
        from PIL import Image as _PIL_Image
        annotation_img = _PIL_Image.fromarray(
            np.clip(composite_arr, 0, 255).astype(np.uint8)
        )

        # Optional signature block (70 % of the time)
        include_sig = bool(child_rng.random() < 0.70)
        sig_lines: list[str] = []
        if include_sig:
            n_sig = int(child_rng.integers(2, 5))
            sig_lines = list(child_rng.choice(
                _HW_SIGNATURE_LINES,
                size=min(n_sig, len(_HW_SIGNATURE_LINES)),
                replace=False,
            ))
            sig_style = sig_style_pool[int(child_rng.integers(0, len(sig_style_pool)))]
            sig_hw = simulate_handwriting(
                list(sig_lines), width=400, height=100,
                color=(10, 10, 150),
                background_color=(255, 255, 255),
                seed=hw_seed + 1, style=sig_style,
            )
            sig_aw, sig_ah = sig_hw.size
            sig_bg = _render_ruled_paper(
                sig_aw, sig_ah, child_rng,
                background=(255, 255, 255),
                ruled=False, has_margin=False,
            )
            sig_arr = (
                np.array(sig_bg, dtype=np.float32) * 0.30
                + np.array(sig_hw, dtype=np.float32) * 0.70
            )
            sig_img = _PIL_Image.fromarray(np.clip(sig_arr, 0, 255).astype(np.uint8))

            total_h = annotation_img.height + sig_img.height + 14
            final = _PIL_Image.new(
                "RGB",
                (max(annotation_img.width, sig_img.width), total_h),
                (255, 255, 255),
            )
            final.paste(annotation_img, (0, 0))
            final.paste(sig_img, (0, annotation_img.height + 14))
        else:
            final = annotation_img

        # Light scan pass over the handwriting
        hw_quality = "high" if layout == "full_page" else "medium"
        scanned = _apply_advanced_scan(final, hw_quality, child_rng)

        img_path = _save_pil(scanned, "hw", out_dir)
        all_lines = hw_lines + sig_lines
        caption = f"Handwritten annotations — {layout} style ({hw_style})"

        variants.append(Variant(
            id=f"hw_{i:04d}_{_uid()[:8]}",
            kind="image",
            metadata={
                "subtype":       "handwriting",
                "hw_style":      hw_style,
                "layout_style":  layout,
                "hw_background": hw_bg,
                "has_signature": include_sig,
                "has_margin":    has_margin,
                "has_holes":     has_holes,
                "line_count":    len(all_lines),
                "caption":       caption,
            },
            render_payload={
                "image_path": img_path.resolve().as_posix(),
                "alt_text":   caption,
                "caption":    caption,
                "format":     "png",
            },
            ground_truth={
                "hw_lines":    hw_lines,
                "sig_lines":   sig_lines,
                "hw_style":    hw_style,
                "layout_style":layout,
                "has_signature":include_sig,
                "caption":     caption,
                "qa_candidates": _handwriting_qa(caption, hw_style, include_sig),
            },
        ))
    return variants


def generate_stamp_mark_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Path | None = None,
) -> list[Variant]:
    """Generate *count* approval stamp and verification mark variants.

    Each variant renders a scanned verification-checklist page, then
    composites one or two rubber-stamp impressions ON TOP of the page
    (just as in a real signed document).

    Stamp improvements:
    - Random rotation (±20°) and position on the page
    - Ink colour matched to stamp type (red approve, blue certify, etc.)
    - Ink bleed via Gaussian blur
    - Occasional double-stamp (slightly offset second impression)
    - Semi-transparent paste for realistic partial ink transfer

    """
    from synthpdf.shared.handwriting import simulate_handwriting
    from PIL import Image

    out_dir = (assets_dir or Path(".")).resolve()
    variants: list[Variant] = []

    for i in range(count):
        child_rng = fork_rng(rng, f"stamp_{i}")

        sq_idx  = int(child_rng.integers(0, len(_SCAN_QUALITY_POOL)))
        quality = _SCAN_QUALITY_POOL[sq_idx]
        hw_seed = int(child_rng.integers(0, 2**31))

        inspector = _INSPECTOR_NAMES[int(child_rng.integers(0, len(_INSPECTOR_NAMES)))]
        ref_no    = int(child_rng.integers(1000, 9999))
        date_str  = f"{int(child_rng.integers(1, 29)):02d}/{int(child_rng.integers(1, 13)):02d}/2024"

        # ---- Verification checklist page ----
        n_items  = int(child_rng.integers(5, len(_VERIFICATION_ITEMS) + 1))
        item_idxs = list(child_rng.choice(len(_VERIFICATION_ITEMS), size=n_items, replace=False))
        items   = [_VERIFICATION_ITEMS[int(j)] for j in item_idxs]
        checked = [it for it in items if child_rng.random() > 0.35]

        checklist_lines = [
            f"VERIFICATION CHECKLIST — Ref: QC-{ref_no}",
            f"Inspector: {inspector}  |  Date: {date_str}",
            "",
        ] + [
            f"[{'X' if it in checked else ' '}]  {it}" for it in items
        ] + [
            "",
            f"Authorised by: _______________________________",
            f"Signature:     _______________________________",
        ]
        checklist_text = "\n".join(checklist_lines)

        # Render the checklist as a full page
        page_img = _render_full_page(checklist_text, child_rng, quality)
        # Apply base scan artefacts first
        page_img = _apply_advanced_scan(page_img, quality, child_rng)

        pw, ph = page_img.size

        # ---- Stamp(s) ----
        n_stamps = int(child_rng.integers(1, 3))
        stamp_records: list[str] = []

        for s_idx in range(n_stamps):
            stamp_text_raw = _STAMP_POOL[int(child_rng.integers(0, len(_STAMP_POOL)))]
            ink = _STAMP_INK.get(stamp_text_raw, (160, 20, 20))
            angle = float(child_rng.uniform(-22.0, 22.0))

            stamp_img = _render_stamp(stamp_text_raw, ink, child_rng, angle=angle)
            sw, sh = stamp_img.size

            # Position: bias toward bottom-right or top-right (realistic stamp placement)
            # Always clamp to keep stamp fully inside the page.
            max_px = max(0, pw - sw)
            max_py = max(0, ph - sh)
            pos_zone = int(child_rng.integers(0, 3))  # 0=top-right, 1=bottom-right, 2=centre
            if pos_zone == 0:
                lo_x = max(0, min(pw - sw - _ML, max_px))
                hi_x = max(lo_x + 1, min(pw - _ML // 2, max_px + 1))
                lo_y, hi_y = _MT, max(_MT + 1, min(_MT + ph // 4, max_py + 1))
            elif pos_zone == 1:
                lo_x = max(0, min(pw - sw - _ML, max_px))
                hi_x = max(lo_x + 1, min(pw - _ML // 2, max_px + 1))
                lo_y = max(0, min(3 * ph // 4, max_py))
                hi_y = max(lo_y + 1, min(ph - sh - _MT, max_py + 1))
            else:
                lo_x = max(0, min(pw // 4, max_px))
                hi_x = max(lo_x + 1, min(3 * pw // 4, max_px + 1))
                lo_y = max(0, min(ph // 4, max_py))
                hi_y = max(lo_y + 1, min(3 * ph // 4, max_py + 1))

            px = int(child_rng.integers(lo_x, hi_x))
            py = int(child_rng.integers(lo_y, hi_y))
            px = max(0, min(px, max_px))
            py = max(0, min(py, max_py))

            # Occasional double-stamp effect (faint second impression)
            if child_rng.random() < 0.30:
                off_x = int(child_rng.integers(-8, 8))
                off_y = int(child_rng.integers(-8, 8))
                ghost = stamp_img.copy()
                # Halve all alpha values for ghost
                ghost_arr = np.array(ghost, dtype=np.uint16)
                ghost_arr[:, :, 3] = ghost_arr[:, :, 3] // 3
                from PIL import Image as _PIL
                ghost = _PIL.fromarray(ghost_arr.astype(np.uint8))
                page_img.paste(ghost,
                               (px + off_x, py + off_y),
                               ghost)

            page_img.paste(stamp_img, (px, py), stamp_img)
            stamp_records.append(stamp_text_raw)

        img_path = _save_pil(page_img, "stamp", out_dir)
        caption  = (
            f"Scanned checklist with approval stamp(s): "
            f"{', '.join(stamp_records)} ({quality} quality)"
        )

        variants.append(Variant(
            id=f"stamp_{i:04d}_{_uid()[:8]}",
            kind="image",
            metadata={
                "subtype":        "stamp_mark",
                "scan_quality":   quality,
                "stamp_count":    len(stamp_records),
                "checklist_items":len(items),
                "checked_count":  len(checked),
                "caption":        caption,
            },
            render_payload={
                "image_path": img_path.resolve().as_posix(),
                "alt_text":   caption,
                "caption":    caption,
                "format":     "png",
            },
            ground_truth={
                "stamp_marks":    stamp_records,
                "scan_quality":   quality,
                "checklist_items":items,
                "checked_items":  checked,
                "unchecked_items":[it for it in items if it not in checked],
                "inspector":      inspector,
                "reference":      f"QC-{ref_no}",
                "caption":        caption,
                "qa_candidates":  _stamp_qa(caption, stamp_records, inspector, f"QC-{ref_no}"),
            },
        ))
    return variants


def generate_scan_pool(
    rng: np.random.Generator,
    count: int,
    assets_dir: Path | None = None,
) -> list[Variant]:
    """Generate a mixed pool of all four scan/handwriting sub-types.

    Split approximately equally: scanned_text → scanned_table →
    handwriting → stamp_mark.  Any remainder goes to scanned_text first.

    Args:
        rng:        Seeded numpy random generator.
        count:      Total number of scan variants to produce.
        assets_dir: Directory where PNG files are written.

    Returns:
        Flat list of ``Variant`` objects.
    """
    rng_st = fork_rng(rng, "pool_scan_text")
    rng_tb = fork_rng(rng, "pool_scan_table")
    rng_hw = fork_rng(rng, "pool_handwriting")
    rng_sm = fork_rng(rng, "pool_stamp_mark")

    base   = count // 4
    extras = count % 4

    n_st = base + (1 if extras > 0 else 0)
    n_tb = base + (1 if extras > 1 else 0)
    n_hw = base + (1 if extras > 2 else 0)
    n_sm = base

    return (
        generate_scanned_text_variants(rng_st, n_st, assets_dir)
        + generate_scanned_table_variants(rng_tb, n_tb, assets_dir)
        + generate_handwriting_variants(rng_hw, n_hw, assets_dir)
        + generate_stamp_mark_variants(rng_sm, n_sm, assets_dir)
    )


# ===========================================================================
# LLM-driven scanned page pipeline
# ===========================================================================
#
# Flow:
#   (1) Consume document plan/brief
#   (2) LLM generates content spec JSON (variant-specific schema)
#   (3) Deterministic Python renders image from spec
#   (4) LLM generates non-overlapping QnA from spec + ground truth
#
# Five variant types:
#   text_document    — formal printed text (memo, report, letter)
#   data_table       — tabular data as a scanned page
#   handwritten_note — handwriting on ruled/blank paper
#   stamped_form     — form/checklist with rubber-stamp overlay
#   annotated_page   — printed page with handwritten review annotations
# ===========================================================================

_SCANNED_LLM_VARIANT_TYPES: list[str] = [
    "text_document",
    "data_table",
    "handwritten_note",
    "stamped_form",
    "annotated_page",
]

# Map legacy config subtype names to LLM variant type names
_SCANNED_SUBTYPE_MAP: dict[str, str] = {
    "text":        "text_document",
    "table":       "data_table",
    "handwriting": "handwritten_note",
    "stamps":      "stamped_form",
    # LLM names pass through unchanged
    "text_document":    "text_document",
    "data_table":       "data_table",
    "handwritten_note": "handwritten_note",
    "stamped_form":     "stamped_form",
    "annotated_page":   "annotated_page",
}


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

def _scanned_content_prompt(
    topic_description: str,
    variant_type: str,
    doc_brief: "dict[str, Any] | None" = None,
    variation_index: int = 0,
) -> str:
    """Return the LLM prompt for generating a scanned-page spec JSON.

    Each variant type has its own detailed schema so the LLM knows exactly
    what structure to produce for deterministic Python rendering.
    ``variation_index`` is injected into the prompt so each call produces
    distinct content even for the same topic and variant type.
    """
    brief_ctx = ""
    if doc_brief:
        brief_ctx = (
            f"\nDocument context:\n"
            f"  Narrative theme: {doc_brief.get('narrative_theme', '')}\n"
        )

    schemas: dict[str, str] = {
        "text_document": """\
{
  "variant_type": "text_document",
  "document_type": "<memo|report_page|letter|policy_excerpt|specification>",
  "title": "<concise document title>",
  "subtitle": "<reference/date/from-to line, e.g. 'Ref: RM-0041 | Date: 14 Mar 2025 | From: Head of Operations'>",
  "sections": [
    {"heading": "<section heading or empty string>", "body": "<paragraph text, 2-5 sentences>"},
    {"heading": "<section heading or empty string>", "body": "<paragraph text>"}
  ],
  "footer_note": "<short footer text, e.g. 'Confidential — Internal Use Only | Page 1 of 3'>",
  "quality_tier": "<pristine|high|medium|aged|poor>"
}

Requirements:
- Include 3-6 sections with realistic headings and substantive body text (2-5 sentences each).
- Ground content firmly in the topic domain — no generic placeholders.
- quality_tier: 'medium' or 'aged' are most common for scanned documents.
""",
        "data_table": """\
{
  "variant_type": "data_table",
  "title": "<table heading — short and descriptive>",
  "caption": "<one sentence describing what the table shows>",
  "headers": ["<col1>", "<col2>", "<col3>"],
  "rows": [
    ["<val>", "<val>", "<val>"],
    "... (8-20 rows total)"
  ],
  "summary_note": "<1-2 sentence observation about the data>",
  "date": "<date string, e.g. 'Q3 2024' or '12 Sep 2024'>",
  "quality_tier": "<pristine|high|medium|aged|poor>"
}

Requirements:
- Include 3-6 columns and 8-20 data rows with realistic, topic-specific values.
- Mix numeric and text columns; at least one column must contain numbers.
- Column headers must be concise (1-4 words each).
- quality_tier: 'medium' or 'aged' most common for scanned tables.
""",
        "handwritten_note": """\
{
  "variant_type": "handwritten_note",
  "title": "<short title line at top, e.g. 'Meeting Notes — 14/03/2025' or 'Action Items'>",
  "lines": [
    "<line 1 of handwritten content>",
    "<line 2>",
    "..."
  ],
  "style": "<neat|messy|fast>",
  "has_signature": <true|false>,
  "signature_name": "<full name if has_signature true, else empty string>",
  "date": "<date string>",
  "paper_type": "<ruled|blank>",
  "quality_tier": "<pristine|high|medium|aged|poor>"
}

Requirements:
- Include 10-25 lines of realistic content (notes, action items, observations, calculations).
- Lines may use abbreviations, short fragments, bullet markers — as real handwriting looks.
- style 'neat' for formal notes; 'messy' or 'fast' for quick jottings.
- paper_type 'ruled' is most common.
- Content must be specific to the topic domain.
""",
        "stamped_form": """\
{
  "variant_type": "stamped_form",
  "title": "<form title, e.g. 'Quality Inspection Report' or 'Expense Approval Form'>",
  "form_fields": [
    {"label": "<field label>", "value": "<filled value>"}
  ],
  "checklist_items": [
    {"text": "<checklist item text>", "checked": <true|false>}
  ],
  "stamps": [
    {"text": "<APPROVED|CERTIFIED|REVIEWED|QC PASS|CONFIDENTIAL|EXECUTED>", "color_key": "<red|blue|green|purple>"}
  ],
  "inspector": "<inspector full name>",
  "reference": "<reference number, e.g. QC-2847>",
  "date": "<date string>",
  "quality_tier": "<pristine|high|medium|aged|poor>"
}

Requirements:
- Include 4-10 form fields with topic-specific labels and realistic filled values.
- Include 5-10 checklist items; a mix of checked and unchecked.
- Include 1-2 stamps appropriate for the form type.
- quality_tier: 'medium' or 'aged' most common for scanned forms.
""",
        "annotated_page": """\
{
  "variant_type": "annotated_page",
  "title": "<document title>",
  "subtitle": "<reference/date line>",
  "sections": [
    {"heading": "<heading>", "body": "<paragraph text, 2-4 sentences>"}
  ],
  "annotations": [
    {
      "type": "<marginal_note|underline|circle|checkmark>",
      "target_hint": "<short phrase from text that this annotation targets>",
      "note_text": "<for marginal_note: 3-12 words of reviewer comment; for others: empty string>"
    }
  ],
  "annotator": "<reviewer name or initials>",
  "date": "<date string>",
  "quality_tier": "<pristine|high|medium|aged|poor>"
}

Requirements:
- Include 2-4 substantive text sections relevant to the topic.
- Include 2-5 annotations of mixed types (at least one marginal_note).
- For marginal_note: note_text should be a realistic reviewer comment (3-12 words).
- quality_tier: 'medium' or 'aged' typical for reviewed documents.
""",
    }

    schema = schemas.get(variant_type, schemas["text_document"])
    return (
        f"You are generating content for a synthetic scanned-document dataset.\n"
        f"Topic domain: {topic_description}\n"
        f"Variation index: {variation_index} — produce content distinct from other variants "
        f"(use different names, dates, figures, and phrasing each time).\n"
        f"{brief_ctx}\n"
        f"Produce a single scanned page specification as a JSON object matching this schema:\n\n"
        f"{schema}\n"
        f"Return ONLY the JSON object. No markdown fences, no commentary.\n"
        f"Make all content specific and realistic — firmly grounded in the topic domain.\n"
        f"Do NOT use placeholder text like 'Lorem ipsum' or generic labels like 'Field 1'."
    )


# ---------------------------------------------------------------------------
# LLM spec generator
# ---------------------------------------------------------------------------

def _llm_generate_scanned_spec(
    client: Any,
    topic: Any,
    variant_type: str,
    rng: "np.random.Generator",
    doc_brief: "dict[str, Any] | None" = None,
    variation_index: int = 0,
) -> "dict[str, Any]":
    """Call the LLM to generate a scanned-page spec for *variant_type*.

    Falls back to a minimal static spec if the LLM call fails or client is None.
    ``variation_index`` is woven into the prompt to ensure distinct content per call.
    """
    import logging as _log

    if client is None:
        return _fallback_scanned_spec(variant_type, rng)

    prompt = _scanned_content_prompt(topic.description, variant_type, doc_brief, variation_index)
    try:
        raw = client.complete(
            prompt, json_mode=True, tier="medium",
            _label=f"llm_scanned_{variant_type}",
        )
        spec: dict[str, Any] = json.loads(raw) if isinstance(raw, str) else raw
        spec["variant_type"] = variant_type  # enforce correct type tag
        return spec
    except Exception as exc:
        _log.getLogger(__name__).warning(
            "LLM scanned spec failed (%s): %s — using fallback", variant_type, exc
        )
        return _fallback_scanned_spec(variant_type, rng)


def _fallback_scanned_spec(
    variant_type: str,
    rng: "np.random.Generator",
) -> "dict[str, Any]":
    """Minimal static spec used when LLM is unavailable.

    Uses RNG to vary names, reference numbers, and dates so repeated calls
    with different RNG states produce distinct (though simple) content.
    """
    quality = _SCAN_QUALITY_POOL[int(rng.integers(0, len(_SCAN_QUALITY_POOL)))]

    # RNG-driven variation helpers
    _NAMES = ["J. Smith", "A. Patel", "M. Chen", "R. Johnson", "L. Torres",
              "D. Nguyen", "C. Williams", "S. Kumar", "F. Okafor", "H. Mueller"]
    _DEPTS = ["Operations", "Finance", "Engineering", "Compliance", "Research",
              "Logistics", "Quality Assurance", "Strategy", "Legal", "Product"]
    _MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _QTRS = ["Q1", "Q2", "Q3", "Q4"]

    name    = _NAMES[int(rng.integers(0, len(_NAMES)))]
    dept    = _DEPTS[int(rng.integers(0, len(_DEPTS)))]
    month   = _MONTHS[int(rng.integers(0, len(_MONTHS)))]
    day     = int(rng.integers(1, 29))
    year    = int(rng.integers(2023, 2026))
    ref_num = int(rng.integers(1000, 9999))
    qtr     = _QTRS[int(rng.integers(0, 4))]
    date_str = f"{day:02d}/{month}/{year}"
    ref_str  = f"INT-{ref_num}"
    qc_ref   = f"QC-{ref_num}"

    if variant_type == "text_document":
        return {
            "variant_type": "text_document",
            "document_type": "memo",
            "title": f"Internal Memorandum — {dept}",
            "subtitle": f"Ref: {ref_str} | Date: {day:02d} {month} {year} | From: {name}",
            "sections": [
                {"heading": "Background",   "body": f"This memorandum from {dept} outlines the current operational status and key findings from the review period ending {month} {year}."},
                {"heading": "Key Findings", "body": f"Analysis conducted by {name} indicates satisfactory performance across all monitored parameters during {qtr} {year}."},
                {"heading": "Next Steps",   "body": f"The {dept} team will proceed with the recommendations outlined in the attached annex by end of {qtr}."},
            ],
            "footer_note": f"Confidential — Internal Use Only | Ref: {ref_str} | Page 1",
            "quality_tier": quality,
        }
    elif variant_type == "data_table":
        base_val = float(rng.uniform(50.0, 300.0))
        rows = [
            [f"{i+1:03d}", f"Item {chr(65+i)}", f"{base_val + i*17.3:.1f}", "Active" if i % 3 != 2 else "Pending"]
            for i in range(5)
        ]
        return {
            "variant_type": "data_table",
            "title": f"{dept} Summary — {qtr} {year}",
            "caption": f"Aggregated metrics prepared by {name} for {qtr} {year}.",
            "headers": ["ID", "Description", "Value", "Status"],
            "rows": rows,
            "summary_note": f"Five records shown as of {date_str}. Values reflect end-of-period aggregation.",
            "date": f"{qtr} {year}",
            "quality_tier": quality,
        }
    elif variant_type == "handwritten_note":
        initials = "".join(p[0] for p in name.split() if p)
        return {
            "variant_type": "handwritten_note",
            "title": f"Meeting Notes — {date_str}",
            "lines": [
                f"Action items from {dept} review ({date_str}):",
                f"- Confirm budget allocation by {_MONTHS[int(rng.integers(0, 12))]} {year}",
                f"- Follow up with vendor re: delivery schedule",
                f"- Update risk register with new items (ref {ref_str})",
                f"- Schedule debrief for next week",
                "",
                f"Note: delays in shipment may affect {qtr} timeline",
                f"Check with legal re: contract clause {int(rng.integers(2, 8))}.{int(rng.integers(1, 5))}",
                f"Approved: proceed with phase {int(rng.integers(2, 4))} implementation",
            ],
            "style": ["neat", "fast", "messy"][int(rng.integers(0, 3))],
            "has_signature": True,
            "signature_name": name,
            "date": date_str,
            "paper_type": "ruled",
            "quality_tier": quality,
        }
    elif variant_type == "stamped_form":
        stamp_text = ["APPROVED", "CERTIFIED", "REVIEWED", "QC PASS"][int(rng.integers(0, 4))]
        stamp_color = ["red", "blue", "green", "purple"][int(rng.integers(0, 4))]
        return {
            "variant_type": "stamped_form",
            "title": f"{dept} Inspection Report",
            "form_fields": [
                {"label": "Inspector",   "value": name},
                {"label": "Reference",   "value": qc_ref},
                {"label": "Date",        "value": date_str},
                {"label": "Department",  "value": dept},
            ],
            "checklist_items": [
                {"text": "Documentation complete",   "checked": bool(rng.integers(0, 2))},
                {"text": "Signatures verified",      "checked": True},
                {"text": "Compliance check passed",  "checked": bool(rng.integers(0, 2))},
                {"text": "Follow-up action required","checked": bool(rng.integers(0, 2))},
            ],
            "stamps": [{"text": stamp_text, "color_key": stamp_color}],
            "inspector": name,
            "reference": qc_ref,
            "date": date_str,
            "quality_tier": quality,
        }
    else:  # annotated_page
        initials = "".join(p[0] for p in name.split() if p)
        return {
            "variant_type": "annotated_page",
            "title": f"{dept} Document Review",
            "subtitle": f"Ref: REV-{ref_num} | Date: {date_str}",
            "sections": [
                {"heading": "Summary",  "body": f"This section summarises the key points under review for {qtr} {year} prepared by {name}."},
                {"heading": "Analysis", "body": f"The {dept} data indicates a need for further investigation of the highlighted operational areas."},
            ],
            "annotations": [
                {"type": "marginal_note", "target_hint": "Summary",               "note_text": "Reviewed - OK"},
                {"type": "underline",     "target_hint": "further investigation", "note_text": ""},
                {"type": "circle",        "target_hint": "key points",            "note_text": ""},
            ],
            "annotator": initials,
            "date": date_str,
            "quality_tier": quality,
        }


# ---------------------------------------------------------------------------
# Text-block builder (spec -> plain string for _render_full_page)
# ---------------------------------------------------------------------------

def _build_text_block_from_spec(spec: "dict[str, Any]") -> str:
    """Convert a text_document or annotated_page spec into a plain-text block.

    The first line becomes the letterhead in _render_full_page; the second
    line is styled as subtitle/reference; subsequent lines are body text.
    """
    lines: list[str] = []

    lines.append(spec.get("title", "Document"))

    subtitle = spec.get("subtitle", spec.get("date", ""))
    lines.append(subtitle if subtitle else "")

    lines.append("")  # spacer

    for sec in spec.get("sections", []):
        heading = sec.get("heading", "")
        body    = sec.get("body", "")
        if heading:
            lines.append(f"Section {heading}")
        if body:
            lines.append(body)
        lines.append("")

    footer = spec.get("footer_note", "")
    if footer:
        lines.append(footer)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Variant-specific renderers
# ---------------------------------------------------------------------------

def _render_handwritten_from_spec(
    spec: "dict[str, Any]",
    rng: "np.random.Generator",
    quality: str,
) -> Any:
    """Render a handwritten note from the LLM spec using simulate_handwriting."""
    from synthpdf.shared.handwriting import simulate_handwriting

    paper_type = spec.get("paper_type", "ruled")
    style = spec.get("style", "neat")
    if style not in ("neat", "messy", "fast"):
        style = "neat"

    # Build the full line list: title + date header + content + signature
    hw_lines: list[str] = []

    title = spec.get("title", "")
    if title:
        hw_lines.append(title)
    date_str = spec.get("date", "")
    if date_str:
        hw_lines.append(date_str)
    if hw_lines:
        hw_lines.append("")  # blank spacer after header

    for line in spec.get("lines", []):
        hw_lines.append(str(line))

    if spec.get("has_signature") and spec.get("signature_name"):
        hw_lines.append("")
        hw_lines.append(f"Signed: {spec['signature_name']}")

    if not hw_lines:
        hw_lines = ["(no content)"]

    # Background paper
    bg_rgb: tuple[int, int, int] = (255, 255, 248)
    bg_img = _render_ruled_paper(
        _PAGE_W, _PAGE_H, rng,
        ruled=(paper_type == "ruled"),
        has_margin=(paper_type == "ruled"),
    )

    # Ink colour: dark blue or black
    _hw_colors = [
        (20, 30, 120),   # dark blue (pen)
        (15, 15, 60),    # navy
        (10, 10, 10),    # black
        (30, 10, 80),    # deep purple-blue
    ]
    hw_color = _hw_colors[int(rng.integers(0, len(_hw_colors)))]
    hw_seed  = int(rng.integers(0, 2**31))

    hw_img = simulate_handwriting(
        hw_lines,
        width=_PAGE_W,
        height=_PAGE_H,
        color=hw_color,
        background_color=bg_rgb,
        seed=hw_seed,
        style=style,
    )

    # simulate_handwriting has its own background; apply scan on top
    final = hw_img.convert("RGB")
    final = _apply_advanced_scan(final, quality, rng)
    return final


def _render_stamped_form_from_spec(
    spec: "dict[str, Any]",
    rng: "np.random.Generator",
    quality: str,
) -> Any:
    """Render a filled form/checklist page with rubber-stamp overlays."""
    from PIL import Image

    # Build the text block representing the form content
    lines: list[str] = []

    lines.append(spec.get("title", "Form"))

    # Subtitle: inspector | ref | date
    sub_parts = [
        p for p in [
            spec.get("inspector", ""),
            spec.get("reference", ""),
            spec.get("date", ""),
        ]
        if p
    ]
    lines.append("  |  ".join(sub_parts) if sub_parts else "")
    lines.append("")

    for field in spec.get("form_fields", []):
        label = str(field.get("label", ""))
        value = str(field.get("value", ""))
        lines.append(f"{label}: {value}")

    if spec.get("form_fields"):
        lines.append("")

    for item in spec.get("checklist_items", []):
        text    = str(item.get("text", ""))
        checked = item.get("checked", False)
        mark    = "X" if checked else " "
        lines.append(f"[{mark}]  {text}")

    if spec.get("checklist_items"):
        lines.append("")
        lines.append("Authorised by: _______________________________")
        lines.append("Signature:     _______________________________")

    text_block = "\n".join(lines)

    # Render base page + scan
    page_img = _render_full_page(text_block, rng, quality)
    page_img = _apply_advanced_scan(page_img, quality, rng)
    pw, ph   = page_img.size

    # Stamp color mapping
    _color_map: dict[str, tuple[int, int, int]] = {
        "red":    (160, 20,  20),
        "blue":   (20,  60,  160),
        "green":  (20,  120, 40),
        "purple": (80,  20,  140),
    }

    for s_spec in spec.get("stamps", []):
        stamp_text_raw = str(s_spec.get("text", "APPROVED")).upper()
        ink = _color_map.get(s_spec.get("color_key", "red"), (160, 20, 20))
        angle = float(rng.uniform(-20.0, 20.0))

        stamp_img = _render_stamp(stamp_text_raw, ink, rng, angle=angle)
        sw, sh    = stamp_img.size

        max_px = max(0, pw - sw)
        max_py = max(0, ph - sh)
        # 70 % chance bottom half; 30 % chance top half
        if rng.random() < 0.7:
            px = int(rng.integers(max(0, max_px // 2), max_px + 1))
            py = int(rng.integers(max(0, max_py * 2 // 3), max_py + 1))
        else:
            px = int(rng.integers(max(0, max_px // 2), max_px + 1))
            py = int(rng.integers(0, max(1, max_py // 3 + 1)))

        if stamp_img.mode == "RGBA":
            page_img = page_img.convert("RGBA")
            overlay  = Image.new("RGBA", page_img.size, (0, 0, 0, 0))
            overlay.paste(stamp_img, (px, py), stamp_img)
            page_img = Image.alpha_composite(page_img, overlay).convert("RGB")
        else:
            page_img.paste(stamp_img, (px, py))

    return page_img.convert("RGB")


def _render_annotated_page_from_spec(
    spec: "dict[str, Any]",
    rng: "np.random.Generator",
    quality: str,
    assets_dir: Path,
) -> Any:
    """Render a printed page with handwritten review annotations overlaid.

    Base text is rendered normally, scan artifacts applied, then annotation
    marks (underlines, circles, checkmarks, marginal notes) are drawn on top.
    Marginal note text is rendered via simulate_handwriting as small PIL images.
    """
    from synthpdf.shared.handwriting import simulate_handwriting
    from PIL import Image, ImageDraw

    # Base page
    text_block = _build_text_block_from_spec(spec)
    img = _render_full_page(text_block, rng, quality)
    img = _apply_advanced_scan(img, quality, rng)

    draw = ImageDraw.Draw(img)
    annotations = spec.get("annotations", [])
    n_ann = len(annotations)

    # Distribute annotation positions evenly through the body area
    body_top = _MT + _LH_TITLE + 60
    body_bot = _MB - 40
    body_h   = body_bot - body_top

    for idx, ann in enumerate(annotations):
        ann_type = ann.get("type", "underline")
        frac  = (idx + 0.5) / max(n_ann, 1)
        ann_y = int(body_top + frac * body_h)

        # Pen colour: dark blue with slight variation
        pr = int(rng.integers(15, 50))
        pg = int(rng.integers(15, 50))
        pb = int(rng.integers(100, 180))
        pen = (pr, pg, pb)

        if ann_type == "underline":
            x1 = _ML + int(rng.integers(0, 80))
            x2 = x1  + int(rng.integers(100, _CW - 100))
            segments = 12
            seg_w    = max(1, (x2 - x1) // segments)
            prev_y   = ann_y
            for s in range(segments):
                sx = x1 + s * seg_w
                ey = ann_y + int(rng.integers(-2, 3))
                draw.line([(sx, prev_y), (sx + seg_w, ey)],
                           fill=pen, width=2)
                prev_y = ey

        elif ann_type == "circle":
            cx = _ML + int(rng.integers(80, _CW // 2))
            cy = ann_y
            rx = int(rng.integers(50, 110))
            ry = int(rng.integers(14, 30))
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                          outline=pen, width=2)

        elif ann_type == "checkmark":
            cx, cy = _ML - 30, ann_y
            sz = 18
            draw.line([(cx,          cy),           (cx + sz // 3, cy + sz // 2)],
                       fill=pen, width=3)
            draw.line([(cx + sz // 3, cy + sz // 2),(cx + sz,      cy - sz // 3)],
                       fill=pen, width=3)

        elif ann_type == "marginal_note":
            note_text = ann.get("note_text", "")
            if note_text:
                note_w, note_h = 180, 50
                hw_seed = int(rng.integers(0, 2**31))
                try:
                    note_img = simulate_handwriting(
                        [note_text],
                        width=note_w,
                        height=note_h,
                        color=pen,
                        background_color=(255, 255, 248),
                        seed=hw_seed,
                        style="fast",
                    )
                    mx = 10
                    my = max(0, ann_y - note_h // 2)
                    img.paste(note_img.convert("RGB"), (mx, my))
                    # Small arrow pointing from note toward text margin
                    ax_start = mx + note_w + 2
                    ax_end   = _ML - 5
                    if ax_start < ax_end:
                        draw.line([(ax_start, ann_y), (ax_end, ann_y)],
                                   fill=pen, width=1)
                        draw.line([(ax_end - 8, ann_y - 4), (ax_end, ann_y)],
                                   fill=pen, width=1)
                        draw.line([(ax_end - 8, ann_y + 4), (ax_end, ann_y)],
                                   fill=pen, width=1)
                except Exception:
                    pass  # best-effort; skip if simulate_handwriting fails

    return img


# ---------------------------------------------------------------------------
# Main renderer dispatcher
# ---------------------------------------------------------------------------

def _render_from_spec(
    spec: "dict[str, Any]",
    rng: "np.random.Generator",
    assets_dir: Path,
) -> Any:
    """Render a scanned page PIL image from the LLM-generated spec.

    Dispatches to the correct renderer based on variant_type and applies
    the 15-stage scan simulation pipeline (except handwriting, which has
    its own scan pass inside _render_handwritten_from_spec).
    """
    quality = spec.get("quality_tier", "medium")
    if quality not in _SCAN_QUALITY_PARAMS:
        quality = "medium"

    vtype = spec.get("variant_type", "text_document")

    if vtype == "text_document":
        text_block = _build_text_block_from_spec(spec)
        img = _render_full_page(text_block, rng, quality)
        img = _apply_advanced_scan(img, quality, rng)

    elif vtype == "data_table":
        sample = {
            "headers": spec.get("headers", ["Col 1", "Col 2"]),
            "rows":    spec.get("rows", []),
            "caption": spec.get("title", spec.get("caption", "Data Table")),
        }
        img = _render_table_page(sample, rng, quality)
        img = _apply_advanced_scan(img, quality, rng)

    elif vtype == "handwritten_note":
        img = _render_handwritten_from_spec(spec, rng, quality)

    elif vtype == "stamped_form":
        img = _render_stamped_form_from_spec(spec, rng, quality)

    elif vtype == "annotated_page":
        img = _render_annotated_page_from_spec(spec, rng, quality, assets_dir)

    else:
        # Unknown — fall back to text document renderer
        text_block = _build_text_block_from_spec(spec)
        img = _render_full_page(text_block, rng, quality)
        img = _apply_advanced_scan(img, quality, rng)

    return img


# ---------------------------------------------------------------------------
# Ground truth builder
# ---------------------------------------------------------------------------

def _build_scanned_gt(
    spec: "dict[str, Any]",
    image_path: Path,
) -> "dict[str, Any]":
    """Build structured ground truth from the LLM-generated spec.

    Returns a flat dict suitable for manifest and QnA generators.
    """
    vtype   = spec.get("variant_type", "text_document")
    quality = spec.get("quality_tier", "medium")

    gt: dict[str, Any] = {
        "variant_type": vtype,
        "scan_quality": quality,
        "image_path":   str(image_path),
        "caption":      spec.get("title", spec.get("caption", "Scanned Document")),
    }

    if vtype == "text_document":
        sections = spec.get("sections", [])
        gt["document_type"]    = spec.get("document_type", "")
        gt["subtitle"]         = spec.get("subtitle", "")
        gt["sections"]         = sections
        gt["full_text"]        = _build_text_block_from_spec(spec)
        gt["section_headings"] = [s.get("heading", "") for s in sections if s.get("heading")]
        gt["footer_note"]      = spec.get("footer_note", "")

    elif vtype == "data_table":
        gt["headers"]      = spec.get("headers", [])
        gt["rows"]         = spec.get("rows", [])
        gt["summary_note"] = spec.get("summary_note", "")
        gt["date"]         = spec.get("date", "")
        gt["row_count"]    = len(spec.get("rows", []))
        gt["col_count"]    = len(spec.get("headers", []))

    elif vtype == "handwritten_note":
        lines = spec.get("lines", [])
        gt["style"]          = spec.get("style", "neat")
        gt["paper_type"]     = spec.get("paper_type", "ruled")
        gt["lines"]          = lines
        gt["line_count"]     = len(lines)
        gt["has_signature"]  = spec.get("has_signature", False)
        gt["signature_name"] = spec.get("signature_name", "")
        gt["date"]           = spec.get("date", "")

    elif vtype == "stamped_form":
        stamps = spec.get("stamps", [])
        gt["form_fields"]    = spec.get("form_fields", [])
        gt["checklist_items"]= spec.get("checklist_items", [])
        gt["stamps"]         = [s.get("text", "") for s in stamps]
        gt["inspector"]      = spec.get("inspector", "")
        gt["reference"]      = spec.get("reference", "")
        gt["date"]           = spec.get("date", "")
        checked = [it for it in spec.get("checklist_items", []) if it.get("checked")]
        gt["checked_count"]  = len(checked)
        gt["total_items"]    = len(spec.get("checklist_items", []))

    elif vtype == "annotated_page":
        gt["sections"]         = spec.get("sections", [])
        gt["annotations"]      = spec.get("annotations", [])
        gt["annotator"]        = spec.get("annotator", "")
        gt["date"]             = spec.get("date", "")
        gt["annotation_types"] = list({a.get("type") for a in spec.get("annotations", [])})
        gt["marginal_notes"]   = [
            a.get("note_text", "")
            for a in spec.get("annotations", [])
            if a.get("type") == "marginal_note" and a.get("note_text")
        ]

    return gt


# ---------------------------------------------------------------------------
# LLM QnA generator for scanned pages
# ---------------------------------------------------------------------------

def _llm_qa_for_scanned_page(
    client: Any,
    topic: Any,
    spec: "dict[str, Any]",
    gt: "dict[str, Any]",
    n: int = 5,
) -> "list[dict[str, Any]]":
    """Generate LLM-driven, non-overlapping QnA for a scanned page.

    Questions must reference the page title as an explicit anchor so they
    are self-sufficient (answerable without seeing the image directly).

    Args:
        client: AzureOpenAIClient (or None -> returns []).
        topic:  TopicSpec with .description.
        spec:   LLM-generated content spec dict.
        gt:     Ground truth dict from _build_scanned_gt().
        n:      Number of QnA pairs to generate.

    Returns:
        List of {"question": str, "answer": str, "assertions": [{"text", "level", "metric"}]}.
    """
    import logging as _log

    if client is None:
        return []

    vtype   = spec.get("variant_type", "text_document")
    caption = gt.get("caption", "scanned document")

    # Build a concise content summary so the LLM has facts to work from
    parts: list[str] = [f"Page type: {vtype}", f"Title/caption: {caption}"]

    if vtype == "text_document":
        for sec in spec.get("sections", [])[:4]:
            h = sec.get("heading", "")
            b = str(sec.get("body", ""))[:140]
            parts.append(f"  Section '{h}': {b}")
        if spec.get("footer_note"):
            parts.append(f"Footer: {spec['footer_note']}")

    elif vtype == "data_table":
        parts.append(f"Headers: {spec.get('headers', [])}")
        for row in spec.get("rows", [])[:6]:
            parts.append(f"  Row: {row}")
        if len(spec.get("rows", [])) > 6:
            parts.append(f"  ... ({len(spec['rows'])} rows total)")
        parts.append(f"Summary: {spec.get('summary_note', '')}")

    elif vtype == "handwritten_note":
        parts.append(f"Style: {spec.get('style', 'neat')}, Date: {spec.get('date', '')}")
        for line in spec.get("lines", [])[:12]:
            parts.append(f"  {line}")
        if spec.get("has_signature"):
            parts.append(f"Signed by: {spec.get('signature_name', '')}")

    elif vtype == "stamped_form":
        parts.append(
            f"Inspector: {spec.get('inspector','')}, "
            f"Ref: {spec.get('reference','')}, "
            f"Date: {spec.get('date','')}"
        )
        for f in spec.get("form_fields", []):
            parts.append(f"  {f.get('label','')}: {f.get('value','')}")
        for it in spec.get("checklist_items", []):
            mark = "CHECKED" if it.get("checked") else "unchecked"
            parts.append(f"  [{mark}] {it.get('text','')}")
        for s in spec.get("stamps", []):
            parts.append(f"  Stamp: {s.get('text','')}")

    elif vtype == "annotated_page":
        for sec in spec.get("sections", [])[:3]:
            parts.append(
                f"  Section '{sec.get('heading','')}': "
                f"{str(sec.get('body',''))[:100]}"
            )
        parts.append(f"Annotator: {spec.get('annotator', '')}")
        for a in spec.get("annotations", []):
            parts.append(
                f"  Annotation ({a.get('type','')}): "
                f"target='{a.get('target_hint','')}' "
                f"note='{a.get('note_text','')}'"
            )

    content_summary = "\n".join(parts)

    prompt = (
        f"You are generating evaluation QnA pairs for a document extraction system.\n"
        f"Topic domain: {topic.description}\n\n"
        f"Scanned page content:\n{content_summary}\n\n"
        f"Generate exactly {n} question-answer pairs about this scanned page.\n"
        f"Rules:\n"
        f"- Every question MUST embed the page title '{caption}' as an explicit anchor "
        f"  (e.g. 'In the scanned document titled \"{caption}\"...' or "
        f"  'According to \"{caption}\"...').\n"
        f"- Questions must be answerable directly from the content above — "
        f"  no external knowledge required.\n"
        f"- Vary question types: factual lookup, count, value extraction, "
        f"  status check, identification.\n"
        f"- No two questions should cover the same fact.\n"
        f"- Answers must be concise (1-2 sentences or a short phrase).\n"
        f"Return a JSON object with key \"qa\" containing an array of {n} objects, "
        f"each with:\n"
        f"  \"question\": string,\n"
        f"  \"answer\": string\n"
        f"Return ONLY the JSON object. No markdown, no commentary."
    )

    try:
        raw  = client.complete(prompt, json_mode=True, tier="easy",
                               _label="llm_scanned_qa")
        data = json.loads(raw) if isinstance(raw, str) else raw
        result: list[dict[str, Any]] = []
        for item in data.get("qa", []):
            if isinstance(item, dict) and item.get("question") and item.get("answer"):
                ans = str(item["answer"]).strip()
                result.append({
                    "question": str(item["question"]).strip(),
                    "answer":   ans,
                    "assertions": [{
                        "text":   f"The response must state '{ans}'.",
                        "level":  "expected",
                        "metric": "Accuracy",
                    }],
                })
        return result
    except Exception as exc:
        _log.getLogger(__name__).warning("LLM scanned QnA failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public API: LLM-driven scanned page generator
# ---------------------------------------------------------------------------

def generate_llm_scanned_variants(
    client: Any,
    topic: Any,
    rng: "np.random.Generator",
    count: int,
    assets_dir: "Path | None" = None,
    subtypes: "list[str] | None" = None,
    doc_brief: "dict[str, Any] | None" = None,
    progress_cb: "Any | None" = None,
) -> "list[Variant]":
    """Generate *count* fully LLM-driven scanned page variants.

    Flow per variant:
      1. LLM generates a content spec JSON (variant-specific schema)
      2. Deterministic Python renders the scanned page image from the spec
      3. LLM generates non-overlapping QnA from the spec + ground truth

    Content, look-and-feel, and QnA are all generated on the fly by the LLM.
    No static templates are used.  Falls back to static specs when client=None.

    Args:
        client:      AzureOpenAIClient. If None, static fallback specs are used.
        topic:       TopicSpec with .id and .description.
        rng:         Seeded numpy random generator.
        count:       Number of scanned variants to produce.
        assets_dir:  Output directory for PNG files.
        subtypes:    Optional allowed variant type list (config subtype names
                     or LLM variant type names).  Defaults to all 5 types.
        doc_brief:   Optional document-level narrative context dict.
        progress_cb: Optional callable(msg: str) for per-variant progress logging.

    Returns:
        List of Variant objects.
    """
    import logging as _log
    import time as _t

    out_dir = (assets_dir or Path(".")).resolve()

    # Resolve subtype names -> LLM variant type names
    active_types: list[str]
    if subtypes:
        active_types = [
            _SCANNED_SUBTYPE_MAP.get(t, t) for t in subtypes
        ]
        active_types = [t for t in active_types if t in _SCANNED_LLM_VARIANT_TYPES]
    else:
        active_types = list(_SCANNED_LLM_VARIANT_TYPES)

    if not active_types:
        active_types = list(_SCANNED_LLM_VARIANT_TYPES)

    variants: list[Variant] = []

    for i in range(count):
        child_rng = fork_rng(rng, f"llm_scan_{i}")
        # Round-robin through active types so each variant is different
        vtype = active_types[i % len(active_types)]
        # Derive a unique variation token from the RNG so each LLM call
        # receives a distinct prompt — prevents response caching and
        # ensures content diversity across documents and variants.
        variation_token = int(child_rng.integers(0, 10_000_000))

        try:
            # Step 1: LLM content spec (falls back internally if LLM fails)
            _t0 = _t.perf_counter()
            spec = _llm_generate_scanned_spec(
                client, topic, vtype, child_rng, doc_brief,
                variation_index=variation_token,
            )
            t_spec = _t.perf_counter() - _t0

            # Step 2: Deterministic render + save
            _t0 = _t.perf_counter()
            img = _render_from_spec(spec, child_rng, out_dir)
            img_path = _save_pil(img, f"scan_llm_{vtype[:4]}", out_dir)
            t_render = _t.perf_counter() - _t0

            # Step 3: Structured ground truth
            gt = _build_scanned_gt(spec, img_path)
            gt["spec"] = spec   # preserve raw json_object for manifest

            # Step 4: LLM QnA
            _t0 = _t.perf_counter()
            qa = _llm_qa_for_scanned_page(client, topic, spec, gt, n=5)
            t_qa = _t.perf_counter() - _t0
            gt["qa_candidates"] = qa

            caption = gt.get("caption", f"Scanned {vtype.replace('_', ' ').title()} {i + 1}")
            quality = spec.get("quality_tier", "medium")

            variants.append(Variant(
                id=f"scan_llm_{i:04d}_{_uid()[:8]}",
                kind="image",
                metadata={
                    "subtype":      f"scanned_{vtype}",
                    "scan_quality": quality,
                    "variant_type": vtype,
                    "caption":      caption,
                },
                render_payload={
                    "image_path": img_path.resolve().as_posix(),
                    "alt_text":   caption,
                    "caption":    caption,
                    "format":     "png",
                },
                ground_truth=gt,
            ))

            if progress_cb is not None:
                _src = "llm" if client is not None else "fallback"
                progress_cb(
                    f"scanned [{i+1}/{count}] {vtype}: "
                    f"spec->{t_spec:.0f}s | render->{t_render:.0f}s | QA({len(qa)})->{t_qa:.0f}s ({_src})"
                )
            try:
                from synthpdf import stats_sink as _ss
                _ss.log_artifact(
                    type="scanned", index=i + 1, total=count,
                    subtype=vtype,
                    spec_time_s=round(t_spec, 1),
                    render_time_s=round(t_render, 1),
                    qa_count=len(qa), qa_time_s=round(t_qa, 1),
                    qa_source="llm" if client is not None else "fallback",
                )
            except Exception:
                pass

        except Exception as exc:
            _log.getLogger(__name__).warning(
                "LLM scanned variant %d (%s) failed: %s — skipping", i, vtype, exc
            )
            if progress_cb is not None:
                progress_cb(f"scanned [{i+1}/{count}] {vtype}: FAILED — skipped")

    return variants
