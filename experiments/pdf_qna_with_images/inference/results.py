# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Results serialization for the inference pipeline.

Writes:
  per_doc/{config}__{doc_id}.json    -- one file per document
  summary.json                        -- aggregated pass rates per model
  summary.xlsx                        -- Excel with Summary + QA Details sheets
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-doc save
# ---------------------------------------------------------------------------

def save_per_doc(result: dict, per_doc_dir: Path) -> None:
    """Write a single document result as JSON (incremental save)."""
    per_doc_dir.mkdir(parents=True, exist_ok=True)
    key = f"{result['config']}__{result['doc_id']}"
    path = per_doc_dir / f"{key}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(all_results: list[dict], model_names: list[str]) -> dict[str, dict]:
    """Compute pass/fail/error counts and pass rate per model + per element_type."""
    totals: dict[str, dict] = {
        name: {"pass": 0, "fail": 0, "error": 0, "unclear": 0, "total": 0}
        for name in model_names
    }
    by_type: dict[str, dict[str, dict]] = {}  # model -> element_type -> stats

    for doc in all_results:
        for item in doc.get("qa_items", []):
            etype = item.get("element_type", "unknown")
            for model_name, judgment in item.get("judgments", {}).items():
                verdict = judgment.get("result", "ERROR").upper()
                if model_name not in totals:
                    totals[model_name] = {"pass": 0, "fail": 0, "error": 0, "unclear": 0, "total": 0}
                totals[model_name]["total"] += 1
                if verdict == "PASS":
                    totals[model_name]["pass"] += 1
                elif verdict == "FAIL":
                    totals[model_name]["fail"] += 1
                elif verdict == "UNCLEAR":
                    totals[model_name]["unclear"] += 1
                else:
                    totals[model_name]["error"] += 1

                # Per element_type breakdown
                bt = by_type.setdefault(model_name, {})
                s = bt.setdefault(etype, {"pass": 0, "fail": 0, "error": 0, "unclear": 0, "total": 0})
                s["total"] += 1
                if verdict == "PASS":
                    s["pass"] += 1
                elif verdict == "FAIL":
                    s["fail"] += 1
                elif verdict == "UNCLEAR":
                    s["unclear"] += 1
                else:
                    s["error"] += 1

    # Calculate pass rates
    for name, s in totals.items():
        t = s["total"]
        s["pass_rate"] = round(s["pass"] / t, 4) if t else 0.0
    for model_bt in by_type.values():
        for s in model_bt.values():
            t = s["total"]
            s["pass_rate"] = round(s["pass"] / t, 4) if t else 0.0

    return {"totals": totals, "by_element_type": by_type}


# ---------------------------------------------------------------------------
# Summary write
# ---------------------------------------------------------------------------

def write_summary(
    all_results: list[dict],
    results_dir: Path,
    model_names: list[str],
) -> None:
    """Write summary.json and summary.xlsx to *results_dir*."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stats = _aggregate(all_results, model_names)

    summary = {
        "n_docs": len(all_results),
        "n_questions_total": sum(
            len(d.get("qa_items", [])) for d in all_results
        ),
        "model_names": model_names,
        "model_stats": stats["totals"],
        "model_stats_by_element_type": stats["by_element_type"],
    }

    (results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s", results_dir / "summary.json")

    _write_excel(all_results, model_names, stats, results_dir)


# ---------------------------------------------------------------------------
# Excel report
# ---------------------------------------------------------------------------

def _write_excel(
    all_results: list[dict],
    model_names: list[str],
    stats: dict,
    results_dir: Path,
) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        logger.warning("openpyxl not installed; skipping Excel report")
        return

    PASS_FILL = PatternFill("solid", fgColor="C6EFCE")   # green
    FAIL_FILL = PatternFill("solid", fgColor="FFC7CE")   # red
    UNCLEAR_FILL = PatternFill("solid", fgColor="FFEB9C") # yellow
    HEADER_FONT = Font(bold=True)

    wb = openpyxl.Workbook()

    # ----------------------------------------------------------------
    # Sheet 1: Summary (pass rates per model)
    # ----------------------------------------------------------------
    ws_sum = wb.active
    ws_sum.title = "Summary"
    hdr = ["Model", "Total Questions", "PASS", "FAIL", "UNCLEAR", "ERROR", "Pass Rate"]
    ws_sum.append(hdr)
    for cell in ws_sum[1]:
        cell.font = HEADER_FONT

    for name in model_names:
        s = stats["totals"].get(name, {})
        ws_sum.append([
            name,
            s.get("total", 0),
            s.get("pass", 0),
            s.get("fail", 0),
            s.get("unclear", 0),
            s.get("error", 0),
            f"{s.get('pass_rate', 0.0):.1%}",
        ])

    ws_sum.column_dimensions["A"].width = 28
    for col in "BCDEFG":
        ws_sum.column_dimensions[col].width = 18

    # ----------------------------------------------------------------
    # Sheet 2: By Element Type
    # ----------------------------------------------------------------
    ws_etype = wb.create_sheet("By Element Type")
    etype_hdr = ["Model", "Element Type", "Total", "PASS", "FAIL", "UNCLEAR", "ERROR", "Pass Rate"]
    ws_etype.append(etype_hdr)
    for cell in ws_etype[1]:
        cell.font = HEADER_FONT

    for model_name in model_names:
        bt = stats["by_element_type"].get(model_name, {})
        for etype, s in sorted(bt.items()):
            ws_etype.append([
                model_name,
                etype,
                s.get("total", 0),
                s.get("pass", 0),
                s.get("fail", 0),
                s.get("unclear", 0),
                s.get("error", 0),
                f"{s.get('pass_rate', 0.0):.1%}",
            ])

    ws_etype.column_dimensions["A"].width = 28
    ws_etype.column_dimensions["B"].width = 20

    # ----------------------------------------------------------------
    # Sheet 3: QA Details (one row per question per doc)
    # ----------------------------------------------------------------
    ws_qa = wb.create_sheet("QA Details")
    qa_hdr = ["Config", "Doc ID", "Q ID", "Element Type", "Question", "Assertion"]
    for m in model_names:
        qa_hdr += [f"{m} | Answer", f"{m} | Verdict", f"{m} | Reason"]
    ws_qa.append(qa_hdr)
    for cell in ws_qa[1]:
        cell.font = HEADER_FONT

    verdict_fills = {"PASS": PASS_FILL, "FAIL": FAIL_FILL, "UNCLEAR": UNCLEAR_FILL}

    for doc in all_results:
        cfg_name = doc.get("config", "")
        doc_id = doc.get("doc_id", "")
        for item in doc.get("qa_items", []):
            row: list[Any] = [
                cfg_name,
                doc_id,
                item.get("id", ""),
                item.get("element_type", ""),
                item.get("question", ""),
                item.get("assertion", ""),
            ]
            for m in model_names:
                ans = item.get("answers", {}).get(m, "")
                jud = item.get("judgments", {}).get(m, {})
                verdict = jud.get("result", "")
                reason = jud.get("reason", "")
                row += [ans, verdict, reason]
            ws_qa.append(row)

            # Colour verdict cells
            row_num = ws_qa.max_row
            base_col = 7  # column G is first model block
            for i, m in enumerate(model_names):
                verdict_col = base_col + i * 3 + 1  # +0=Answer, +1=Verdict, +2=Reason
                verdict = row[6 + i * 3 + 1]
                fill = verdict_fills.get(verdict)
                if fill:
                    ws_qa.cell(row=row_num, column=verdict_col).fill = fill

    # Freeze header row + auto-width for first 6 cols
    ws_qa.freeze_panes = "A2"
    for col_idx, width in enumerate([16, 22, 8, 16, 60, 60], start=1):
        ws_qa.column_dimensions[get_column_letter(col_idx)].width = width
    # Auto-width per-model answer columns
    base_col = 7
    for i in range(len(model_names)):
        ws_qa.column_dimensions[get_column_letter(base_col + i * 3)].width = 40     # Answer
        ws_qa.column_dimensions[get_column_letter(base_col + i * 3 + 1)].width = 10  # Verdict
        ws_qa.column_dimensions[get_column_letter(base_col + i * 3 + 2)].width = 40  # Reason

    path = results_dir / "summary.xlsx"
    wb.save(str(path))
    logger.info("Wrote %s", path)
