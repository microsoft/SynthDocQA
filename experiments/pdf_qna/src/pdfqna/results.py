# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
Results persistence - per-doc JSON saves and aggregated Excel/JSON summary.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill

    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False
    logger.warning("openpyxl not installed; Excel output will be skipped")

# Verdict -> background fill colour (ARGB hex, no alpha prefix)
_VERDICT_FILL = {
    "PASS": "C6EFCE",
    "PARTIAL": "FFEB9C",
    "FAIL": "FFC7CE",
    "ERROR": "D9D9D9",
    "SKIP": "BDD7EE",   # light blue — rate-limited, will be retried
}

# Quality -> numeric score for aggregate statistics
_QUALITY_SCORE = {"excellent": 1.0, "good": 0.75, "fair": 0.5, "poor": 0.25, "error": 0.0, "skip": 0.0}


# ---------------------------------------------------------------------------
# Per-doc persistence
# ---------------------------------------------------------------------------


def load_all_per_doc(per_doc_dir: Path) -> list[dict[str, Any]]:
    """
    Load all per-doc results from .jsonl files in per_doc_dir.

    Each .jsonl file holds one line per question result.  Lines are grouped
    by (config, doc_id) and reconstructed into the standard per-doc structure
    expected by write_summary().
    """
    doc_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    doc_meta: dict[tuple[str, str], dict[str, str]] = {}

    for p in sorted(per_doc_dir.glob("*.jsonl")):
        for raw in p.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                key = (entry["config"], entry["doc_id"])
                doc_groups[key].append(
                    {
                        "question_id": entry.get("question_id", ""),
                        "question": entry.get("question", ""),
                        "element_type": entry.get("element_type", "unknown"),
                        "assertion": entry.get("assertion", ""),
                        "model_results": entry.get("model_results", {}),
                    }
                )
                doc_meta[key] = {
                    "config": entry["config"],
                    "doc_id": entry["doc_id"],
                    "md_path": entry.get("md_path", ""),
                    "markdown_source": entry.get("markdown_source", "dataset"),
                }
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Could not parse line in %s: %s", p.name, exc)

    return [
        {**doc_meta[key], "qa_results": qa_results}
        for key, qa_results in doc_groups.items()
    ]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_summary(
    all_results: list[dict[str, Any]],
    results_dir: Path,
    model_names: list[str],
) -> None:
    """Write summary.json and summary.xlsx from all per-doc results."""
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = _flatten_rows(all_results)
    agg = _aggregate(rows, model_names)

    # JSON summary
    json_path = results_dir / "summary.json"
    json_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    # Excel summary
    if _HAS_OPENPYXL:
        _write_excel(agg, rows, results_dir / "summary.xlsx", model_names)
    else:
        logger.warning("Skipping Excel output (openpyxl not installed)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_rows(all_results: list[dict]) -> list[dict]:
    """Flatten into one row per (doc, question, prediction model, judge)."""
    rows: list[dict] = []
    for doc in all_results:
        config = doc.get("config", "")
        doc_id = doc.get("doc_id", "")
        for qa in doc.get("qa_results", []):
            question = qa.get("question", "")
            q_id = qa.get("question_id", "")
            element_type = qa.get("element_type", "unknown")
            assertion = qa.get("assertion", "")
            for model_name, mr in qa.get("model_results", {}).items():
                judges = mr.get("judges")
                if not judges:
                    judges = {"judge_1": mr.get("judge", {})}
                for judge_name, judge in judges.items():
                    verdict = judge.get("verdict", "ERROR")
                    rows.append({
                        "config": config,
                        "doc_id": doc_id,
                        "question_id": q_id,
                        "question": question,
                        "element_type": element_type,
                        "assertion": assertion,
                        "model": model_name,
                        "judge": judge_name,
                        "judge_deployment": judge.get("judge_deployment", ""),
                        "predicted": mr.get("predicted", ""),
                        "response_time": mr.get("response_time", 0.0),
                        "prompt_tokens": mr.get("prompt_tokens", 0),
                        "completion_tokens": mr.get("completion_tokens", 0),
                        "qa_error": mr.get("error") or "",
                        "model_refuse": mr.get("model_refuse", False),
                        "skip_reason": mr.get("skip_reason", ""),
                        "verdict": verdict,
                        "confidence": judge.get("confidence", 0.0),
                        "quality": judge.get("quality", "error"),
                        "quality_score": _QUALITY_SCORE.get(
                            judge.get("quality", "error"), 0.0
                        ),
                        "reasoning": judge.get("reasoning", ""),
                        "missing_aspects": "; ".join(
                            judge.get("missing_aspects", [])
                        ),
                        "judge_error": judge.get("error") or "",
                        "judge_time": judge.get("judge_time", 0.0),
                    })
    return rows


def _aggregate(rows: list[dict], model_names: list[str]) -> dict:
    """Build aggregated pass-rate stats by model, config, and element_type."""

    def _counter():
        return {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0, "SKIP": 0,
                "total": 0, "conf_sum": 0.0, "qual_sum": 0.0}

    by_model: dict[str, dict] = defaultdict(_counter)
    by_judge_model: dict[str, dict] = defaultdict(_counter)
    by_config_model: dict[str, dict] = defaultdict(_counter)
    by_etype_model: dict[str, dict] = defaultdict(_counter)

    for r in rows:
        m = r["model"]
        v = r["verdict"] if r["verdict"] in ("PASS", "PARTIAL", "FAIL", "SKIP") else "ERROR"
        cfg = r["config"]
        et = r["element_type"]
        judge = r["judge"]

        for d in [
            by_model[m],
            by_judge_model[f"{judge}::{m}"],
            by_config_model[f"{cfg}::{m}"],
            by_etype_model[f"{et}::{m}"],
        ]:
            d[v] += 1
            d["total"] += 1
            d["conf_sum"] += r["confidence"]
            d["qual_sum"] += r["quality_score"]

    def _summarise(d: dict) -> dict:
        total = d["total"] or 1
        pass_rate = round((d["PASS"] + 0.5 * d["PARTIAL"]) / total, 4)
        return {
            "total": d["total"],
            "PASS": d["PASS"],
            "PARTIAL": d["PARTIAL"],
            "FAIL": d["FAIL"],
            "ERROR": d["ERROR"],
            "SKIP": d["SKIP"],
            "pass_rate": pass_rate,
            "avg_confidence": round(d["conf_sum"] / total, 4),
            "avg_quality_score": round(d["qual_sum"] / total, 4),
        }

    return {
        "total_rows": len(rows),
        "by_model": {k: _summarise(v) for k, v in by_model.items()},
        "by_judge_model": {k: _summarise(v) for k, v in by_judge_model.items()},
        "by_config_model": {k: _summarise(v) for k, v in by_config_model.items()},
        "by_element_type_model": {k: _summarise(v) for k, v in by_etype_model.items()},
    }


def _write_excel(
    agg: dict,
    rows: list[dict],
    path: Path,
    model_names: list[str],
) -> None:
    """Write a four-sheet Excel workbook."""
    wb = openpyxl.Workbook()

    # ---- Sheet 1: Model Summary ----
    ws1 = wb.active
    ws1.title = "Model Summary"
    _header(ws1, ["Model", "Total", "PASS", "PARTIAL", "FAIL", "ERROR", "SKIP",
                   "Pass Rate", "Avg Confidence", "Avg Quality Score"])
    for model, s in sorted(agg["by_model"].items()):
        ws1.append([model, s["total"], s["PASS"], s["PARTIAL"], s["FAIL"],
                    s["ERROR"], s["SKIP"], s["pass_rate"], s["avg_confidence"],
                    s["avg_quality_score"]])

    # ---- Sheet 2: By Judge and Model ----
    ws2 = wb.create_sheet("By Judge and Model")
    _header(ws2, ["Judge", "Model", "Total", "PASS", "PARTIAL", "FAIL",
                  "ERROR", "SKIP", "Pass Rate", "Avg Confidence", "Avg Quality Score"])
    for key, s in sorted(agg["by_judge_model"].items()):
        judge, model = key.split("::", 1)
        ws2.append([judge, model, s["total"], s["PASS"], s["PARTIAL"], s["FAIL"],
                    s["ERROR"], s["SKIP"], s["pass_rate"], s["avg_confidence"],
                    s["avg_quality_score"]])

    # ---- Sheet 3: By Config ----
    ws3 = wb.create_sheet("By Config")
    _header(ws3, ["Config", "Model", "Total", "PASS", "PARTIAL", "FAIL",
                   "ERROR", "SKIP", "Pass Rate", "Avg Confidence", "Avg Quality Score"])
    for key, s in sorted(agg["by_config_model"].items()):
        cfg, model = key.split("::", 1)
        ws3.append([cfg, model, s["total"], s["PASS"], s["PARTIAL"], s["FAIL"],
                    s["ERROR"], s["SKIP"], s["pass_rate"], s["avg_confidence"],
                    s["avg_quality_score"]])

    # ---- Sheet 4: By Element Type ----
    ws4 = wb.create_sheet("By Element Type")
    _header(ws4, ["Element Type", "Model", "Total", "PASS", "PARTIAL", "FAIL",
                   "ERROR", "SKIP", "Pass Rate", "Avg Confidence", "Avg Quality Score"])
    for key, s in sorted(agg["by_element_type_model"].items()):
        et, model = key.split("::", 1)
        ws4.append([et, model, s["total"], s["PASS"], s["PARTIAL"], s["FAIL"],
                    s["ERROR"], s["SKIP"], s["pass_rate"], s["avg_confidence"],
                    s["avg_quality_score"]])

    # ---- Sheet 5: QA Details ----
    ws5 = wb.create_sheet("QA Details")
    detail_cols = [
        "Config", "Doc ID", "Q ID", "Question", "Element Type", "Model",
        "Judge", "Judge Deployment", "Predicted Answer", "Assertion", "Verdict", "Confidence",
        "Quality", "Reasoning", "Missing Aspects",
        "Response Time (s)", "Prompt Tokens", "Completion Tokens",
        "Judge Time (s)", "QA Error", "Judge Error", "Model Refuse", "Skip Reason",
    ]
    _header(ws5, detail_cols)
    verdict_col_idx = detail_cols.index("Verdict") + 1  # 1-based

    for r in rows:
        ws5.append([
            r["config"], r["doc_id"], r["question_id"], r["question"],
            r["element_type"], r["model"], r["judge"], r["judge_deployment"],
            r["predicted"], r["assertion"],
            r["verdict"], r["confidence"], r["quality"], r["reasoning"],
            r["missing_aspects"], r["response_time"], r["prompt_tokens"],
            r["completion_tokens"], r["judge_time"], r["qa_error"], r["judge_error"],
            r.get("model_refuse", False), r.get("skip_reason", ""),
        ])
        # Colour-code the Verdict cell
        cell = ws5.cell(row=ws5.max_row, column=verdict_col_idx)
        hex_color = _VERDICT_FILL.get(r["verdict"], "D9D9D9")
        cell.fill = PatternFill(
            start_color=hex_color, end_color=hex_color, fill_type="solid"
        )

    # Auto-width columns (capped at 60)
    for ws in (ws1, ws2, ws3, ws4, ws5):
        for col in ws.columns:
            width = max((len(str(cell.value or "")) for cell in col), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 60)

    wb.save(path)
    logger.info("Wrote Excel report to %s", path)


def _header(ws: Any, cols: list[str]) -> None:
    """Write a bold blue header row."""
    ws.append(cols)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")
