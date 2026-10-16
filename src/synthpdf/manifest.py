# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Manifest generation for synthpdf.

Per-document:  output/manifests/{doc_id}.json  — rich, self-sufficient JSON
               with a clear position -> ground-truth mapping for every element,
               per-section QA candidates, full text content, detailed image
               metadata, and a variants catalogue keyed by variant_id.

Run summary:   output/manifest.jsonl  — one JSON line per document with
               lightweight fields for easy bulk iteration.

Quest output:  output/quest/{doc_id}.xlsx  — per-document XLSX with two sheets:
                 QUERIES: Query, Query Segment, Query Author, Query Expected Answer,
                          Assertion, Assertion Level, Assertion Metric, Query Type,
                          File IDs
                 FILES:   File ID, File Name, SharePoint Path

               output/quest/{doc_id}.json  — per-document JSON query set in the
               standard format (query / refs / assertions / Id).

               output/quest/{doc_id}.tsv   — TSV with columns:
                 query, segment, annotation (JSON array), tags

               output/quest/{doc_id}.yml   — YAML list of
                 {query, assertions:[{level,text,metric}]}
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from synthpdf.types import DocumentSpec, Variant

# ---------------------------------------------------------------------------
# Excel cell sanitization
# ---------------------------------------------------------------------------

# XML 1.0 disallows these code-points in any context; openpyxl raises
# IllegalCharacterError when they appear in cell values.  Strip them so
# LLM-generated text (which may contain \x0b vertical-tab, \x0c form-feed,
# etc.) never crashes the XLSX writer.
_XML_ILLEGAL = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f"
    r"\ud800-\udfff\ufffe\uffff]"
)

def _xl_safe(value: Any) -> Any:
    """Strip XML-illegal characters from string values; pass non-strings through."""
    if isinstance(value, str):
        return _XML_ILLEGAL.sub("", value)
    return value

# ---------------------------------------------------------------------------
# Quest column constants
# ---------------------------------------------------------------------------

_QUEST_QUERIES_COLUMNS = [
    "Query",
    "Query Segment",
    "Query Author",
    "Query Expected Answer",
    "Assertion",
    "Assertion Level",
    "Assertion Metric",
    "Query Type",
    "File IDs",
]

_QUEST_FILES_COLUMNS = [
    "File ID",
    "File Name",
    "SharePoint Path",
]

# Query authors written into every row
_QUERY_AUTHORS = "synthpdf"

# Map section_type -> default Query_Segment and element_type
_STYPE_QUERY_SEGMENT: dict[str, str] = {
    "table":           "Table",
    "image":           "Image",
    "text":            "Text",
    "annotation":      "Text",
    "checklist":       "Form",
    "button":          "Form",
    "form":            "Form",
    "fillable_form":   "Form",
    "filled_form":     "Form",
    "filled_fillable": "Form",
}
_STYPE_ELEMENT_TYPE: dict[str, str] = {
    "table":           "table",
    "image":           "figure",
    "text":            "text_block",
    "annotation":      "annotation",
    "checklist":       "form",
    "button":          "form",
    "form":            "form",
    "fillable_form":   "form",
    "filled_form":     "form",
    "filled_fillable": "form",
}

_FORM_SUBTYPES = frozenset((
    "checklist", "button", "form", "fillable_form",
    "filled_form", "filled_fillable",
))

_SCANNED_SUBTYPES = frozenset((
    # Random-path legacy names
    "scanned_text", "scanned_table", "handwriting", "stamp_mark",
    # LLM-path names (metadata subtype = f"scanned_{vtype}")
    "scanned_text_document", "scanned_data_table",
    "scanned_handwritten_note", "scanned_stamped_form", "scanned_annotated_page",
))

_INFOGRAPHIC_SUBTYPES = frozenset((
    "stat_dashboard", "timeline", "comparison", "process_steps",
    "hub_spoke", "magazine", "pictograph", "floor_plan",
))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_structure(
    sections_gt: list[dict[str, Any]],
    variants_by_id: dict[str, Variant],
) -> list[dict[str, Any]]:
    """Convert sections_gt into a rich, self-contained structure list.

    Each entry describes one section in document order and contains all
    the data needed to evaluate that element without cross-referencing
    other parts of the manifest.
    """
    structure: list[dict[str, Any]] = []

    for sec in sections_gt:
        stype = sec["section_type"]
        position = sec["position"]
        variant_ids = sec.get("variant_ids", [])
        vid = variant_ids[0] if variant_ids else None
        v = variants_by_id.get(vid) if vid else None

        entry: dict[str, Any] = {
            "position": position,
            "type": stype,
        }
        if vid:
            entry["variant_id"] = vid

        # --- Table ---
        if stype == "table":
            entry["table_number"] = sec["table_number"]
            for gt in sec.get("ground_truth", []):
                entry["caption"] = gt.get("caption", f"Table {sec['table_number']}")
                entry["columns"] = gt.get("schema", [])
                entry["row_count"] = len(gt.get("cells", []))
                entry["cells"] = gt.get("cells", [])
                entry["derived_answers"] = gt.get("derived_answers", {})
                entry["qa_candidates"] = gt.get("qa_candidates", [])
                break

        # --- Image / Figure (chart, scanned, or infographic) ---
        elif stype == "image":
            entry["figure_number"] = sec["figure_number"]
            if v:
                entry["image_path"] = v.render_payload.get("image_path", "")
            v_subtype = (v.metadata.get("subtype", "") if v else "")

            if v_subtype in _SCANNED_SUBTYPES:
                # Scanned document sub-branch
                entry["subtype"] = v_subtype
                for gt in sec.get("ground_truth", []):
                    entry["caption"]      = gt.get("caption", f"Figure {sec['figure_number']}")
                    entry["scan_quality"] = gt.get("scan_quality", "")
                    # Raw json_object spec (stored by generate_llm_scanned_variants)
                    entry["spec"]         = gt.get("spec", {})

                    # Use variant_type from gt to sub-branch (LLM path stores this).
                    # Fall back to v_subtype for legacy random-path variants.
                    vt = gt.get("variant_type", "")

                    if vt == "text_document" or v_subtype == "scanned_text":
                        entry["columns"]       = gt.get("section_headings", [])
                        entry["cells"]         = gt.get("sections", [])
                        entry["full_text"]     = gt.get("full_text", "")
                        entry["section_count"] = len(gt.get("sections", []))

                    elif vt == "data_table" or v_subtype == "scanned_table":
                        entry["columns"]  = gt.get("headers", [])
                        entry["cells"]    = gt.get("rows", [])
                        entry["row_count"] = gt.get("row_count", 0)
                        entry["col_count"] = gt.get("col_count", 0)
                        entry["summary"]   = gt.get("summary_note", "")

                    elif vt == "handwritten_note" or v_subtype == "handwriting":
                        entry["columns"]      = ["line"]
                        entry["cells"]        = [[ln] for ln in gt.get("lines", [])]
                        entry["line_count"]   = gt.get("line_count", 0)
                        entry["hw_style"]     = gt.get("style", gt.get("hw_style", ""))
                        entry["has_signature"] = gt.get("has_signature", False)

                    elif vt == "stamped_form" or v_subtype == "stamp_mark":
                        entry["columns"] = ["label", "value"]
                        entry["cells"]   = [
                            [f.get("label", ""), f.get("value", "")]
                            for f in gt.get("form_fields", [])
                        ]
                        entry["stamps"]        = gt.get("stamps", [])
                        entry["inspector"]     = gt.get("inspector", "")
                        entry["reference"]     = gt.get("reference", "")
                        entry["checked_items"] = [
                            it for it in gt.get("checklist_items", [])
                            if it.get("checked")
                        ]

                    elif vt == "annotated_page":
                        sections = gt.get("sections", [])
                        entry["columns"]          = [s.get("heading", "") for s in sections]
                        entry["cells"]            = sections
                        entry["annotations"]      = gt.get("annotations", [])
                        entry["annotator"]        = gt.get("annotator", "")
                        entry["annotation_types"] = gt.get("annotation_types", [])

                    entry["qa_candidates"] = gt.get("qa_candidates", [])
                    break

            elif v_subtype in _INFOGRAPHIC_SUBTYPES:
                # Infographic sub-branch
                entry["subtype"] = v_subtype
                for gt in sec.get("ground_truth", []):
                    entry["caption"] = gt.get("caption", f"Figure {sec['figure_number']}")
                    entry["title"]   = gt.get("title", "")
                    entry["theme"]   = gt.get("theme", "")
                    if v_subtype == "stat_dashboard":
                        entry["subtitle"]     = gt.get("subtitle", "")
                        entry["period_badge"] = gt.get("period_badge", "")
                        entry["kpis"]         = gt.get("kpis", [])
                        entry["chart"]        = gt.get("chart", {})
                        entry["callouts"]     = gt.get("callouts", [])
                    elif v_subtype == "floor_plan":
                        entry["scale"]      = gt.get("scale", "")
                        entry["rooms"]      = gt.get("rooms", [])
                        entry["room_count"] = gt.get("room_count", 0)
                    elif v_subtype == "timeline":
                        entry["milestones"] = gt.get("milestones", [])
                    elif v_subtype == "comparison":
                        entry["options"] = gt.get("options", [])
                    elif v_subtype == "process_steps":
                        entry["steps"] = gt.get("steps", [])
                    elif v_subtype == "hub_spoke":
                        entry["center"] = gt.get("center", "")
                        entry["spokes"] = gt.get("spokes", [])
                    elif v_subtype == "magazine":
                        entry["hero_number"] = gt.get("hero_number", "")
                        entry["hero_label"]  = gt.get("hero_label", "")
                    elif v_subtype == "pictograph":
                        entry["unit_label"] = gt.get("unit_label", "")
                        entry["rows"]       = gt.get("rows", [])
                    entry["qa_candidates"] = gt.get("qa_candidates", [])
                    break

            else:
                # Regular chart / figure
                for gt in sec.get("ground_truth", []):
                    entry["caption"]      = gt.get("caption", f"Figure {sec['figure_number']}")
                    entry["chart_type"]   = gt.get("chart_type", "")
                    entry["chart_title"]  = gt.get("title", "")
                    entry["x_label"]      = gt.get("x_label", "")
                    entry["y_label"]      = gt.get("y_label", "")
                    entry["categories"]   = gt.get("categories", [])
                    entry["series_names"] = gt.get("series_names", [])
                    entry["value_range"]  = gt.get("value_range", [])
                    entry["trend"]        = gt.get("trend", "")
                    entry["chart_data"]   = gt.get("chart_data", {})
                    entry["qa_candidates"] = gt.get("qa_candidates", [])
                    break

        # --- Form subtypes ---
        elif stype in _FORM_SUBTYPES:
            for gt in sec.get("ground_truth", []):
                entry["title"]   = gt.get("title", "")
                entry["caption"] = gt.get("caption", entry["title"])
                if stype == "checklist":
                    entry["total_items"]         = gt.get("total_items", 0)
                    entry["checked_count"]        = gt.get("checked_count", 0)
                    entry["unchecked_count"]      = gt.get("unchecked_count", 0)
                    entry["completion_rate_pct"]  = gt.get("completion_rate_pct", 0.0)
                    entry["checked_items"]        = gt.get("checked_items", [])
                    entry["unchecked_items"]      = gt.get("unchecked_items", [])
                elif stype == "button":
                    entry["button_count"] = gt.get("button_count", 0)
                    entry["buttons"]      = gt.get("buttons", [])
                    entry["context"]      = gt.get("context", "")
                elif stype in ("form", "fillable_form", "filled_form", "filled_fillable"):
                    entry["section_count"]        = gt.get("section_count", 0)
                    entry["field_count"]          = gt.get("field_count", 0)
                    entry["required_field_count"] = gt.get("required_field_count", 0)
                    entry["sections"]             = gt.get("sections", [])
                    entry["required_fields"]      = gt.get("required_fields", [])
                    entry["description"]          = gt.get("description", "")
                    if stype in ("fillable_form", "filled_fillable"):
                        entry["form_number"]  = gt.get("form_number", "")
                        entry["instructions"] = gt.get("instructions", "")
                    if stype in ("filled_form", "filled_fillable"):
                        entry["filled"]      = True
                        entry["fill_style"]  = gt.get("fill_style", {})
                entry["qa_candidates"] = gt.get("qa_candidates", [])
                break

        # --- Text / Annotation ---
        else:
            for gt in sec.get("ground_truth", []):
                entry["heading"]      = gt.get("heading", "")
                entry["word_count"]   = gt.get("word_count", 0)
                raw_text = gt.get("text_content", "")
                entry["text_content"] = [p for p in raw_text.split("\n\n") if p.strip()] if raw_text else []
                entry["qa_candidates"] = gt.get("qa_candidates", [])
                break

        structure.append(entry)

    return structure


def _build_annotation_qa_candidates(
    inline_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build QA candidates from inline annotation specs (post-content pass).

    Each annotation type generates a distinct question/answer pair so
    the Quest file can evaluate whether a model correctly reads the
    visual annotations in the rendered PDF.
    """
    qa: list[dict[str, Any]] = []
    for spec in inline_annotations:
        ann_type = spec.get("type", "")
        text     = spec.get("text", "")
        extra    = spec.get("extra", {})

        # Build a semantic location string from section context when available,
        # falling back to page number so questions are always specific.
        ctx_type  = extra.get("context_type", "")
        ctx_label = extra.get("context_label", "")
        page      = spec.get("page")

        if ctx_type == "table" and ctx_label:
            loc = f" in {ctx_label}"
        elif ctx_type == "text" and ctx_label:
            loc = f" in the section titled \"{ctx_label}\""
        elif isinstance(page, int):
            loc = f" on page {page + 1}"
        else:
            loc = ""

        if ann_type == "highlight" and text:
            qa.append({
                "question":      f"What text is marked with a coloured highlight{loc}?",
                "answer":        text,
                "query_segment": "Annotation",
                "query_type":    "CIQ",
                "assertions": [{
                    "level":  "expected",
                    "metric": "Accuracy",
                    "text":   f"The response must mention the highlighted phrase: '{text}'.",
                }],
            })

        elif ann_type == "text_color" and text:
            qa.append({
                "question":      f"What text is rendered in a visually distinct colour (not the default black){loc}?",
                "answer":        text,
                "query_segment": "Annotation",
                "query_type":    "CIQ",
                "assertions": [{
                    "level":  "expected",
                    "metric": "Accuracy",
                    "text":   f"The response must mention the coloured phrase: '{text}'.",
                }],
            })

        elif ann_type == "circle" and text:
            qa.append({
                "question":      f"What text has a drawn oval or circle around it{loc}?",
                "answer":        text,
                "query_segment": "Annotation",
                "query_type":    "CIQ",
                "assertions": [{
                    "level":  "expected",
                    "metric": "Accuracy",
                    "text":   f"The response must identify the text enclosed by the drawn oval: '{text}'.",
                }],
            })

        elif ann_type == "margin_callout":
            comment = extra.get("comment", "")
            anchor  = text[:40] if text else ""
            if comment:
                q = (
                    f"A short note is written in the left margin of the document"
                    f" next to the underlined text '{anchor}'{loc}. What does that margin note say?"
                    if anchor else
                    f"A short note is written in the left margin of the document{loc}. What does it say?"
                )
                qa.append({
                    "question":      q,
                    "answer":        comment,
                    "query_segment": "Annotation",
                    "query_type":    "CIQ",
                    "assertions": [{
                        "level":  "expected",
                        "metric": "Accuracy",
                        "text":   f"The response must convey the margin note text: '{comment[:80]}'.",
                    }],
                })

        elif ann_type == "sticky_note":
            note = extra.get("note", text)
            if note:
                q = (
                    f"A sticky-note icon is attached near the text '{text[:40]}'{loc}."
                    f" What text does the sticky note contain?"
                    if text else
                    f"A sticky-note icon appears in the document{loc}. What text does it contain?"
                )
                qa.append({
                    "question":      q,
                    "answer":        note,
                    "query_segment": "Annotation",
                    "query_type":    "CIQ",
                    "assertions": [{
                        "level":  "expected",
                        "metric": "Accuracy",
                        "text":   f"The response must mention the sticky-note content: '{note[:80]}'.",
                    }],
                })

        elif ann_type == "score_callout":
            score   = extra.get("score", "")
            comment = extra.get("comment", "")
            if score:
                qa.append({
                    "question":      (
                        "Near the bottom of the last page, a numerical rating is shown "
                        "inside a drawn oval (e.g. 7/10). What is that rating?"
                    ),
                    "answer":        score,
                    "query_segment": "Annotation",
                    "query_type":    "CIQ",
                    "assertions": [{
                        "level":  "expected",
                        "metric": "Accuracy",
                        "text":   f"The response must state the numerical rating '{score}' shown in the oval badge.",
                    }],
                })
            if comment:
                qa.append({
                    "question":      (
                        "Near the bottom of the last page, a short feedback comment appears "
                        "alongside a numerical rating oval. What does that comment say?"
                    ),
                    "answer":        comment,
                    "query_segment": "Annotation",
                    "query_type":    "CIQ",
                    "assertions": [{
                        "level":  "expected",
                        "metric": "Accuracy",
                        "text":   f"The response must mention the feedback comment: '{comment}'.",
                    }],
                })

    return qa


def _build_variants_catalogue(variants_used: list[Variant]) -> dict[str, Any]:
    """Build a dict keyed by variant_id describing each variant's kind and metadata."""
    catalogue: dict[str, Any] = {}
    for v in variants_used:
        entry: dict[str, Any] = {
            "kind": v.kind,
            "perturbations": v.metadata.get("perturbations", []),
        }

        if v.kind == "text":
            entry["heading"]    = v.metadata.get("heading", "")
            entry["word_count"] = v.ground_truth.get("word_count", 0)
            entry["text_type"]  = v.metadata.get("text_type", "")
            entry["topic_id"]   = v.metadata.get("topic_id", "")

        elif v.kind == "table":
            entry["base_type"]  = v.metadata.get("base_type", "")
            entry["row_count"]  = v.metadata.get("row_count", 0)
            entry["col_count"]  = v.metadata.get("col_count", 0)
            entry["columns"]    = v.ground_truth.get("schema", [])
            entry["topic_id"]   = v.metadata.get("topic_id", "")

        elif v.kind == "image":
            entry["chart_type"]  = v.metadata.get("chart_type", "")
            entry["chart_title"] = v.ground_truth.get("title", "")
            entry["x_label"]     = v.ground_truth.get("x_label", "")
            entry["y_label"]     = v.ground_truth.get("y_label", "")
            entry["categories"]  = v.ground_truth.get("categories", [])
            entry["series_names"]= v.ground_truth.get("series_names", [])
            entry["value_range"] = v.ground_truth.get("value_range", [])
            entry["trend"]       = v.ground_truth.get("trend", "")
            entry["filename"]    = v.metadata.get("filename", "")
            entry["image_path"]  = v.render_payload.get("image_path", "")
            entry["topic_id"]    = v.metadata.get("topic_id", "")

        elif v.kind == "annotation":
            entry["text_type"] = v.metadata.get("text_type", "")

        catalogue[v.id] = entry

    return catalogue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_document_manifest(
    manifests_dir: Path,
    doc_spec: DocumentSpec,
    markdown_path: Path | None,
    pdf_path: Path,
    assets: list[Path],
    variants_used: list[Variant],
    ground_truth: dict[str, Any],
    render_metadata: dict[str, Any],
    inline_annotations: list[dict[str, Any]] | None = None,
    document_brief: dict[str, Any] | None = None,
) -> Path:
    """Write a rich per-document JSON manifest and return its path.

    Parameters
    ----------
    inline_annotations:
        Optional list of serialised ``AnnotationSpec`` dicts produced by the
        inline annotation pass.  When provided they are written to the
        ``inline_annotations`` key in the manifest and their QA candidates are
        merged into the top-level ``qa_candidates`` list.
    document_brief:
        Optional dict produced by ``ContentGenerator.generate_document_brief()``.
        Contains the coherence metadata that ties all artifacts together:
        ``narrative_theme``, per-section ``key_facts`` + ``data_references``,
        per-table ``key_data``, and per-chart axis/purpose specs.
        Stored verbatim under the ``document_brief`` key in the manifest.
    """
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifests_dir / f"{doc_spec.doc_id}.json"

    variants_by_id = {v.id: v for v in variants_used}
    sections_gt: list[dict[str, Any]] = ground_truth.get("sections", [])
    inline_ann: list[dict[str, Any]] = inline_annotations or []

    # Merge annotation QA into content QA, deduplicating by question text.
    # Content QA takes priority; annotation entries with the same question are dropped.
    # Also deduplicate within content_qa itself (same question may appear from multiple
    # sources: merge_facts, _llm_qa_for_table, template QA, etc.).
    raw_content_qa: list[dict[str, Any]] = ground_truth.get("all_qa_candidates", [])
    seen_q: set[str] = set()
    content_qa: list[dict[str, Any]] = []
    for qa in raw_content_qa:
        key = qa.get("question", "").strip().lower()
        if key and key not in seen_q:
            seen_q.add(key)
            content_qa.append(qa)
    ann_qa = [
        qa for qa in _build_annotation_qa_candidates(inline_ann)
        if qa.get("question", "").strip().lower() not in seen_q
    ]
    all_qa = content_qa + ann_qa

    manifest = {
        "doc_id": doc_spec.doc_id,
        "title": doc_spec.title or doc_spec.doc_id,
        "subtitle": doc_spec.subtitle or "",
        "topic_id": ground_truth.get("topic_id"),
        "recipe_name": doc_spec.recipe_name,
        "seed": doc_spec.seed,
        "document_brief": document_brief or {},
        "files": {
            "pdf": str(pdf_path),
            "markdown": str(markdown_path) if markdown_path is not None else None,
            "assets": [str(a) for a in assets],
        },
        "render": {
            "backend": render_metadata.get("backend", ""),
            "target_pages": render_metadata.get("target_pages"),
            "actual_page_count": render_metadata.get("page_count"),
        },
        "structure": _build_structure(sections_gt, variants_by_id),
        "variants": _build_variants_catalogue(variants_used),
        "inline_annotations": inline_ann,
        "qa_candidates": all_qa,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return out_path


def _annotation_quest_rows(
    doc_spec: DocumentSpec,
    pdf_path: Path,
    inline_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build flat quest rows (one per assertion) for inline annotations."""
    filename = pdf_path.name
    data_source = doc_spec.recipe_name or "synthpdf"
    rows: list[dict[str, Any]] = []

    for qa in _build_annotation_qa_candidates(inline_annotations):
        question = qa.get("question", "")
        if not question:
            continue
        for a in qa.get("assertions", []):
            text = a.get("text", "")
            if not text:
                continue
            rows.append({
                "Query":                  question,
                "Query Segment":          qa.get("query_segment", "Annotation"),
                "Query Author":           _QUERY_AUTHORS,
                "Query Expected Answer":  qa.get("answer", ""),
                "Assertion":              text,
                "Assertion Level":        a.get("level", "expected"),
                "Assertion Metric":       a.get("metric", "Accuracy"),
                "Query Type":             qa.get("query_type", "CIQ"),
                "_filename":              filename,
                "_variant_id":            "",
                "_element_type":          "annotation",
                "_data_source":           data_source,
            })

    return rows


def _quest_rows_for_doc(
    doc_spec: DocumentSpec,
    pdf_path: Path,
    sections_gt: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build flat quest rows (one per assertion) for a single document.

    Each row includes all fields for both the QUERIES Excel sheet and the
    JSON output (variant_id, element_type, data_source).
    """
    filename = pdf_path.name
    data_source = doc_spec.recipe_name or "synthpdf"

    rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()  # dedup by normalised question text across all sections

    for sec in sections_gt:
        stype = sec.get("section_type", "text")
        variant_ids = sec.get("variant_ids", [])
        variant_id = variant_ids[0] if variant_ids else ""

        query_segment_default = _STYPE_QUERY_SEGMENT.get(stype, "Text")
        element_type = _STYPE_ELEMENT_TYPE.get(stype, "text_block")

        for gt in sec.get("ground_truth", []):
            is_complex_table = bool(gt.get("complex_type") or gt.get("multipage"))
            if stype == "table" and is_complex_table:
                query_segment_default = "Complex Table"

            for qa in gt.get("qa_candidates", []):
                question = qa.get("question", "")
                if not question:
                    continue
                q_key = question.strip().lower()
                if q_key in seen_questions:
                    continue
                seen_questions.add(q_key)

                query_segment = qa.get("query_segment", query_segment_default)
                query_type    = qa.get("query_type", "CIQ")

                assertions_raw = qa.get("assertions", [])
                if not assertions_raw:
                    answer = qa.get("answer", "")
                    if answer:
                        assertions_raw = [{
                            "level": "expected",
                            "metric": "Accuracy",
                            "text": f"The response must accurately answer: {answer}",
                        }]

                for a in assertions_raw:
                    text = a.get("text", "")
                    if not text:
                        continue
                    rows.append({
                        # Excel QUERIES sheet fields
                        "Query":                  question,
                        "Query Segment":          query_segment,
                        "Query Author":           _QUERY_AUTHORS,
                        "Query Expected Answer":  qa.get("answer", ""),
                        "Assertion":              text,
                        "Assertion Level":        a.get("level", "expected"),
                        "Assertion Metric":       a.get("metric", "Accuracy"),
                        "Query Type":             query_type,
                        # JSON / lookup fields (not written to Excel directly)
                        "_filename":              filename,
                        "_variant_id":            variant_id,
                        "_element_type":          element_type,
                        "_data_source":           data_source,
                    })

    return rows


def _write_quest_xlsx(
    quest_dir: Path,
    doc_spec: DocumentSpec,
    pdf_path: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write the two-sheet Quest Excel file and return its path."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment

    quest_dir.mkdir(parents=True, exist_ok=True)
    out_path = quest_dir / f"{doc_spec.doc_id}.xlsx"

    wb = openpyxl.Workbook()

    # ---- Sheet 1: QUERIES ----
    ws_q = wb.active
    ws_q.title = "QUERIES"

    header_font  = Font(bold=True, color="FFFFFF")
    header_fill  = PatternFill("solid", fgColor="1F4E79")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    wrap         = Alignment(vertical="top", wrap_text=True)

    for col_idx, col_name in enumerate(_QUEST_QUERIES_COLUMNS, start=1):
        cell = ws_q.cell(row=1, column=col_idx, value=col_name)
        cell.font  = header_font
        cell.fill  = header_fill
        cell.alignment = header_align

    file_id = 1  # single document per file -> always File ID 1
    for row_data in rows:
        ws_q.append([
            _xl_safe(row_data.get("Query", "")),
            _xl_safe(row_data.get("Query Segment", "")),
            _xl_safe(row_data.get("Query Author", "")),
            _xl_safe(row_data.get("Query Expected Answer", "")),
            _xl_safe(row_data.get("Assertion", "")),
            _xl_safe(row_data.get("Assertion Level", "")),
            _xl_safe(row_data.get("Assertion Metric", "")),
            _xl_safe(row_data.get("Query Type", "")),
            file_id,
        ])
        for cell in ws_q[ws_q.max_row]:
            cell.alignment = wrap

    # Auto-size QUERIES columns (cap at 80)
    for col in ws_q.columns:
        max_len = max((len(str(c.value or "")) for c in col), default=10)
        ws_q.column_dimensions[col[0].column_letter].width = min(max_len + 2, 80)
    ws_q.freeze_panes = "A2"

    # ---- Sheet 2: FILES ----
    ws_f = wb.create_sheet(title="FILES")

    for col_idx, col_name in enumerate(_QUEST_FILES_COLUMNS, start=1):
        cell = ws_f.cell(row=1, column=col_idx, value=col_name)
        cell.font  = header_font
        cell.fill  = header_fill
        cell.alignment = header_align

    ws_f.append([file_id, pdf_path.name, ""])
    for cell in ws_f[ws_f.max_row]:
        cell.alignment = wrap

    for col in ws_f.columns:
        max_len = max((len(str(c.value or "")) for c in col), default=10)
        ws_f.column_dimensions[col[0].column_letter].width = min(max_len + 2, 80)
    ws_f.freeze_panes = "A2"

    wb.save(str(out_path))
    return out_path


def _write_quest_json(
    quest_dir: Path,
    doc_spec: DocumentSpec,
    pdf_path: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write the per-document JSON query set and return its path."""
    quest_dir.mkdir(parents=True, exist_ok=True)
    out_path = quest_dir / f"{doc_spec.doc_id}.json"

    # Group rows by (filename, query) preserving insertion order
    groups: OrderedDict[tuple[str, str], list[dict]] = OrderedDict()
    for row in rows:
        key = (row["_filename"], row["Query"])
        groups.setdefault(key, []).append(row)

    items: list[dict[str, Any]] = []
    for q_num, ((filename, query), group_rows) in enumerate(groups.items(), start=1):
        first = group_rows[0]

        ref = {
            "comment":      "",
            "filePath":     filename,
            "debugInfo":    "",
            "element_type": first["_element_type"],
            "Artifact_ID":  first["_variant_id"],
            "data_source":  first["_data_source"],
        }

        assertions: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in group_rows:
            level = row.get("Assertion Level", "expected")
            text  = row.get("Assertion", "")
            if not text:
                continue
            dedup_key = (level, text)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            assertions.append({
                "level": level,
                "owner": _QUERY_AUTHORS,
                "text":  text,
            })

        items.append({
            "query":      query,
            "refs":       [ref],
            "assertions": assertions,
            "Id":         f"Q{q_num}",
        })

    out_path.write_text(
        json.dumps(items, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def _spo_id_from_filename(filename: str) -> str:
    """Generate a stable SPO-style file ID from a filename."""
    raw = hashlib.sha256(filename.encode()).digest()[:18]
    return "SPO_" + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _write_quest_tsv(
    quest_dir: Path,
    doc_spec: DocumentSpec,
    rows: list[dict[str, Any]],
) -> Path:
    """Write the per-document TSV query file and return its path.

    Columns: query, segment, annotation, tags
    One row per unique query (not per assertion).  The annotation cell is a
    JSON array (double-quoted inside TSV) referencing the companion XLSX file.
    """
    import csv

    quest_dir.mkdir(parents=True, exist_ok=True)
    out_path = quest_dir / f"{doc_spec.doc_id}.tsv"

    xlsx_filename = f"{doc_spec.doc_id}.xlsx"
    annotation_json = json.dumps([{
        "Id":                   _spo_id_from_filename(xlsx_filename),
        "QueryAnnotationType":  "File",
        "Text":                 xlsx_filename,
        "Type":                 "LocalFile",
    }])

    # One row per unique query — preserve first-seen order
    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        q = row.get("Query", "")
        if q and q not in seen:
            seen.add(q)
            unique_rows.append(row)

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["query", "segment", "annotation", "tags"])
        for row in unique_rows:
            writer.writerow([
                row.get("Query", ""),
                row.get("Query Segment", ""),
                annotation_json,
                row.get("Query Type", "CIQ"),
            ])

    return out_path


def _write_quest_yml(
    quest_dir: Path,
    doc_spec: DocumentSpec,
    rows: list[dict[str, Any]],
) -> Path:
    """Write the per-document YAML query file and return its path.

    Each entry: query text + list of assertions (level / text / metric).
    One entry per unique query; assertions are deduplicated within each query.
    """
    quest_dir.mkdir(parents=True, exist_ok=True)
    out_path = quest_dir / f"{doc_spec.doc_id}.yml"

    # Group assertions by query, preserving insertion order
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        q = row.get("Query", "")
        if q:
            groups.setdefault(q, []).append(row)

    items: list[dict[str, Any]] = []
    for query, group_rows in groups.items():
        assertions: list[dict[str, Any]] = []
        seen_a: set[tuple[str, str]] = set()
        for row in group_rows:
            level  = row.get("Assertion Level", "expected")
            text   = row.get("Assertion", "")
            metric = row.get("Assertion Metric", "Accuracy")
            if not text:
                continue
            key = (level, text)
            if key in seen_a:
                continue
            seen_a.add(key)
            assertions.append({"level": level, "text": text, "metric": metric})
        items.append({"query": query, "assertions": assertions})

    try:
        import yaml  # PyYAML
        with open(out_path, "w", encoding="utf-8") as fh:
            yaml.dump(
                items, fh,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
    except ImportError:
        # Fallback: minimal hand-rolled YAML (no block-scalar wrapping)
        def _esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"')

        lines: list[str] = []
        for item in items:
            lines.append(f'- query: "{_esc(item["query"])}"')
            lines.append("  assertions:")
            for a in item["assertions"]:
                lines.append(f'  - level: {a["level"]}')
                lines.append(f'    text: "{_esc(a["text"])}"')
                lines.append(f'    metric: {a["metric"]}')
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out_path


def write_quest_file(
    quest_dir: Path,
    doc_spec: DocumentSpec,
    pdf_path: Path,
    sections_gt: list[dict[str, Any]],
    inline_annotations: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Write the per-document Quest XLSX, JSON, TSV, and YML files.

    Creates:
      quest_dir/{doc_id}.xlsx  -- two-sheet workbook (QUERIES + FILES)
      quest_dir/{doc_id}.json  -- JSON query set (query / refs / assertions / Id)
      quest_dir/{doc_id}.tsv   -- TSV query set (query / segment / annotation / tags)
      quest_dir/{doc_id}.yml   -- YAML query set (query + assertions list)

    Returns (xlsx_path, json_path, tsv_path, yml_path).
    """
    rows = _quest_rows_for_doc(doc_spec, pdf_path, sections_gt)
    if inline_annotations:
        rows += _annotation_quest_rows(doc_spec, pdf_path, inline_annotations)
    xlsx_path = _write_quest_xlsx(quest_dir, doc_spec, pdf_path, rows)
    json_path = _write_quest_json(quest_dir, doc_spec, pdf_path, rows)
    tsv_path  = _write_quest_tsv(quest_dir, doc_spec, rows)
    yml_path  = _write_quest_yml(quest_dir, doc_spec, rows)
    return xlsx_path, json_path, tsv_path, yml_path


# ---------------------------------------------------------------------------
# Backward-compat alias (kept so external scripts importing the old name
# continue to work; new code should call write_quest_file directly).
# ---------------------------------------------------------------------------

def write_seval_file(
    seval_dir: Path,
    doc_spec: DocumentSpec,
    pdf_path: Path,
    sections_gt: list[dict[str, Any]],
) -> Path:
    """Deprecated alias for write_quest_file. Writes to quest/ directory."""
    xlsx_path, *_ = write_quest_file(seval_dir, doc_spec, pdf_path, sections_gt)
    return xlsx_path


def consolidate_seval_xlsx(seval_dir: Path) -> None:
    """No-op. Consolidated SEVAL files are no longer generated."""
    return None


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def append_run_summary(
    summary_path: Path,
    doc_spec: DocumentSpec,
    pdf_path: Path,
    manifest_path: Path,
    render_metadata: dict[str, Any],
    topic_id: str | None,
) -> None:
    """Append a single lightweight line to the run-level manifest.jsonl."""
    entry = {
        "doc_id": doc_spec.doc_id,
        "title": doc_spec.title or doc_spec.doc_id,
        "topic_id": topic_id,
        "recipe_name": doc_spec.recipe_name,
        "pdf_path": str(pdf_path),
        "manifest_path": str(manifest_path),
        "target_pages": render_metadata.get("target_pages"),
        "actual_page_count": render_metadata.get("page_count"),
        "n_tables": render_metadata.get("n_tables", 0),
        "n_figures": render_metadata.get("n_figures", 0),
        "n_text_sections": render_metadata.get("n_text", 0),
    }
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_variants_catalog(
    out_path: Path,
    all_variants: list[Variant],
) -> None:
    """Write the full variant catalog as a JSON file."""
    catalog = {
        "total": len(all_variants),
        "variants": [v.to_dict() for v in all_variants],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2, ensure_ascii=False)
