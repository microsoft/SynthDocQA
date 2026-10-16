# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
evaluate.py -- pdfplumber output evaluation against ground-truth manifests.

Computes the same metrics as docling/evaluate.py and mineru/evaluate.py so
results are directly comparable across parsers.  Only the data-extraction
layer differs:

  pdfplumber input   : <doc_id>/<doc_id>.json
  Tables             : content[i].tables[j].rows  — 2-D list of strings,
                       no HTML parsing required
  Text               : content[i].text concatenated across all pages
  Images             : NOT extracted by pdfplumber; image recall/F1 = 0
  Captions           : NOT associated with tables or images by pdfplumber;
                       caption NED metrics are suppressed (None)
  Reading order      : sequential index of tables across pages
                       (page order * position within page by bbox top coord)

Parser capability notes:
  pdfplumber is a rule-based text/table extractor with no ML models.
  It intentionally does not detect or extract image regions.

  - Image metrics (image_recall, image_f1, img_caption_*, img_ocr_*):
    all worst-case or None when GT has image elements.
  - Caption metrics (img_caption_ned, agg_header_ned_avg):
    pdfplumber emits no caption text alongside tables or images, so these
    are suppressed rather than penalised with artificial zeros.

Metrics (identical labels to docling/evaluate.py and mineru/evaluate.py):
  Table Detection     table_recall, table_precision, table_f1
  Table Content       agg_cell_ned_avg, agg_cell_ned_median, agg_cell_bow_f1
  Table Structure     agg_row_count_sim, agg_col_count_sim, agg_structure_sim
  Table Alignment     agg_header_ned_avg, alignment_score_avg
  Text Extraction     text_ned, text_wer, text_bow_f1
  Image / Figure      image_recall, image_precision, image_f1
                      img_caption_ned, img_caption_found
                      img_ocr_ned, img_ocr_bow_f1, img_ocr_wer, img_ocr_text_found
  Reading Order       reading_order_kendall_tau
  Coverage            element_recall

Dependencies: numpy, scipy

Usage:
    python evaluate.py \\
        --gt-dir   Q:/path/to/cfg_001_tables_basic_only/ \\
        --pred-dir Q:/path/to/pdfplumber/output/ \\
        [--output  results/pdfplumber_cfg001.json]
"""

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau


# ---------------------------------------------------------------------------
# String utilities  (identical to docling/evaluate.py and mineru/evaluate.py)
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Lowercase, strip basic markdown formatting, collapse whitespace."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", str(text))
    text = re.sub(r"\*(.+?)\*",     r"\1", text)
    text = re.sub(r"`(.+?)`",        r"\1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def ned(a: str, b: str) -> float:
    """Normalized Edit Distance in [0,1]. 0=identical, 1=completely different."""
    a, b = _clean(a), _clean(b)
    if not a and not b:
        return 0.0
    return round(1.0 - difflib.SequenceMatcher(None, a, b).ratio(), 6)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate = word-level edit distance / |ref tokens|."""
    r = _clean(ref).split()
    h = _clean(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return round(prev[-1] / len(r), 6)


def _count_sim(gt_n: int, pred_n: int):
    """
    Bounded count similarity in [0,1]: min(gt, pred) / max(gt, pred).
    Returns None when gt_n==0 (no ground truth to compare against).
    """
    if gt_n == 0 and pred_n == 0:
        return 1.0
    if gt_n == 0:
        return None
    if pred_n == 0:
        return 0.0
    return round(min(gt_n, pred_n) / max(gt_n, pred_n), 6)


def bow_f1(ref: str, hyp: str) -> float:
    """Bag-of-words token F1 between two strings."""
    rc = Counter(_clean(ref).split())
    hc = Counter(_clean(hyp).split())
    if not rc and not hc:
        return 1.0
    if not rc or not hc:
        return 0.0
    common = sum((rc & hc).values())
    prec = common / sum(hc.values())
    rec  = common / sum(rc.values())
    return round(2 * prec * rec / (prec + rec), 6) if (prec + rec) else 0.0


# ---------------------------------------------------------------------------
# GT manifest helpers  (identical to docling/evaluate.py and mineru/evaluate.py)
# ---------------------------------------------------------------------------

def load_gt(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gt_tables(m: dict) -> list:
    return [e for e in m.get("structure", []) if e["type"] == "table"]


def gt_texts(m: dict) -> list:
    return [e for e in m.get("structure", []) if e["type"] == "text"]


def gt_images(m: dict) -> list:
    return [e for e in m.get("structure", []) if e["type"] == "image"]


# ---------------------------------------------------------------------------
# pdfplumber JSON helpers
# ---------------------------------------------------------------------------

def load_pdfplumber(path: Path) -> dict:
    """Load a pdfplumber <doc_id>.json file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pdfplumber_tables(doc: dict) -> list:
    """
    Collect all table dicts across all pages in reading order.
    Injects '_idx' (global sequential index) for reading-order scoring and
    '_page' for diagnostics.

    Within each page, tables are already ordered by pdfplumber's bbox top
    coordinate (top-of-page = 0), giving natural reading order.
    """
    result = []
    idx = 0
    for page in doc.get("content", []):
        for tbl in page.get("tables", []):
            result.append({**tbl, "_idx": idx, "_page": page["page"]})
            idx += 1
    return result


def pdfplumber_body_texts(doc: dict) -> list:
    """
    Return one pseudo-element per page whose 'text' field is the full
    page text.  Matches the interface expected by eval_texts().
    pdfplumber does not separate body text from table text at extraction
    time; both are included in page['text'].
    """
    return [{"text": page.get("text", "")} for page in doc.get("content", [])]


def pdfplumber_pictures(doc: dict) -> list:
    """
    pdfplumber does not extract image regions.
    Always returns an empty list so image metrics correctly show 0 recall.
    """
    return []


def _pdfplumber_tbl_header_row(tbl: dict) -> list:
    """First row of a pdfplumber table (treated as column headers)."""
    rows = tbl.get("rows", [])
    return rows[0] if rows else []


def _pdfplumber_tbl_data_grid(tbl: dict) -> list:
    """Data rows (all rows after the first) from a pdfplumber table."""
    rows = tbl.get("rows", [])
    return rows[1:] if len(rows) > 1 else []


# ---------------------------------------------------------------------------
# GT image text  (identical to docling/evaluate.py and mineru/evaluate.py)
# ---------------------------------------------------------------------------

def _gt_image_text(img: dict) -> str:
    """Build flat GT text from chart/scanned image fields."""
    parts = []
    for field in ("chart_title", "x_label", "y_label"):
        v = img.get(field, "")
        if v:
            parts.append(v)
    for cat in img.get("categories", []):
        if cat:
            parts.append(cat)
    for series in img.get("series_names", []):
        if series:
            parts.append(series)
    vr = img.get("value_range")
    if vr and len(vr) == 2:
        parts.append(str(vr[0]))
        parts.append(str(vr[1]))
    full_text = img.get("full_text", "")
    if full_text:
        parts.append(full_text)
    else:
        cells = img.get("cells", [])
        for row in cells:
            if isinstance(row, dict):
                parts.append(row.get("heading", ""))
                parts.append(row.get("body", ""))
            elif isinstance(row, list):
                parts.extend(c for c in row if c)
            elif isinstance(row, str) and row:
                parts.append(row)
        spec = img.get("spec") or {}
        for line in spec.get("lines", []):
            if line:
                parts.append(line)
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Table alignment  (same Hungarian algorithm as other evaluators)
# ---------------------------------------------------------------------------

def _header_sim(gt_cols: list, pred_hdrs: list) -> float:
    if not gt_cols and not pred_hdrs:
        return 0.5
    if not gt_cols or not pred_hdrs:
        return 0.0
    return 1.0 - ned(" | ".join(gt_cols), " | ".join(pred_hdrs))


def align_tables(gt_tbls: list, pred_tbls: list, min_sim: float = 0.3) -> list:
    """
    Optimal bipartite matching of GT tables to pdfplumber tables.

    Similarity: column-header NED (w=0.7 when GT has columns) blended with
    row-count proximity (w=0.3), solved via Hungarian algorithm.
    Pairs below min_sim are dropped.
    """
    if not gt_tbls or not pred_tbls:
        return []

    n_gt, n_pred = len(gt_tbls), len(pred_tbls)
    sim = np.zeros((n_gt, n_pred))

    for i, gt in enumerate(gt_tbls):
        gt_cols = gt.get("columns", [])
        gt_rows = gt.get("row_count", 0)
        for j, pred in enumerate(pred_tbls):
            h_sim = _header_sim(gt_cols, _pdfplumber_tbl_header_row(pred))
            pred_rows = len(pred.get("rows", []))
            if gt_rows and pred_rows:
                size_sim = 1.0 - abs(gt_rows - pred_rows) / max(gt_rows, pred_rows)
                w_hdr = 0.7 if gt_cols else 0.0
                h_sim = w_hdr * h_sim + (1.0 - w_hdr) * size_sim
            sim[i, j] = h_sim

    row_ind, col_ind = linear_sum_assignment(-sim)
    return [
        (int(r), int(c), float(sim[r, c]))
        for r, c in zip(row_ind, col_ind)
        if sim[r, c] >= min_sim
    ]


# ---------------------------------------------------------------------------
# Per-pair table metrics
# ---------------------------------------------------------------------------

def eval_table_pair(gt: dict, pred: dict) -> dict:
    """
    All metrics for one matched (GT table, pdfplumber table) pair.

    pdfplumber rows are already 2-D lists of strings — no parsing required.
    Cell values may contain embedded newlines (\n); _clean() collapses them.
    """
    res = {}

    gt_rows  = gt.get("row_count", 0)
    gt_cols  = len(gt.get("columns", []))
    pred_all_rows = pred.get("rows", [])
    pred_rows = len(pred_all_rows)
    pred_cols = max((len(r) for r in pred_all_rows), default=0)

    res["gt_rows"]       = gt_rows
    res["gt_cols"]       = gt_cols
    res["pred_rows"]     = pred_rows
    res["pred_cols"]     = pred_cols
    res["row_count_sim"] = _count_sim(gt_rows, pred_rows)
    res["col_count_sim"] = _count_sim(gt_cols, pred_cols)

    sims = [s for s in (res["row_count_sim"], res["col_count_sim"]) if s is not None]
    res["structure_sim"] = round(sum(sims) / len(sims), 6) if sims else None

    # Header NED
    gt_col_list = gt.get("columns", [])
    pred_hdrs   = _pdfplumber_tbl_header_row(pred)
    if gt_col_list and pred_hdrs:
        paired_neds = [ned(g, p) for g, p in zip(gt_col_list, pred_hdrs)]
        res["header_ned_avg"] = round(float(np.mean(paired_neds)), 6)
    else:
        res["header_ned_avg"] = None

    # Cell-level NED (positional alignment on data rows)
    gt_grid   = gt.get("cells", [])
    pred_grid = _pdfplumber_tbl_data_grid(pred)

    if gt_grid and pred_grid:
        cell_neds = []
        n_rows = min(len(gt_grid), len(pred_grid))
        for ri in range(n_rows):
            n_c = min(len(gt_grid[ri]), len(pred_grid[ri]))
            for ci in range(n_c):
                cell_neds.append(ned(gt_grid[ri][ci], pred_grid[ri][ci]))
        if cell_neds:
            res["cell_ned_avg"]    = round(float(np.mean(cell_neds)),   6)
            res["cell_ned_median"] = round(float(np.median(cell_neds)), 6)
        else:
            res["cell_ned_avg"]    = None
            res["cell_ned_median"] = None
        gt_flat   = " ".join(c for row in gt_grid   for c in row)
        pred_flat = " ".join(c for row in pred_grid for c in row)
        res["cell_bow_f1"]    = bow_f1(gt_flat, pred_flat)
        res["cells_compared"] = len(cell_neds)
        res["cells_gt"]       = sum(len(r) for r in gt_grid)
        res["cells_pred"]     = sum(len(r) for r in pred_grid)
    else:
        res["cell_ned_avg"]    = None
        res["cell_ned_median"] = None
        res["cell_bow_f1"]     = None
        res["cells_compared"]  = 0
        res["cells_gt"]        = sum(len(r) for r in gt_grid)
        res["cells_pred"]      = sum(len(r) for r in pred_grid)

    return res


# ---------------------------------------------------------------------------
# Text section metrics
# ---------------------------------------------------------------------------

def eval_texts(gt_txts: list, pred_pages: list) -> dict:
    """
    Document-level text NED / WER / BOW-F1.

    GT  = concatenated text_content paragraph lists from GT text elements.
    Pred = concatenated page-level text strings from pdfplumber output.
          Note: pdfplumber page text includes table cell content as well
          as body text, since it does not separate the two at extraction time.
    """
    if not gt_txts:
        return {}
    gt_str = " ".join(
        para for el in gt_txts for para in el.get("text_content", [])
    )
    pred_str = " ".join(el.get("text", "") for el in pred_pages)
    if not gt_str.strip():
        return {}
    return {
        "text_ned":    ned(gt_str, pred_str),
        "text_wer":    wer(gt_str, pred_str),
        "text_bow_f1": bow_f1(gt_str, pred_str),
    }


# ---------------------------------------------------------------------------
# Image / figure extraction metrics
# ---------------------------------------------------------------------------

def eval_images(gt_imgs: list, pred_pics: list) -> dict:
    """
    Document-level image/figure extraction metrics.

    pdfplumber does not detect image regions, so pred_pics is always [].
    When GT has image elements:
      - image_recall = 0.0, image_precision = 0.0, image_f1 = 0.0
      - caption / OCR metrics = None (no pairs to evaluate)
    This correctly reflects pdfplumber's lack of image extraction capability.
    """
    if not gt_imgs:
        return {}

    n_gt, n_pred = len(gt_imgs), len(pred_pics)
    res = {
        "gt_image_count":      n_gt,
        "pred_image_count":    n_pred,
        "image_matched_count": 0,
        "image_recall":        0.0,
        "image_precision":     0.0,
        "image_f1":            0.0,
        "image_pairs":         [],
    }

    # No matches possible — all image content metrics are suppressed
    for k in ("caption_ned", "caption_found",
              "ocr_ned", "ocr_bow_f1", "ocr_wer", "ocr_text_found"):
        res[f"img_{k}"] = None

    return res


# ---------------------------------------------------------------------------
# Reading order: Kendall's tau
# ---------------------------------------------------------------------------

def eval_reading_order(
    matches: list,
    gt_tbls: list,
    pred_tbls: list,
) -> dict:
    """
    Kendall's tau on the reading order of matched tables.

    GT order  = structure element 'position' field (1-indexed manifest order).
    Pred order = '_idx' (global sequential index across all pages, assigned
                 in page order then by bbox top coordinate within each page).
    Requires >= 2 matched tables.
    """
    if len(matches) < 2:
        return {}

    gt_pos, pred_pos = [], []
    for gi, pi, _ in matches:
        gt_p   = gt_tbls[gi].get("position", gi)
        pred_p = pred_tbls[pi].get("_idx", pi)
        gt_pos.append(gt_p)
        pred_pos.append(pred_p)

    if len(gt_pos) < 2:
        return {}

    tau, pval = kendalltau(gt_pos, pred_pos)
    return {
        "reading_order_kendall_tau": round(float(tau),  6),
        "reading_order_pvalue":      round(float(pval), 6),
        "reading_order_n":           len(gt_pos),
    }


# ---------------------------------------------------------------------------
# Document-level evaluation
# ---------------------------------------------------------------------------

def eval_document(manifest: dict, doc: dict) -> dict:
    """
    Evaluate a single document: GT manifest vs pdfplumber JSON output.
    Returns a flat dict of metrics (same keys as docling/evaluate.py).
    """
    doc_id = manifest.get("doc_id", "?")
    res    = {"doc_id": doc_id}

    gt_tbls   = gt_tables(manifest)
    gt_txts   = gt_texts(manifest)
    gt_imgs   = gt_images(manifest)
    pred_tbls = pdfplumber_tables(doc)
    pred_pages = pdfplumber_body_texts(doc)
    pred_pics  = pdfplumber_pictures(doc)   # always []

    # -- table detection --------------------------------------------------
    n_gt, n_pred = len(gt_tbls), len(pred_tbls)
    matches = align_tables(gt_tbls, pred_tbls)
    n_match = len(matches)

    res.update({
        "gt_table_count":   n_gt,
        "pred_table_count": n_pred,
        "matched_count":    n_match,
    })

    recall    = n_match / n_gt   if n_gt   else 0.0
    precision = n_match / n_pred if n_pred else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    res.update({
        "table_recall":    round(recall,    6),
        "table_precision": round(precision, 6),
        "table_f1":        round(f1,        6),
    })

    # -- per-pair metrics -------------------------------------------------
    pairs, align_scores = [], []
    for gi, pi, score in matches:
        align_scores.append(score)
        pair = eval_table_pair(gt_tbls[gi], pred_tbls[pi])
        pair["gt_idx"]          = gi
        pair["pred_idx"]        = pi
        pair["gt_caption"]      = gt_tbls[gi].get("caption", "")
        pair["alignment_score"] = round(score, 6)
        pairs.append(pair)

    res["table_pairs"]         = pairs
    res["alignment_score_avg"] = (
        round(float(np.mean(align_scores)), 6) if align_scores else None
    )
    for k in ("cell_ned_avg", "cell_ned_median", "cell_bow_f1",
              "header_ned_avg", "row_count_sim", "col_count_sim", "structure_sim"):
        vals = [p[k] for p in pairs if p.get(k) is not None]
        res[f"agg_{k}"] = round(float(np.mean(vals)), 6) if vals else None

    # -- text metrics -----------------------------------------------------
    res.update(eval_texts(gt_txts, pred_pages))
    res["gt_text_count"]   = len(gt_txts)
    res["pred_text_count"] = len(pred_pages)

    # -- image metrics ----------------------------------------------------
    img_metrics = eval_images(gt_imgs, pred_pics)
    res.update(img_metrics)

    # -- reading order ----------------------------------------------------
    res.update(eval_reading_order(matches, gt_tbls, pred_tbls))

    # -- element coverage -------------------------------------------------
    n_match_imgs = img_metrics.get("image_matched_count", 0)
    total_gt     = len(gt_tbls) + len(gt_txts) + len(gt_imgs)
    matched_all  = n_match + n_match_imgs
    res["element_recall"] = round(matched_all / total_gt, 6) if total_gt else None

    return res


# ---------------------------------------------------------------------------
# Corpus-level aggregation  (identical to docling/evaluate.py)
# ---------------------------------------------------------------------------

def aggregate(docs: list) -> dict:
    """Macro-average all scalar metrics across documents."""
    all_keys: set = set()
    for d in docs:
        all_keys.update(k for k, v in d.items() if isinstance(v, (int, float)))
    agg: dict = {}
    for k in sorted(all_keys):
        vals = [d[k] for d in docs if isinstance(d.get(k), (int, float))]
        agg[k] = round(float(np.mean(vals)), 6) if vals else None
    return agg


# ---------------------------------------------------------------------------
# Grouped output  (identical structure to docling/evaluate.py)
# ---------------------------------------------------------------------------

_METRIC_DIRECTIONS: dict = {
    "table_recall":              "+",
    "table_precision":           "+",
    "table_f1":                  "+",
    "image_recall":              "+",
    "image_precision":           "+",
    "image_f1":                  "+",
    "agg_cell_ned_avg":          "-",
    "agg_cell_ned_median":       "-",
    "agg_cell_bow_f1":           "+",
    "agg_header_ned_avg":        "-",
    "alignment_score_avg":       "+",
    "agg_structure_sim":         "+",
    "agg_row_count_sim":         "+",
    "agg_col_count_sim":         "+",
    "img_caption_ned":           "-",
    "img_caption_found":         "+",
    "img_ocr_ned":               "-",
    "img_ocr_bow_f1":            "+",
    "img_ocr_wer":               "+",   # 1/(1+WER)
    "img_ocr_text_found":        "+",
    "text_ned":                  "-",
    "text_wer":                  "+",   # 1/(1+WER)
    "text_bow_f1":               "+",
    "reading_order_kendall_tau": "+",
    "reading_order_pvalue":      "-",
    "element_recall":            "+",
}

_WER_SCORE_KEYS: frozenset = frozenset({"text_wer", "img_ocr_wer"})
_NORMALIZE_TAU:  frozenset = frozenset({"reading_order_kendall_tau"})


def _to_grouped(flat: dict) -> dict:
    """
    Convert flat metrics dict to grouped structure with _directions labels.
    All scores normalized to [0,1]:
      WER  -> 1/(1+raw)  [0,inf) -> (0,1]; no info lost
      tau  -> (tau+1)/2  [-1,1]  -> [0,1]
    """
    def _pick(keys: list) -> dict:
        values: dict = {}
        dirs:   dict = {}
        for k in keys:
            v = flat.get(k)
            if v is None:
                continue
            v = float(v)
            if k in _WER_SCORE_KEYS:
                v = 1.0 / (1.0 + v)
            elif k in _NORMALIZE_TAU:
                v = (v + 1.0) / 2.0
            values[k] = round(v, 6)
            dirs[k]   = _METRIC_DIRECTIONS.get(k, "?")
        if not values:
            return None
        return {**values, "_directions": dirs}

    out: dict = {}
    if "doc_id" in flat:
        out["doc_id"] = flat["doc_id"]

    out["element_counts"] = {
        "gt_tables":      flat.get("gt_table_count"),
        "pred_tables":    flat.get("pred_table_count"),
        "matched_tables": flat.get("matched_count"),
        "gt_images":      flat.get("gt_image_count"),
        "pred_images":    flat.get("pred_image_count"),
        "matched_images": flat.get("image_matched_count"),
        "gt_texts":       flat.get("gt_text_count"),
        "pred_texts":     flat.get("pred_text_count"),
    }
    out["detection"]       = _pick(["table_recall", "table_precision", "table_f1",
                                     "image_recall", "image_precision", "image_f1"])
    out["table_content"]   = _pick(["agg_cell_ned_avg", "agg_cell_ned_median",
                                     "agg_cell_bow_f1", "agg_header_ned_avg",
                                     "alignment_score_avg"])
    out["table_structure"] = _pick(["agg_structure_sim", "agg_row_count_sim",
                                     "agg_col_count_sim"])
    out["image_content"]   = _pick(["img_caption_ned", "img_caption_found",
                                     "img_ocr_ned", "img_ocr_bow_f1",
                                     "img_ocr_wer", "img_ocr_text_found"])
    out["text_content"]    = _pick(["text_ned", "text_wer", "text_bow_f1"])

    ro = _pick(["reading_order_kendall_tau", "reading_order_pvalue"])
    if ro is not None:
        n = flat.get("reading_order_n")
        if n is not None:
            ro["reading_order_n"] = int(n)
    out["reading_order"] = ro

    out["coverage"] = _pick(["element_recall"])

    if "table_pairs" in flat:
        out["table_pairs"] = flat["table_pairs"]
    if "image_pairs" in flat:
        out["image_pairs"] = flat["image_pairs"]

    return out


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_SUMMARY_LAYOUT = [
    ("TABLE DETECTION", [
        ("table_recall",              "Table Recall",           "% GT tables found             (up better)"),
        ("table_precision",           "Table Precision",        "% detections that match GT    (up better)"),
        ("table_f1",                  "Table F1",               "harmonic mean                 (up better)"),
        ("gt_table_count",            "  GT Tables (avg)",      ""),
        ("pred_table_count",          "  Pred Tables (avg)",    ""),
        ("matched_count",             "  Matched (avg)",        ""),
    ]),
    ("TABLE CONTENT  (matched pairs)", [
        ("agg_cell_ned_avg",          "Cell NED (avg)",         "char-edit dist per cell       (down better)"),
        ("agg_cell_ned_median",       "Cell NED (median)",      ""),
        ("agg_cell_bow_f1",           "Cell BOW-F1",            "bag-of-words cell overlap     (up better)"),
        ("agg_header_ned_avg",        "Header NED",             "col-header char accuracy      (down better)"),
        ("alignment_score_avg",       "Alignment Score (avg)",  "GT to pred match quality      (up better)"),
    ]),
    ("TABLE STRUCTURE  (matched pairs)", [
        ("agg_structure_sim",         "Structure Sim",          "combined row+col count match  (up better)"),
        ("agg_row_count_sim",         "  Row Count Sim",        "min/max row count ratio       (up better)"),
        ("agg_col_count_sim",         "  Col Count Sim",        "min/max col count ratio       (up better)"),
    ]),
    ("IMAGE / FIGURE EXTRACTION", [
        ("image_recall",         "Figure Recall",          "% GT figures found            (up better)"),
        ("image_precision",      "Figure Precision",       "% detections are valid        (up better)"),
        ("image_f1",             "Figure F1",              "harmonic mean                 (up better)"),
        ("img_caption_ned",      "Caption NED",            "caption char accuracy         (down better)"),
        ("img_caption_found",    "  Caption Found rate",   "% GT captions extracted       (up better)"),
        ("img_ocr_ned",          "OCR Text NED",           "image text accuracy           (down better)"),
        ("img_ocr_bow_f1",       "OCR Text BOW-F1",        "image text word overlap       (up better)"),
        ("img_ocr_wer",          "OCR WER Score",          "1/(1+WER): higher=fewer errs  (up better)"),
        ("img_ocr_text_found",   "  OCR Text Found rate",  "% images w/ text extracted   (up better)"),
    ]),
    ("TEXT EXTRACTION", [
        ("text_ned",                  "Text NED",               "char-edit distance            (down better)"),
        ("text_wer",                  "Text WER Score",         "1/(1+WER): higher=fewer errs (up better)"),
        ("text_bow_f1",               "Text BOW-F1",            "word overlap                  (up better)"),
    ]),
    ("READING ORDER", [
        ("reading_order_kendall_tau", "Kendall's tau",          "rank correlation              (up better)"),
        ("reading_order_n",           "  Pairs used",           ""),
    ]),
    ("COVERAGE", [
        ("element_recall",            "Element Recall",         "% GT elements recovered       (up better)"),
    ]),
]


def print_summary(agg: dict, n_docs: int, gt_dir: Path, pred_dir: Path) -> None:
    W = 72
    print(f"\n{'='*W}")
    print(f"  PDFPLUMBER EVALUATION SUMMARY  --  {n_docs} document(s)")
    print(f"  GT  : {gt_dir}")
    print(f"  Pred: {pred_dir}")
    print(f"{'='*W}")

    for section, rows in _SUMMARY_LAYOUT:
        lines = []
        for key, label, hint in rows:
            val = agg.get(key)
            if val is None:
                continue
            if key in _WER_SCORE_KEYS:
                val = 1.0 / (1.0 + float(val))
            elif key in _NORMALIZE_TAU:
                val = (float(val) + 1.0) / 2.0
            val_s = f"{val:.4f}" if isinstance(val, float) else f"{val:.1f}"
            lines.append(f"  {label:<34} {val_s:>8}   {hint}")
        if lines:
            print(f"\n  {section}")
            print(f"  {'-'*66}")
            for ln in lines:
                print(ln)

    print(f"\n{'='*W}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pdfplumber output against GT manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gt-dir", required=True, type=Path,
        help="GT config directory (must contain a manifests/ sub-folder).",
    )
    parser.add_argument(
        "--pred-dir", required=True, type=Path,
        help="pdfplumber output directory (contains <doc_id>/<doc_id>.json).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Write full JSON report to this path.",
    )
    args = parser.parse_args()

    manifest_dir = args.gt_dir / "manifests"
    if not manifest_dir.is_dir():
        print(f"[error] {manifest_dir} not found", file=sys.stderr)
        sys.exit(1)

    manifests = sorted(manifest_dir.glob("*.json"))
    if not manifests:
        print(f"[error] no manifest JSON files in {manifest_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[pdfplumber-eval] gt={args.gt_dir}  pred={args.pred_dir}")

    doc_results = []
    for mf in manifests:
        doc_id    = mf.stem
        pred_json = args.pred_dir / doc_id / f"{doc_id}.json"
        if not pred_json.exists():
            print(f"  [skip] {doc_id}  (no prediction at {pred_json})")
            continue

        print(f"  [eval] {doc_id}")
        manifest = load_gt(mf)
        doc      = load_pdfplumber(pred_json)
        res      = eval_document(manifest, doc)
        doc_results.append(res)

        cnav   = res.get("agg_cell_ned_avg")
        cnav_s = f"{cnav:.3f}" if cnav is not None else "N/A"
        print(
            f"         tables gt={res['gt_table_count']}  "
            f"pred={res['pred_table_count']}  "
            f"matched={res['matched_count']}  "
            f"f1={res['table_f1']:.3f}  "
            f"cell_ned={cnav_s}"
        )

    if not doc_results:
        print("[error] no documents evaluated", file=sys.stderr)
        sys.exit(1)

    agg    = aggregate(doc_results)
    report = {
        "parser":              "pdfplumber",
        "documents_evaluated": len(doc_results),
        "aggregate":           _to_grouped(agg),
        "per_document":        [_to_grouped(r) for r in doc_results],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[report written to {args.output}]")

    print_summary(agg, len(doc_results), args.gt_dir, args.pred_dir)


if __name__ == "__main__":
    main()
