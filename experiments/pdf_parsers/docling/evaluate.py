# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
evaluate.py -- Docling output evaluation against ground-truth manifests.

Metrics
-------
  Table Detection     table_recall, table_precision, table_f1
  Table Content       cell_ned_avg, cell_ned_median, cell_bow_f1
  Table Structure     row_count_mae, col_count_mae
  Table Alignment     header_ned_avg, alignment_score_avg
  Text Extraction     text_ned, text_wer, text_bow_f1  (when GT text exists)
  Reading Order       reading_order_kendall_tau
  Element Coverage    element_recall

Alignment strategy
------------------
GT tables with non-empty columns[] are matched to Docling tables by
column-header NED similarity (joined header strings).  For tables with no
column metadata, a row-count proximity term is blended in.  Final assignment
uses scipy.optimize.linear_sum_assignment (optimal bipartite matching).
Pairs below a similarity threshold of 0.3 are dropped (false negatives).

Cell content comparison uses positional alignment (row i, col j) on data
rows only (Docling header row is excluded).  NED is computed via
difflib.SequenceMatcher.ratio() -- no external edit-distance library needed.

Dependencies: numpy, scipy  (both available in the chai conda env)

Usage
-----
    python evaluate.py \\
        --gt-dir   Q:/path/to/cfg_001_tables_basic_only/ \\
        --pred-dir Q:/path/to/docling/output/ \\
        [--output  results/docling_cfg001.json]
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
# String utilities
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Lowercase, strip basic markdown formatting, collapse whitespace."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", str(text))   # bold
    text = re.sub(r"\*(.+?)\*",     r"\1", text)          # italic
    text = re.sub(r"`(.+?)`",        r"\1", text)          # inline code
    return re.sub(r"\s+", " ", text).strip().lower()


def ned(a: str, b: str) -> float:
    """
    Normalized Edit Distance in [0, 1] via SequenceMatcher ratio.
    0 = identical, 1 = completely different.
    """
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
    # Standard DP (O(|r||h|) time, O(|h|) space)
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return round(prev[-1] / len(r), 6)


def _count_sim(gt_n: int, pred_n: int) -> float:
    """
    Bounded count similarity in [0, 1]: min(gt, pred) / max(gt, pred).

    1.0 = exact match, degrades linearly as counts diverge.
    Returns None when gt_n == 0 (GT has no information to compare against).
    """
    if gt_n == 0 and pred_n == 0:
        return 1.0
    if gt_n == 0:
        return None   # cannot score; GT lacks ground truth for this dimension
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
# GT manifest helpers
# ---------------------------------------------------------------------------

def load_gt(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gt_tables(m: dict) -> list[dict]:
    """Structure elements of type 'table'."""
    return [e for e in m.get("structure", []) if e["type"] == "table"]


def gt_texts(m: dict) -> list[dict]:
    """Structure elements of type 'text'."""
    return [e for e in m.get("structure", []) if e["type"] == "text"]


def gt_images(m: dict) -> list[dict]:
    """Structure elements of type 'image' (scanned pages, infographics, figures)."""
    return [e for e in m.get("structure", []) if e["type"] == "image"]


# ---------------------------------------------------------------------------
# Docling JSON helpers
# ---------------------------------------------------------------------------

def load_docling(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def docling_tables(doc: dict) -> list[dict]:
    return doc.get("tables", [])


def docling_body_texts(doc: dict) -> list[dict]:
    """Body-order text items, excluding page headers/footers/footnotes/captions."""
    body_refs = {ref["$ref"] for ref in doc.get("body", {}).get("children", [])}
    exclude_labels = {"page_header", "page_footer", "footnote", "caption"}
    return [
        t for t in doc.get("texts", [])
        if t.get("self_ref", "") in body_refs
        and t.get("label", "") not in exclude_labels
    ]


def _tbl_header_row(tbl: dict) -> list[str]:
    """First-row cell texts from a Docling table."""
    grid = tbl.get("data", {}).get("grid", [])
    return [c.get("text", "") for c in grid[0]] if grid else []


def _tbl_data_grid(tbl: dict) -> list[list[str]]:
    """
    Data rows as a 2D list of strings, excluding the header row if
    Docling tagged row-0 cells as column_header=True.
    """
    grid = tbl.get("data", {}).get("grid", [])
    if not grid:
        return []
    start = 1 if any(c.get("column_header", False) for c in grid[0]) else 0
    return [[c.get("text", "") for c in row] for row in grid[start:]]


# ---------------------------------------------------------------------------
# Table alignment: GT to Docling optimal bipartite matching
# ---------------------------------------------------------------------------

def _header_sim(gt_cols: list[str], pred_hdrs: list[str]) -> float:
    """
    Similarity in [0,1] between GT column list and Docling header row.
    Uses NED on the joined strings.
    """
    if not gt_cols and not pred_hdrs:
        return 0.5   # neutral -- cannot discriminate
    if not gt_cols or not pred_hdrs:
        return 0.0
    return 1.0 - ned(" | ".join(gt_cols), " | ".join(pred_hdrs))


def align_tables(
    gt_tbls: list[dict],
    pred_tbls: list[dict],
    min_sim: float = 0.3,
) -> list[tuple[int, int, float]]:
    """
    Optimal bipartite matching of GT tables to Docling tables.

    Similarity matrix:
      - Primary signal: column-header NED similarity (weight 0.7 when columns
        are present in GT)
      - Secondary signal: row-count proximity (weight 0.3; sole signal when GT
        has no column metadata)
    Solved with Hungarian algorithm via scipy.optimize.linear_sum_assignment.

    Returns list of (gt_idx, pred_idx, similarity) for pairs >= min_sim.
    """
    if not gt_tbls or not pred_tbls:
        return []

    n_gt, n_pred = len(gt_tbls), len(pred_tbls)
    sim = np.zeros((n_gt, n_pred))

    for i, gt in enumerate(gt_tbls):
        gt_cols = gt.get("columns", [])
        gt_rows = gt.get("row_count", 0)
        for j, pred in enumerate(pred_tbls):
            h_sim = _header_sim(gt_cols, _tbl_header_row(pred))
            pred_rows = pred.get("data", {}).get("num_rows", 0)
            if gt_rows and pred_rows:
                size_sim = 1.0 - abs(gt_rows - pred_rows) / max(gt_rows, pred_rows)
                # Use header when columns available; rely purely on size otherwise
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
    Compute all metrics for one matched (GT table, Docling table) pair.

    Cell comparison uses positional alignment: row i col j of GT is compared
    to row i col j of Docling (after stripping the Docling header row).
    Only the overlap region (min_rows x min_cols) is scored.
    """
    res = {}

    # -- structure counts --------------------------------------------------
    gt_rows  = gt.get("row_count", 0)
    gt_cols  = len(gt.get("columns", []))
    pred_rows = pred.get("data", {}).get("num_rows", 0)
    pred_cols = pred.get("data", {}).get("num_cols", 0)

    res["gt_rows"]        = gt_rows
    res["gt_cols"]        = gt_cols
    res["pred_rows"]      = pred_rows
    res["pred_cols"]      = pred_cols
    res["row_count_sim"]  = _count_sim(gt_rows, pred_rows)  # 1.0 = exact row match
    res["col_count_sim"]  = _count_sim(gt_cols, pred_cols)  # 1.0 = exact col match

    sims = [s for s in (res["row_count_sim"], res["col_count_sim"]) if s is not None]
    res["structure_sim"] = round(sum(sims) / len(sims), 6) if sims else None

    # -- column header NED -------------------------------------------------
    gt_col_list = gt.get("columns", [])
    pred_hdrs   = _tbl_header_row(pred)
    if gt_col_list and pred_hdrs:
        paired_neds = [ned(g, p) for g, p in zip(gt_col_list, pred_hdrs)]
        res["header_ned_avg"] = round(float(np.mean(paired_neds)), 6)
    else:
        res["header_ned_avg"] = None

    # -- cell-level NED (positional alignment on data rows) ----------------
    gt_grid   = gt.get("cells", [])      # list of rows; header excluded
    pred_grid = _tbl_data_grid(pred)     # list of rows; header stripped

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

def eval_texts(gt_txts: list[dict], pred_txts: list[dict]) -> dict:
    """
    Document-level text NED / WER / BOW-F1.

    GT text  = concatenation of all text_content paragraph lists.
    Pred text = concatenation of body-level Docling text items.
    Returns an empty dict when no GT text elements exist.
    """
    if not gt_txts:
        return {}
    gt_str = " ".join(
        para
        for el in gt_txts
        for para in el.get("text_content", [])
    )
    pred_str = " ".join(t.get("text", "") for t in pred_txts)
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

def docling_body_pictures(doc: dict) -> list[dict]:
    """Body-level pictures only (excludes furniture items such as logos)."""
    body_refs = {ref["$ref"] for ref in doc.get("body", {}).get("children", [])}
    return [
        p for p in doc.get("pictures", [])
        if p.get("self_ref", "") in body_refs
    ]


def _gt_image_text(img: dict) -> str:
    """
    Build a flat GT text string from an image element, drawing from every
    text-bearing field regardless of image subtype.

    Chart images          chart_title, x_label, y_label, categories,
                          series_names, value_range endpoints
    scanned_text_document full_text (complete formatted body)
    scanned_data_table    cells rows (list-of-lists)
    scanned_* with spec   spec.lines
    Cells as section dicts {"heading", "body"} are handled explicitly.
    """
    parts = []

    # -- chart structural labels (present whenever chart_type is set) ------
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

    # -- scanned / text content -------------------------------------------
    # full_text is the richest single source (present on scanned_text_document)
    full_text = img.get("full_text", "")
    if full_text:
        parts.append(full_text)
    else:
        cells = img.get("cells", [])
        for row in cells:
            if isinstance(row, dict):
                # scanned_text_document section: {"heading": ..., "body": ...}
                parts.append(row.get("heading", ""))
                parts.append(row.get("body", ""))
            elif isinstance(row, list):
                # standard table-style row: ["val1", "val2", ...]
                parts.extend(c for c in row if c)
            elif isinstance(row, str) and row:
                parts.append(row)

        # handwritten notes / scanned docs store text in spec.lines
        spec = img.get("spec") or {}
        for line in spec.get("lines", []):
            if line:
                parts.append(line)

    return " ".join(p for p in parts if p)


def _pic_caption_text(pic: dict, texts_by_ref: dict) -> str:
    """
    Resolve caption text for a Docling picture.
    Captions are stored as $ref pointers to text items in the texts array.
    """
    parts = []
    for cap in pic.get("captions", []):
        ref = cap.get("$ref", "")
        t = texts_by_ref.get(ref)
        if t:
            parts.append(t.get("text", ""))
        elif "text" in cap:
            parts.append(cap["text"])
    return " ".join(parts).strip()


def _pic_ocr_text(pic: dict, texts_by_ref: dict) -> str:
    """
    Concatenate text from a Docling picture's children refs.
    These are OCR-extracted or embedded text items within the picture region.
    """
    parts = [
        texts_by_ref[ref["$ref"]].get("text", "")
        for ref in pic.get("children", [])
        if ref.get("$ref", "") in texts_by_ref
    ]
    return " ".join(parts).strip()


def align_images(
    gt_imgs: list[dict],
    pred_pics: list[dict],
    texts_by_ref: dict,
    min_sim: float = 0.2,
) -> list[tuple[int, int, float]]:
    """
    Match GT image elements to Docling body pictures.

    Primary signal: caption NED similarity (both sides must have a caption).
    Fallback: sequential order (i-th GT image -> i-th picture in body order)
    when neither side has captions, with a neutral score of 0.5.

    min_sim is lower than for tables because partial caption matches are
    still informative (e.g. "Figure 1:" prefix may match even if text differs).
    """
    if not gt_imgs or not pred_pics:
        return []

    n_gt, n_pred = len(gt_imgs), len(pred_pics)
    gt_caps   = [img.get("caption", "") for img in gt_imgs]
    pred_caps = [_pic_caption_text(p, texts_by_ref) for p in pred_pics]

    if any(gt_caps) and any(pred_caps):
        sim = np.zeros((n_gt, n_pred))
        for i, gc in enumerate(gt_caps):
            for j, pc in enumerate(pred_caps):
                if gc and pc:
                    sim[i, j] = 1.0 - ned(gc, pc)
                elif not gc and not pc:
                    sim[i, j] = 0.3   # neutral: neither has a caption
                # else: one side has caption, other doesn't -> 0.0
        row_ind, col_ind = linear_sum_assignment(-sim)
        return [
            (int(r), int(c), float(sim[r, c]))
            for r, c in zip(row_ind, col_ind)
            if sim[r, c] >= min_sim
        ]
    else:
        # No captions available on either side: fall back to sequential order
        n = min(n_gt, n_pred)
        return [(i, i, 0.5) for i in range(n)]


def eval_image_pair(gt_img: dict, pic: dict, texts_by_ref: dict) -> dict:
    """
    Metrics for one matched (GT image element, Docling picture) pair.

    caption_ned         NED between GT caption and Docling caption text.
                        1.0 when Docling finds no caption at all.
    caption_found       1 if Docling extracted any caption text, else 0.
    ocr_ned             NED between GT cell text and Docling OCR children text.
                        1.0 when no OCR text extracted (complete miss).
    ocr_bow_f1          BOW-F1 on the same pair (more lenient than NED).
    ocr_wer             Word Error Rate on the same pair.
    ocr_text_found      1 if Docling extracted any OCR text from this image, else 0.
    """
    res = {}

    # -- caption ----------------------------------------------------------
    gt_cap   = gt_img.get("caption", "")
    pred_cap = _pic_caption_text(pic, texts_by_ref)
    if gt_cap:
        res["caption_ned"]   = ned(gt_cap, pred_cap) if pred_cap else 1.0
        res["caption_found"] = 1 if pred_cap else 0
    else:
        res["caption_ned"]   = None
        res["caption_found"] = None

    # -- OCR / embedded text ----------------------------------------------
    gt_text  = _gt_image_text(gt_img)
    ocr_text = _pic_ocr_text(pic, texts_by_ref)

    if gt_text:
        if ocr_text:
            res["ocr_ned"]    = ned(gt_text, ocr_text)
            res["ocr_bow_f1"] = bow_f1(gt_text, ocr_text)
            res["ocr_wer"]    = wer(gt_text, ocr_text)
        else:
            res["ocr_ned"]    = 1.0
            res["ocr_bow_f1"] = 0.0
            res["ocr_wer"]    = 1.0
        res["ocr_text_found"] = 1 if ocr_text else 0
    else:
        res["ocr_ned"]        = None
        res["ocr_bow_f1"]     = None
        res["ocr_wer"]        = None
        res["ocr_text_found"] = None

    res["subtype"] = gt_img.get("subtype", "unknown")
    return res


def eval_images(
    gt_imgs: list[dict],
    pred_pics: list[dict],
    doc: dict,
) -> dict:
    """
    Document-level image/figure extraction metrics.
    Returns an empty dict when the GT has no image elements (skips section
    in the summary so it doesn't pollute table-only configs).
    """
    if not gt_imgs:
        return {}

    texts_by_ref = {t.get("self_ref", ""): t for t in doc.get("texts", [])}

    n_gt, n_pred = len(gt_imgs), len(pred_pics)
    res = {
        "gt_image_count":   n_gt,
        "pred_image_count": n_pred,
    }

    matches    = align_images(gt_imgs, pred_pics, texts_by_ref)
    n_match    = len(matches)
    res["image_matched_count"] = n_match

    recall    = n_match / n_gt    if n_gt    else 0.0
    precision = n_match / n_pred  if n_pred  else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    res.update({
        "image_recall":    round(recall,    6),
        "image_precision": round(precision, 6),
        "image_f1":        round(f1,        6),
    })

    # Per-pair metrics
    pairs = []
    for gi, pi, score in matches:
        pair = eval_image_pair(gt_imgs[gi], pred_pics[pi], texts_by_ref)
        pair["gt_idx"]          = gi
        pair["pred_idx"]        = pi
        pair["gt_caption"]      = gt_imgs[gi].get("caption", "")
        pair["alignment_score"] = round(score, 6)
        pairs.append(pair)

    res["image_pairs"] = pairs

    # Aggregate per-pair scalars
    for k in ("caption_ned", "caption_found",
              "ocr_ned", "ocr_bow_f1", "ocr_wer", "ocr_text_found"):
        vals = [p[k] for p in pairs if p.get(k) is not None]
        res[f"img_{k}"] = round(float(np.mean(vals)), 6) if vals else None

    return res


# ---------------------------------------------------------------------------
# Reading order: Kendall's tau
# ---------------------------------------------------------------------------

def eval_reading_order(
    matches: list[tuple[int, int, float]],
    gt_tbls: list[dict],
    pred_tbls: list[dict],
    doc: dict,
) -> dict:
    """
    Kendall's tau on the reading order of matched tables.

    GT order  = structure element 'position' field (1-indexed manifest order).
    Pred order = index in body.children (Docling reading sequence).
    Requires >= 2 matched tables with a known body position.
    """
    if len(matches) < 2:
        return {}

    body_ro  = [ref["$ref"] for ref in doc.get("body", {}).get("children", [])]
    ref_rank = {ref: idx for idx, ref in enumerate(body_ro)}

    gt_pos, pred_pos = [], []
    for gi, pi, _ in matches:
        gt_p     = gt_tbls[gi].get("position", gi)
        pred_ref = pred_tbls[pi].get("self_ref", "")
        pred_p   = ref_rank.get(pred_ref, -1)
        if pred_p >= 0:
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
    Evaluate a single document: GT manifest vs Docling JSON output.
    Returns a flat dict of metrics plus 'table_pairs' for per-pair detail.
    """
    doc_id = manifest.get("doc_id", "?")
    res    = {"doc_id": doc_id}

    gt_tbls   = gt_tables(manifest)
    gt_txts   = gt_texts(manifest)
    gt_imgs   = gt_images(manifest)
    pred_tbls = docling_tables(doc)
    pred_txts = docling_body_texts(doc)
    pred_pics = docling_body_pictures(doc)

    # -- table detection --------------------------------------------------
    n_gt, n_pred = len(gt_tbls), len(pred_tbls)
    matches = align_tables(gt_tbls, pred_tbls)
    n_match = len(matches)

    res.update({
        "gt_table_count":   n_gt,
        "pred_table_count": n_pred,
        "matched_count":    n_match,
    })

    recall    = n_match / n_gt    if n_gt    else 0.0
    precision = n_match / n_pred  if n_pred  else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
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

    # Macro-average each per-pair scalar over matched pairs
    for k in ("cell_ned_avg", "cell_ned_median", "cell_bow_f1",
              "header_ned_avg", "row_count_sim", "col_count_sim", "structure_sim"):
        vals = [p[k] for p in pairs if p.get(k) is not None]
        res[f"agg_{k}"] = round(float(np.mean(vals)), 6) if vals else None

    # -- text metrics -----------------------------------------------------
    res.update(eval_texts(gt_txts, pred_txts))
    res["gt_text_count"]   = len(gt_txts)
    res["pred_text_count"] = len(pred_txts)

    # -- image metrics ----------------------------------------------------
    img_metrics = eval_images(gt_imgs, pred_pics, doc)
    res.update(img_metrics)

    # -- reading order ----------------------------------------------------
    res.update(eval_reading_order(matches, gt_tbls, pred_tbls, doc))

    # -- element coverage (tables + texts + images) -----------------------
    n_match_imgs = img_metrics.get("image_matched_count", 0)
    total_gt     = len(gt_tbls) + len(gt_txts) + len(gt_imgs)
    matched_all  = n_match + n_match_imgs
    res["element_recall"] = round(matched_all / total_gt, 6) if total_gt else None

    return res


# ---------------------------------------------------------------------------
# Corpus-level aggregation
# ---------------------------------------------------------------------------

def aggregate(docs: list[dict]) -> dict:
    """Macro-average all scalar metrics across documents."""
    all_keys: set[str] = set()
    for d in docs:
        all_keys.update(k for k, v in d.items() if isinstance(v, (int, float)))

    agg: dict = {}
    for k in sorted(all_keys):
        vals = [d[k] for d in docs if isinstance(d.get(k), (int, float))]
        agg[k] = round(float(np.mean(vals)), 6) if vals else None
    return agg


# ---------------------------------------------------------------------------
# Grouped output
# ---------------------------------------------------------------------------

# Direction label for every scored metric:
#   "+" = positive metric — closer to 1 is better
#   "-" = negative metric — closer to 0 is better
_METRIC_DIRECTIONS: dict = {
    # detection
    "table_recall":              "+",
    "table_precision":           "+",
    "table_f1":                  "+",
    "image_recall":              "+",
    "image_precision":           "+",
    "image_f1":                  "+",
    # table content
    "agg_cell_ned_avg":          "-",   # char edit distance: lower = more accurate
    "agg_cell_ned_median":       "-",
    "agg_cell_bow_f1":           "+",   # word overlap: higher = better
    "agg_header_ned_avg":        "-",   # header char edit distance: lower = better
    "alignment_score_avg":       "+",   # GT-to-pred match quality: higher = better
    # table structure
    "agg_structure_sim":         "+",   # row+col count similarity: higher = better
    "agg_row_count_sim":         "+",
    "agg_col_count_sim":         "+",
    # image content
    "img_caption_ned":           "-",   # caption char edit distance: lower = better
    "img_caption_found":         "+",   # fraction of captions extracted: higher = better
    "img_ocr_ned":               "-",   # OCR char edit distance: lower = better
    "img_ocr_bow_f1":            "+",   # OCR word overlap: higher = better
    "img_ocr_wer":               "+",   # 1/(1+WER): higher = fewer word errors
    "img_ocr_text_found":        "+",   # fraction with any OCR text: higher = better
    # text content
    "text_ned":                  "-",
    "text_wer":                  "+",   # 1/(1+WER): higher = fewer word errors
    "text_bow_f1":               "+",
    # reading order
    "reading_order_kendall_tau": "+",   # normalized (tau+1)/2 to [0,1]; higher = better
    "reading_order_pvalue":      "-",   # lower p-value = stronger correlation
    # coverage
    "element_recall":            "+",
}

# WER can exceed 1.0 when the parser over-generates (many insertions).
# Clamping to 1 would make WER=1.4 and WER=5.0 indistinguishable.
# Instead use 1/(1+WER), a monotonic bijection [0,∞) -> (0,1], which:
#   - preserves the full ranking (no information lost)
#   - maps WER=0 -> 1.0 (perfect), WER=1 -> 0.5, WER=5 -> 0.167
#   - flips direction to "+" (higher score = fewer word errors)
# Raw WER is kept in the flat dict for aggregation arithmetic.
_WER_SCORE_KEYS: frozenset = frozenset({"text_wer", "img_ocr_wer"})

# Kendall's tau lives in [-1, 1].  Map to [0, 1] via (tau + 1) / 2 so it reads
# as: 0.0 = perfectly reversed order, 0.5 = no correlation, 1.0 = perfect order.
_NORMALIZE_TAU: frozenset = frozenset({"reading_order_kendall_tau"})


def _to_grouped(flat: dict) -> dict:
    """
    Convert a flat metrics dict (from eval_document or aggregate) into a
    grouped structure with labelled sections for readability.

    Normalization applied to make every score safely in [0, 1]:
      WER metrics        bijection:  1 / (1 + raw)   [0,inf) -> (0,1]; no info lost
      Kendall's tau      rescaled:   (tau + 1) / 2   [-1,1]  -> [0,1]

    Each section includes a "_directions" dict that labels every metric as:
      "+"  positive metric — closer to 1 is better
      "-"  negative metric — closer to 0 is better
    Raw count keys (element_counts, reading_order_n) carry no direction label.

    Works for both per-document results and corpus-level aggregates.
    Sections with no available data are set to null.
    """
    def _pick(keys: list) -> dict:
        """Build a section dict with normalized values + _directions."""
        values: dict = {}
        dirs:   dict = {}
        for k in keys:
            v = flat.get(k)
            if v is None:
                continue
            v = float(v)
            if k in _WER_SCORE_KEYS:
                v = 1.0 / (1.0 + v)   # [0,inf) -> (0,1]; higher = better
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

    # Raw counts — integers, not [0,1] scored metrics; no direction labels
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

    # reading_order: scored metrics via _pick, then append the pair count separately
    ro = _pick(["reading_order_kendall_tau", "reading_order_pvalue"])
    if ro is not None:
        n = flat.get("reading_order_n")
        if n is not None:
            ro["reading_order_n"] = int(n)   # count, not a scored metric
    out["reading_order"] = ro

    out["coverage"]        = _pick(["element_recall"])

    # Per-pair detail — only present in per-document results
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
        ("table_f1",                  "Table F1",               "harmonic mean of above two    (up better)"),
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
        ("img_ocr_ned",          "OCR Text NED",           "image text char accuracy      (down better)"),
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
    print(f"  DOCLING EVALUATION SUMMARY  --  {n_docs} document(s)")
    print(f"  GT  : {gt_dir}")
    print(f"  Pred: {pred_dir}")
    print(f"{'='*W}")

    for section, rows in _SUMMARY_LAYOUT:
        lines = []
        for key, label, hint in rows:
            val = agg.get(key)
            if val is None:
                continue
            # Apply the same normalization as _to_grouped so the summary
            # matches the JSON output (WER -> 1/(1+WER), tau -> (tau+1)/2)
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
        description="Evaluate Docling output against GT manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gt-dir", required=True, type=Path,
        help="GT config directory (must contain a manifests/ sub-folder).",
    )
    parser.add_argument(
        "--pred-dir", required=True, type=Path,
        help="Docling output directory (contains <doc_id>/<doc_id>.json).",
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

    print(f"[docling-eval] gt={args.gt_dir}  pred={args.pred_dir}")

    doc_results = []
    for mf in manifests:
        doc_id    = mf.stem
        pred_json = args.pred_dir / doc_id / f"{doc_id}.json"
        if not pred_json.exists():
            print(f"  [skip] {doc_id}  (no prediction at {pred_json})")
            continue

        print(f"  [eval] {doc_id}")
        manifest = load_gt(mf)
        doc      = load_docling(pred_json)
        res      = eval_document(manifest, doc)
        doc_results.append(res)

        cnav = res.get("agg_cell_ned_avg")
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
        "parser":              "docling",
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
