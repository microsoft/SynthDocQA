# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
evaluate.py -- pymupdf4llm output evaluation against ground-truth manifests.

Computes the same metrics as docling/evaluate.py and mineru/evaluate.py so
results are directly comparable.  Only the data-extraction layer differs.

pymupdf4llm input file : <doc_id>/<doc_id>_pages.json
  Saved by parse.py as a list of per-page dicts from to_markdown(page_chunks=True).

Two backends are supported (detected automatically from the JSON):

  Layout backend (default when pymupdf.layout is available):
    Each page dict has keys: metadata, toc_items, page_boxes, text
    page_boxes is a list of {index, class, bbox, pos} dicts.
    pos = [start, stop] char offsets into the page "text" string.
    Table content is extracted from text[pos[0]:pos[1]] and parsed as markdown.
    Images / figures are counted via page_boxes entries with class "picture".
    Body text is collected from page_boxes entries with text-type classes.

  Legacy (rag) backend:
    Each page dict has keys: metadata, toc_items, tables, images, graphics,
                              words, text
    Table count comes from page["tables"]; content is parsed from page "text".
    Image count comes from page["images"].
    Body text is stripped from page "text" (table markdown removed).

Notes on OCR and captions for images:
  pymupdf4llm does not extract text from inside image/figure regions and
  does not produce explicit caption objects.  When the GT has image text or
  a caption, worst-case scores are reported so the metric correctly penalises
  the missing extraction.

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
        --pred-dir Q:/path/to/pymupdf4llm/output/ \\
        [--output  results/pymupdf4llm_cfg001.json]
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
# pymupdf4llm pages.json helpers
# ---------------------------------------------------------------------------

def load_pymupdf(path: Path) -> list:
    """Load a pymupdf4llm *_pages.json file (list of page dicts)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_layout_backend(pages: list) -> bool:
    """True when the layout backend was used (page_boxes key present)."""
    return bool(pages) and "page_boxes" in pages[0]


# ---------------------------------------------------------------------------
# Markdown table parsing
# ---------------------------------------------------------------------------

def _is_separator_row(cells: list) -> bool:
    """True if every cell looks like a markdown table separator (---, :--:, etc.)."""
    return bool(cells) and all(re.match(r"^:?-+:?$", c.strip()) for c in cells)


def _parse_md_table(md_text: str) -> list:
    """
    Parse a markdown table block into a 2D list of cell strings.

    Returns all rows in order: row 0 is the header row, rows 1+ are data rows.
    Separator rows (containing only dashes/colons) are skipped.

    pymupdf4llm table_to_markdown format:
        |col1|col2|col3|
        |---|---|---|
        |val1|val2|val3|
    """
    rows = []
    for line in md_text.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        # Remove empty strings from leading/trailing |
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if not cells:
            continue
        if _is_separator_row(cells):
            continue
        rows.append([c.strip() for c in cells])
    return rows


def _extract_page_md_tables(text: str) -> list:
    """
    Extract all markdown tables from a page text block.
    Returns a list of 2D cell grids (one per table, same order as in text).
    """
    tables = []
    in_table = False
    current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            current_lines.append(stripped)
            in_table = True
        elif in_table:
            if current_lines:
                parsed = _parse_md_table("\n".join(current_lines))
                if parsed:
                    tables.append(parsed)
            current_lines = []
            in_table = False

    if current_lines:
        parsed = _parse_md_table("\n".join(current_lines))
        if parsed:
            tables.append(parsed)

    return tables


# ---------------------------------------------------------------------------
# pymupdf4llm element extraction (handles both layout and rag backends)
# ---------------------------------------------------------------------------

# Box classes counted as body text in the layout backend
_TEXT_CLASSES = frozenset({"text", "title", "section-header", "list-item", "footnote"})

# Box classes counted as pictures/figures in the layout backend.
# "table-fallback" is remapped to "table" in page_boxes by pymupdf4llm,
# so it never appears as a picture class here.
_PICTURE_CLASSES = frozenset({"picture"})


def pymupdf_tables(pages: list) -> list:
    """
    All tables across all pages, with injected _idx (sequential reading order)
    and _parsed (2D cell grid: row 0 = header, rows 1+ = data).

    Layout backend:  tables identified by page_boxes entries with class "table".
                     Markdown is sliced from page text via pos offsets.
    Rag backend:     tables counted from page["tables"]; markdown parsed from
                     the full page text.
    """
    result = []
    idx = 0

    for page_num, page in enumerate(pages):
        text = page.get("text", "")

        if _is_layout_backend(pages):
            for box in page.get("page_boxes", []):
                if box.get("class") == "table":
                    pos = box.get("pos", [0, len(text)])
                    table_md = text[pos[0]:pos[1]]
                    parsed = _parse_md_table(table_md)
                    result.append({
                        "_idx":   idx,
                        "_page":  page_num,
                        "_parsed": parsed,
                        "bbox":   box.get("bbox"),
                    })
                    idx += 1
        else:
            # Rag backend: page["tables"] gives count; content in page text
            n_tables = len(page.get("tables", []))
            md_tables = _extract_page_md_tables(text)
            for i in range(n_tables):
                parsed = md_tables[i] if i < len(md_tables) else []
                result.append({
                    "_idx":   idx,
                    "_page":  page_num,
                    "_parsed": parsed,
                    "bbox":   (page.get("tables", [{}])[i].get("bbox")
                               if i < len(page.get("tables", [])) else None),
                })
                idx += 1

    return result


def pymupdf_body_texts(pages: list) -> list:
    """
    Collect body text strings from all pages (tables and pictures excluded).

    Layout backend:  text sliced from page text via pos offsets for text-type
                     boxes (text, title, section-header, list-item, footnote).
    Rag backend:     table markdown blocks and image references stripped from
                     the full page text.
    """
    parts = []

    for page in pages:
        text = page.get("text", "")

        if _is_layout_backend(pages):
            for box in page.get("page_boxes", []):
                if box.get("class") in _TEXT_CLASSES:
                    pos = box.get("pos", [0, len(text)])
                    segment = text[pos[0]:pos[1]].strip()
                    if segment:
                        parts.append(segment)
        else:
            # Strip table rows and image references from page text
            lines = [l for l in text.splitlines() if not l.strip().startswith("|")]
            cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", "\n".join(lines)).strip()
            if cleaned:
                parts.append(cleaned)

    return parts


def pymupdf_pictures(pages: list) -> list:
    """
    All detected images/figures across all pages, with injected _idx.

    Layout backend:  page_boxes entries with class in {"picture", "formula",
                     "table-fallback"}.
    Rag backend:     items in page["images"].
    """
    result = []
    idx = 0

    for page_num, page in enumerate(pages):
        if _is_layout_backend(pages):
            for box in page.get("page_boxes", []):
                if box.get("class") in _PICTURE_CLASSES:
                    result.append({
                        "_idx":  idx,
                        "_page": page_num,
                        "bbox":  box.get("bbox"),
                        "class": box.get("class"),
                    })
                    idx += 1
        else:
            for img in page.get("images", []):
                result.append({**img, "_idx": idx, "_page": page_num})
                idx += 1

    return result


def _pymupdf_tbl_header_row(tbl: dict) -> list:
    """Column header strings from the parsed table (first row of _parsed)."""
    parsed = tbl.get("_parsed", [])
    return parsed[0] if parsed else []


def _pymupdf_tbl_data_grid(tbl: dict) -> list:
    """Data rows from the parsed table (all rows after the header)."""
    parsed = tbl.get("_parsed", [])
    return parsed[1:] if len(parsed) > 1 else []


# ---------------------------------------------------------------------------
# GT image text helper  (identical to docling/evaluate.py and mineru/evaluate.py)
# ---------------------------------------------------------------------------

def _gt_image_text(img: dict) -> str:
    """
    Build a flat GT text string from chart/scanned image fields.
    Covers chart_title, x_label, y_label, categories, series_names,
    value_range, full_text, cells (list-of-rows or section dicts), spec.lines.
    """
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
# Table alignment (same Hungarian algorithm as docling/evaluate.py)
# ---------------------------------------------------------------------------

def _header_sim(gt_cols: list, pred_hdrs: list) -> float:
    if not gt_cols and not pred_hdrs:
        return 0.5
    if not gt_cols or not pred_hdrs:
        return 0.0
    return 1.0 - ned(" | ".join(gt_cols), " | ".join(pred_hdrs))


def align_tables(gt_tbls: list, pred_tbls: list, min_sim: float = 0.3) -> list:
    """
    Optimal bipartite matching of GT tables to pymupdf4llm tables.

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
            h_sim = _header_sim(gt_cols, _pymupdf_tbl_header_row(pred))
            pred_rows = max(len(pred.get("_parsed", [])) - 1, 0)
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
    All metrics for one matched (GT table, pymupdf4llm table) pair.
    Identical metric definitions to docling/evaluate.py and mineru/evaluate.py;
    cell content is sourced from the parsed markdown grid (_parsed).
    """
    res = {}

    gt_rows = gt.get("row_count", 0)
    gt_cols = len(gt.get("columns", []))

    parsed  = pred.get("_parsed", [])
    # Row 0 of _parsed is the header; data rows start at index 1
    pred_rows = max(len(parsed) - 1, 0)
    pred_cols = len(parsed[0]) if parsed else 0

    res["gt_rows"]       = gt_rows
    res["gt_cols"]       = gt_cols
    res["pred_rows"]     = pred_rows
    res["pred_cols"]     = pred_cols
    res["row_count_sim"] = _count_sim(gt_rows, pred_rows)
    res["col_count_sim"] = _count_sim(gt_cols, pred_cols)

    sims = [s for s in (res["row_count_sim"], res["col_count_sim"]) if s is not None]
    res["structure_sim"] = round(sum(sims) / len(sims), 6) if sims else None

    gt_col_list = gt.get("columns", [])
    pred_hdrs   = _pymupdf_tbl_header_row(pred)
    if gt_col_list and pred_hdrs:
        paired_neds = [ned(g, p) for g, p in zip(gt_col_list, pred_hdrs)]
        res["header_ned_avg"] = round(float(np.mean(paired_neds)), 6)
    else:
        res["header_ned_avg"] = None

    gt_grid   = gt.get("cells", [])
    pred_grid = _pymupdf_tbl_data_grid(pred)

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

def eval_texts(gt_txts: list, pred_parts: list) -> dict:
    """
    Document-level text NED / WER / BOW-F1.
    GT  = concatenated text_content paragraph lists.
    Pred = concatenated body text strings collected from page boxes.
    """
    if not gt_txts:
        return {}
    gt_str = " ".join(
        para for el in gt_txts for para in el.get("text_content", [])
    )
    pred_str = " ".join(pred_parts)
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

def align_images(gt_imgs: list, pred_pics: list, min_sim: float = 0.2) -> list:
    """
    Match GT image elements to pymupdf4llm pictures.

    pymupdf4llm does not produce caption objects, so caption-based matching
    is not possible.  Sequential order matching is used: the i-th GT image
    is paired with the i-th detected picture, up to min(n_gt, n_pred).
    A neutral score of 0.5 is assigned to each pair.
    """
    if not gt_imgs or not pred_pics:
        return []
    n = min(len(gt_imgs), len(pred_pics))
    return [(i, i, 0.5) for i in range(n)]


def eval_image_pair(gt_img: dict, _pic: dict) -> dict:
    """
    Metrics for one matched (GT image, pymupdf4llm picture) pair.

    caption_ned / caption_found:
      pymupdf4llm does not extract structured captions.  When GT has a caption,
      worst-case scores are reported (caption_ned=1.0, caption_found=0) so the
      metric correctly penalises the absence of caption extraction.

    ocr_ned / ocr_bow_f1 / ocr_wer / ocr_text_found:
      pymupdf4llm does not perform in-region OCR on figures.  When GT has image
      text content, worst-case scores are reported (ned=1, bow=0, wer=1, found=0).
    """
    res = {}

    gt_cap = gt_img.get("caption", "")
    if gt_cap:
        res["caption_ned"]   = 1.0
        res["caption_found"] = 0
    else:
        res["caption_ned"]   = None
        res["caption_found"] = None

    gt_text = _gt_image_text(gt_img)
    if gt_text:
        res["ocr_ned"]        = 1.0
        res["ocr_bow_f1"]     = 0.0
        res["ocr_wer"]        = 1.0
        res["ocr_text_found"] = 0
    else:
        res["ocr_ned"]        = None
        res["ocr_bow_f1"]     = None
        res["ocr_wer"]        = None
        res["ocr_text_found"] = None

    res["subtype"] = gt_img.get("subtype", "unknown")
    return res


def eval_images(gt_imgs: list, pred_pics: list) -> dict:
    """
    Document-level image/figure extraction metrics.
    Returns empty dict when GT has no image elements.
    """
    if not gt_imgs:
        return {}

    n_gt, n_pred = len(gt_imgs), len(pred_pics)
    res = {
        "gt_image_count":   n_gt,
        "pred_image_count": n_pred,
    }

    matches = align_images(gt_imgs, pred_pics)
    n_match = len(matches)
    res["image_matched_count"] = n_match

    recall    = n_match / n_gt   if n_gt   else 0.0
    precision = n_match / n_pred if n_pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    res.update({
        "image_recall":    round(recall,    6),
        "image_precision": round(precision, 6),
        "image_f1":        round(f1,        6),
    })

    pairs = []
    for gi, pi, score in matches:
        pair = eval_image_pair(gt_imgs[gi], pred_pics[pi])
        pair["gt_idx"]          = gi
        pair["pred_idx"]        = pi
        pair["gt_caption"]      = gt_imgs[gi].get("caption", "")
        pair["alignment_score"] = round(score, 6)
        pairs.append(pair)

    res["image_pairs"] = pairs

    for k in ("caption_ned", "caption_found",
              "ocr_ned", "ocr_bow_f1", "ocr_wer", "ocr_text_found"):
        vals = [p[k] for p in pairs if p.get(k) is not None]
        res[f"img_{k}"] = round(float(np.mean(vals)), 6) if vals else None

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
    Pred order = '_idx' (sequential index across all pages in document order).
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

def eval_document(manifest: dict, pages: list) -> dict:
    """
    Evaluate a single document: GT manifest vs pymupdf4llm pages list.
    Returns a flat dict of metrics (same keys as docling/evaluate.py).
    """
    doc_id = manifest.get("doc_id", "?")
    res    = {"doc_id": doc_id}

    gt_tbls   = gt_tables(manifest)
    gt_txts   = gt_texts(manifest)
    gt_imgs   = gt_images(manifest)
    pred_tbls = pymupdf_tables(pages)
    pred_txts = pymupdf_body_texts(pages)
    pred_pics = pymupdf_pictures(pages)

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
    res.update(eval_texts(gt_txts, pred_txts))
    res["gt_text_count"]   = len(gt_txts)
    res["pred_text_count"] = len(pred_txts)

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
    print(f"  PYMUPDF4LLM EVALUATION SUMMARY  --  {n_docs} document(s)")
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
        description="Evaluate pymupdf4llm output against GT manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gt-dir", required=True, type=Path,
        help="GT config directory (must contain a manifests/ sub-folder).",
    )
    parser.add_argument(
        "--pred-dir", required=True, type=Path,
        help="pymupdf4llm output directory (contains <doc_id>/<doc_id>_pages.json).",
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

    print(f"[pymupdf4llm-eval] gt={args.gt_dir}  pred={args.pred_dir}")

    doc_results = []
    for mf in manifests:
        doc_id    = mf.stem
        pred_json = args.pred_dir / doc_id / f"{doc_id}_pages.json"
        if not pred_json.exists():
            print(f"  [skip] {doc_id}  (no prediction at {pred_json})")
            continue

        print(f"  [eval] {doc_id}")
        manifest = load_gt(mf)
        pages    = load_pymupdf(pred_json)
        res      = eval_document(manifest, pages)
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
        "parser":              "pymupdf4llm",
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
