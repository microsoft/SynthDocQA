# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Inline annotation text picker.

After document content is generated the pipeline calls ``pick_annotation_specs``
with the ground-truth sections and the inline annotation config.  This module
selects random phrases from the already-generated text, assigns annotation
styles, and returns a list of ``AnnotationSpec`` objects that the PDF layer
then renders directly onto the finished PDF.

No phrase is predetermined: every run picks different text depending on the
document RNG seed, so annotation targets are unique per document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Markdown / markup stripping
# ---------------------------------------------------------------------------

# Patterns to remove before phrase selection so picked text matches PDF content
_MD_STRIP = [
    re.compile(r'\*\*(.+?)\*\*'),   # **bold**
    re.compile(r'\*(.+?)\*'),        # *italic*
    re.compile(r'__(.+?)__'),        # __bold__
    re.compile(r'_(.+?)_'),          # _italic_
    re.compile(r'#{1,6}\s+'),        # ## headings
    re.compile(r'`(.+?)`'),          # `code`
    re.compile(r'\[(.+?)\]\(.+?\)'), # [link](url) -> keep label
]


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting marks, keeping the plain words."""
    for pattern in _MD_STRIP:
        text = pattern.sub(r'\1' if pattern.groups else '', text)
    # Collapse leftover punctuation clusters and whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AnnotationSpec:
    """One inline annotation to be applied to the rendered PDF."""

    annotation_id: str
    type: str           # highlight | circle | margin_callout | score_callout
                        # | text_color | sticky_note
    text: str           # phrase picked from document content (empty for score_callout)
    color: str          # colour name (resolved to RGB in pdf_layer.py)
    paragraph_text: str # full paragraph the phrase came from
    extra: dict = field(default_factory=dict)
    page: int | None = None   # set by pdf_layer after annotation is placed


# ---------------------------------------------------------------------------
# Score-callout comment bank
# ---------------------------------------------------------------------------

_SCORE_COMMENTS = [
    "Lacks sufficient depth.",
    "Key terms are not defined.",
    "Good overview but misses detail.",
    "Repeats ideas too often.",
    "Needs better structure.",
    "References are missing.",
    "Claims are not well supported.",
    "Could use clearer examples.",
    "Writing is clear but too brief.",
    "More specific data would help.",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_corpus(sections_gt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect clean, plain-text phrases from all text AND table sections.

    Returns a list of dicts, each with:
      "text"          -- the candidate phrase / paragraph
      "context_type"  -- "text" or "table"
      "context_label" -- section heading (text) or "Table N" (table)

    Markdown formatting is stripped so picked phrases match rendered PDF
    content exactly (enabling PyMuPDF text search to locate them).
    """
    # Regex to detect purely numeric / currency / percentage cells that make
    # poor annotation targets (e.g. "3,450", "$12.5M", "87%")
    _NUMERIC_ONLY = re.compile(r'^[\d\s.,+\-()%$€£¥/\\]+$')

    corpus: list[dict[str, Any]] = []

    for sec in sections_gt:
        stype = sec.get("section_type", "text")

        # ── Prose sections ──────────────────────────────────────────────────
        if stype in ("text", "annotation"):
            # Grab the section heading for semantic context
            heading = ""
            for gt in sec.get("ground_truth", []):
                heading = _strip_markdown(gt.get("heading", "")).strip()
                break
            # Truncate long headings so QA questions stay readable
            label = (heading[:60] + "...") if len(heading) > 60 else heading

            for gt in sec.get("ground_truth", []):
                raw = gt.get("text_content", "")
                raw_paras = raw if isinstance(raw, list) else (raw.split("\n\n") if raw else [])
                for para in raw_paras:
                    para = _strip_markdown(para.strip())
                    words = para.split()
                    if len(words) >= 4 and not re.search(r'[*#`_\[\]|]', para):
                        corpus.append({
                            "text":          para,
                            "context_type":  "text",
                            "context_label": label,
                        })

        # ── Table sections ───────────────────────────────────────────────────
        # Only cell DATA values are used — column headers are excluded because
        # they tend to be generic labels (Year, Status, etc.) rather than
        # meaningful generated content.
        elif stype == "table":
            table_num = sec.get("table_number", 0)
            label = f"Table {table_num}" if table_num else "a table"

            for gt in sec.get("ground_truth", []):
                for row in gt.get("cells", []):
                    for cell in row:
                        phrase = _strip_markdown(str(cell).strip())
                        if (phrase and phrase.isascii()
                                and not _NUMERIC_ONLY.match(phrase)
                                and not re.search(r'[*#`_\[\]|]', phrase)
                                # Hyphens cause line-breaks in narrow table cells,
                                # splitting the text and breaking PDF text search.
                                and "-" not in phrase
                                and len(phrase) >= 2):
                            corpus.append({
                                "text":          phrase,
                                "context_type":  "table",
                                "context_label": label,
                            })

    return corpus


def _pick_color(subtype: str, cfg: Any, rng: np.random.Generator) -> str:
    """Return a random colour name from the appropriate palette."""
    if subtype == "highlight":
        palette = cfg.highlight_colors
    elif subtype == "circle":
        palette = cfg.circle_colors
    elif subtype == "text_color":
        palette = cfg.text_colors
    elif subtype in ("margin_callout", "sticky_note", "score_callout"):
        palette = cfg.callout_colors
    else:
        palette = ["blue"]
    return str(palette[int(rng.integers(0, len(palette)))])


def _pick_phrase(
    corpus: list[dict[str, Any]],
    min_w: int,
    max_w: int,
    rng: np.random.Generator,
    used: set[str],
) -> tuple[str, str, str, str] | None:
    """Return (phrase, paragraph, context_type, context_label) or None."""
    indices = list(range(len(corpus)))
    rng.shuffle(indices)
    for idx in indices:
        entry = corpus[idx]
        para = entry["text"]
        words = para.split()
        if len(words) < min_w:
            continue
        cap = min(max_w, len(words))
        n = int(rng.integers(min_w, cap + 1))
        start = int(rng.integers(0, len(words) - n + 1))
        phrase = " ".join(words[start: start + n])
        if phrase in used:
            continue
        # Skip phrases with non-ASCII characters (em-dashes, smart quotes,
        # unicode hyphens etc.) that won't survive PDF text search reliably
        if not phrase.isascii():
            continue
        # Skip phrases containing hyphens — a hyphenated word at a line-break
        # in the PDF causes the text to be split, breaking page.search_for()
        if "-" in phrase:
            continue
        return phrase, para, entry["context_type"], entry["context_label"]
    return None


def _make_score_callout(
    cfg: Any, rng: np.random.Generator, doc_id: str, idx: int
) -> AnnotationSpec:
    score = int(rng.integers(4, 10))
    comment = str(_SCORE_COMMENTS[int(rng.integers(0, len(_SCORE_COMMENTS)))])
    color = _pick_color("score_callout", cfg, rng)
    return AnnotationSpec(
        annotation_id=f"{doc_id}_annot_{idx:03d}",
        type="score_callout",
        text="",
        color=color,
        paragraph_text="",
        extra={"score": f"{score}/10", "comment": comment},
    )


def _make_spec(
    subtype: str,
    phrase: str,
    para: str,
    context_type: str,
    context_label: str,
    cfg: Any,
    rng: np.random.Generator,
    doc_id: str,
    idx: int,
) -> AnnotationSpec:
    color = _pick_color(subtype, cfg, rng)
    extra: dict[str, Any] = {
        "context_type":  context_type,
        "context_label": context_label,
    }

    if subtype == "text_color":
        extra["strikethrough"] = bool(rng.integers(0, 2))

    elif subtype == "margin_callout":
        # Use first ~15 words of the paragraph as the margin comment
        words = para.split()
        snippet = " ".join(words[:15])
        if len(words) > 15:
            snippet += "..."
        extra["comment"] = snippet

    elif subtype == "sticky_note":
        extra["note"] = phrase[:120]

    return AnnotationSpec(
        annotation_id=f"{doc_id}_annot_{idx:03d}",
        type=subtype,
        text=phrase,
        color=color,
        paragraph_text=para,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pick_annotation_specs(
    sections_gt: list[dict[str, Any]],
    inline_cfg: Any,          # InlineAnnotationConfig
    rng: np.random.Generator,
    doc_id: str,
) -> list[AnnotationSpec]:
    """Pick annotation specs from already-generated content.

    Parameters
    ----------
    sections_gt:
        The ``gt["sections"]`` list returned by ``build_document()`` or
        ``build_mixed_document()``.  Only text/annotation section types are
        searched for candidate phrases.
    inline_cfg:
        An ``InlineAnnotationConfig`` instance from the dataset config.
    rng:
        A forked document RNG (seeded deterministically per document).
    doc_id:
        Document identifier used to build unique annotation IDs.

    Returns
    -------
    list[AnnotationSpec]
        One spec per requested annotation, ordered by annotation index.
        Specs have ``page=None``; the PDF layer fills this in after placement.
    """
    corpus = _flatten_corpus(sections_gt)
    if not corpus and not any(s == "score_callout" for s in inline_cfg.subtypes):
        return []

    subtypes = list(inline_cfg.subtypes)
    if not subtypes:
        return []

    min_w = inline_cfg.text_selection.min_words
    max_w = inline_cfg.text_selection.max_words

    specs: list[AnnotationSpec] = []
    used_phrases: set[str] = set()

    for i in range(inline_cfg.count):
        subtype = subtypes[i % len(subtypes)]

        if subtype == "score_callout":
            specs.append(_make_score_callout(inline_cfg, rng, doc_id, i))
            continue

        # Keep phrases short so PyMuPDF can find them in the rendered PDF.
        # Longer phrases risk not matching due to ligatures / hyphenation.
        # margin_callout anchor is still short; its *comment* uses the full para.
        eff_max = min(max_w, 4)  # hard cap at 4 words for reliable PDF search

        result = _pick_phrase(corpus, min_w, eff_max, rng, used_phrases)
        if result is None:
            continue  # no suitable phrase found (e.g. very short document)

        phrase, para, ctx_type, ctx_label = result
        used_phrases.add(phrase)
        specs.append(_make_spec(subtype, phrase, para, ctx_type, ctx_label, inline_cfg, rng, doc_id, i))

    return specs
