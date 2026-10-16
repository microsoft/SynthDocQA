# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Question-guided page selection for long PDF documents.

Strategy (applied in order until pages are found):

  1. Title extraction + exact search
     Parse the element title from the question (e.g., "In the table titled
     'Revenue Summary'..." -> "Revenue Summary"), then find pages whose
     text contains that title string.  Returns matching pages plus a
     configurable window of neighbouring pages (for multi-page elements).

  2. BM25 keyword ranking
     Score all pages against the full question text using BM25 and return
     the top-K scoring pages.  Works well when the title is not directly
     searchable (e.g., scanned pages, annotation "this element" questions).

  3. First-N fallback
     Return pages [0, 1, ..., fallback_n-1] if both methods yield nothing.

Why this works for our dataset
-------------------------------
Every quest question embeds the element caption as an anchor:
  TABLE       : 'In the table titled "XYZ", ...'
  FIGURE      : 'In the chart titled "ABC", ...'
  TEXT_BLOCK  : 'According to the "DEF" section, ...'
  FORM        : 'In the \'GHI\' form, ...'
  ANNOTATION  : '\'this element\' (the JKL table)' (fallback to BM25)
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Title extraction from question text
# ---------------------------------------------------------------------------

# Ordered from most-specific to least-specific
_TITLE_PATTERNS: list[re.Pattern] = [
    # "titled 'X'" or 'titled "X"' — table / figure / chart
    re.compile(r'titled?\s+["\u201c\u2018]([^"\u201d\u2019]{10,})["\u201d\u2019]', re.IGNORECASE),
    # '"X" section' — text_block sections
    re.compile(r'["\u201c]([^"\u201d]{10,})["\u201d]\s+section', re.IGNORECASE),
    # "'X' form" — forms (single quotes)
    re.compile(r"['\u2018]([^'\u2019]{10,})['\u2019]\s+form", re.IGNORECASE),
    # "'X' (this element)" or "this element (the X)" — annotations
    re.compile(r"['\u2018]([^'\u2019]{8,})['\u2019]\s+(?:colored\s+)?(?:table|chart|figure)", re.IGNORECASE),
    re.compile(r"\(the\s+([^)]{8,}(?:table|chart|figure|overview|report)[^)]*)\)", re.IGNORECASE),
    # Generic fallback: first double-quoted string >= 12 chars
    re.compile(r'["\u201c]([^"\u201d]{12,})["\u201d]'),
]


def extract_title(question: str) -> Optional[str]:
    """Extract the element title/caption from a quest question string.

    Returns the title string, or None if no title pattern matches.
    """
    for pat in _TITLE_PATTERNS:
        m = pat.search(question)
        if m:
            title = m.group(1).strip()
            # Skip very short or generic matches
            if len(title) >= 8:
                return title
    return None


# ---------------------------------------------------------------------------
# Page search by title string
# ---------------------------------------------------------------------------

def find_pages_by_title(
    page_texts: list[str],
    title: str,
    window_before: int = 1,
    window_after: int = 2,
    min_match_ratio: float = 0.6,
) -> list[int]:
    """Find pages whose text contains *title* (or a long prefix of it).

    Uses a sliding prefix match so that titles truncated by PDF extraction
    still hit the correct page.

    Parameters
    ----------
    page_texts:      Per-page extracted text strings (index = page number).
    title:           Element title to search for.
    window_before:   Extra pages to include before each match.
    window_after:    Extra pages to include after each match (useful for
                     multi-page tables where caption is on page N but data
                     continues to page N+1, N+2).
    min_match_ratio: Fraction of the title that must appear verbatim on a
                     page for it to count as a match.

    Returns
    -------
    Sorted list of page indices (0-based) to include.
    """
    title_lower = title.lower().strip()
    # Try progressively shorter prefixes (helps when PDF text truncates long titles)
    min_len = max(12, int(len(title_lower) * min_match_ratio))

    match_pages: list[int] = []
    for i, text in enumerate(page_texts):
        text_lower = text.lower()
        # Full title match
        if title_lower in text_lower:
            match_pages.append(i)
            continue
        # Prefix match (try 90%, 80%, 70%, min_match_ratio of title)
        for ratio in (0.9, 0.8, 0.7, min_match_ratio):
            prefix_len = max(min_len, int(len(title_lower) * ratio))
            if prefix_len >= len(title_lower):
                continue
            if title_lower[:prefix_len] in text_lower:
                match_pages.append(i)
                break

    if not match_pages:
        return []

    # Expand each match with a window
    pages: set[int] = set()
    n = len(page_texts)
    for idx in match_pages:
        for j in range(max(0, idx - window_before), min(n, idx + window_after + 1)):
            pages.add(j)

    return sorted(pages)


# ---------------------------------------------------------------------------
# BM25 page ranking (no external dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer (strips punctuation, keeps alphanumerics)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_top_pages(
    page_texts: list[str],
    query: str,
    top_k: int = 6,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[int]:
    """Return the *top_k* page indices most relevant to *query* via BM25.

    Returns pages sorted by score descending (highest first).
    Pages with zero score are excluded.
    """
    if not page_texts:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return list(range(min(top_k, len(page_texts))))

    # Tokenize all pages once
    tokenized_pages = [_tokenize(t) for t in page_texts]
    n = len(page_texts)
    avg_len = sum(len(t) for t in tokenized_pages) / max(n, 1)

    # Compute document frequency per term
    df: dict[str, int] = {}
    for tokens in tokenized_pages:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    scores: list[float] = []
    for i, tokens in enumerate(tokenized_pages):
        doc_len = len(tokens)
        tf_map = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            # IDF (smoothed)
            idf = math.log((n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
            # BM25 TF normalisation
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))
            score += idf * tf_norm
        scores.append(score)

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return [i for i in ranked[:top_k] if scores[i] > 0]


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def select_relevant_pages(
    page_texts: list[str],
    question: str,
    top_k: int = 8,
    window_before: int = 1,
    window_after: int = 2,
    fallback_n: int = 5,
) -> list[int]:
    """Select the most relevant pages for a question from a long PDF.

    Parameters
    ----------
    page_texts:   Per-page text extracted from the PDF (index = page 0-based).
    question:     The quest question string.
    top_k:        Maximum pages to return from BM25 fallback.
    window_before / window_after:
                  Pages to include around each title-matched page.
    fallback_n:   Number of pages to return if all methods find nothing.

    Returns
    -------
    Sorted list of page indices (0-based).  Never empty (falls back to
    [0 .. min(fallback_n, len)-1]).
    """
    n = len(page_texts)
    if n == 0:
        return []

    # 1. Title-based exact search
    title = extract_title(question)
    if title:
        pages = find_pages_by_title(
            page_texts, title,
            window_before=window_before,
            window_after=window_after,
        )
        if pages:
            logger.debug(
                "Title match '%s...' -> pages %s",
                title[:40], pages,
            )
            return pages[:top_k]  # cap in case title appears on many pages
        else:
            logger.debug("Title '%s...' not found in page texts; falling back to BM25", title[:40])

    # 2. BM25 ranking
    pages = bm25_top_pages(page_texts, question, top_k=top_k)
    if pages:
        logger.debug("BM25 top pages: %s", pages)
        return pages

    # 3. First-N fallback
    logger.debug("No signal found; using first %d pages", fallback_n)
    return list(range(min(fallback_n, n)))


# ---------------------------------------------------------------------------
# Diagnostics helper (for testing / logging)
# ---------------------------------------------------------------------------

def explain_page_selection(
    page_texts: list[str],
    question: str,
    top_k: int = 8,
) -> dict:
    """Return a diagnostic dict explaining how pages were selected."""
    title = extract_title(question)
    title_pages: list[int] = []
    bm25_pages: list[int] = []

    if title:
        title_pages = find_pages_by_title(page_texts, title)

    if not title_pages:
        bm25_pages = bm25_top_pages(page_texts, question, top_k=top_k)

    method = "title" if title_pages else ("bm25" if bm25_pages else "fallback")
    selected = title_pages or bm25_pages or list(range(min(5, len(page_texts))))

    return {
        "extracted_title": title,
        "method": method,
        "title_match_pages": title_pages,
        "bm25_top_pages": bm25_pages,
        "selected_pages": selected[:top_k],
    }
