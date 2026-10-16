# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Markdown document builder.

Loads a variant catalog, samples a document structure from a recipe,
selects variants, and renders Markdown using Jinja2 templates.

Text sections are the spine of the document. Tables and images are
supporting elements embedded in the text flow without hard separators
between consecutive text blocks.
"""
from __future__ import annotations

import re
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from synthpdf.config import DatasetConfig, RecipeConfig
from synthpdf.rng import fork_rng
from synthpdf.types import DocumentSpec, SectionSpec, Variant


_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _dedup_qa(qa_iter: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return QA candidates with duplicate questions removed (first occurrence kept)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for qa in qa_iter:
        key = qa.get("question", "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(qa)
    return out


def _renumber_variant(v: Variant, label: str, n: int) -> Variant:
    """Return a copy of the variant with caption renumbered to document order.

    Strips any existing "Table X:" / "Figure X:" prefix so the number always
    reflects the variant's sequential position in *this* document.
    """
    old_cap = v.render_payload.get("caption", "")
    body = re.sub(r"^(?:Table|Figure|Image)\s+\d+[:.]\s*", "", old_cap).strip()
    new_cap = f"{label} {n}: {body}" if body else f"{label} {n}"
    new_payload = {**v.render_payload, "caption": new_cap}
    new_gt = {**v.ground_truth, "caption": new_cap} if "caption" in v.ground_truth else v.ground_truth
    return Variant(id=v.id, kind=v.kind, metadata=v.metadata,
                   render_payload=new_payload, ground_truth=new_gt)


def _get_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        keep_trailing_newline=True,
    )


def _sample_section_type(rng: np.random.Generator, recipe: RecipeConfig) -> str:
    """Sample a section type from the recipe's layout probabilities."""
    types = list(recipe.layout_probabilities.keys())
    probs = [recipe.layout_probabilities[t] for t in types]
    total = sum(probs)
    probs = [p / total for p in probs]
    idx = int(rng.choice(len(types), p=probs))
    return types[idx]


def _select_variants_for_section(
    section_type: str,
    pool: dict[str, list[Variant]],
    rng: np.random.Generator,
    n: int = 1,
) -> list[Variant]:
    """Pick *n* variants from the appropriate pool."""
    pool_key = {
        "table": "tables",
        "image": "images",
        "text": "texts",
        "annotation": "annotations",
    }.get(section_type, "texts")
    available = pool.get(pool_key, [])
    if not available:
        return []
    n = min(n, len(available))
    indices = list(rng.choice(len(available), size=n, replace=False))
    return [available[int(i)] for i in indices]


def _render_section(
    env: Environment,
    section_type: str,
    variants: list[Variant],
) -> str:
    """Render a single section using the appropriate Jinja template."""
    template_map = {
        "table": "sections/table_section.md.j2",
        "image": "sections/image_section.md.j2",
        "text": "sections/text_section.md.j2",
        "annotation": "sections/text_section.md.j2",
    }
    tmpl_name = template_map.get(section_type, "sections/text_section.md.j2")
    tmpl = env.get_template(tmpl_name)
    return tmpl.render(variants=variants, section_type=section_type)


def build_document(
    doc_spec: DocumentSpec,
    variant_pool: dict[str, list[Variant]],
    output_dir: Path,
    recipe: RecipeConfig,
    doc_rng: np.random.Generator,
) -> tuple[Path, list[Variant], dict[str, Any]]:
    """Build one Markdown document from the spec.

    Sections are passed to the base template as {"type": str, "content": str}
    dicts so the template can apply separators selectively — only around
    tables and images, not between consecutive text blocks.

    Returns:
        (md_path, variants_used, document_ground_truth)
    """
    env = _get_jinja_env()
    sections_data: list[dict[str, str]] = []   # {"type": ..., "content": ...}
    variants_used: list[Variant] = []
    sections_gt: list[dict[str, Any]] = []

    # Sequential per-type counters so "Table 1, Table 2…" and "Figure 1, Figure 2…"
    # always reflect appearance order in this specific document.
    table_n = 0
    image_n = 0
    position = 0  # 1-based position across all section types

    # Build per-type shuffled deques so each variant is used at most once before
    # cycling.  This prevents identical text/table/image blocks from appearing
    # multiple times in the same document even when the pool is smaller than the
    # requested section count.
    _KEY_MAP = {
        "table": "tables",
        "image": "images",
        "text": "texts",
        "annotation": "annotations",
    }
    _pool_queues: dict[str, deque] = {}
    for _pkey in ("texts", "tables", "images", "annotations"):
        _items = list(variant_pool.get(_pkey, []))
        if _items:
            _q_rng = fork_rng(doc_rng, f"queue_{_pkey}")
            if _pkey == "images":
                # Stratified interleaving: split the images pool by subtype
                # (chart / scanned / infographic), shuffle each group independently,
                # then merge round-robin so every subtype appears early in the queue
                # regardless of pool proportions or RNG seed.
                _sc_ids = {v.id for v in variant_pool.get("scanned",      [])}
                _in_ids = {v.id for v in variant_pool.get("infographics", [])}
                _g_ch = [v for v in _items if v.id not in _sc_ids and v.id not in _in_ids]
                _g_sc = [v for v in _items if v.id in _sc_ids]
                _g_in = [v for v in _items if v.id in _in_ids]
                _ordered: list = []
                for _grp in (_g_ch, _g_sc, _g_in):
                    if _grp:
                        _sp = list(_q_rng.permutation(len(_grp)))
                        _grp[:] = [_grp[_j] for _j in _sp]
                # Round-robin interleave: 1 chart, 1 scanned, 1 infographic, repeat
                _bufs = [deque(_g) for _g in (_g_ch, _g_sc, _g_in) if _g]
                while _bufs:
                    for _buf in list(_bufs):
                        _ordered.append(_buf.popleft())
                        if not _buf:
                            _bufs.remove(_buf)
                _pool_queues[_pkey] = deque(_ordered)
            else:
                _perm = list(_q_rng.permutation(len(_items)))
                _pool_queues[_pkey] = deque(_items[_j] for _j in _perm)
        else:
            _pool_queues[_pkey] = deque()

    for i, section in enumerate(doc_spec.sections):
        sec_rng = fork_rng(doc_rng, f"section_{i}")

        if section.variant_ids:
            id_set = set(section.variant_ids)
            pool_key = _KEY_MAP.get(section.section_type, "texts")
            pool_list = variant_pool.get(pool_key, [])
            selected = [v for v in pool_list if v.id in id_set]
        else:
            pool_key = _KEY_MAP.get(section.section_type, "texts")
            queue = _pool_queues.get(pool_key, deque())
            if not queue:
                continue
            selected = [queue.popleft()]

        if not selected:
            continue

        # Renumber table/image captions to document-order sequential numbers
        if section.section_type == "table":
            table_n += 1
            selected = [_renumber_variant(v, "Table", table_n) for v in selected]
        elif section.section_type == "image":
            image_n += 1
            selected = [_renumber_variant(v, "Figure", image_n) for v in selected]

        section.variant_ids = [v.id for v in selected]
        position += 1

        rendered_md = _render_section(env, section.section_type, selected)
        sections_data.append({"type": section.section_type, "content": rendered_md})
        variants_used.extend(selected)

        gt_entry: dict[str, Any] = {
            "position": position,
            "section_type": section.section_type,
            "variant_ids": [v.id for v in selected],
            "ground_truth": [v.ground_truth for v in selected],
        }
        if section.section_type == "table":
            gt_entry["table_number"] = table_n
        elif section.section_type == "image":
            gt_entry["figure_number"] = image_n
        elif section.section_type == "text" and selected:
            # Promote form variants placed in text slots to their actual subtype
            # so the manifest and quest file carry the correct element type.
            from synthpdf.variants.forms import FORM_SUBTYPES as _FORM_SUBTYPES
            _fsub = selected[0].metadata.get("subtype", "")
            if _fsub in _FORM_SUBTYPES:
                gt_entry["section_type"] = _fsub
        sections_gt.append(gt_entry)

    # Optionally append annotation section (draw from same deque to avoid repeats)
    _ann_queue = _pool_queues.get("annotations", deque())
    if (
        _ann_queue
        and doc_rng.random() < recipe.annotation_attachment_probability
    ):
        anns = [_ann_queue.popleft()]
        if anns:
            position += 1
            rendered_md = _render_section(env, "annotation", anns)
            sections_data.append({"type": "annotation", "content": rendered_md})
            variants_used.extend(anns)
            sections_gt.append({
                "position": position,
                "section_type": "annotation",
                "variant_ids": [v.id for v in anns],
                "ground_truth": [v.ground_truth for v in anns],
            })

    title = doc_spec.title or doc_spec.doc_id
    subtitle = doc_spec.subtitle or ""

    md_path = output_dir / "markdown" / f"{doc_spec.doc_id}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    render_sections_to_markdown(title, subtitle, sections_data, md_path)

    document_gt = {
        "doc_id": doc_spec.doc_id,
        "title": title,
        "subtitle": subtitle,
        "sections": sections_gt,
        "all_qa_candidates": _dedup_qa(
            qa
            for v in variants_used
            for qa in v.ground_truth.get("qa_candidates", [])
        ),
        # Temporary: stores rendered section dicts so the CLI fill loop can
        # interleave new sections throughout the document instead of appending.
        # Stripped from gt before the manifest is written.
        "_sections_data": sections_data,
    }

    return md_path, variants_used, document_gt


def render_sections_to_markdown(
    title: str,
    subtitle: str,
    sections_data: list[dict[str, str]],
    md_path: Path,
) -> None:
    """Re-render a list of section dicts (``{"type": ..., "content": ...}``) to
    the markdown file, replacing its current contents.

    Used by the CLI fill loop to interleave new sections into an existing
    document without having to re-parse the raw markdown text.
    """
    env = _get_jinja_env()
    base_tmpl = env.get_template("base.md.j2")
    full_md = base_tmpl.render(title=title, subtitle=subtitle, sections=sections_data)
    md_path.write_text(full_md, encoding="utf-8")


def sample_doc_spec(
    doc_id: str,
    seed: int,
    recipe: RecipeConfig,
    rng: np.random.Generator,
) -> DocumentSpec:
    """Sample a DocumentSpec from a recipe using the given RNG."""
    n_sections = int(
        rng.integers(recipe.section_count_min, recipe.section_count_max + 1)
    )
    sections: list[SectionSpec] = []
    for i in range(n_sections):
        s_type = _sample_section_type(rng, recipe)
        sections.append(
            SectionSpec(
                section_type=s_type,  # type: ignore[arg-type]
                variant_ids=[],
                layout={},
            )
        )
    return DocumentSpec(
        doc_id=doc_id,
        seed=seed,
        recipe_name=recipe.name,
        sections=sections,
    )
