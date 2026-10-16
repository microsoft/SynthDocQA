# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Internal document pipeline used by the unconditional public wrapper."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import click

from synthpdf.config import AzureOpenAIConfig, DatasetConfig, RecipeConfig
from synthpdf.rng import make_rng, fork_rng
from synthpdf.types import Variant

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

# Shared LLM client injected by run_config() for batch runs (avoids re-authentication).
_SHARED_LLM_CLIENT: Any | None = None


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _hms(seconds: float) -> str:
    """Format seconds as h:mm:ss or m:ss."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s ({seconds:.1f}s)"


def _phase(label: str, t0: float) -> float:
    """Print elapsed time for a phase and return the current timestamp."""
    elapsed = time.perf_counter() - t0
    click.echo(f"      ↳ {label}: {_hms(elapsed)}")
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Variant pool generation helpers
# ---------------------------------------------------------------------------

def _vc(cfg: DatasetConfig) -> Any:
    """Return the effective VariantsConfig — new detailed config or built from legacy counts."""
    if cfg.variants is not None:
        return cfg.variants
    from synthpdf.config import (
        VariantsConfig, TableVariantConfig, TableBasicConfig,
        TableMultipageConfig, TableComplexConfig,
        ImageVariantConfig, TextVariantConfig, FormVariantConfig,
        AnnotationVariantConfig, ScannedVariantConfig, InfographicVariantConfig,
    )
    vc = cfg.variant_counts
    return VariantsConfig(
        tables=TableVariantConfig(
            basic=TableBasicConfig(count=vc.tables),
            multipage=TableMultipageConfig(count=vc.multipage_tables),
            complex=TableComplexConfig(count=vc.complex_tables),
        ),
        images=ImageVariantConfig(count=vc.images),
        texts=TextVariantConfig(count=vc.texts),
        annotations=AnnotationVariantConfig(count=vc.annotations),
        forms=FormVariantConfig(count=vc.forms),
        scanned=ScannedVariantConfig(count=vc.scanned),
        infographics=InfographicVariantConfig(count=vc.infographics),
    )


def _gen_pool_from_subtypes(generators_map: dict, subtypes: list, rng: Any,
                             count: int, assets_dir: Path) -> list[Variant]:
    """Generate *count* variants cycling through the enabled *subtypes*."""
    active = [(k, g) for k, g in generators_map.items() if k in subtypes]
    if not active or count <= 0:
        return []
    result: list[Variant] = []
    per = max(1, count // len(active))
    remainder = count - per * len(active)
    for i, (name, gen) in enumerate(active):
        n = per + (1 if i < remainder else 0)
        result.extend(gen(fork_rng(rng, name), n, assets_dir))
    return result


def _generate_variant_pool(cfg: DatasetConfig) -> dict[str, list[Variant]]:
    """Generate all variant pools according to config (random fallback path)."""
    from synthpdf.variants.tables import generate_table_variants, generate_multipage_table_variants
    from synthpdf.variants.complex_tables import generate_complex_table_variants
    from synthpdf.variants.images import generate_image_variants
    from synthpdf.variants.text_annotations import generate_text_variants
    from synthpdf.variants.forms import (
        generate_checklist_variants, generate_button_variants,
        generate_form_variants, generate_fillable_form_variants,
        generate_filled_form_variants, generate_filled_fillable_variants,
    )
    from synthpdf.variants.scanned import (
        generate_scanned_text_variants, generate_scanned_table_variants,
        generate_handwriting_variants, generate_stamp_mark_variants,
    )
    from synthpdf.variants.infographics import (
        generate_stat_dashboard_variants, generate_timeline_variants,
        generate_comparison_variants, generate_process_step_variants,
        generate_hub_spoke_variants, generate_magazine_variants,
        generate_pictograph_variants, generate_floor_plan_variants,
    )
    from synthpdf.variants.annotations import (
        generate_highlight_variants, generate_callout_variants,
        generate_label_variants, generate_sticky_note_variants,
        generate_freetext_variants, generate_review_variants,
        generate_annotated_report_variants, generate_colored_table_variants,
    )

    v = _vc(cfg)
    root_rng = make_rng(cfg.global_seed)
    assets_dir = cfg.output_dir / "assets"

    # ── Tables ──────────────────────────────────────────────────────────────
    tables: list[Variant] = []
    if v.tables.enabled:
        if v.tables.basic.enabled and v.tables.basic.count > 0:
            click.echo(f"  tables.basic: {v.tables.basic.count}")
            tables += list(generate_table_variants(
                fork_rng(root_rng, "tables"), v.tables.basic.count, assets_dir))
        if v.tables.multipage.enabled and v.tables.multipage.count > 0:
            click.echo(f"  tables.multipage: {v.tables.multipage.count}")
            tables += list(generate_multipage_table_variants(
                fork_rng(root_rng, "multipage_tables"), v.tables.multipage.count, assets_dir))
        if v.tables.complex.enabled and v.tables.complex.count > 0:
            click.echo(f"  tables.complex: {v.tables.complex.count}")
            tables += list(generate_complex_table_variants(
                fork_rng(root_rng, "complex_tables"), v.tables.complex.count, assets_dir))

    # ── Images ──────────────────────────────────────────────────────────────
    images: list[Variant] = []
    if v.images.enabled and v.images.count > 0:
        click.echo(f"  images: {v.images.count}")
        images += list(generate_image_variants(
            fork_rng(root_rng, "images"), v.images.count, assets_dir))

    # ── Texts ───────────────────────────────────────────────────────────────
    texts: list[Variant] = []
    if v.texts.enabled and v.texts.count > 0:
        click.echo(f"  texts: {v.texts.count} (subtypes: {v.texts.subtypes})")
        texts += list(generate_text_variants(
            fork_rng(root_rng, "texts"), v.texts.count, assets_dir))

    # ── Forms ───────────────────────────────────────────────────────────────
    forms: list[Variant] = []
    if v.forms.enabled and v.forms.count > 0:
        click.echo(f"  forms: {v.forms.count} (subtypes: {v.forms.subtypes})")
        forms = _gen_pool_from_subtypes({
            "checklist":      generate_checklist_variants,
            "button":         generate_button_variants,
            "form":           generate_form_variants,
            "fillable":       generate_fillable_form_variants,
            "filled_form":    generate_filled_form_variants,
            "filled_fillable": generate_filled_fillable_variants,
        }, v.forms.subtypes, fork_rng(root_rng, "forms"), v.forms.count, assets_dir)

    # ── Scanned ─────────────────────────────────────────────────────────────
    scanned: list[Variant] = []
    if v.scanned.enabled and v.scanned.count > 0:
        click.echo(f"  scanned: {v.scanned.count} (subtypes: {v.scanned.subtypes})")
        scanned = _gen_pool_from_subtypes({
            "text":        generate_scanned_text_variants,
            "table":       generate_scanned_table_variants,
            "handwriting": generate_handwriting_variants,
            "stamps":      generate_stamp_mark_variants,
        }, v.scanned.subtypes, fork_rng(root_rng, "scanned"), v.scanned.count, assets_dir)

    # ── Infographics ─────────────────────────────────────────────────────────
    infographics: list[Variant] = []
    if v.infographics.enabled and v.infographics.count > 0:
        click.echo(f"  infographics: {v.infographics.count} (subtypes: {v.infographics.subtypes})")
        infographics = _gen_pool_from_subtypes({
            "stat_dashboard": generate_stat_dashboard_variants,
            "timeline":       generate_timeline_variants,
            "comparison":     generate_comparison_variants,
            "process_steps":  generate_process_step_variants,
            "hub_spoke":      generate_hub_spoke_variants,
            "magazine":       generate_magazine_variants,
            "pictograph":     generate_pictograph_variants,
            "floor_plan":     generate_floor_plan_variants,
        }, v.infographics.subtypes, fork_rng(root_rng, "infographics"), v.infographics.count, assets_dir)

    # ── Annotations ──────────────────────────────────────────────────────────
    annotations: list[Variant] = []
    if v.annotations.enabled and v.annotations.count > 0:
        click.echo(f"  annotations: {v.annotations.count} (subtypes: {v.annotations.subtypes})")
        annotations = _gen_pool_from_subtypes({
            "highlights":       generate_highlight_variants,
            "callouts":         generate_callout_variants,
            "labels":           generate_label_variants,
            "sticky_notes":     generate_sticky_note_variants,
            "freetext":         generate_freetext_variants,
            "review":           generate_review_variants,
            "annotated_report": generate_annotated_report_variants,
            "colored_table":    generate_colored_table_variants,
        }, v.annotations.subtypes, fork_rng(root_rng, "annotations"), v.annotations.count, assets_dir)

    return {
        "tables":       tables,
        "images":       images + scanned + infographics,
        "texts":        texts + forms,
        "annotations":  annotations,
        "forms":        forms,
        "scanned":      scanned,
        "infographics": infographics,
    }


def _effective_recipe(recipe: RecipeConfig, vc: Any) -> RecipeConfig:
    """Return a recipe copy with layout_probabilities zeroed for disabled variant types.

    Prevents wasted LLM calls and incorrect section composition when a config
    disables a variant type (e.g. ``variants.tables.enabled: false``) while the
    recipe still lists a non-zero probability for that type.
    """
    # The "image" layout slot is shared by seaborn charts, scanned pages, and
    # infographic pages.  Only zero it out when ALL three are disabled so that
    # configs like cfg_077 (images.enabled=false but scanned/infographics=true)
    # still get image section slots for their scanned/infographic variants.
    image_slot_active = (
        vc.images.enabled
        or vc.scanned.enabled
        or vc.infographics.enabled
    )
    type_enabled = {
        "table": vc.tables.enabled,
        "image": image_slot_active,
        "text":  vc.texts.enabled,
    }
    disabled = {k for k, enabled in type_enabled.items() if not enabled}
    if not disabled:
        return recipe  # nothing disabled, use as-is

    new_probs = {k: v for k, v in recipe.layout_probabilities.items() if k not in disabled}
    if not new_probs:
        # All types disabled — fall back to text so the builder can still run.
        new_probs = {"text": 1.0}
    return recipe.model_copy(update={"layout_probabilities": new_probs})


def _scale_sections(
    target_pages: int,
    recipe: RecipeConfig,
    use_llm: bool = True,
) -> dict[str, int]:
    """Return per-type variant pool counts and the target document section count.

    Only types present in recipe.layout_probabilities receive non-zero counts.
    Missing types default to 0.0 (never 0.3 or 0.4).

    LLM-mode caps prevent runaway API call counts:
      text   <= 2   (each requires 2 LLM calls)
      table  <= 6   (each requires 2 LLM calls)
      image  <= 4

    Calibrated heuristics per section (pages consumed in rendered PDF):
      text ~0.50 pp (LLM) / 0.30 pp (random)
      table ~0.80 pp
      image ~1.00 pp
      annotation ~0.10 pp

    These heuristics are intentionally slightly conservative to avoid
    generating far more pages than requested.  The pool counts (texts,
    tables, images) are LLM-capped, but ``_doc_section_count`` — the value
    used for ``adapted_recipe.section_count_range`` — is derived from the
    page-budget formula and is NOT capped by pool sizes, so the document
    builder repeats pool variants as needed to fill the target page count.

    Multipage tables are computed first (hard cap: 2) and their expected
    page contribution (~3 pp each) is subtracted from the target before
    computing regular section counts.
    """
    probs = recipe.layout_probabilities
    total = sum(probs.values()) or 1.0

    # Use 0.0 as default — only count types that the recipe explicitly lists
    text_frac       = probs.get("text",       0.0) / total
    table_frac      = probs.get("table",      0.0) / total
    image_frac      = probs.get("image",      0.0) / total
    annotation_frac = probs.get("annotation", 0.0) / total

    # Calibrated pages-per-section heuristics (validated against observed output).
    # These are higher than the old 0.65/0.6/0.8 values, which underestimated
    # actual page consumption and caused target_pages to be badly missed.
    text_pp = 0.50 if use_llm else 0.30

    # --- Multipage tables: reserve budget upfront, hard cap at 2 ---
    # Each multipage table spans ~3 pages (50-80 rows @ 20 rows/page).
    n_mp_tables = min(2, max(1, round(target_pages / 20)))
    remaining_pages = max(4, target_pages - n_mp_tables * 3)

    avg_pp = (
        text_frac * text_pp
        + table_frac * 0.80
        + image_frac * 1.00
        + annotation_frac * 0.10
    )
    # n_sections: how many document sections are needed to fill remaining_pages.
    # Capped at 300 to prevent runaway on very large targets.
    n_sections = max(4, min(300, int(remaining_pages / max(avg_pp, 0.1))))

    # Raw pool counts from page-budget formula
    n_texts_raw  = max(1, round(n_sections * text_frac))  if text_frac  > 0 else 0
    n_tables_raw = max(1, round(n_sections * table_frac)) if table_frac > 0 else 0
    n_images_raw = max(1, round(n_sections * image_frac)) if image_frac > 0 else 0

    # Auxiliary pool types: small fixed counts based on remaining page budget.
    # Computed BEFORE n_images so we can guarantee seaborn charts outnumber
    # the combined aux pool (scanned + infographics each get _aux items).
    _aux = max(2, min(5, round(remaining_pages / 10)))

    # Hard caps for LLM mode: keep API call count manageable.
    # The variant pool is sampled WITH REPLACEMENT when building docs,
    # so fewer unique variants does not mean fewer pages — just less variety.
    if use_llm:
        n_texts  = min(2, n_texts_raw)
        n_tables = min(6, n_tables_raw)
        # Chart images share the "images" pool with scanned + infographics (_aux each).
        # Ensure seaborn chart count exceeds combined aux count so they are reliably
        # selected when the document assembler draws image slots at random.
        n_images = max(min(6, n_images_raw), _aux * 2 + 1)
    else:
        n_texts  = n_texts_raw
        n_tables = n_tables_raw
        n_images = n_images_raw

    return {
        "texts":  n_texts,
        "tables": n_tables,
        "images": n_images,
        "annotations":      max(1, round(n_sections * annotation_frac)) if annotation_frac > 0 else 2,
        "forms":            _aux,
        "scanned":          _aux,
        "infographics":     _aux,
        "multipage_tables": n_mp_tables,
        "complex_tables":   min(4, max(1, round(remaining_pages / 8))),
        # Key fix: document section count is the page-budget value, NOT sum(pool counts).
        # Pool counts are LLM-capped to 2/6/4 but doc sections repeat from pool as needed.
        "_doc_section_count": n_sections,
    }


def _generate_llm_variant_pool(
    client: Any,
    topic: Any,
    rng: Any,
    counts: dict[str, int],
    assets_dir: Path,
    max_workers: int = 12,
    vc: Any = None,   # VariantsConfig — passed from CLI to honour subtype flags
) -> dict[str, list[Variant]]:
    """Generate per-document LLM-driven variant pools.

    LLM is used for texts, tables, and images.  Forms, scanned, infographics,
    and annotations use their random generators (no LLM needed).
    """
    from synthpdf.llm.content_generator import ContentGenerator
    from synthpdf.variants.tables import generate_llm_table_variants, generate_llm_multipage_table_variants
    from synthpdf.variants.complex_tables import generate_complex_table_variants
    from synthpdf.variants.images import generate_llm_image_variants
    from synthpdf.variants.text_annotations import (
        generate_llm_text_variants,
        generate_llm_structured_text_block_variants,
    )
    from synthpdf.variants.forms import (
        generate_checklist_variants, generate_button_variants,
        generate_form_variants, generate_fillable_form_variants,
        generate_filled_form_variants, generate_filled_fillable_variants,
    )
    from synthpdf.variants.scanned import (
        generate_scanned_text_variants, generate_scanned_table_variants,
        generate_handwriting_variants, generate_stamp_mark_variants,
        generate_llm_scanned_variants,
    )
    from synthpdf.variants.infographics import (
        generate_stat_dashboard_variants, generate_timeline_variants,
        generate_comparison_variants, generate_process_step_variants,
        generate_hub_spoke_variants, generate_magazine_variants,
        generate_pictograph_variants, generate_floor_plan_variants,
        generate_llm_stat_dashboard_variants, generate_llm_floor_plan_variants,
        generate_llm_timeline_variants, generate_llm_comparison_variants,
        generate_llm_process_step_variants, generate_llm_hub_spoke_variants,
        generate_llm_magazine_variants, generate_llm_pictograph_variants,
    )
    from synthpdf.variants.annotations import (
        generate_highlight_variants, generate_callout_variants,
        generate_label_variants, generate_sticky_note_variants,
        generate_freetext_variants, generate_review_variants,
        generate_annotated_report_variants, generate_colored_table_variants,
    )

    # Respect enabled flags from VariantsConfig — counts should already be 0
    # for disabled types (filtered by _effective_recipe), but guard here too.
    n_texts  = counts["texts"]  if (not vc or vc.texts.enabled)  else 0
    n_tables = counts["tables"] if (not vc or vc.tables.enabled) else 0
    n_images = counts["images"] if (not vc or vc.images.enabled) else 0
    _n_sc = counts.get("scanned", 0)
    _n_in = counts.get("infographics", 0)
    # Each artifact: 2 LLM calls (content/spec + QA). +3 for brief + title/subtitle.
    total_calls = (n_texts + n_tables + n_images + _n_sc + _n_in) * 2 + 3
    click.echo(
        f"    Artifacts: {n_texts} text, {n_tables} table, {n_images} charts, "
        f"{_n_sc} scanned, {_n_in} infographic  ->  ~{total_calls} LLM calls  (all parallel)"
    )

    pool_t0 = time.perf_counter()
    generator = ContentGenerator(client, topic)

    # Split text count: up to 1/3 are LLM structured blocks (email/policy/spec/etc.),
    # the remainder are LLM prose sections driven by the document outline.
    n_structured_texts = min(n_texts // 3, 3) if n_texts >= 3 else 0
    n_prose_texts = n_texts - n_structured_texts

    # Document Brief: one coherent story spine before parallel generation.
    # Replaces generate_document_outline() — sections carry key_facts and
    # data_references that tie prose, tables, and charts together.
    t = time.perf_counter()
    if n_prose_texts + n_tables + n_images > 0:
        brief = generator.generate_document_brief(n_prose_texts, n_tables, n_images)
    else:
        brief = {"narrative_theme": topic.description, "sections": [], "tables": [], "charts": []}
    outline = [s["heading"] for s in brief.get("sections", [])]
    click.echo(
        f"    Brief ({n_prose_texts} prose + {n_structured_texts} structured, "
        f"{n_tables} tables, {n_images} charts): {_hms(time.perf_counter() - t)}"
    )
    click.echo(f"    Theme: {brief.get('narrative_theme', '')[:100]}")
    click.echo(f"    Sections: {outline}")

    # LLM variants in parallel
    import concurrent.futures as _cf
    type_times: dict[str, float] = {}

    def _gen_texts() -> list[Any]:
        t0 = time.perf_counter()
        r: list[Any] = []
        if n_prose_texts > 0:
            r += generate_llm_text_variants(
                client, topic, fork_rng(rng, "llm_texts"),
                n_prose_texts, outline=outline, max_workers=max_workers,
                section_briefs=brief.get("sections", []),
            )
        if n_structured_texts > 0:
            r += generate_llm_structured_text_block_variants(
                client, topic, fork_rng(rng, "llm_struct_texts"), n_structured_texts,
            )
        type_times["texts"] = time.perf_counter() - t0
        return r

    def _gen_tables() -> list[Any]:
        t0 = time.perf_counter()
        r = generate_llm_table_variants(
            client, topic, fork_rng(rng, "llm_tables"),
            n_tables, max_workers=max_workers,
            table_briefs=brief.get("tables", []),
        )
        type_times["tables"] = time.perf_counter() - t0
        return r

    def _gen_images() -> list[Any]:
        t0 = time.perf_counter()
        r = generate_llm_image_variants(
            client, topic, fork_rng(rng, "llm_images"),
            n_images, assets_dir, max_workers=max_workers,
            chart_briefs=brief.get("charts", []),
        )
        type_times["images"] = time.perf_counter() - t0
        return r

    # Submit all three in parallel, then collect with a hard wall-clock timeout.
    # timeout_seconds * (max_retries+1) + 120s buffer covers worst-case retry chains.
    # Subtype flags (resolved here so lambdas can capture them).
    # Use enabled flag + non-empty subtypes check so that an empty subtypes list []
    # (from a disabled variant type) never collapses to None and triggers defaults.
    _DEFAULT_FORM_SUB    = ["checklist","button","form","fillable"]
    _DEFAULT_SCANNED_SUB = ["text","table","handwriting","stamps"]
    _DEFAULT_INFOGR_SUB  = ["stat_dashboard","timeline","comparison","process_steps",
                             "hub_spoke","magazine","pictograph","floor_plan"]
    _DEFAULT_ANN_SUB     = ["highlights","callouts","labels","sticky_notes",
                             "freetext","review","annotated_report","colored_table"]

    if vc and vc.forms.enabled and vc.forms.subtypes:
        form_subtypes = vc.forms.subtypes
    elif vc and not vc.forms.enabled:
        form_subtypes = []          # disabled — _gen_pool_from_subtypes returns [] immediately
    else:
        form_subtypes = _DEFAULT_FORM_SUB

    if vc and vc.scanned.enabled and vc.scanned.subtypes:
        scanned_subtypes = vc.scanned.subtypes
    elif vc and not vc.scanned.enabled:
        scanned_subtypes = []       # disabled — generate_llm_scanned_variants gets count=0 anyway
    else:
        scanned_subtypes = _DEFAULT_SCANNED_SUB

    if vc and vc.infographics.enabled and vc.infographics.subtypes:
        infographic_sub = vc.infographics.subtypes
    elif vc and not vc.infographics.enabled:
        infographic_sub = []        # disabled
    else:
        infographic_sub = _DEFAULT_INFOGR_SUB

    if vc and vc.annotations.enabled and vc.annotations.subtypes:
        ann_subtypes = vc.annotations.subtypes
    elif vc and not vc.annotations.enabled:
        ann_subtypes = []           # disabled
    else:
        ann_subtypes = _DEFAULT_ANN_SUB

    # Helper generators for the remaining pool types
    def _gen_mp_tables() -> list[Any]:
        t0 = time.perf_counter()
        r = generate_llm_multipage_table_variants(
            fork_rng(rng, "llm_mp_tables"), counts.get("multipage_tables", 1), generator, assets_dir)
        type_times["mp_tables"] = time.perf_counter() - t0
        return r

    def _gen_cx_tables() -> list[Any]:
        t0 = time.perf_counter()
        r = generate_complex_table_variants(
            fork_rng(rng, "llm_cx_tables"), counts.get("complex_tables", 1), assets_dir,
            client=client, topic=topic)
        type_times["cx_tables"] = time.perf_counter() - t0
        return r

    def _gen_forms() -> list[Any]:
        t0 = time.perf_counter()
        r = _gen_pool_from_subtypes({
            "checklist":       lambda r, n, a: generate_checklist_variants(r, n, a, client=client, topic=topic),
            "button":          lambda r, n, a: generate_button_variants(r, n, a, client=client, topic=topic),
            "form":            lambda r, n, a: generate_form_variants(r, n, a, client=client, topic=topic),
            "fillable":        lambda r, n, a: generate_fillable_form_variants(r, n, a, client=client, topic=topic),
            "filled_form":     lambda r, n, a: generate_filled_form_variants(r, n, a, client=client, topic=topic),
            "filled_fillable": lambda r, n, a: generate_filled_fillable_variants(r, n, a, client=client, topic=topic),
        }, form_subtypes, fork_rng(rng, "llm_forms"), counts.get("forms", 4), assets_dir)
        type_times["forms"] = time.perf_counter() - t0
        return r

    def _gen_scanned() -> list[Any]:
        t0 = time.perf_counter()
        r = generate_llm_scanned_variants(
            client=client,
            topic=topic,
            rng=fork_rng(rng, "llm_scanned"),
            count=counts.get("scanned", 3),
            assets_dir=assets_dir,
            subtypes=scanned_subtypes or None,   # empty list -> None uses defaults (guarded above)
            doc_brief=brief if brief else None,
            progress_cb=lambda msg: click.echo(f"      {msg}"),
        )
        type_times["scanned"] = time.perf_counter() - t0
        return r

    def _gen_infographics() -> list[Any]:
        t0 = time.perf_counter()
        r = _gen_pool_from_subtypes({
            "stat_dashboard": lambda r, n, a: generate_llm_stat_dashboard_variants(client, topic, r, n, a),
            "floor_plan":     lambda r, n, a: generate_llm_floor_plan_variants(client, topic, r, n, a),
            "timeline":       lambda r, n, a: generate_llm_timeline_variants(client, topic, r, n, a),
            "comparison":     lambda r, n, a: generate_llm_comparison_variants(client, topic, r, n, a),
            "process_steps":  lambda r, n, a: generate_llm_process_step_variants(client, topic, r, n, a),
            "hub_spoke":      lambda r, n, a: generate_llm_hub_spoke_variants(client, topic, r, n, a),
            "magazine":       lambda r, n, a: generate_llm_magazine_variants(client, topic, r, n, a),
            "pictograph":     lambda r, n, a: generate_llm_pictograph_variants(client, topic, r, n, a),
        }, infographic_sub, fork_rng(rng, "llm_infographics"), counts.get("infographics", 2), assets_dir)
        type_times["infographics"] = time.perf_counter() - t0
        return r

    def _gen_annotations() -> list[Any]:
        t0 = time.perf_counter()
        r = _gen_pool_from_subtypes({
            "highlights":       lambda r, n, a: generate_highlight_variants(r, n, a, client=client, topic=topic),
            "callouts":         lambda r, n, a: generate_callout_variants(r, n, a, client=client, topic=topic),
            "labels":           lambda r, n, a: generate_label_variants(r, n, a, client=client, topic=topic),
            "sticky_notes":     lambda r, n, a: generate_sticky_note_variants(r, n, a, client=client, topic=topic),
            "freetext":         lambda r, n, a: generate_freetext_variants(r, n, a, client=client, topic=topic),
            "review":           lambda r, n, a: generate_review_variants(r, n, a, client=client, topic=topic),
            "annotated_report": lambda r, n, a: generate_annotated_report_variants(r, n, a, client=client, topic=topic),
            "colored_table":    lambda r, n, a: generate_colored_table_variants(r, n, a, client=client, topic=topic),
        }, ann_subtypes, fork_rng(rng, "llm_annotations"), counts["annotations"], assets_dir)
        type_times["annotations"] = time.perf_counter() - t0
        return r

    # All 9 pool types run in parallel — wall time is the slowest single pool.
    # max_workers=9 is safe because all work is I/O-bound (LLM HTTP calls).
    _timeout_cfg = getattr(client, '_timeout_seconds', 120.0)
    _retries_cfg = getattr(client, '_max_retries', 3)
    _wall_timeout = _timeout_cfg * (_retries_cfg + 1) + 120.0

    _pool_jobs = [
        ("texts",        _gen_texts),
        ("tables",       _gen_tables),
        ("images",       _gen_images),
        ("mp_tables",    _gen_mp_tables),
        ("cx_tables",    _gen_cx_tables),
        ("forms",        _gen_forms),
        ("scanned",      _gen_scanned),
        ("infographics", _gen_infographics),
        ("annotations",  _gen_annotations),
    ]
    _pool_results: dict[str, list[Any]] = {}

    with _cf.ThreadPoolExecutor(max_workers=len(_pool_jobs)) as pool:
        _futures = {name: pool.submit(fn) for name, fn in _pool_jobs}
        for name, fut in _futures.items():
            try:
                _pool_results[name] = fut.result(timeout=_wall_timeout)
            except _cf.TimeoutError:
                click.echo(f"  WARNING: {name} generation timed out — skipping.", err=True)
                _pool_results[name] = []
            except Exception as exc:
                click.echo(f"  WARNING: {name} generation failed: {exc} — skipping.", err=True)
                _pool_results[name] = []

    texts        = _pool_results["texts"]
    tables       = _pool_results["tables"]
    images       = _pool_results["images"]
    mp_tables    = _pool_results["mp_tables"]
    cx_tables    = _pool_results["cx_tables"]
    forms        = _pool_results["forms"]
    scanned      = _pool_results["scanned"]
    infographics = _pool_results["infographics"]
    annotations  = _pool_results["annotations"]

    click.echo(
        f"    Text: {_hms(type_times.get('texts',0))}  |  "
        f"Tables: {_hms(type_times.get('tables',0))}  |  "
        f"Images: {_hms(type_times.get('images',0))}  |  "
        f"CxTbl: {_hms(type_times.get('cx_tables',0))}  |  "
        f"Forms: {_hms(type_times.get('forms',0))}  |  "
        f"Scanned: {_hms(type_times.get('scanned',0))}  |  "
        f"Infogr: {_hms(type_times.get('infographics',0))}  |  "
        f"Ann: {_hms(type_times.get('annotations',0))}  (all parallel)"
    )
    click.echo(f"    Variant pool total: {_hms(time.perf_counter() - pool_t0)}")
    try:
        from synthpdf import stats_sink as _ss
        _ss.set_type_times(type_times)
    except Exception:
        pass
    _n_ch = len(images); _n_sc = len(scanned); _n_in = len(infographics)
    click.echo(
        f"    Images pool: {_n_ch} charts, {_n_sc} scanned, {_n_in} infographic"
        f" = {_n_ch + _n_sc + _n_in} total"
    )

    return {
        "tables":          list(tables) + list(mp_tables) + list(cx_tables),
        "images":          list(images) + list(scanned) + list(infographics),
        "texts":           list(texts) + list(forms),
        "annotations":     annotations,
        "forms":           forms,
        "scanned":         scanned,
        "infographics":    infographics,
        "_document_brief": brief,   # Document-coherence metadata (not a Variant list)
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_config(config: Path, llm_client: Any | None = None, num_documents: int | None = None) -> None:
    """Run an internally sampled configuration; not a public conditioning interface."""
    global _SHARED_LLM_CLIENT
    _SHARED_LLM_CLIENT = llm_client
    try:
        _generate_docs(config, num_documents)
    finally:
        _SHARED_LLM_CLIENT = None

def _generate_docs(config: Path, num_documents: int | None = None) -> None:
    """Compose and render documents, writing manifest entries."""
    from synthpdf.manifest import write_document_manifest, append_run_summary, write_quest_file
    from synthpdf.compose.markdown_builder import build_document, sample_doc_spec
    from synthpdf.render.md_to_pdf import render_markdown_to_pdf
    from synthpdf.render.pdf_annotate import apply_pdf_annotations

    cfg = DatasetConfig.from_yaml(config)
    if num_documents is not None:
        cfg = cfg.model_copy(update={"num_documents": num_documents})

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = cfg.output_dir / "assets"

    # Determine if LLM mode is active.
    # When called via run_config(), a shared client is injected here so that
    # batch calls authenticate only once.
    import synthpdf._pipeline as _cli_mod
    llm_client: Any = _cli_mod._SHARED_LLM_CLIENT
    topics: list[Any] = []

    if llm_client is not None:
        click.echo("Azure OpenAI client provided externally (shared across batch).")
    elif cfg.azure_openai is not None:
        from synthpdf.llm.client import AzureOpenAIClient
        llm_client = AzureOpenAIClient(cfg.azure_openai)
        click.echo("Azure OpenAI client initialized.")

    if cfg.topics_file is not None:
        from synthpdf.llm.content_generator import load_topics
        topics_path = Path(cfg.topics_file)
        if not topics_path.is_absolute():
            # Prefer CWD-relative (project root); fall back to config-file-relative
            cwd_relative = Path.cwd() / topics_path
            topics_path = cwd_relative if cwd_relative.exists() else config.parent / topics_path
        topics = load_topics(topics_path)
        click.echo(f"Loaded {len(topics)} topics from {topics_path}.")

    summary_path = cfg.output_dir / "manifest.jsonl"
    manifests_dir = cfg.output_dir / "manifests"
    quest_dir = cfg.output_dir / "quest"
    if summary_path.exists():
        summary_path.unlink()

    recipes = cfg.recipes or [RecipeConfig(name="default")]

    root_rng = make_rng(cfg.global_seed)
    doc_rng_base = fork_rng(root_rng, "documents")

    # Pre-generate global variant pool as fallback
    global_pool: dict[str, list[Variant]] | None = None

    # Build template index once (None if not configured / disabled)
    template_index = _make_template_index(cfg, config_path=config)

    # Resolve title_pages_dir
    _te_base_full = _resolve_te_base(cfg, config_path=config)
    _tp_dir = (_te_base_full / "title_pages") if _te_base_full is not None else None
    title_pages_dir_full = _tp_dir if (_tp_dir is not None and _tp_dir.exists()) else None

    run_t0 = time.perf_counter()

    for doc_idx in range(cfg.num_documents):
        doc_t0 = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            from synthpdf import stats_sink as _ss
            _ss.clear()
        except Exception:
            pass

        recipe = recipes[doc_idx % len(recipes)]
        doc_seed = int(doc_rng_base.integers(0, 2**31))
        doc_id = f"doc_{doc_idx:04d}_s{doc_seed}"

        doc_rng = make_rng(doc_seed)

        # Determine topic for this document
        topic = None
        if topics:
            topic_idx = int(doc_rng.integers(0, len(topics)))
            topic = topics[topic_idx]

        # Determine target page count and section counts
        if cfg.page_count is not None:
            target_pages = cfg.page_count
            pmin = pmax = cfg.page_count
        else:
            pmin, pmax = cfg.page_count_range[0], cfg.page_count_range[1]
            target_pages = int(doc_rng.integers(pmin, pmax + 1))
        use_llm = llm_client is not None and topic is not None
        # Filter recipe probabilities to respect variants.X.enabled flags.
        # This prevents wasted LLM calls and incorrect section composition
        # when a config disables a type (e.g. tables.enabled: false) while the
        # recipe still lists a non-zero probability for that type.
        effective_recipe = _effective_recipe(recipe, _vc(cfg))
        counts = _scale_sections(target_pages, effective_recipe, use_llm=use_llm)
        # Honor config-specified counts for auxiliary pools (scanned, infographics, forms).
        # _scale_sections always returns a positive _aux heuristic for these types regardless
        # of enabled flags, so we must explicitly zero disabled types and override counts
        # for enabled ones.
        if use_llm:
            _vc_obj = _vc(cfg)
            if _vc_obj is not None:
                if _vc_obj.scanned.enabled and _vc_obj.scanned.count > 0:
                    counts["scanned"]      = _vc_obj.scanned.count
                else:
                    counts["scanned"]      = 0   # disabled — prevent spurious generation
                if _vc_obj.infographics.enabled and _vc_obj.infographics.count > 0:
                    counts["infographics"] = _vc_obj.infographics.count
                else:
                    counts["infographics"] = 0   # disabled — prevent spurious generation
                if _vc_obj.forms.enabled and _vc_obj.forms.count > 0:
                    counts["forms"]        = _vc_obj.forms.count
                else:
                    counts["forms"]        = 0   # disabled — prevent spurious generation
        # Use page-budget section count (not sum of pool counts which are LLM-capped)
        doc_section_count = counts.pop("_doc_section_count")
        total_sections = sum(v for k, v in counts.items() if k != "annotations")

        # Build adapted recipe with fixed section count and filtered probabilities
        adapted_recipe = effective_recipe.model_copy(
            update={"section_count_range": [doc_section_count, doc_section_count]}
        )

        # Generate variant pool
        if llm_client is not None and topic is not None:
            click.echo(
                f"\n[{doc_idx+1}/{cfg.num_documents}] {doc_id} "
                f"(topic={topic.id}, pages~{target_pages})"
            )
            click.echo(f"  Generating LLM variants...")
            t = time.perf_counter()
            pool_rng = fork_rng(doc_rng, "pool")
            max_workers = cfg.azure_openai.max_parallel_llm_calls if cfg.azure_openai else 1
            pool = _generate_llm_variant_pool(
                llm_client, topic, pool_rng, counts, assets_dir,
                max_workers=max_workers,
                vc=_vc(cfg),
            )
            timings["llm_variants"] = time.perf_counter() - t
            llm_brief = pool.pop("_document_brief", {})
        else:
            click.echo(
                f"\n[{doc_idx+1}/{cfg.num_documents}] {doc_id} "
                f"(pages~{target_pages}) — using random variants"
            )
            t = time.perf_counter()
            pool = _get_global_pool(cfg, global_pool)
            global_pool = pool
            timings["llm_variants"] = time.perf_counter() - t
            llm_brief = {}

        # Build doc spec
        doc_rng2 = make_rng(doc_seed)
        doc_spec = sample_doc_spec(doc_id, doc_seed, adapted_recipe, doc_rng2)

        # Generate title and subtitle.
        # Priority: LLM (fresh, unique per doc) > topic description > doc_id fallback.
        t = time.perf_counter()
        if llm_client is not None and topic is not None:
            from synthpdf.llm.content_generator import ContentGenerator
            title_gen = ContentGenerator(llm_client, topic)
            try:
                doc_spec.title = title_gen.generate_document_title()
            except Exception:
                doc_spec.title = topic.description.split(".")[0].strip()
            try:
                doc_spec.subtitle = title_gen.generate_document_subtitle()
            except Exception:
                doc_spec.subtitle = ""
        elif topic is not None:
            # No LLM: use the topic description as a readable title / subtitle
            doc_spec.title = topic.id.replace("_", " ").title()
            doc_spec.subtitle = topic.description.split(".")[0].strip()
        # else: title stays as whatever sample_doc_spec set (may be None; _render_title_page
        # handles None gracefully via fallback text)
        timings["title_subtitle"] = time.perf_counter() - t
        click.echo(f"  Title: {doc_spec.title!r}")

        t = time.perf_counter()
        doc_rng3 = make_rng(doc_seed)

        # Resolve branding spec for markdown rendering (logo file / logos_dir)
        branding_spec = None
        branding_cfg = cfg.rendering.branding
        if branding_cfg is not None and branding_cfg.enabled:
            from synthpdf.render.branding import BrandingSpec, select_logo
            logo_path = None
            if branding_cfg.logo_file:
                p = Path(branding_cfg.logo_file)
                if not p.is_absolute():
                    # Resolve relative paths against the config file directory
                    p = config.parent / p
                if p.exists():
                    logo_path = p
            if logo_path is None and branding_cfg.logos_dir:
                logos_dir_p = Path(branding_cfg.logos_dir)
                if not logos_dir_p.is_absolute():
                    logos_dir_p = config.parent / logos_dir_p
                brand_rng = fork_rng(doc_rng, "branding")
                logo_path = select_logo(logos_dir_p, brand_rng)
            if logo_path is not None:
                branding_spec = BrandingSpec(
                    logo_path=logo_path,
                    placement=branding_cfg.placement,
                    position=branding_cfg.position,
                    pages=branding_cfg.pages,
                    logo_height_pt=branding_cfg.logo_height_pt,
                )

        pdf_path = cfg.output_dir / "pdfs" / f"{doc_id}.pdf"
        css_file = Path(cfg.rendering.css_file) if cfg.rendering.css_file else None
        md_path = None  # only set for pure-markdown builds
        t = time.perf_counter()

        # --- Choose builder ---
        use_template = template_index is not None and len(template_index) > 0

        if use_template:
            # Mixed build: styled title page + markdown content + ~10% Jinja2 template pages
            te_cfg = cfg.template_engine
            click.echo(f"  Building mixed document (title page + markdown + ~10% template pages)...")
            try:
                from synthpdf.template_engine.doc_builder import build_mixed_document
                pdf_path, variants_used, gt = build_mixed_document(
                    doc_spec=doc_spec,
                    variant_pool=pool,
                    template_index=template_index,
                    output_dir=cfg.output_dir,
                    doc_rng=doc_rng3,
                    recipe=adapted_recipe,
                    html_backend=te_cfg.backend,
                    md_backend=cfg.rendering.backend,
                    template_fraction=0.10,
                    branding_spec=branding_spec,
                    css_file=css_file,
                    title_pages_dir=title_pages_dir_full,
                    page_layout=getattr(te_cfg, "page_layout", "portrait"),
                )
                n_tpl = len(gt.get("template_pages", []))
                click.echo(
                    f"  Brand: {gt.get('brand', '?')} | "
                    f"template pages: {n_tpl} | "
                    f"total pages: {gt.get('page_count', '?')}"
                )
                render_meta = {"page_count": gt.get("page_count", -1), "backend": "mixed"}
            except Exception as exc:
                if te_cfg.fallback_to_markdown:
                    click.echo(f"  WARNING: mixed build failed ({exc}); falling back to plain markdown.", err=True)
                    use_template = False
                else:
                    raise
            timings["build_document"] = time.perf_counter() - t

        if not use_template:
            # Pre-fill text pool: generate enough unique text variants to cover
            # every text section in the spec.  By adding them to the pool before
            # build_document the deque shuffle in build_document spreads them
            # naturally throughout the document instead of concentrating them at
            # the tail via the post-render fill loop.
            if llm_client is not None and topic is not None:
                _n_text_needed = sum(1 for s in doc_spec.sections if s.section_type == "text")
                _n_text_have = len(pool.get("texts", []))
                _text_shortfall = max(0, _n_text_needed - _n_text_have)
                if _text_shortfall > 0:
                    click.echo(f"  Pre-fill: {_text_shortfall} text variant(s) needed for even spread...")
                    from synthpdf.variants.text_annotations import generate_llm_text_variants as _gltv
                    _pf_heads_src = [
                        "Additional Analysis", "Further Discussion", "Supplementary Details",
                        "Extended Review", "Complementary Findings", "Background Context",
                        "Detailed Examination", "In-Depth Assessment", "Supporting Evidence",
                        "Comprehensive Overview",
                    ]
                    _pf_outline = [_pf_heads_src[i % len(_pf_heads_src)] for i in range(_text_shortfall)]
                    _pf_rng = fork_rng(doc_rng, "pre_fill_pool")
                    _pf_workers = cfg.azure_openai.max_parallel_llm_calls if cfg.azure_openai else 1
                    _pf_variants = _gltv(
                        llm_client, topic, _pf_rng, _text_shortfall,
                        outline=_pf_outline,
                        max_workers=_pf_workers,
                    )
                    pool = {**pool, "texts": list(pool.get("texts", [])) + list(_pf_variants)}
                    click.echo(f"    Pre-fill added {len(_pf_variants)} text variant(s) to pool.")

            md_path, variants_used, gt = build_document(
                doc_spec, pool, cfg.output_dir, adapted_recipe, doc_rng3
            )
            timings["build_document"] = time.perf_counter() - t
            # Render markdown → PDF
            try:
                render_meta = render_markdown_to_pdf(
                    md_path, pdf_path,
                    backend=cfg.rendering.backend,
                    css_file=css_file,
                    branding_spec=branding_spec,
                )
            except Exception as exc:
                click.echo(f"  WARNING: PDF render failed: {exc}", err=True)
                render_meta = {"error": str(exc), "page_count": -1}

        # Page-count safety net: interleaved fill loop (up to 3 passes) — markdown mode only.
        # New sections are inserted at random positions in the existing section list so
        # fill content is spread throughout the document, not concentrated at the end.
        if not use_template and llm_client is not None and topic is not None:
            from synthpdf.llm.content_generator import ContentGenerator as _FillCG
            from synthpdf.compose.markdown_builder import render_sections_to_markdown as _rsm
            _fill_gen = _FillCG(llm_client, topic)
            _fill_headings = [
                "Additional Analysis", "Further Discussion", "Supplementary Details",
                "Extended Review", "Complementary Findings", "Background Context",
                "Detailed Examination", "In-Depth Assessment", "Supporting Evidence",
                "Comprehensive Overview",
            ]
            # Live section list: starts from what build_document assembled.
            # We mutate this list in-place across passes so each pass sees the
            # full current document structure when picking insertion positions.
            _live_sections: list[dict] = list(gt.get("_sections_data", []))
            _fill_title    = gt.get("title", doc_spec.title or doc_spec.doc_id)
            _fill_subtitle = gt.get("subtitle", doc_spec.subtitle or "")
            _fill_pass_rng = fork_rng(doc_rng, "fill_interleave")

            for _pass in range(3):
                _actual = render_meta.get("page_count", 0)
                if not isinstance(_actual, int) or _actual <= 0 or _actual >= target_pages:
                    break
                _pp = max(_actual / max(len(_live_sections), 1), 0.4)
                _n_fill = min(max(1, math.ceil((target_pages - _actual) / _pp)), 20)
                click.echo(
                    f"  Page shortfall ({_actual}/{target_pages}), "
                    f"~{_pp:.2f}pp/section -> adding {_n_fill} fill sections (interleaved)..."
                )
                # Generate fill text and insert each new section at a random position.
                _global_fi = _pass * 10
                for _fi in range(_n_fill):
                    _fheading = _fill_headings[(_global_fi + _fi) % len(_fill_headings)]
                    _ftext = _fill_gen.generate_text_section(_fheading, 600)
                    _ftext_md = f"**{_fheading}**\n\n{_ftext.lstrip()}"
                    _pos = int(_fill_pass_rng.integers(0, len(_live_sections) + 1))
                    _live_sections.insert(_pos, {"type": "text", "content": _ftext_md})
                # Rewrite the markdown from the updated section list and re-render.
                if md_path is not None:
                    _rsm(_fill_title, _fill_subtitle, _live_sections, md_path)
                try:
                    render_meta = render_markdown_to_pdf(
                        md_path, pdf_path,
                        backend=cfg.rendering.backend,
                        css_file=css_file,
                        branding_spec=branding_spec,
                    )
                except Exception as _fexc:
                    click.echo(f"  WARNING: fill render failed: {_fexc}", err=True)
                    break
                click.echo(
                    f"  Fill pass {_pass+1}: {render_meta.get('page_count', '?')} pages"
                )

        timings["render_pdf"] = time.perf_counter() - t

        # Hard page trim: if the rendered PDF exceeds pmax, truncate to pmax pages.
        _actual_pages = render_meta.get("page_count", 0)
        if isinstance(_actual_pages, int) and _actual_pages > 0 and _actual_pages > pmax:
            try:
                from pypdf import PdfWriter, PdfReader
                _reader = PdfReader(str(pdf_path))
                _writer = PdfWriter()
                for _pi in range(min(pmax, len(_reader.pages))):
                    _writer.add_page(_reader.pages[_pi])
                with open(pdf_path, "wb") as _fp:
                    _writer.write(_fp)
                click.echo(f"  Trimmed PDF: {_actual_pages} -> {pmax} pages (pmax).")
                render_meta = {**render_meta, "page_count": pmax}
            except Exception as _trim_exc:
                click.echo(f"  WARNING: page trim failed: {_trim_exc}", err=True)

        # Optional PDF annotation pass (legacy variant-pool annotations)
        ann_variants = [v for v in variants_used if v.kind == "annotation"]
        if cfg.rendering.enable_annotations and ann_variants:
            t = time.perf_counter()
            try:
                apply_pdf_annotations(pdf_path, ann_variants)
            except Exception as exc:
                click.echo(f"  WARNING: annotation pass failed: {exc}", err=True)
            timings["annotations"] = time.perf_counter() - t

        # Inline annotation pass (post-content: picks phrases from generated text)
        inline_ann_dicts: list[dict] = []
        _vc_cfg = _vc(cfg)
        if (
            _vc_cfg.annotations.inline.enabled
            and pdf_path.exists()
            and gt.get("sections")
        ):
            from synthpdf.annotation.text_picker import pick_annotation_specs
            from synthpdf.annotation.pdf_layer import apply_inline_annotations

            inline_cfg = _vc_cfg.annotations.inline
            ann_rng = fork_rng(doc_rng, "inline_annotations")
            t = time.perf_counter()
            try:
                inline_specs = pick_annotation_specs(
                    gt.get("sections", []),
                    inline_cfg,
                    ann_rng,
                    doc_id,
                )
                if inline_specs:
                    page_map = apply_inline_annotations(pdf_path, inline_specs)
                    # Back-fill page numbers onto each spec
                    for spec in inline_specs:
                        spec.page = page_map.get(spec.annotation_id)
                    placed = sum(1 for s in inline_specs if s.page is not None)
                    click.echo(
                        f"  Inline annotations: {placed}/{len(inline_specs)} placed "
                        f"({', '.join(s.type for s in inline_specs)})"
                    )
                    # Serialise to dicts — only include annotations that were
                    # actually placed in the PDF (page is not None).  Un-placed
                    # specs are silently discarded so the manifest and QA never
                    # reference annotations that don't exist in the PDF.
                    inline_ann_dicts = [
                        {
                            "annotation_id": s.annotation_id,
                            "type":          s.type,
                            "text":          s.text,
                            "color":         s.color,
                            "extra":         s.extra,
                            "page":          s.page,
                        }
                        for s in inline_specs
                        if s.page is not None
                    ]
            except Exception as exc:
                click.echo(f"  WARNING: inline annotation pass failed: {exc}", err=True)
            timings["inline_annotations"] = time.perf_counter() - t

        # Collect asset paths
        assets = [
            Path(v.render_payload["image_path"])
            for v in variants_used
            if v.kind == "image" and "image_path" in v.render_payload
        ]

        # Add topic_id to ground truth
        if topic is not None:
            gt["topic_id"] = topic.id

        # Count section types for summary manifest
        structure = gt.get("sections", [])
        n_tables  = sum(1 for s in structure if s.get("section_type") == "table")
        n_figures = sum(1 for s in structure if s.get("section_type") == "image")
        n_text    = sum(1 for s in structure if s.get("section_type") == "text")

        rich_render_meta = {
            **render_meta,
            "target_pages": target_pages,
            "backend": cfg.rendering.backend,
            "n_tables": n_tables,
            "n_figures": n_figures,
            "n_text": n_text,
            "logo": branding_spec.logo_path.name if branding_spec else None,
        }

        # Strip build-only internal keys before writing to manifest.
        gt.pop("_sections_data", None)

        t = time.perf_counter()
        doc_manifest_path = write_document_manifest(
            manifests_dir,
            doc_spec=doc_spec,
            markdown_path=md_path,
            pdf_path=pdf_path,
            assets=assets,
            variants_used=variants_used,
            ground_truth=gt,
            render_metadata=rich_render_meta,
            inline_annotations=inline_ann_dicts if inline_ann_dicts else None,
            document_brief=llm_brief if llm_brief else None,
        )

        write_quest_file(
            quest_dir,
            doc_spec=doc_spec,
            pdf_path=pdf_path,
            sections_gt=gt.get("sections", []),
            inline_annotations=inline_ann_dicts if inline_ann_dicts else None,
        )

        append_run_summary(
            summary_path,
            doc_spec=doc_spec,
            pdf_path=pdf_path,
            manifest_path=doc_manifest_path,
            render_metadata=rich_render_meta,
            topic_id=topic.id if topic else None,
        )
        timings["manifest_quest"] = time.perf_counter() - t

        # Per-document timing summary
        doc_total = time.perf_counter() - doc_t0
        page_count = render_meta.get("page_count", "?")
        click.echo(
            f"\n  -- Timing: {doc_id} "
            f"(pages target={target_pages} actual={page_count}) --"
        )
        phase_order = [
            ("llm_variants",       "LLM variant generation"),
            ("title_subtitle",     "Title + subtitle (LLM)"),
            ("build_document",     "Build document (compose)"),
            ("render_pdf",         "Render to PDF"),
            ("annotations",        "PDF annotations (legacy)"),
            ("inline_annotations", "Inline annotations (post-content)"),
            ("manifest_quest",     "Write manifest + Quest"),
        ]
        for key, label in phase_order:
            if key in timings:
                pct = timings[key] / doc_total * 100
                click.echo(f"    {label:<32} {_hms(timings[key]):>12}  ({pct:4.1f}%)")
        click.echo(f"    {'TOTAL':<32} {_hms(doc_total):>12}")

        # Write per-document stats JSON
        try:
            import json as _json
            import datetime as _dt
            from synthpdf import stats_sink as _ss

            _vu = locals().get("variants_used", [])
            _rm = locals().get("render_meta", {})
            _rrm = locals().get("rich_render_meta", {})
            _lb = locals().get("llm_brief", {})
            _vs = locals().get("variants_used", [])

            artifacts_in_pdf: dict[str, int] = {}
            for _v in _vs:
                artifacts_in_pdf[_v.kind] = artifacts_in_pdf.get(_v.kind, 0) + 1

            _n_tx = counts.get("texts", 0)
            _n_tb = counts.get("tables", 0)
            _n_im = counts.get("images", 0)
            _n_sc_c = counts.get("scanned", 0)
            _n_in_c = counts.get("infographics", 0)
            _llm_calls = _n_tx * 2 + _n_tb * 2 + _n_im * 2 + _n_sc_c * 2 + _n_in_c * 2 + 3

            doc_stats = {
                "doc_id":       doc_id,
                "config_name":  Path(cfg.output_dir).name,
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "topic_id":     topic.id if topic else None,
                "seed":         doc_seed,
                "recipe_name":  doc_spec.recipe_name,
                "title":        doc_spec.title or "",
                "theme_brief":  (_lb.get("narrative_theme") or "")[:200],
                "pool_requested": {k: v for k, v in counts.items()
                                   if not k.startswith("_")},
                "llm_calls_estimated": _llm_calls,
                "artifacts_in_pdf":       artifacts_in_pdf,
                "artifacts_in_pdf_total": sum(artifacts_in_pdf.values()),
                "render": {
                    "backend":      _rrm.get("backend", _rm.get("backend", "")),
                    "target_pages": target_pages,
                    "actual_pages": _rm.get("page_count"),
                    "logo":         _rrm.get("logo"),
                },
                "phase_timings_s": {
                    **{k: round(v, 2) for k, v in timings.items()},
                    "total": round(doc_total, 2),
                },
                "type_generation_times_s": {
                    k: round(v, 2) for k, v in _ss.get_type_times().items()
                },
                "per_artifact": _ss.get_log(),
            }

            _stats_dir = Path(cfg.output_dir) / "stats"
            _stats_dir.mkdir(parents=True, exist_ok=True)

            # Per-doc file (one JSON per document)
            (_stats_dir / f"{doc_id}.json").write_text(
                _json.dumps(doc_stats, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Running per-config aggregate — upserted after every PDF so that
            # a mid-run crash still leaves stats for all completed documents.
            _agg_path = _stats_dir / "job_stats.json"
            try:
                _agg = _json.loads(_agg_path.read_text(encoding="utf-8")) \
                       if _agg_path.exists() else {}
            except Exception:
                _agg = {}
            _agg[doc_id] = doc_stats
            _agg_path.write_text(
                _json.dumps(_agg, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as _se:
            import logging as _logging
            _logging.getLogger(__name__).warning("Stats write failed: %s", _se)

    run_total = time.perf_counter() - run_t0
    click.echo(f"\nDone. Total run time: {_hms(run_total)}")
    click.echo(f"  PDFs:      {cfg.output_dir / 'pdfs'}/")
    click.echo(f"  Manifests: {manifests_dir}/")
    click.echo(f"  Quest:     {quest_dir}/")
    click.echo(f"  Summary:   {summary_path}")


def _get_global_pool(
    cfg: DatasetConfig,
    existing: dict[str, list[Variant]] | None,
) -> dict[str, list[Variant]]:
    """Return the cached global pool, generating it lazily on first call."""
    if existing is not None:
        return existing
    click.echo("  Generating global random variant pool...")
    return _generate_variant_pool(cfg)


def _resolve_te_base(cfg: DatasetConfig, config_path: Path | None = None) -> "Path | None":
    """Resolve the template_engine base directory to an absolute Path, or None."""
    te = cfg.template_engine
    if te is None or not te.enabled:
        return None
    base = Path(te.templates_dir)
    if not base.is_absolute():
        if config_path is not None:
            cfg_relative = (config_path.parent / base).resolve()
            base = cfg_relative if cfg_relative.exists() else (Path.cwd() / base).resolve()
        else:
            base = (Path.cwd() / base).resolve()
    return base


def _make_template_index(cfg: DatasetConfig, config_path: Path | None = None) -> Any | None:
    """Build a TemplateIndex if template_engine config is present and enabled."""
    te = cfg.template_engine
    if te is None or not te.enabled:
        return None
    from synthpdf.template_engine.index import TemplateIndex
    base = _resolve_te_base(cfg, config_path)
    if base is None:
        return None
    specs_dir     = base / te.specs_subdir
    templates_dir = base / te.templates_subdir
    if not specs_dir.exists() or not templates_dir.exists():
        click.echo(
            f"  WARNING: template_engine dirs not found "
            f"(specs={specs_dir}, html={templates_dir}); falling back to markdown.",
            err=True,
        )
        return None
    idx = TemplateIndex(specs_dir, templates_dir)
    click.echo(f"  TemplateIndex loaded: {len(idx)} templates "
               f"({len(idx.available_roles)} roles, {len(idx.available_industries)} industries)")
    return idx

