# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Unconditional, LLM-backed batch generation using packaged resources."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import yaml

_RESOURCES = Path(__file__).parent / "resources"
_TOPICS_FILE = _RESOURCES / "theme" / "topics.yaml"
_TEMPLATES_DIR = _RESOURCES / "jinja2_templates"
_LOGOS_DIR = _RESOURCES / "placeholders"


def _connection_config() -> dict[str, Any]:
    """Read public connection settings without serializing credentials."""
    from dotenv import load_dotenv
    from synthpdf.config import AzureOpenAIConfig

    load_dotenv(Path.cwd() / ".env")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    if not endpoint or not deployment:
        raise ValueError(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT before generation."
        )
    return AzureOpenAIConfig(
        endpoint=endpoint,
        deployment=deployment,
        deployment_hard=os.environ.get("AZURE_OPENAI_DEPLOYMENT_HARD"),
        deployment_medium=os.environ.get("AZURE_OPENAI_DEPLOYMENT_MEDIUM"),
        deployment_easy=os.environ.get("AZURE_OPENAI_DEPLOYMENT_EASY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        tenant_id=os.environ.get("AZURE_TENANT_ID"),
    ).model_dump(exclude_none=True, exclude={"api_key"})


def generate_documents(
    num_pages: int | None = None,
    num_pdfs: int | None = None,
    output_dir: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate an unconditional batch, preserving the existing sampling sequence."""
    from synthpdf._pipeline import run_config

    for name, value in (("num_pages", num_pages), ("num_pdfs", num_pdfs)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)
                             or not 0 <= seed < 2**31):
        raise ValueError("seed must be an integer between 0 and 2147483647.")
    _connection_config()
    rng_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    rng = random.Random(rng_seed)
    num_pages = num_pages if num_pages is not None else rng.randint(4, 20)
    num_pdfs = num_pdfs if num_pdfs is not None else rng.randint(2, 6)
    output_dir = Path(output_dir) if output_dir is not None else (
        Path.cwd() / "output" / "generated" / f"random_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = output_dir.resolve()
    config_dict = _assemble_config(num_pages, num_pdfs, output_dir, rng_seed, rng)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")
    t0 = time.perf_counter()
    run_config(config_path, llm_client=None, num_documents=num_pdfs)
    summary = output_dir / "manifest.jsonl"
    return {
        "config_path": config_path,
        "output_dir": output_dir,
        "manifest_jsonl": summary if summary.exists() else None,
        "documents": _collect_outputs(output_dir),
        "elapsed_seconds": time.perf_counter() - t0,
    }
_ALL_COMPLEX_TABLE_SUBTYPES = [
    "merged_cells", "multi_row_headers", "column_groups", "group_summary_rows",
    "nested_table", "footnoted_table", "matrix_table", "ragged_colspan",
    "stacked_comparison", "transposed_metrics",
]
_ALL_IMAGE_SUBTYPES = ["bar_chart", "line_chart", "scatter_chart", "pie_chart", "diagram"]
_ALL_IMAGE_PERTURBS = ["blur", "rotation", "low_contrast", "watermark", "occlusion"]
_ALL_TEXT_SUBTYPES = ["email", "policy", "spec", "bullet_list", "footnote"]
_ALL_TEXT_PERTURBS = ["typos", "abbreviations", "unicode_punctuation", "date_formats"]
_ALL_ANNOTATION_SUBTYPES = [
    "highlights", "callouts", "labels", "sticky_notes",
    "freetext", "review", "annotated_report",
]
_ALL_FORM_SUBTYPES = ["checklist", "button", "form", "fillable"]
_ALL_SCANNED_SUBTYPES = ["text", "table", "handwriting", "stamps"]
_ALL_INFOGRAPHIC_SUBTYPES = [
    "stat_dashboard", "timeline", "comparison", "process_steps",
    "hub_spoke", "magazine", "pictograph", "floor_plan",
]

_RECIPE_STYLES = [
    {
        "name": "mixed_document",
        "allowed_variant_pools": ["tables", "images", "texts"],
        "section_count_range": [6, 12],
        "layout_probabilities": {"text": 0.55, "table": 0.25, "image": 0.20},
        "annotation_attachment_probability": 0.20,
    },
    {
        "name": "text_heavy",
        "allowed_variant_pools": ["texts", "tables"],
        "section_count_range": [8, 16],
        "layout_probabilities": {"text": 0.75, "table": 0.25},
        "annotation_attachment_probability": 0.10,
    },
    {
        "name": "data_focused",
        "allowed_variant_pools": ["tables", "images", "texts"],
        "section_count_range": [6, 12],
        "layout_probabilities": {"text": 0.40, "table": 0.40, "image": 0.20},
        "annotation_attachment_probability": 0.10,
    },
]


# ---------------------------------------------------------------------------
# Azure OpenAI block (reads from env vars; silently omitted if not set)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Random config builder
# ---------------------------------------------------------------------------

def _rand_subset(items: list, rng: random.Random, min_k: int = 1) -> list:
    """Return a random non-empty subset of *items*."""
    k = rng.randint(max(1, min_k), len(items))
    return rng.sample(items, k)


def _build_random_variants(rng: random.Random) -> dict[str, Any]:
    """Build a fully randomised variants block."""
    return {
        "tables": {
            "enabled": rng.random() > 0.2,
            "basic": {
                "enabled": True,
                "count": rng.randint(5, 25),
            },
            "multipage": {
                "enabled": rng.random() > 0.4,
                "count": rng.randint(1, 5),
                "rows_per_page": rng.randint(15, 30),
                "row_range": [rng.randint(30, 50), rng.randint(60, 100)],
            },
            "complex": {
                "enabled": rng.random() > 0.3,
                "count": rng.randint(3, 15),
                "subtypes": _rand_subset(_ALL_COMPLEX_TABLE_SUBTYPES, rng, min_k=2),
            },
        },
        "images": {
            "enabled": rng.random() > 0.2,
            "count": rng.randint(4, 18),
            "subtypes": _rand_subset(_ALL_IMAGE_SUBTYPES, rng),
            "perturbations": _rand_subset(_ALL_IMAGE_PERTURBS, rng),
        },
        "texts": {
            "enabled": True,  # always have some text
            "count": rng.randint(10, 30),
            "subtypes": _rand_subset(_ALL_TEXT_SUBTYPES, rng),
            "perturbations": _rand_subset(_ALL_TEXT_PERTURBS, rng),
        },
        "annotations": {
            "enabled": rng.random() > 0.3,
            "count": rng.randint(5, 20),
            "subtypes": _rand_subset(_ALL_ANNOTATION_SUBTYPES, rng),
        },
        "forms": {
            "enabled": rng.random() > 0.4,
            "count": rng.randint(3, 12),
            "subtypes": _rand_subset(_ALL_FORM_SUBTYPES, rng),
        },
        "scanned": {
            "enabled": rng.random() > 0.4,
            "count": rng.randint(2, 10),
            "subtypes": _rand_subset(_ALL_SCANNED_SUBTYPES, rng),
        },
        "infographics": {
            "enabled": rng.random() > 0.3,
            "count": rng.randint(3, 12),
            "subtypes": _rand_subset(_ALL_INFOGRAPHIC_SUBTYPES, rng),
        },
    }


def _random_recipe(rng: random.Random) -> dict[str, Any]:
    """Pick a random recipe style."""
    return dict(rng.choice(_RECIPE_STYLES))


def _assemble_config(
    num_pages: int,
    num_pdfs: int,
    output_dir: Path,
    seed: int,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Assemble a complete config dict ready for YAML serialisation."""

    if rng is None:
        rng = random.Random(seed)

    variants_block = _build_random_variants(rng)
    recipe_block = [_random_recipe(rng)]

    # Normalise variants_block: convert nested "basic"/"multipage"/"complex" dicts
    # stored flat (only count + enabled) into the expected sub-key structure.
    tables_raw = variants_block.get("tables", {})
    if isinstance(tables_raw, dict):
        # Ensure basic/multipage/complex sub-keys exist with at least enabled+count
        for sub in ("basic", "multipage", "complex"):
            if sub not in tables_raw:
                tables_raw[sub] = {"enabled": False, "count": 0}

    # Detect if template engine can be used
    te_enabled = _TEMPLATES_DIR.exists() and (_TEMPLATES_DIR / "specs").exists()

    config: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_documents": num_pdfs,
        "global_seed": seed,
        "page_count": num_pages,
        "topics_file": str(_TOPICS_FILE) if _TOPICS_FILE.exists() else None,
        "azure_openai": _connection_config(),
        "variants": variants_block,
        "rendering": {
            "backend": "playwright",
            "page_size": "A4",
            "enable_annotations": False,
            "branding": {
                "logos_dir": str(_LOGOS_DIR) if _LOGOS_DIR.exists() else None,
                "placement": rng.choice(["header", "footer", "both"]),
                "position": rng.choice(["left", "center", "right"]),
                "pages": "all",
                "logo_height_pt": rng.uniform(24.0, 35.0),
                "enabled": _LOGOS_DIR.exists(),
            },
        },
        "template_engine": {
            "enabled": te_enabled,
            "templates_dir": str(_TEMPLATES_DIR),
            "specs_subdir": "specs",
            "templates_subdir": "templates",
            "backend": "playwright",
            "pages_per_doc_range": [3, 6],
            "fallback_to_markdown": True,
        },
        "recipes": recipe_block,
    }

    # Remove None values at top level to keep YAML clean
    config = {k: v for k, v in config.items() if v is not None}

    return config


def _collect_outputs(output_dir: Path) -> list[dict[str, Any]]:
    """Walk *output_dir* and collect per-document output paths."""
    pdfs_dir = output_dir / "pdfs"
    manifests_dir = output_dir / "manifests"
    quest_dir = output_dir / "quest"
    markdown_dir = output_dir / "markdown"

    documents: list[dict[str, Any]] = []

    if not pdfs_dir.exists():
        return documents

    for pdf_path in sorted(pdfs_dir.glob("*.pdf")):
        doc_id = pdf_path.stem
        manifest_path = manifests_dir / f"{doc_id}.json"
        quest_xlsx = quest_dir / f"{doc_id}.xlsx"
        quest_json = quest_dir / f"{doc_id}.json"
        md_path = markdown_dir / f"{doc_id}.md"

        documents.append({
            "doc_id": doc_id,
            "pdf_path": pdf_path if pdf_path.exists() else None,
            "manifest_path": manifest_path if manifest_path.exists() else None,
            "quest_xlsx_path": quest_xlsx if quest_xlsx.exists() else None,
            "quest_json_path": quest_json if quest_json.exists() else None,
            "markdown_path": md_path if md_path.exists() else None,
        })

    return documents
