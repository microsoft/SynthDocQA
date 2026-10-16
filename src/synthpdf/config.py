# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Pydantic configuration models for the synthpdf package."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} / $VAR placeholders in string values."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


class VariantCounts(BaseModel):
    """Legacy flat count model — kept for backward compatibility."""
    tables: int = 20
    images: int = 10
    texts: int = 30
    annotations: int = 15
    forms: int = 10
    scanned: int = 8
    infographics: int = 10
    multipage_tables: int = 5
    complex_tables: int = 10


# ---------------------------------------------------------------------------
# Detailed per-type variant config (new: use ``variants:`` key in YAML)
# ---------------------------------------------------------------------------

class TableBasicConfig(BaseModel):
    enabled: bool = True
    count: int = 20


class TableMultipageConfig(BaseModel):
    enabled: bool = True
    count: int = 2
    rows_per_page: int = 20
    row_range: list[int] = Field(default_factory=lambda: [40, 80])


class TableComplexConfig(BaseModel):
    enabled: bool = True
    count: int = 10
    # Subtypes to include (all enabled by default).
    subtypes: list[str] = Field(default_factory=lambda: [
        "merged_cells", "multi_row_headers", "column_groups", "group_summary_rows",
        "nested_table", "footnoted_table", "matrix_table", "ragged_colspan",
        "stacked_comparison", "transposed_metrics",
    ])


class TableVariantConfig(BaseModel):
    enabled: bool = True
    basic: TableBasicConfig = Field(default_factory=TableBasicConfig)
    multipage: TableMultipageConfig = Field(default_factory=TableMultipageConfig)
    complex: TableComplexConfig = Field(default_factory=TableComplexConfig)


class ImageVariantConfig(BaseModel):
    enabled: bool = True
    count: int = 10
    # Chart subtypes for random (non-LLM) generation.
    subtypes: list[str] = Field(default_factory=lambda: [
        "bar_chart", "line_chart", "scatter_chart", "pie_chart", "diagram",
    ])
    # Perturbations applied to rendered chart images.
    perturbations: list[str] = Field(default_factory=lambda: [
        "blur", "rotation", "low_contrast", "watermark", "occlusion",
    ])


class TextVariantConfig(BaseModel):
    enabled: bool = True
    count: int = 30
    # Text block subtypes.
    subtypes: list[str] = Field(default_factory=lambda: [
        "email", "policy", "spec", "bullet_list", "footnote",
    ])
    # Text perturbations applied post-generation.
    perturbations: list[str] = Field(default_factory=lambda: [
        "typos", "abbreviations", "unicode_punctuation", "date_formats",
    ])


class FormVariantConfig(BaseModel):
    enabled: bool = True
    count: int = 10
    subtypes: list[str] = Field(default_factory=lambda: [
        "checklist", "button", "form", "fillable",
    ])


class ScannedVariantConfig(BaseModel):
    enabled: bool = True
    count: int = 8
    subtypes: list[str] = Field(default_factory=lambda: [
        "text", "table", "handwriting", "stamps",
    ])


class InlineAnnotationTextSelectionConfig(BaseModel):
    """Controls how phrases are picked from generated content for inline annotations."""
    min_words: int = 1
    max_words: int = 6


class InlineAnnotationConfig(BaseModel):
    """Post-content annotation pass: picks text on the fly and overlays it on the PDF.

    Unlike the pre-generated annotation variant pool (which produces section-level
    annotation blocks), these annotations are applied AFTER the document is rendered
    by searching for the selected phrase inside the finished PDF and drawing the
    annotation directly via PyMuPDF.
    """
    enabled: bool = False
    count: int = 4
    subtypes: list[str] = Field(default_factory=lambda: [
        "highlight",        # coloured background over a phrase
        "circle",           # hand-drawn oval around a phrase
        "margin_callout",   # blue side-note in the left margin
        "score_callout",    # oval score badge + comment at page bottom
        "text_color",       # coloured highlight (simulates coloured text)
        "sticky_note",      # PDF sticky-note icon above a phrase
    ])
    text_selection: InlineAnnotationTextSelectionConfig = Field(
        default_factory=InlineAnnotationTextSelectionConfig
    )
    # Colour palettes per annotation family (names resolved in pdf_layer.py)
    highlight_colors: list[str] = Field(
        default_factory=lambda: ["yellow", "orange", "green", "pink"]
    )
    circle_colors: list[str] = Field(
        default_factory=lambda: ["orange", "red", "purple"]
    )
    text_colors: list[str] = Field(
        default_factory=lambda: ["blue", "teal", "red"]
    )
    callout_colors: list[str] = Field(
        default_factory=lambda: ["blue", "teal"]
    )


class AnnotationVariantConfig(BaseModel):
    enabled: bool = True
    count: int = 15
    subtypes: list[str] = Field(default_factory=lambda: [
        "highlights", "callouts", "labels", "sticky_notes",
        "freetext", "review", "annotated_report",
    ])
    # Post-content inline annotation pass (NEW)
    inline: InlineAnnotationConfig = Field(default_factory=InlineAnnotationConfig)


class InfographicVariantConfig(BaseModel):
    enabled: bool = True
    count: int = 10
    subtypes: list[str] = Field(default_factory=lambda: [
        "stat_dashboard", "timeline", "comparison", "process_steps",
        "hub_spoke", "magazine", "pictograph", "floor_plan",
    ])


class VariantsConfig(BaseModel):
    """Full per-type variant configuration.  Use ``variants:`` in YAML.
    Takes precedence over the legacy ``variant_counts:`` key when present.
    """
    tables: TableVariantConfig = Field(default_factory=TableVariantConfig)
    images: ImageVariantConfig = Field(default_factory=ImageVariantConfig)
    texts: TextVariantConfig = Field(default_factory=TextVariantConfig)
    annotations: AnnotationVariantConfig = Field(default_factory=AnnotationVariantConfig)
    forms: FormVariantConfig = Field(default_factory=FormVariantConfig)
    scanned: ScannedVariantConfig = Field(default_factory=ScannedVariantConfig)
    infographics: InfographicVariantConfig = Field(default_factory=InfographicVariantConfig)


class LogoBrandingConfig(BaseModel):
    """Configuration for per-document logo branding."""

    logos_dir: str | None = None   # directory to randomly pick a logo from
    logo_file: str | None = None   # specific logo file (takes precedence over logos_dir)
    placement: str = "header"      # "header" | "footer" | "both"
    position: str = "right"        # "left" | "center" | "right"
    pages: str = "all"             # "all" | "first" | "rest"
    logo_height_pt: float = 30.0   # logo height in points (~1 cm); HTML uses 40 px equivalent
    enabled: bool = True


class TemplateEngineConfig(BaseModel):
    """Configuration for Jinja2 HTML template-based document generation."""

    templates_dir: str                            # base directory (contains specs/ and templates/)
    specs_subdir: str = "specs"                   # relative sub-path to spec JSONs
    templates_subdir: str = "templates"           # relative sub-path to HTML files
    enabled: bool = False
    backend: str = "playwright"                   # "playwright" | "weasyprint"
    pages_per_doc_range: list[int] = Field(default_factory=lambda: [3, 8])
    fallback_to_markdown: bool = True             # fall back to markdown builder on error
    fancy_mode: bool = False                      # use full-CSS3 fancy archetypes via Playwright
    page_layout: Literal["portrait", "landscape"] = "portrait"  # page orientation for fancy mode


class RenderingConfig(BaseModel):
    backend: str = "weasyprint"
    css_file: str | None = None
    page_size: str = "A4"
    enable_annotations: bool = False
    branding: LogoBrandingConfig | None = None


class AzureOpenAIConfig(BaseModel):
    endpoint: str
    deployment: str = "gpt-4o"
    # Per-tier deployments (fall back to `deployment` if not set).
    # HARD  → complex multi-step reasoning (QA + assertions)
    # MEDIUM → structured data generation (tables)
    # EASY  → fast generation (title, subtitle, outline, prose, chart spec)
    deployment_hard: str | None = None
    deployment_medium: str | None = None
    deployment_easy: str | None = None
    api_version: str = "2024-02-01"
    tenant_id: str | None = None  # falls back to AZURE_TENANT_ID env var; required for Entra ID
    api_key: str | None = None    # falls back to AZURE_OPENAI_API_KEY; if absent uses Entra ID
    max_tokens: int = 16000
    temperature: float = 0.7
    max_retries: int = 5          # retries on connection/timeout errors
    timeout_seconds: float = 120.0  # per-request timeout
    max_parallel_llm_calls: int = 12  # concurrent LLM API calls per document
    verify_ssl: bool = True       # set False to disable SSL verification (corp MITM proxies)
    ssl_ca_bundle: str | None = None  # path to custom CA bundle (PEM file)
    http_proxy: str | None = None     # explicit proxy URL, e.g. "http://proxy:8080"
                                      # falls back to HTTPS_PROXY / HTTP_PROXY env vars

    @field_validator("endpoint", mode="before")
    @classmethod
    def _validate_endpoint(cls, v: Any) -> Any:
        if not isinstance(v, str) or not v:
            raise ValueError("azure_openai.endpoint must be a non-empty string.")
        if "${" in v or v.startswith("$"):
            raise ValueError(
                f"azure_openai.endpoint still contains an unresolved placeholder: {v!r}. "
                "Set the AZURE_OPENAI_ENDPOINT environment variable."
            )
        if not v.startswith(("http://", "https://")):
            # Auto-prepend https:// for bare hostnames like "myname.openai.azure.com"
            v = "https://" + v
        return v

    @field_validator("tenant_id", "deployment_hard", "deployment_medium", "deployment_easy", mode="before")
    @classmethod
    def _coerce_optional_str(cls, v: Any) -> Any:
        """Convert empty strings and unresolved ${VAR} placeholders to None."""
        if v is None or v == "":
            return None
        if isinstance(v, str) and ("${" in v or v.startswith("$")):
            return None
        return v

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _coerce_max_tokens(cls, v: Any) -> Any:
        if v is None:
            return 16000
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return 16000  # unresolved placeholder or empty string
        return v

    @field_validator("temperature", mode="before")
    @classmethod
    def _coerce_temperature(cls, v: Any) -> Any:
        if v is None:
            return 0.7
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return 0.7  # unresolved placeholder or empty string
        return v


class RecipeConfig(BaseModel):
    name: str
    allowed_variant_pools: list[str] = Field(
        default_factory=lambda: ["tables", "images", "texts"]
    )
    section_count_range: list[int] = Field(default_factory=lambda: [3, 8])
    layout_probabilities: dict[str, float] = Field(
        default_factory=lambda: {
            "table": 0.4,
            "image": 0.3,
            "text": 0.3,
        }
    )
    annotation_attachment_probability: float = 0.2

    @property
    def section_count_min(self) -> int:
        return self.section_count_range[0]

    @property
    def section_count_max(self) -> int:
        return self.section_count_range[1]


class DatasetConfig(BaseModel):
    output_dir: Path = Path("output")
    num_documents: int = 10
    global_seed: int = 42
    # Detailed per-type config (new). Takes precedence over variant_counts when set.
    variants: VariantsConfig | None = None
    # Legacy flat counts — kept for backward compatibility.
    variant_counts: VariantCounts = Field(default_factory=VariantCounts)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    recipes: list[RecipeConfig] = Field(default_factory=list)
    page_count: int | None = None                                   # fixed page count (takes priority)
    page_count_range: list[int] = Field(default_factory=lambda: [4, 20])  # fallback range
    topics_file: str | None = None
    azure_openai: AzureOpenAIConfig | None = None
    template_engine: TemplateEngineConfig | None = None

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetConfig":
        import yaml

        # Load .env file if present so ${VAR} placeholders can be expanded below.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # python-dotenv not installed; rely on shell env vars

        with open(path) as f:
            data = yaml.safe_load(f)
        # Expand ${VAR} / $VAR placeholders throughout the config
        data = _expand_env_vars(data)
        # Convert nested dicts to model instances
        if "variants" in data and data["variants"] is not None:
            # Pydantic v2 handles nested dict -> model conversion automatically.
            data["variants"] = VariantsConfig.model_validate(data["variants"])
        if "variant_counts" in data:
            data["variant_counts"] = VariantCounts(**data["variant_counts"])
        if "rendering" in data:
            rendering_data = data["rendering"]
            if isinstance(rendering_data, dict):
                branding_raw = rendering_data.pop("branding", None)
                rendering_obj = RenderingConfig(**rendering_data)
                if branding_raw is not None:
                    rendering_obj = rendering_obj.model_copy(
                        update={"branding": LogoBrandingConfig(**branding_raw)}
                    )
                data["rendering"] = rendering_obj
            else:
                data["rendering"] = rendering_data
        if "recipes" in data:
            data["recipes"] = [RecipeConfig(**r) for r in data["recipes"]]
        if "azure_openai" in data and data["azure_openai"] is not None:
            data["azure_openai"] = AzureOpenAIConfig(**data["azure_openai"])
        if "template_engine" in data and data["template_engine"] is not None:
            data["template_engine"] = TemplateEngineConfig(**data["template_engine"])
        return cls(**data)
