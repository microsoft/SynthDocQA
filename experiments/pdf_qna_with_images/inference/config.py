# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Model registry and runtime configuration for the inference pipeline."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VARIANT_CONFIGS_V2 = Path("dataset")
INFERENCE_RESULTS_DIR = Path("results") / "page_images"


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------

def _expand_env(value: str) -> str:
    """Expand ${VAR} placeholders using environment variables."""
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


@dataclass
class ModelSpec:
    """One model endpoint configuration for Azure AI Foundry.

    client_type:
      "azure_openai"  -- AzureOpenAI client (for GPT deployments on Azure OpenAI
                         Service).  Requires endpoint + api_version.
      "serverless"    -- OpenAI-compatible client pointing at an Azure AI Foundry
                         serverless endpoint (Claude, Mistral, Phi, LLaMA, MAI
                         etc.).  The endpoint should be the base URL that the SDK
                         will append /chat/completions to.

    input_mode:
      "images"  -- PDF pages are rendered to PNG and sent as base64 image_url
                   content items.  Use for vision-capable models.
      "text"    -- Text is extracted from the PDF with PyMuPDF and sent as plain
                   text.  Use for text-only models.
    """

    name: str                           # Human-friendly label, used as dict key
    deployment: str                     # Deployment name (Azure OAI) or model name (serverless)
    endpoint: str                       # Azure AI Foundry endpoint URL
    api_version: str = "2024-12-01-preview"
    client_type: str = "azure_openai"   # "azure_openai" | "serverless"
    input_mode: str = "images"          # "images" | "text"
    max_tokens: int = 4096
    temperature: float = 0.0
    api_key_env: str = "AZURE_OPENAI_API_KEY"  # env var name holding the key
    verify_ssl: bool = False            # disable SSL verification (corp proxy)
    # Page selection (images mode only)
    top_k_pages: int = 8               # max pages to send per question after selection
    page_selection_window_before: int = 1  # extra pages before each title-matched page
    page_selection_window_after: int = 2   # extra pages after (catches multi-page elements)

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env) or None

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        d = dict(d)
        if "endpoint" in d:
            d["endpoint"] = _expand_env(str(d["endpoint"]))
        if "deployment" in d:
            d["deployment"] = _expand_env(str(d["deployment"]))
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# InferenceConfig
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Runtime settings for the inference + judge pipeline."""

    qa_models: list[ModelSpec] = field(default_factory=list)
    judge_model: Optional[ModelSpec] = None

    results_dir: Path = field(default_factory=lambda: INFERENCE_RESULTS_DIR)
    max_workers: int = 4            # parallel questions per document
    max_qa_per_doc: int = 0        # 0 = all questions
    config_filter: Optional[str] = None  # glob pattern for dataset config dirs
    resume: bool = True            # skip docs already written to per_doc/
    dataset_dir: Path = field(default_factory=lambda: VARIANT_CONFIGS_V2)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_models_config(path: Path) -> InferenceConfig:
    """Load InferenceConfig from a YAML file.

    Expected YAML shape::

        qa_models:
          - name: gpt-4o
            deployment: gpt-4o
            endpoint: "${AZURE_OPENAI_ENDPOINT}"
            client_type: azure_openai
            input_mode: images
            ...

        judge_model:
          name: gpt-4o-judge
          deployment: gpt-4o
          ...

        inference:              # optional overrides
          max_workers: 4
          max_qa_per_doc: 0
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    qa_models = [ModelSpec.from_dict(m) for m in raw.get("qa_models", [])]
    judge_raw = raw.get("judge_model")
    judge_model = ModelSpec.from_dict(judge_raw) if judge_raw else None

    inf = raw.get("inference", {})
    return InferenceConfig(
        qa_models=qa_models,
        judge_model=judge_model,
        max_workers=int(inf.get("max_workers", 4)),
        max_qa_per_doc=int(inf.get("max_qa_per_doc", 0)),
    )
