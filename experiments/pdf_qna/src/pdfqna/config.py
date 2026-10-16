# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
Configuration for the PDF QnA evaluation pipeline.

All model names and endpoints are read from the user-supplied --env file.

The .env file drives everything for the QnA pipeline:
  AZURE_OPENAI_API_VERSION  -> API version shared across all Azure endpoints
  USE_ENTRA_ID              -> true (Entra ID) or false (API key)
  AZURE_OPENAI_MAX_TOKENS   -> token limit (default 8000)
  AZURE_OPENAI_TEMPERATURE  -> temperature (default 0.0)

  QA_<NAME>_ENDPOINT        -> endpoint for one QA model (each model has its own)
  QA_<NAME>_DEPLOYMENT      -> deployment name for that model

    JUDGE_<N>_ENDPOINT        -> endpoint for judge N
    JUDGE_<N>_DEPLOYMENT      -> deployment name for judge N

Legacy JUDGE_ENDPOINT + JUDGE_DEPLOYMENT is also accepted as judge_1.

Claude models are detected automatically by "claude" appearing in the deployment value
and are invoked via `claude --model <name> -p` subprocess instead of the Azure client.

Note: AZURE_OPENAI_DEPLOYMENT_* variables are used by the PDF generation pipeline
and are intentionally ignored here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None  # type: ignore[assignment]


def _parse_env_file(path: Path) -> dict[str, str]:
    """
    Parse KEY=VALUE pairs directly from a .env file.

    Used as the primary source of config values — more reliable than relying
    on load_dotenv populating os.environ, which can behave differently across
    python-dotenv versions and environments.

    Rules:
    - Lines starting with # are comments
    - Inline # comments are stripped
    - Values may be quoted with " or '
    - Blank lines are skipped
    """
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if not key:
            continue
        # Strip inline comment — only when not inside quotes
        value = rest
        if not (value.startswith('"') or value.startswith("'")):
            value = value.split("#")[0]
        value = value.strip().strip('"').strip("'")
        result[key] = value
    return result


@dataclass
class ModelSpec:
    """One QA model deployment read from the .env file."""
    name: str           # lower-cased suffix, e.g. "pro", "hard", "claude_sonnet"
    deployment: str     # Azure deployment name OR claude model id, e.g. "gpt-5.4" / "claude-sonnet-4-5"
    endpoint: str       # Azure endpoint URL (unused for Claude models)
    api_version: str
    use_entra_id: bool = True
    max_tokens: int = 8000
    temperature: float = 0.0
    auth_scope: str = "https://cognitiveservices.azure.com/.default"
    api_mode: str = "chat_completions"

    @property
    def is_claude(self) -> bool:
        """True when this deployment is a Claude model invoked via the local claude CLI."""
        return "claude" in self.deployment.lower()


@dataclass
class EvalConfig:
    """Complete runtime configuration — models from env, dataset/run from CLI."""
    models: list[ModelSpec]     # QA models (from QA_*_ENDPOINT + QA_*_DEPLOYMENT pairs)
    judges: list[ModelSpec]     # judge models (from JUDGE_<N>_* pairs)
    env_path: Path              # path to the .env file that was loaded

    # Set after CLI parsing
    dataset_base_dir: Optional[Path] = None
    config_filter: Optional[str] = None
    azure_tenant_id: str = ""
    important_queries: Optional[set] = None   # set of query text strings
    max_docs: Optional[int] = None
    max_qa_per_doc: Optional[int] = None
    max_total_qa: Optional[int] = None
    markdown_source: str = "dataset"
    output_dir: Path = field(default_factory=lambda: Path("results"))
    workers: int = 5
    resume: bool = True
    log_level: str = "INFO"

    @property
    def judge(self) -> ModelSpec:
        """Return the first judge for compatibility with older callers."""
        return self.judges[0]


def load_config(env_path: str | Path) -> EvalConfig:
    """
    Build EvalConfig by reading the single shared .env file.

    QA models come exclusively from QA_<NAME>_ENDPOINT + QA_<NAME>_DEPLOYMENT pairs.
    Judge models come from JUDGE_<N>_ENDPOINT + JUDGE_<N>_DEPLOYMENT pairs.
    Legacy JUDGE_ENDPOINT + JUDGE_DEPLOYMENT is accepted as judge_1.
    AZURE_OPENAI_DEPLOYMENT_* variables are used by the PDF generation pipeline
    and are intentionally ignored here.
    """
    env_path = Path(env_path).resolve()
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")

    # Parse the .env file directly — this is the primary source of all config values.
    # We do NOT rely on load_dotenv populating os.environ, which can silently produce
    # different results across python-dotenv versions and environments.
    env = _parse_env_file(env_path)

    # Also push the values into os.environ so that azure-identity and other libraries
    # that read env vars (e.g. AZURE_TENANT_ID) pick them up correctly.
    if _load_dotenv is not None:
        _load_dotenv(env_path, override=True)
    else:
        for k, v in env.items():
            os.environ.setdefault(k, v)

    def _get(key: str, default: str = "") -> str:
        return env.get(key, os.environ.get(key, default))

    api_version = _get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    use_entra_id = _get("USE_ENTRA_ID", "true").lower() == "true"
    azure_tenant_id = _get("AZURE_TENANT_ID", "")
    max_tokens = int(_get("AZURE_OPENAI_MAX_TOKENS", "8000"))
    temperature = float(_get("AZURE_OPENAI_TEMPERATURE", "0.0"))
    log_level = _get("LOG_LEVEL", "INFO")

    # QA models — exclusively from QA_<NAME>_ENDPOINT + QA_<NAME>_DEPLOYMENT pairs
    models: list[ModelSpec] = []
    for key, ep_value in env.items():
        if not (key.startswith("QA_") and key.endswith("_ENDPOINT")):
            continue
        if not ep_value.strip():
            continue
        short = key[len("QA_"):-len("_ENDPOINT")]          # e.g. "DEEPSEEK_V4_FLASH"
        dep_value = env.get(f"QA_{short}_DEPLOYMENT", "").strip()
        if not dep_value:
            continue                                         # skip if DEPLOYMENT not set
        models.append(
            ModelSpec(
                name=short.lower(),                         # e.g. "deepseek_v4_flash"
                deployment=dep_value,
                endpoint=ep_value.strip().rstrip("/"),
                api_version=api_version,
                use_entra_id=use_entra_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )

    if not models:
        raise ValueError(
            f"No QA models found in {env_path}.\n"
            "Add at least one QA_<NAME>_ENDPOINT + QA_<NAME>_DEPLOYMENT pair."
        )

    # Judge models — numbered pairs are preferred.  Keep support for the old
    # unnumbered pair so existing private configs continue to work.
    judge_numbers = sorted(
        {
            int(key[len("JUDGE_"):-len("_ENDPOINT")])
            for key in env
            if key.startswith("JUDGE_")
            and key.endswith("_ENDPOINT")
            and key[len("JUDGE_"):-len("_ENDPOINT")].isdigit()
        }
    )
    judges: list[ModelSpec] = []
    for number in judge_numbers:
        endpoint_key = f"JUDGE_{number}_ENDPOINT"
        deployment_key = f"JUDGE_{number}_DEPLOYMENT"
        endpoint = _get(endpoint_key, "").strip().rstrip("/")
        deployment = _get(deployment_key, "").strip()
        if bool(endpoint) != bool(deployment):
            raise ValueError(f"{endpoint_key} and {deployment_key} must both be set.")
        if endpoint and deployment:
            auth_scope = _get(
                f"JUDGE_{number}_AUTH_SCOPE",
                "https://cognitiveservices.azure.com/.default",
            ).strip()
            api_mode = _get(
                f"JUDGE_{number}_API_MODE", "chat_completions"
            ).strip().lower()
            if api_mode not in ("chat_completions", "responses"):
                raise ValueError(
                    f"JUDGE_{number}_API_MODE must be 'chat_completions' or 'responses'."
                )
            judges.append(
                ModelSpec(
                    name=f"judge_{number}",
                    deployment=deployment,
                    endpoint=endpoint,
                    api_version=api_version,
                    use_entra_id=use_entra_id,
                    # Reasoning judges may consume substantial hidden reasoning
                    # tokens before emitting the small JSON verdict.
                    max_tokens=max_tokens,
                    temperature=0.0,
                    auth_scope=auth_scope,
                    api_mode=api_mode,
                )
            )

    if not judges:
        judge_endpoint = _get("JUDGE_ENDPOINT", "").strip().rstrip("/")
        judge_deployment = _get("JUDGE_DEPLOYMENT", "").strip()
        if not judge_endpoint or not judge_deployment:
            raise ValueError(
                "At least one JUDGE_<N>_ENDPOINT + JUDGE_<N>_DEPLOYMENT pair "
                "must be set in the .env file."
            )
        judges.append(
            ModelSpec(
                name="judge_1",
                deployment=judge_deployment,
                endpoint=judge_endpoint,
                api_version=api_version,
                use_entra_id=use_entra_id,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        )

    return EvalConfig(
        models=models,
        judges=judges,
        env_path=env_path,
        azure_tenant_id=azure_tenant_id,
        log_level=log_level,
    )
