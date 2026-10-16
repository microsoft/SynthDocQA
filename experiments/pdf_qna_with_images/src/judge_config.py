# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Configuration for the four LLM judges used by direct-PDF QA evaluation."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class JudgeSpec:
    """One JUDGE_<N> endpoint/deployment pair from the shared dotenv file."""

    name: str
    deployment: str
    endpoint: str
    api_version: str
    use_entra_id: bool = True
    max_tokens: int = 8000
    temperature: float = 0.0
    auth_scope: str = "https://cognitiveservices.azure.com/.default"
    api_mode: str = "chat_completions"


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse uncommented KEY=VALUE entries from a dotenv file."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if not value.startswith(("'", '"')):
            value = value.split("#", 1)[0].strip()
        values[key.strip()] = value.strip("'\"")
    return values


def load_judge_specs(env_path: str | Path) -> tuple[list[JudgeSpec], dict[str, str]]:
    """Load every enabled numbered JUDGE_<N> pair from the shared dotenv file."""
    path = Path(env_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Environment file was not found: {path}")

    values = parse_env_file(path)
    load_dotenv(path, override=True)
    api_version = values.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    use_entra_id = values.get("USE_ENTRA_ID", "true").lower() == "true"
    max_tokens = int(values.get("AZURE_OPENAI_MAX_TOKENS", "8000"))

    numbers = sorted(
        int(key[len("JUDGE_") : -len("_ENDPOINT")])
        for key in values
        if key.startswith("JUDGE_")
        and key.endswith("_ENDPOINT")
        and key[len("JUDGE_") : -len("_ENDPOINT")].isdigit()
    )
    judges: list[JudgeSpec] = []
    for number in numbers:
        prefix = f"JUDGE_{number}"
        endpoint = values.get(f"{prefix}_ENDPOINT", "").strip().rstrip("/")
        deployment = values.get(f"{prefix}_DEPLOYMENT", "").strip()
        if bool(endpoint) != bool(deployment):
            raise ValueError(
                f"{prefix}_ENDPOINT and {prefix}_DEPLOYMENT must both be set."
            )
        if not endpoint:
            continue
        api_mode = values.get(f"{prefix}_API_MODE", "chat_completions").lower()
        if api_mode not in {"chat_completions", "responses"}:
            raise ValueError(
                f"{prefix}_API_MODE must be chat_completions or responses."
            )
        judges.append(
            JudgeSpec(
                name=f"judge_{number}",
                deployment=deployment,
                endpoint=endpoint,
                api_version=api_version,
                use_entra_id=use_entra_id,
                max_tokens=max_tokens,
                auth_scope=values.get(
                    f"{prefix}_AUTH_SCOPE",
                    "https://cognitiveservices.azure.com/.default",
                ).strip(),
                api_mode=api_mode,
            )
        )

    if not judges:
        raise ValueError(f"No enabled JUDGE_<N> pairs were found in {path}")
    return judges, values
