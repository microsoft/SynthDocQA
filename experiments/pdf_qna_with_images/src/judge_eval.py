# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Judge existing direct-PDF QA JSONL results in place."""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import openai
from azure.identity import (
    AzureCliCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI

from judge_config import JudgeSpec, load_judge_specs
from judge_runner import judge_answer


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = Path("experiment.env")
DEFAULT_RESULTS_DIR = (
    WORKSPACE_ROOT / "pdf_qna_with_images" / "results" / "per_doc"
)
logger = logging.getLogger("pdf_qna_with_images.judges")
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
_TOKEN_LOCK = threading.Lock()


def _get_token(credential: Any, scope: str) -> str:
    with _TOKEN_LOCK:
        cached = _TOKEN_CACHE.get(scope)
        if cached and cached[1] - time.time() > 300:
            return cached[0]
        token = credential.get_token(scope)
        _TOKEN_CACHE[scope] = (token.token, float(token.expires_on))
        return token.token


def _is_serverless(endpoint: str) -> bool:
    return "/openai/v1" in endpoint.lower()


def build_judge_client(spec: JudgeSpec, tenant_id: str = "") -> Any:
    """Build an endpoint-appropriate client with API-key or Entra authentication."""
    if _is_serverless(spec.endpoint):
        if spec.use_entra_id:
            if spec.auth_scope == "https://ai.azure.com/.default":
                provider = get_bearer_token_provider(
                    DefaultAzureCredential(), spec.auth_scope
                )
                return openai.OpenAI(base_url=spec.endpoint, api_key=provider)
            credential = (
                AzureCliCredential(tenant_id=tenant_id)
                if tenant_id
                else DefaultAzureCredential()
            )

            class BearerAuth(httpx.Auth):
                def auth_flow(self, request: httpx.Request):
                    request.headers["Authorization"] = (
                        f"Bearer {_get_token(credential, spec.auth_scope)}"
                    )
                    yield request

            return openai.OpenAI(
                base_url=spec.endpoint,
                api_key="placeholder",
                http_client=httpx.Client(auth=BearerAuth()),
            )
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not configured.")
        return openai.OpenAI(base_url=spec.endpoint, api_key=api_key)

    if spec.use_entra_id:
        credential = (
            AzureCliCredential(tenant_id=tenant_id)
            if tenant_id
            else DefaultAzureCredential()
        )

        def token_provider() -> str:
            return _get_token(
                credential, "https://cognitiveservices.azure.com/.default"
            )

        return AzureOpenAI(
            azure_endpoint=spec.endpoint,
            api_version=spec.api_version,
            azure_ad_token_provider=token_provider,
        )
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not configured.")
    return AzureOpenAI(
        azure_endpoint=spec.endpoint,
        api_version=spec.api_version,
        api_key=api_key,
    )


def build_judge_clients(
    judges: list[JudgeSpec], tenant_id: str
) -> dict[str, Any]:
    """Build one client for each unique (endpoint, auth scope)."""
    clients: dict[str, Any] = {}
    for judge in judges:
        if judge.endpoint not in clients:
            clients[judge.endpoint] = build_judge_client(judge, tenant_id)
    return clients


def _synthetic_judgment(qa_result: dict[str, Any]) -> dict[str, Any] | None:
    if qa_result.get("error"):
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "reasoning": f"QA call failed: {qa_result['error']}",
            "missing_aspects": [],
            "quality": "error",
            "error": None,
            "judge_time": 0.0,
        }
    if not qa_result.get("predicted"):
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "reasoning": "QA result has no predicted answer.",
            "missing_aspects": [],
            "quality": "error",
            "error": "missing predicted answer",
            "judge_time": 0.0,
        }
    return None


def _call_one_judge(
    client: Any,
    judge: JudgeSpec,
    question: str,
    assertion: str,
    predicted: str,
    label: str,
    judge_run_id: str,
) -> tuple[str, dict[str, Any]]:
    result = judge_answer(
        client, judge, question, assertion, predicted, label
    )
    result["judge_name"] = judge.name
    result["judge_deployment"] = judge.deployment
    if judge_run_id:
        result["judge_run_id"] = judge_run_id
    return judge.name, result


def evaluate_entry(
    entry: dict[str, Any],
    judges: list[JudgeSpec],
    clients: dict[str, Any],
    workers: int,
    rerun_judges: set[str] | None = None,
    judge_run_id: str = "",
) -> bool:
    """Add missing judge results to one question record."""
    changed = False
    forced = rerun_judges or set()
    pending: list[tuple[str, dict[str, Any], JudgeSpec]] = []
    for model_name, qa_result in entry.get("model_results", {}).items():
        model_judges = qa_result.setdefault("judges", {})
        synthetic = _synthetic_judgment(qa_result)
        for judge in judges:
            existing = model_judges.get(judge.name)
            if synthetic is not None and existing and judge.name not in forced:
                continue
            if (
                judge.name in forced
                and judge_run_id
                and existing
                and existing.get("judge_run_id") == judge_run_id
                and existing.get("verdict") != "ERROR"
            ):
                continue
            if judge.name not in forced and existing and existing.get("verdict") != "ERROR":
                continue
            if synthetic is not None:
                result = dict(synthetic)
                result["judge_name"] = judge.name
                result["judge_deployment"] = judge.deployment
                if judge.name in forced and judge_run_id:
                    result["judge_run_id"] = judge_run_id
                model_judges[judge.name] = result
                changed = True
            else:
                pending.append((model_name, qa_result, judge))

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _call_one_judge,
                    clients[judge.endpoint],
                    judge,
                    entry.get("question", ""),
                    entry.get("assertion", ""),
                    str(qa_result["predicted"]),
                    f"{entry.get('question_id', '?')}|{judge.name}|{model_name}",
                    judge_run_id if judge.name in forced else "",
                ): (model_name, qa_result, judge)
                for model_name, qa_result, judge in pending
            }
            for future in as_completed(futures):
                model_name, qa_result, judge = futures[future]
                judge_name, result = future.result()
                qa_result["judges"][judge_name] = result
                changed = True
                logger.info(
                    "%s  %s  %s  %s",
                    entry.get("question_id", "?"),
                    model_name,
                    judge.name,
                    result.get("verdict"),
                )
    return changed


def evaluate_results_directory(
    results_dir: Path,
    judges: list[JudgeSpec],
    clients: dict[str, Any],
    workers: int = 8,
    rerun_judges: set[str] | None = None,
    judge_run_id: str = "",
    file_pattern: str = "*.jsonl",
) -> tuple[int, int]:
    """Judge all existing result files in place with atomic per-question checkpoints."""
    paths = sorted(results_dir.glob(file_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No JSONL files matching {file_pattern!r} in {results_dir}"
        )
    files_changed = 0
    questions_changed = 0
    for file_index, path in enumerate(paths, start=1):
        lines = path.read_text(encoding="utf-8").splitlines()
        file_changed = False
        logger.info("[%d/%d] %s", file_index, len(paths), path.name)
        for line_index, raw_line in enumerate(lines):
            if not raw_line.strip():
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as error:
                logger.warning("Leaving malformed %s:%d: %s", path.name, line_index + 1, error)
                continue
            if evaluate_entry(
                entry,
                judges,
                clients,
                workers,
                rerun_judges,
                judge_run_id,
            ):
                lines[line_index] = json.dumps(entry, ensure_ascii=False)
                file_changed = True
                questions_changed += 1
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
                os.replace(temporary, path)
        if file_changed:
            files_changed += 1
    return files_changed, questions_changed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate direct-PDF QA predictions with all configured LLM judges."
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--file-pattern", default="*.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--judges",
        help="Optional comma-separated judge names; default is all configured judges.",
    )
    parser.add_argument(
        "--rerun-judges",
        help="Comma-separated judge names to replace, including existing successes.",
    )
    parser.add_argument(
        "--judge-run-id",
        help="Stable ID required for resumable --rerun-judges execution.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    judges, env = load_judge_specs(args.env_file)
    selected = {name.strip() for name in (args.judges or "").split(",") if name.strip()}
    if selected:
        judges = [judge for judge in judges if judge.name in selected]
    configured = {judge.name for judge in judges}
    if selected - configured:
        parser.error("Unknown judge(s): " + ", ".join(sorted(selected - configured)))

    rerun = {
        name.strip()
        for name in (args.rerun_judges or "").split(",")
        if name.strip()
    }
    if rerun - configured:
        parser.error("Unknown rerun judge(s): " + ", ".join(sorted(rerun - configured)))
    if rerun and not args.judge_run_id:
        parser.error("--judge-run-id is required with --rerun-judges")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("openai", "httpx", "httpcore", "azure", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger.info(
        "judges: %s",
        ", ".join(f"{judge.name}={judge.deployment}" for judge in judges),
    )
    clients = build_judge_clients(judges, env.get("AZURE_TENANT_ID", ""))
    files, questions = evaluate_results_directory(
        args.results_dir,
        judges,
        clients,
        workers=max(1, args.workers),
        rerun_judges=rerun,
        judge_run_id=args.judge_run_id or "",
        file_pattern=args.file_pattern,
    )
    logger.info(
        "Done. Updated %d file(s), %d question record(s) in %s",
        files,
        questions,
        args.results_dir,
    )


if __name__ == "__main__":
    main()
