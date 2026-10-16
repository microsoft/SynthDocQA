# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS_DIR = WORKSPACE_ROOT / "pdf_qna" / "results" / "per_doc"
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "pdf_qna_with_images" / "results" / "per_doc"
DEFAULT_ENV_PATH = Path("experiment.env")


@dataclass(frozen=True)
class QaModelSpec:
    """One enabled QA_*_ENDPOINT + QA_*_DEPLOYMENT pair."""

    name: str
    endpoint: str
    deployment: str

    @property
    def is_claude_cli(self) -> bool:
        return self.endpoint.lower() == "claude-cli"


@dataclass(frozen=True)
class UploadedPdfSet:
    """Endpoint-local file IDs for one PDF plus its local path."""

    pdf_path: Path
    file_ids: dict[str, str]
    upload_errors: dict[str, str]


def _parse_env_file(path: Path) -> dict[str, str]:
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


def load_qa_model_specs(env_path: Path) -> tuple[list[QaModelSpec], dict[str, str]]:
    """Load every enabled QA_* model pair from the shared dotenv file."""
    env_path = env_path.resolve()
    if not env_path.is_file():
        raise FileNotFoundError(f"Environment file was not found: {env_path}")
    values = _parse_env_file(env_path)
    load_dotenv(env_path, override=True)

    models: list[QaModelSpec] = []
    for key, endpoint in values.items():
        if not (key.startswith("QA_") and key.endswith("_ENDPOINT")):
            continue
        suffix = key[len("QA_") : -len("_ENDPOINT")]
        deployment = values.get(f"QA_{suffix}_DEPLOYMENT", "").strip()
        if endpoint.strip() and deployment:
            models.append(
                QaModelSpec(
                    name=suffix.lower(),
                    endpoint=endpoint.strip().rstrip("/"),
                    deployment=deployment,
                )
            )
    if not models:
        raise ValueError(f"No enabled QA_* model pairs were found in {env_path}")
    return models, values


def _split_deployments(value: str | None) -> list[str]:
    """Parse a comma-separated deployment list while preserving order."""
    if not value:
        return []
    return list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def configured_deployments(cli_value: str | None = None) -> list[str]:
    """Return Azure deployment names from the CLI or environment.

    AZURE_OPENAI_DEPLOYMENTS accepts a comma-separated list. The singular
    AZURE_OPENAI_DEPLOYMENT variable remains supported for compatibility.
    """
    deployments = _split_deployments(cli_value)
    if not deployments:
        deployments = _split_deployments(os.getenv("AZURE_OPENAI_DEPLOYMENTS"))
    if not deployments:
        deployments = _split_deployments(os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    if not deployments:
        raise ValueError(
            "No models are configured. Pass --models deployment1,deployment2 "
            "or set AZURE_OPENAI_DEPLOYMENTS."
        )
    return deployments


class AzurePdfQA:
    """Ask questions against a directly uploaded PDF with Azure OpenAI.

    The PDF is uploaded once and the file ID is reused with every deployment
    on the same Azure OpenAI resource. A deployment must support the Responses
    API and PDF file input. The batch runner records unsupported deployments as
    per-model errors without stopping the rest of the run.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment_name: str | None = None,
        deployment_names: Sequence[str] | None = None,
        *,
        use_entra_id: bool = False,
        tenant_id: str = "",
    ) -> None:
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")

        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not configured.")
        if not api_key and use_entra_id:
            try:
                from azure.identity import AzureCliCredential
            except ImportError as error:
                raise RuntimeError(
                    "azure-identity is required when USE_ENTRA_ID=true."
                ) from error
            credential = AzureCliCredential(tenant_id=tenant_id or None)

            def api_key_provider() -> str:
                return credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                ).token

            api_key = api_key_provider
        if not api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is not configured and USE_ENTRA_ID is false."
            )

        requested = list(deployment_names or [])
        if deployment_name:
            requested.insert(0, deployment_name)
        if not requested:
            requested = _split_deployments(os.getenv("AZURE_OPENAI_DEPLOYMENTS"))
        if not requested:
            requested = _split_deployments(os.getenv("AZURE_OPENAI_DEPLOYMENT"))
        if not requested:
            raise ValueError(
                "No Azure OpenAI deployment is configured. Set "
                "AZURE_OPENAI_DEPLOYMENTS or AZURE_OPENAI_DEPLOYMENT."
            )
        self.deployment_names = list(dict.fromkeys(requested))

        base_url = endpoint.rstrip("/")
        if not base_url.lower().endswith("/openai/v1"):
            base_url += "/openai/v1"
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{base_url}/",
            max_retries=3,
            timeout=300.0,
        )

    @property
    def deployment_name(self) -> str:
        """Return the default deployment for backward compatibility."""
        if not self.deployment_names:
            raise ValueError("No deployment was supplied to AzurePdfQA.")
        return self.deployment_names[0]

    def upload_pdf(self, pdf_path: str | Path) -> str:
        """Upload a local PDF once and return its Azure OpenAI file ID."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF was not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are accepted: {path}")

        with path.open("rb") as pdf_file:
            uploaded_file = self.client.files.create(
                file=pdf_file,
                purpose="assistants",
            )
        return uploaded_file.id

    def ask(
        self,
        file_id: str,
        question: str,
        *,
        deployment_name: str | None = None,
        include_page_numbers: bool = False,
    ) -> str:
        """Ask one deployment one question against an uploaded PDF."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        deployment = deployment_name or self.deployment_name
        instructions = (
            "Answer the question using only the attached PDF. Do not use outside "
            "knowledge or invent facts. Be precise and concise. If the answer is "
            "not present, say: The answer was not found in the provided PDF."
        )
        if include_page_numbers:
            instructions += " Cite relevant PDF pages as [Page 3] or [Pages 3-5]."

        response = self.client.responses.create(
            model=deployment,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": file_id},
                        {
                            "type": "input_text",
                            "text": f"{instructions}\n\nQuestion:\n{question}",
                        },
                    ],
                }
            ],
        )
        return response.output_text

    def ask_pdf(
        self,
        pdf_path: str | Path,
        question: str,
        *,
        deployment_name: str | None = None,
        delete_after: bool = True,
    ) -> str:
        """Upload a PDF and ask one question in one convenience operation."""
        file_id = self.upload_pdf(pdf_path)
        try:
            return self.ask(
                file_id,
                question,
                deployment_name=deployment_name,
            )
        finally:
            if delete_after:
                self.delete_file(file_id)

    def ask_multiple_separately(
        self,
        file_id: str,
        questions: Iterable[str],
        *,
        deployment_name: str | None = None,
        include_page_numbers: bool = False,
    ) -> list[dict[str, str]]:
        """Ask multiple questions independently using one uploaded PDF."""
        results: list[dict[str, str]] = []
        for question in questions:
            question = question.strip()
            if question:
                results.append(
                    {
                        "question": question,
                        "answer": self.ask(
                            file_id,
                            question,
                            deployment_name=deployment_name,
                            include_page_numbers=include_page_numbers,
                        ),
                    }
                )
        return results

    def ask_multiple_together(
        self,
        file_id: str,
        questions: Iterable[str],
        *,
        deployment_name: str | None = None,
        include_page_numbers: bool = False,
    ) -> str:
        """Ask several numbered questions in one request."""
        cleaned_questions = [
            question.strip()
            for question in questions
            if question and question.strip()
        ]
        if not cleaned_questions:
            raise ValueError("At least one question is required.")

        numbered_questions = "\n".join(
            f"{index}. {question}"
            for index, question in enumerate(cleaned_questions, start=1)
        )
        combined_question = (
            "Answer every numbered question separately and preserve the "
            f"numbering.\n\n{numbered_questions}"
        )
        return self.ask(
            file_id,
            combined_question,
            deployment_name=deployment_name,
            include_page_numbers=include_page_numbers,
        )

    def ask_all_models(
        self,
        file_id: str,
        question: str,
    ) -> dict[str, dict[str, Any]]:
        """Ask every configured deployment and isolate per-model failures."""
        if not self.deployment_names:
            raise ValueError("At least one deployment is required.")

        results: dict[str, dict[str, Any]] = {}
        for deployment in self.deployment_names:
            started = time.perf_counter()
            try:
                answer = self.ask(
                    file_id,
                    question,
                    deployment_name=deployment,
                )
                results[deployment] = {
                    "predicted": answer,
                    "response_time": round(time.perf_counter() - started, 3),
                    "error": None,
                }
            except Exception as error:
                results[deployment] = {
                    "predicted": None,
                    "response_time": round(time.perf_counter() - started, 3),
                    "error": api_error_message(error),
                }
        return results

    def delete_file(self, file_id: str) -> None:
        """Delete an uploaded PDF from Azure OpenAI file storage."""
        self.client.files.delete(file_id)


class AzurePdfQAFleet:
    """Run all QA_* models, grouping Azure deployments by endpoint."""

    def __init__(
        self,
        models: Sequence[QaModelSpec],
        *,
        use_entra_id: bool,
        tenant_id: str = "",
    ) -> None:
        self.models = list(models)
        self.deployment_names = [model.name for model in self.models]
        deployments_by_endpoint: dict[str, list[str]] = {}
        for model in self.models:
            if not model.is_claude_cli:
                deployments_by_endpoint.setdefault(model.endpoint, []).append(
                    model.deployment
                )
        self.clients = {
            endpoint: AzurePdfQA(
                endpoint=endpoint,
                deployment_names=deployments,
                use_entra_id=use_entra_id,
                tenant_id=tenant_id,
            )
            for endpoint, deployments in deployments_by_endpoint.items()
        }

    def upload_pdf(self, pdf_path: str | Path) -> UploadedPdfSet:
        path = Path(pdf_path).resolve()
        file_ids: dict[str, str] = {}
        errors: dict[str, str] = {}
        for endpoint, client in self.clients.items():
            try:
                file_ids[endpoint] = client.upload_pdf(path)
            except Exception as error:
                errors[endpoint] = api_error_message(error)
        return UploadedPdfSet(path, file_ids, errors)

    @staticmethod
    def _ask_claude(model: QaModelSpec, pdf_path: Path, question: str) -> str:
        executable = shutil.which("claude")
        if not executable:
            raise RuntimeError("claude CLI was not found in PATH.")
        command = [executable, "--model", model.deployment, "-p"]
        if sys.platform == "win32" and executable.lower().endswith((".cmd", ".bat")):
            command = ["cmd", "/c", *command]
        prompt = (
            f"Read the PDF file at {pdf_path}. Answer using only that PDF. "
            "Be precise and concise; do not invent facts.\n\n"
            f"Question: {question}"
        )
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "claude CLI failed")
        if not result.stdout.strip():
            raise RuntimeError("claude CLI returned an empty response")
        return result.stdout.strip()

    def ask_all_models(
        self,
        uploaded: UploadedPdfSet,
        question: str,
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for model in self.models:
            started = time.perf_counter()
            try:
                if model.is_claude_cli:
                    answer = self._ask_claude(model, uploaded.pdf_path, question)
                else:
                    upload_error = uploaded.upload_errors.get(model.endpoint)
                    if upload_error:
                        raise RuntimeError(upload_error)
                    file_id = uploaded.file_ids.get(model.endpoint)
                    if not file_id:
                        raise RuntimeError("PDF upload did not return a file ID.")
                    answer = self.clients[model.endpoint].ask(
                        file_id,
                        question,
                        deployment_name=model.deployment,
                    )
                results[model.name] = {
                    "deployment": model.deployment,
                    "endpoint": model.endpoint,
                    "predicted": answer,
                    "response_time": round(time.perf_counter() - started, 3),
                    "error": None,
                }
            except Exception as error:
                results[model.name] = {
                    "deployment": model.deployment,
                    "endpoint": model.endpoint,
                    "predicted": None,
                    "response_time": round(time.perf_counter() - started, 3),
                    "error": api_error_message(error),
                }
        return results

    def delete_file(self, uploaded: UploadedPdfSet) -> None:
        for endpoint, file_id in uploaded.file_ids.items():
            self.clients[endpoint].delete_file(file_id)


def api_error_message(error: Exception) -> str:
    """Return a useful error message without exposing credentials."""
    if isinstance(error, AuthenticationError):
        return "Authentication failed. Check the Azure OpenAI API key."
    if isinstance(error, RateLimitError):
        return f"Azure OpenAI rate/capacity limit reached: {error.message}"
    if isinstance(error, BadRequestError):
        return f"Azure rejected the request: {error.message}"
    if isinstance(error, APIConnectionError):
        return "Could not connect to the Azure OpenAI endpoint."
    if isinstance(error, APIStatusError):
        return f"Azure OpenAI returned HTTP {error.status_code}: {error.message}"
    return f"Unexpected error: {error}"


def print_api_error(error: Exception) -> None:
    print(api_error_message(error))


def load_questions(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load and de-duplicate questions from a per_doc JSONL result file."""
    questions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with jsonl_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}: {error}"
                ) from error
            question = str(item.get("question", "")).strip()
            question_id = str(item.get("question_id", "")).strip()
            if not question:
                continue
            key = (question_id, question)
            if key not in seen:
                seen.add(key)
                questions.append(item)
    return questions


def resolve_pdf_path(item: dict[str, Any], dataset_root: Path) -> Path:
    """Resolve the PDF corresponding to one per_doc question entry."""
    explicit_path = item.get("pdf_path")
    if explicit_path and Path(explicit_path).is_file():
        return Path(explicit_path)

    md_path = item.get("md_path")
    if md_path:
        markdown_path = Path(md_path)
        if markdown_path.parent.name.lower() == "markdown":
            candidate = markdown_path.parent.parent / "pdfs" / f"{markdown_path.stem}.pdf"
            if candidate.is_file():
                return candidate

    config = str(item.get("config", ""))
    doc_id = str(item.get("doc_id", ""))
    candidate = dataset_root / config / "pdfs" / f"{doc_id}.pdf"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Could not find the PDF for config={config!r}, doc_id={doc_id!r}; "
        f"expected {candidate}"
    )


def _safe_output_name(input_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", input_path.name)


def run_question_directory(
    pdf_qa: AzurePdfQA | AzurePdfQAFleet,
    questions_dir: Path = DEFAULT_QUESTIONS_DIR,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    file_pattern: str = "*.jsonl",
    max_questions: int = 0,
    keep_uploaded_files: bool = False,
) -> tuple[int, int]:
    """Run per_doc questions against matching PDFs for all deployments.

    Returns (documents_processed, questions_processed). Output is flushed after
    every question, making completed records usable if a later request fails.
    """
    input_paths = sorted(questions_dir.glob(file_pattern))
    if not input_paths:
        raise FileNotFoundError(
            f"No question files matching {file_pattern!r} in {questions_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    documents_processed = 0
    questions_processed = 0
    for input_path in input_paths:
        items = load_questions(input_path)
        if max_questions > 0:
            items = items[:max_questions]
        if not items:
            continue

        pdf_path = resolve_pdf_path(items[0], dataset_root)
        output_path = output_dir / _safe_output_name(input_path)
        print(f"\n{input_path.name}: uploading {pdf_path.name}")
        uploaded_pdf = pdf_qa.upload_pdf(pdf_path)
        try:
            with output_path.open("w", encoding="utf-8") as output:
                for index, item in enumerate(items, start=1):
                    question = str(item["question"]).strip()
                    print(f"  [{index}/{len(items)}] {item.get('question_id', '')}")
                    result = {
                        "config": item.get("config"),
                        "doc_id": item.get("doc_id"),
                        "pdf_path": str(pdf_path),
                        "question_id": item.get("question_id"),
                        "question": question,
                        "element_type": item.get("element_type"),
                        "assertion": item.get("assertion"),
                        "model_results": pdf_qa.ask_all_models(uploaded_pdf, question),
                    }
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    questions_processed += 1
            documents_processed += 1
        finally:
            if not keep_uploaded_files:
                try:
                    pdf_qa.delete_file(uploaded_pdf)
                except Exception as error:
                    print(f"  Warning: could not delete uploaded file: {error}")

    return documents_processed, questions_processed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload each dataset PDF directly to Azure OpenAI and ask the "
            "questions in pdf_qna/results/per_doc with every configured model."
        )
    )
    parser.add_argument(
        "--models",
        help=(
            "Optional comma-separated QA model names or deployment names to run. "
            "By default every enabled QA_* pair in --env-file is used."
        ),
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--questions-dir", type=Path, default=DEFAULT_QUESTIONS_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--file-pattern",
        default="*.jsonl",
        help='Input filename glob, for example "cfg_001*.jsonl".',
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Maximum questions per PDF; 0 means all.",
    )
    parser.add_argument(
        "--keep-uploaded-files",
        action="store_true",
        help="Do not delete PDFs from Azure file storage after each document.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    try:
        models, env = load_qa_model_specs(args.env_file)
        requested = set(_split_deployments(args.models))
        if requested:
            models = [
                model
                for model in models
                if model.name in requested or model.deployment in requested
            ]
            if not models:
                raise ValueError("--models did not match any enabled QA_* model.")
        print(
            "QA models: "
            + ", ".join(f"{model.name} ({model.deployment})" for model in models)
        )
        pdf_qa = AzurePdfQAFleet(
            models,
            use_entra_id=env.get("USE_ENTRA_ID", "true").lower() == "true",
            tenant_id=env.get("AZURE_TENANT_ID", ""),
        )
        documents, questions = run_question_directory(
            pdf_qa,
            questions_dir=args.questions_dir,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            file_pattern=args.file_pattern,
            max_questions=args.max_questions,
            keep_uploaded_files=args.keep_uploaded_files,
        )
        print(
            f"\nCompleted {questions} questions across {documents} PDFs and "
            f"{len(models)} deployments. Results: {args.output_dir}"
        )
    except Exception as error:
        print_api_error(error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
