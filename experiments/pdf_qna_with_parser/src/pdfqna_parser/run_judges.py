# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Evaluate parser-based QnA predictions with the configured judge models."""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PDF_QNA_SRC = _REPO_ROOT / "pdf_qna" / "src"
if str(_PDF_QNA_SRC) not in sys.path:
    sys.path.insert(0, str(_PDF_QNA_SRC))

from pdfqna.config import EvalConfig, ModelSpec, load_config  # pyright: ignore[reportMissingImports]  # noqa: E402
from pdfqna.judge_runner import judge_answer  # pyright: ignore[reportMissingImports]  # noqa: E402
from pdfqna.qa_runner import build_clients  # pyright: ignore[reportMissingImports]  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_ENV = Path("experiment.env")
_DEFAULT_PER_DOC = (
    _REPO_ROOT / "pdf_qna_with_parser" / "results" / "docling" / "per_doc"
)
_VALID_VERDICTS = {"PASS", "PARTIAL", "FAIL"}


def _synthetic_judgment(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return a no-call verdict for predictions that cannot be judged."""
    if result.get("status") == "skip":
        return {
            "verdict": "SKIP",
            "confidence": 0.0,
            "reasoning": f"Skipped: {result.get('skip_reason', 'rate_limit')}",
            "missing_aspects": [],
            "quality": "skip",
            "error": None,
            "judge_time": 0.0,
        }
    if result.get("error"):
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "reasoning": f"QA call failed: {result['error']}",
            "missing_aspects": [],
            "quality": "error",
            "error": None,
            "judge_time": 0.0,
        }
    if not result.get("predicted"):
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "reasoning": "QA model returned no answer.",
            "missing_aspects": [],
            "quality": "error",
            "error": "empty prediction",
            "judge_time": 0.0,
        }
    return None


def _call_judge(
    clients: dict[str, Any],
    spec: ModelSpec,
    question: str,
    assertion: str,
    predicted: str,
    label: str,
) -> tuple[str, dict[str, Any]]:
    judgment = judge_answer(
        client=clients[spec.endpoint],
        spec=spec,
        question=question,
        assertion=assertion,
        predicted=predicted,
        label=label,
    )
    judgment["judge_name"] = spec.name
    judgment["judge_deployment"] = spec.deployment
    return spec.name, judgment


def evaluate_entry(
    entry: dict[str, Any],
    judges: list[ModelSpec],
    clients: dict[str, Any],
    workers: int,
    retry_errors: bool = True,
) -> bool:
    """Add missing judge results to one question entry; return whether it changed."""
    changed = False
    pending: list[tuple[str, dict[str, Any], ModelSpec]] = []
    question = entry.get("question", "")
    assertion = entry.get("assertion", "")
    question_id = entry.get("question_id", "unknown")

    for model_name, prediction in entry.get("model_results", {}).items():
        judgment_map = prediction.setdefault("judges", {})
        if "judge" in prediction and "judge_1" not in judgment_map:
            legacy = dict(prediction["judge"])
            legacy.setdefault("judge_name", "judge_1")
            judgment_map["judge_1"] = legacy
            changed = True

        synthetic = _synthetic_judgment(prediction)
        for spec in judges:
            current = judgment_map.get(spec.name)
            if current and current.get("verdict") in _VALID_VERDICTS:
                continue
            if current and current.get("verdict") == "ERROR" and not retry_errors:
                continue
            if synthetic is not None:
                result = dict(synthetic)
                result["judge_name"] = spec.name
                result["judge_deployment"] = spec.deployment
                if current != result:
                    judgment_map[spec.name] = result
                    changed = True
            else:
                pending.append((model_name, prediction, spec))

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _call_judge,
                    clients,
                    spec,
                    question,
                    assertion,
                    prediction["predicted"],
                    f"{question_id}|{spec.name}|{model_name}",
                ): (model_name, prediction, spec)
                for model_name, prediction, spec in pending
            }
            for future in as_completed(futures):
                model_name, prediction, spec = futures[future]
                judge_name, result = future.result()
                prediction["judges"][judge_name] = result
                changed = True
                logger.info("  %s  %s  %s", question_id, model_name, spec.name)
    return changed


def _checkpoint(path: Path, lines: list[str]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def evaluate_file(
    path: Path,
    judges: list[ModelSpec],
    clients: dict[str, Any],
    workers: int,
    retry_errors: bool = True,
    max_questions: int | None = None,
) -> tuple[int, int]:
    """Judge the latest record for each question and checkpoint after each update."""
    lines = path.read_text(encoding="utf-8").splitlines()
    latest_line: dict[str, int] = {}
    parsed: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Leaving malformed %s:%d unchanged: %s", path.name, index + 1, exc)
            continue
        parsed[index] = entry
        question_id = entry.get("question_id")
        if question_id:
            latest_line[question_id] = index

    changed_questions = 0
    processed_questions = 0
    for line_index in sorted(latest_line.values()):
        if max_questions is not None and processed_questions >= max_questions:
            break
        processed_questions += 1
        entry = parsed[line_index]
        if evaluate_entry(entry, judges, clients, workers, retry_errors):
            lines[line_index] = json.dumps(entry, ensure_ascii=False)
            _checkpoint(path, lines)
            changed_questions += 1
    return changed_questions, processed_questions


def judge_directory(
    per_doc_dir: Path,
    cfg: EvalConfig,
    clients: dict[str, Any],
    config_filter: str | None = None,
    max_files: int | None = None,
    max_questions: int | None = None,
    retry_errors: bool = True,
) -> tuple[int, int]:
    """Run a resumable judge pass over parser result files."""
    if not per_doc_dir.is_dir():
        raise FileNotFoundError(f"Per-document results directory not found: {per_doc_dir}")
    paths = sorted(per_doc_dir.glob("*.jsonl"))
    if config_filter:
        paths = [path for path in paths if fnmatch.fnmatch(path.name, config_filter)]
    if max_files:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No matching JSONL files found in {per_doc_dir}")

    changed_total = 0
    processed_total = 0
    for file_number, path in enumerate(paths, 1):
        if max_questions is not None and processed_total >= max_questions:
            break
        remaining = None if max_questions is None else max_questions - processed_total
        logger.info("[%d/%d] %s", file_number, len(paths), path.name)
        changed, processed = evaluate_file(
            path,
            cfg.judges,
            clients,
            cfg.workers,
            retry_errors=retry_errors,
            max_questions=remaining,
        )
        changed_total += changed
        processed_total += processed
    return changed_total, processed_total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the four configured judges over parser QnA prediction JSONL files."
    )
    parser.add_argument("--env", default=str(_DEFAULT_ENV), metavar="PATH")
    parser.add_argument("--per-doc-dir", default=str(_DEFAULT_PER_DOC), metavar="PATH")
    parser.add_argument(
        "--judges",
        default="judge_1,judge_2,judge_3,judge_4",
        metavar="NAMES",
        help="Comma-separated configured judges (default: all four)",
    )
    parser.add_argument("--config-filter", metavar="PATTERN")
    parser.add_argument("--max-files", type=int, metavar="N")
    parser.add_argument("--max-questions", type=int, metavar="N")
    parser.add_argument("--workers", type=int, default=5, metavar="N")
    parser.add_argument(
        "--no-retry-errors",
        action="store_true",
        help="Preserve existing ERROR judge responses instead of retrying them",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.env)
    requested = {name.strip() for name in args.judges.split(",") if name.strip()}
    configured = {spec.name: spec for spec in cfg.judges}
    unknown = requested - configured.keys()
    if unknown:
        parser.error("Unconfigured judge(s): " + ", ".join(sorted(unknown)))
    cfg.judges = [spec for spec in cfg.judges if spec.name in requested]
    if not cfg.judges:
        parser.error("At least one judge must be selected")
    cfg.models = []
    cfg.workers = args.workers

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("openai", "httpx", "httpcore", "azure", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info(
        "Judges (%d): %s",
        len(cfg.judges),
        ", ".join(f"{spec.name}={spec.deployment}" for spec in cfg.judges),
    )
    clients = build_clients(cfg)
    changed, processed = judge_directory(
        Path(args.per_doc_dir),
        cfg,
        clients,
        config_filter=args.config_filter,
        max_files=args.max_files,
        max_questions=args.max_questions,
        retry_errors=not args.no_retry_errors,
    )
    logger.info("Done. Processed %d question(s); updated %d", processed, changed)


if __name__ == "__main__":
    main()