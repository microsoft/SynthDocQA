# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
Main evaluation runner.

Reads all model and API settings from the shared .env file.
Dataset paths and run options come from CLI arguments.

Usage:
    python -m pdfqna --env path/to/.env --dataset-dir path/to/variant_configs_v2
    python -m pdfqna --env path/to/.env --dataset-dir ... --config-filter "cfg_001*" --max-docs 5
    python -m pdfqna --env path/to/.env --dataset-dir ... --markdown-source mineru
"""
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

from .config import EvalConfig, ModelSpec, load_config
from .judge_runner import judge_answer
from .md_loader import find_markdown, load_markdown
from .qa_runner import build_clients, run_qa
from .results import write_summary

logger = logging.getLogger(__name__)

_DEFAULT_ENV = Path("experiment.env")
_DEFAULT_DATASET = Path("dataset")


# ---------------------------------------------------------------------------
# Doc discovery
# ---------------------------------------------------------------------------


def discover_docs(cfg: EvalConfig) -> list[tuple[str, str, Path, Path]]:
    """
    Find all (config_name, doc_id, md_path, quest_path) tuples.
    Applies config_filter and max_docs limits from cfg.
    """
    base = cfg.dataset_base_dir
    if not base or not base.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {base}\n"
            "Pass --dataset-dir pointing to variant_configs_v2/"
        )

    config_dirs = sorted(d for d in base.iterdir() if d.is_dir())
    if cfg.config_filter:
        config_dirs = [d for d in config_dirs if fnmatch.fnmatch(d.name, cfg.config_filter)]

    docs: list[tuple[str, str, Path, Path]] = []
    for config_dir in config_dirs:
        quest_dir = config_dir / "quest"
        if not quest_dir.exists():
            continue
        for quest_file in sorted(quest_dir.glob("*.json")):
            doc_id = quest_file.stem
            md_path = find_markdown(
                base, config_dir.name, doc_id, cfg.markdown_source
            )
            if md_path is None:
                logger.warning(
                    "No markdown for %s / %s (source=%r) — skipping",
                    config_dir.name, doc_id, cfg.markdown_source,
                )
                continue
            docs.append((config_dir.name, doc_id, md_path, quest_file))
            if cfg.max_docs and len(docs) >= cfg.max_docs:
                return docs
    return docs


# ---------------------------------------------------------------------------
# Quest loading
# ---------------------------------------------------------------------------


def load_quest(quest_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(quest_path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else [raw]


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------


def _call_one_model(
    clients: dict[str, Any],
    spec: ModelSpec,
    md_content: str,
    question: str,
    label: str,
) -> tuple[str, dict[str, Any]]:
    """Worker: call one QA model and return (model_name, result)."""
    result = run_qa(clients, spec, md_content, question, label)
    return spec.name, result


def _process_question(
    q: dict[str, Any],
    q_idx: int,
    md_content: str,
    cfg: EvalConfig,
    clients: dict[str, Any],   # endpoint -> AzureOpenAI client
    existing_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run QA models in parallel for one question, then judge each answer.

    If existing_results is provided (from a previous partial run), models that
    already have a non-skip result are reused as-is (including their judge verdict).
    Only models with status=="skip" or absent from existing_results are re-run.
    """
    question = q.get("query", "")
    q_id = q.get("Id", f"Q{q_idx + 1}")
    refs = q.get("refs", [{}])
    element_type = refs[0].get("element_type", "unknown") if refs else "unknown"
    assertions = q.get("assertions", [])
    assertion_text = assertions[0].get("text", "") if assertions else ""

    existing = existing_results or {}

    # Step 1 — determine which models need to (re-)run
    models_to_run = [
        spec for spec in cfg.models
        if spec.name not in existing or existing[spec.name].get("status") == "skip"
    ]

    new_results: dict[str, Any] = {}
    if models_to_run:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {
                pool.submit(
                    _call_one_model,
                    clients,
                    spec,
                    md_content,
                    question,
                    f"{q_id}|{spec.name}",
                ): spec.name
                for spec in models_to_run
            }
            for future in as_completed(futures):
                model_name, qa_result = future.result()
                new_results[model_name] = qa_result

    # Merge: existing non-skip results + freshly computed results
    model_results: dict[str, Any] = {}
    for spec in cfg.models:
        name = spec.name
        if name in new_results:
            model_results[name] = new_results[name]
        elif name in existing:
            model_results[name] = existing[name]

    # Step 2 — judge only missing (judge, model) pairs. Existing legacy
    # `judge` results are migrated to judges.judge_1 and kept for compatibility.
    _evaluate_model_results(
        model_results=model_results,
        question=question,
        assertion=assertion_text,
        q_id=q_id,
        cfg=cfg,
        clients=clients,
    )

    return {
        "question_id": q_id,
        "question": question,
        "element_type": element_type,
        "assertion": assertion_text,
        "model_results": model_results,
    }


# ---------------------------------------------------------------------------
# Per-doc inference
# ---------------------------------------------------------------------------


def infer_doc(
    config_name: str,
    doc_id: str,
    md_path: Path,
    quest_path: Path,
    cfg: EvalConfig,
    clients: dict[str, Any],
    per_doc_dir: Path,
    max_qa: int | None = None,
) -> dict[str, Any]:
    """
    Run inference for one document: all questions x all models x judge.

    Results are appended to a .jsonl file immediately after each question
    so that a crashed or interrupted run can resume from the exact question
    it left off rather than restarting the whole document.
    """
    jsonl_path = per_doc_dir / f"{config_name}__{doc_id}.jsonl"

    # Load already-completed questions from a previous (partial) run.
    # "done"    = fully complete (no skipped models)
    # "partial" = has at least one model with status=="skip" — will be retried
    # Last entry wins for both dicts (handles overwrites from repeated runs).
    done: dict[str, dict[str, Any]] = {}
    partial: dict[str, dict[str, Any]] = {}
    if cfg.resume and jsonl_path.exists():
        for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                q_id = entry.get("question_id", "")
                if not q_id:
                    continue
                model_results = entry.get("model_results", {})
                qa_entry = {
                    "question_id": q_id,
                    "question": entry.get("question", ""),
                    "element_type": entry.get("element_type", "unknown"),
                    "assertion": entry.get("assertion", ""),
                    "model_results": model_results,
                }
                has_skips = any(mr.get("status") == "skip" for mr in model_results.values())
                if has_skips:
                    partial[q_id] = qa_entry
                    done.pop(q_id, None)
                else:
                    done[q_id] = qa_entry
                    partial.pop(q_id, None)
            except json.JSONDecodeError:
                pass

    # Load questions first (fast) so we can check the cache before paying
    # the cost of reading the (potentially large) markdown file.
    questions = load_quest(quest_path)

    # Filter to important queries only (matched by query text)
    if cfg.important_queries:
        before = len(questions)
        questions = [
            q for q in questions
            if q.get("query", "") in cfg.important_queries
        ]
        logger.info(
            "  important-queries filter: %d → %d questions", before, len(questions)
        )

    # max_qa parameter overrides cfg.max_qa_per_doc (used for global total cap)
    effective_max_qa = max_qa if max_qa is not None else cfg.max_qa_per_doc
    if effective_max_qa:
        questions = questions[:effective_max_qa]

    # Pre-compute question IDs so we can check the cache without loading markdown.
    q_ids = [q.get("Id", f"Q{i + 1}") for i, q in enumerate(questions)]

    # Fast path: every question is fully cached — skip markdown I/O entirely.
    if all(qid in done for qid in q_ids):
        qa_results = []
        for i, (q, qid) in enumerate(zip(questions, q_ids)):
            el_type = (q.get("refs") or [{}])[0].get("element_type", "?")
            logger.info("  Q%d/%d  %s  [%s]  (cached)", i + 1, len(questions), qid, el_type)
            qa_results.append(done[qid])
        return {
            "config": config_name,
            "doc_id": doc_id,
            "md_path": str(md_path),
            "markdown_source": cfg.markdown_source,
            "qa_results": qa_results,
        }

    # At least one question needs work — now load the markdown.
    md_content = load_markdown(md_path)

    qa_results: list[dict[str, Any]] = []

    # Open in append mode when resuming (preserve existing lines), write mode otherwise
    file_mode = "a" if cfg.resume else "w"
    with jsonl_path.open(file_mode, encoding="utf-8") as f:
        for i, (q, q_id) in enumerate(zip(questions, q_ids)):
            el_type = (q.get("refs") or [{}])[0].get("element_type", "?")

            if q_id in done:
                logger.info("  Q%d/%d  %s  [%s]  (cached)", i + 1, len(questions), q_id, el_type)
                qa_results.append(done[q_id])
                continue

            existing = None
            if q_id in partial:
                existing = partial[q_id]["model_results"]
                skip_count = sum(1 for mr in existing.values() if mr.get("status") == "skip")
                logger.info(
                    "  Q%d/%d  %s  [%s]  (retrying %d skip(s))",
                    i + 1, len(questions), q_id, el_type, skip_count,
                )
            else:
                logger.info("  Q%d/%d  %s  [%s]", i + 1, len(questions), q_id, el_type)

            qa_result = _process_question(
                q, i, md_content, cfg, clients, existing_results=existing
            )
            qa_results.append(qa_result)

            # Write immediately — flush so the line is on disk even if we crash
            entry = {
                "config": config_name,
                "doc_id": doc_id,
                "md_path": str(md_path),
                "markdown_source": cfg.markdown_source,
                **qa_result,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

    return {
        "config": config_name,
        "doc_id": doc_id,
        "md_path": str(md_path),
        "markdown_source": cfg.markdown_source,
        "qa_results": qa_results,
    }


def _non_call_judge_result(qa_result: dict[str, Any]) -> dict[str, Any] | None:
    """Return a synthetic verdict when a QA output cannot be judged."""
    if qa_result.get("status") == "skip":
        return {
            "verdict": "SKIP",
            "confidence": 0.0,
            "reasoning": f"Skipped: {qa_result.get('skip_reason', 'rate_limit')}",
            "missing_aspects": [],
            "quality": "skip",
            "error": None,
            "judge_time": 0.0,
        }
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
    if "predicted" not in qa_result:
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "reasoning": "QA result has no predicted answer.",
            "missing_aspects": [],
            "quality": "error",
            "error": "missing predicted field",
            "judge_time": 0.0,
        }
    return None


def _call_one_judge(
    clients: dict[str, Any],
    spec: ModelSpec,
    question: str,
    assertion: str,
    predicted: str,
    label: str,
    judge_run_id: str = "",
) -> tuple[str, dict[str, Any]]:
    result = judge_answer(
        client=clients[spec.endpoint],
        spec=spec,
        question=question,
        assertion=assertion,
        predicted=predicted,
        label=label,
    )
    result["judge_name"] = spec.name
    result["judge_deployment"] = spec.deployment
    if judge_run_id:
        result["judge_run_id"] = judge_run_id
    return spec.name, result


def _evaluate_model_results(
    model_results: dict[str, Any],
    question: str,
    assertion: str,
    q_id: str,
    cfg: EvalConfig,
    clients: dict[str, Any],
    rerun_judges: set[str] | None = None,
    judge_run_id: str = "",
) -> bool:
    """Add all missing judge results without making any QA model calls."""
    changed = False
    pending: list[tuple[str, dict[str, Any], ModelSpec]] = []
    forced = rerun_judges or set()

    for model_name, qa_result in model_results.items():
        judges = qa_result.setdefault("judges", {})
        if "judge" in qa_result and "judge_1" not in judges:
            legacy = dict(qa_result["judge"])
            legacy.setdefault("judge_name", "judge_1")
            if cfg.judges and cfg.judges[0].name == "judge_1":
                legacy.setdefault("judge_deployment", cfg.judges[0].deployment)
            judges["judge_1"] = legacy
            changed = True

        synthetic = _non_call_judge_result(qa_result)
        for spec in cfg.judges:
            existing_judge = judges.get(spec.name)
            if (
                spec.name in forced
                and judge_run_id
                and existing_judge
                and existing_judge.get("judge_run_id") == judge_run_id
                and existing_judge.get("verdict") != "ERROR"
            ):
                continue
            if spec.name not in forced and existing_judge and (
                spec.name == "judge_1" or existing_judge.get("verdict") != "ERROR"
            ):
                continue
            if synthetic is not None:
                result = dict(synthetic)
                result["judge_name"] = spec.name
                result["judge_deployment"] = spec.deployment
                if spec.name in forced and judge_run_id:
                    result["judge_run_id"] = judge_run_id
                judges[spec.name] = result
                changed = True
            else:
                pending.append((model_name, qa_result, spec))

    if pending:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {
                pool.submit(
                    _call_one_judge,
                    clients,
                    spec,
                    question,
                    assertion,
                    qa_result["predicted"],
                    f"{q_id}|{spec.name}|{model_name}",
                    judge_run_id if spec.name in forced else "",
                ): (model_name, qa_result, spec)
                for model_name, qa_result, spec in pending
            }
            for future in as_completed(futures):
                model_name, qa_result, spec = futures[future]
                judge_name, result = future.result()
                qa_result["judges"][judge_name] = result
                changed = True
                logger.info("  %s  %s  %s", q_id, model_name, spec.name)

    return changed


def evaluate_existing_jsonl(
    cfg: EvalConfig,
    clients: dict[str, Any],
    per_doc_dir: Path,
    rerun_judges: set[str] | None = None,
    judge_run_id: str = "",
) -> tuple[int, int]:
    """Judge predictions already stored in per-doc JSONL files, in place."""
    paths = sorted(per_doc_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No .jsonl files found in {per_doc_dir}")

    files_changed = 0
    questions_changed = 0
    for file_index, path in enumerate(paths, 1):
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        file_changed = False
        logger.info("[%d/%d] judging existing predictions: %s", file_index, len(paths), path.name)

        for line_index, raw in enumerate(raw_lines):
            line_number = line_index + 1
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Leaving malformed %s:%d unchanged: %s", path.name, line_number, exc)
                continue

            changed = _evaluate_model_results(
                model_results=entry.get("model_results", {}),
                question=entry.get("question", ""),
                assertion=entry.get("assertion", ""),
                q_id=entry.get("question_id", f"line_{line_number}"),
                cfg=cfg,
                clients=clients,
                rerun_judges=rerun_judges,
                judge_run_id=judge_run_id,
            )
            if changed:
                file_changed = True
                questions_changed += 1
                raw_lines[line_index] = json.dumps(entry, ensure_ascii=False)

                # Checkpoint every question atomically. A long multi-file run can
                # be interrupted and resumed without losing completed API calls.
                temp_path = path.with_suffix(path.suffix + ".tmp")
                temp_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
                os.replace(temp_path, path)

        if file_changed:
            files_changed += 1

    return files_changed, questions_changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdfqna",
        description=(
            "Evaluate QA models on a markdown-based PDF dataset.\n"
            "Model names and the Azure endpoint are read from the shared .env file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default=str(_DEFAULT_ENV),
        metavar="PATH",
        help=(
            "Path to the experiment .env file "
            "(default: experiment.env)"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(_DEFAULT_DATASET),
        metavar="PATH",
        help="Dataset root with per-config markdown/ and quest/ directories",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent.parent.parent / "pdf_qna" / "results"),
        metavar="PATH",
        help="Where to write results/ (default: pdf_qna/results/)",
    )
    parser.add_argument(
        "--markdown-source",
        default="dataset",
        metavar="SOURCE",
        help=(
            "Which markdown subfolder to use: 'dataset' (ground-truth) "
            "or a parser name like 'mineru', 'docling' (default: dataset)"
        ),
    )
    parser.add_argument(
        "--config-filter",
        metavar="PATTERN",
        help="fnmatch pattern to limit configs, e.g. 'cfg_001*' or 'cfg_00[12]*'",
    )
    parser.add_argument(
        "--important-queries",
        metavar="PATH",
        help="Path to MOST_IMPORTANT_queries.json — only those questions will be evaluated",
    )
    parser.add_argument("--max-docs", type=int, metavar="N", help="Limit total docs")
    parser.add_argument("--max-qa", type=int, metavar="N", help="Limit questions per doc")
    parser.add_argument(
        "--max-total-qa",
        type=int,
        metavar="N",
        help="Limit total questions across all docs (spread across docs in order)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        metavar="N",
        help="Parallel threads for QA model calls per question (default: 5)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process all docs, ignoring previously cached results",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help=(
            "Evaluate predictions already in OUTPUT_DIR/per_doc with all missing "
            "JUDGE_<N> models; never call QA prediction models"
        ),
    )
    parser.add_argument(
        "--rerun-judges",
        metavar="NAMES",
        help=(
            "Comma-separated judge names to replace in --judge-only mode, "
            "for example: judge_2,judge_3,judge_4"
        ),
    )
    parser.add_argument(
        "--judge-run-id",
        metavar="ID",
        help="Stable ID used to resume an interrupted --rerun-judges operation",
    )
    args = parser.parse_args(argv)

    # Load config from the shared .env file
    cfg = load_config(args.env)

    # Build important-queries filter set if requested.
    # Keyed by query text (not doc_id+question_id) because Id numbering may
    # differ between the important-queries file and the per-doc quest files.
    if args.important_queries:
        iq_path = Path(args.important_queries)
        raw = json.loads(iq_path.read_text(encoding="utf-8"))
        cfg.important_queries = {
            entry["query"]
            for entry in raw
            if entry.get("query")
        }
        logger.info(
            "important-queries filter: %d queries loaded from %s",
            len(cfg.important_queries), iq_path.name,
        )

    # Apply CLI overrides for runtime settings
    cfg.dataset_base_dir = Path(args.dataset_dir)
    cfg.output_dir = Path(args.output_dir)
    cfg.markdown_source = args.markdown_source
    cfg.config_filter = args.config_filter
    cfg.max_docs = args.max_docs
    cfg.max_qa_per_doc = args.max_qa
    cfg.max_total_qa = args.max_total_qa
    cfg.workers = args.workers
    cfg.resume = not args.no_resume

    # Configure logging:
    # - pdfqna loggers at the configured level (INFO by default)
    # - all third-party libraries (openai, httpx, httpcore, azure) silenced to WARNING
    #   so their per-request DEBUG traces don't flood the console
    our_level = getattr(logging, cfg.log_level, logging.INFO)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("pdfqna").setLevel(our_level)
    # Silence noisy third-party loggers regardless of cfg.log_level
    for _noisy in ("openai", "httpx", "httpcore", "azure", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    logger.info("env: %s", cfg.env_path)
    logger.info(
        "models (%d): %s",
        len(cfg.models),
        ", ".join(m.name for m in cfg.models),
    )
    logger.info(
        "judges (%d): %s  source: %s",
        len(cfg.judges),
        ", ".join(f"{j.name}={j.deployment}" for j in cfg.judges),
        cfg.markdown_source,
    )

    # Build one Azure client per unique endpoint (QA_* models each have their own)
    clients = build_clients(cfg)

    if args.judge_only:
        per_doc_dir = cfg.output_dir / "per_doc"
        rerun_judges = {
            name.strip()
            for name in (args.rerun_judges or "").split(",")
            if name.strip()
        }
        configured_judges = {judge.name for judge in cfg.judges}
        unknown_judges = rerun_judges - configured_judges
        if unknown_judges:
            parser.error(
                "--rerun-judges contains unconfigured judge(s): "
                + ", ".join(sorted(unknown_judges))
            )
        if "judge_1" in rerun_judges:
            parser.error("judge_1 is the preserved baseline and cannot be force-rerun")
        if rerun_judges and not args.judge_run_id:
            parser.error("--judge-run-id is required with --rerun-judges")
        files_changed, questions_changed = evaluate_existing_jsonl(
            cfg,
            clients,
            per_doc_dir,
            rerun_judges=rerun_judges,
            judge_run_id=args.judge_run_id or "",
        )
        logger.info(
            "Done. Updated %d file(s), %d question record(s) in %s",
            files_changed,
            questions_changed,
            per_doc_dir,
        )
        return

    # Discover docs
    docs = discover_docs(cfg)
    if not docs:
        logger.warning(
            "No docs found in %s (filter=%r). Check --dataset-dir and --config-filter.",
            cfg.dataset_base_dir, cfg.config_filter,
        )
        sys.exit(0)
    logger.info("Found %d doc(s) to process", len(docs))

    per_doc_dir = cfg.output_dir / "per_doc"
    per_doc_dir.mkdir(parents=True, exist_ok=True)

    # Main loop — resume is handled per-question inside infer_doc
    all_results: list[dict[str, Any]] = []
    total_qa_done = 0
    for i, (config_name, doc_id, md_path, quest_path) in enumerate(docs, 1):
        if cfg.max_total_qa and total_qa_done >= cfg.max_total_qa:
            break

        # Compute how many questions this doc is allowed to process
        max_qa_this_doc = cfg.max_qa_per_doc
        if cfg.max_total_qa:
            remaining = cfg.max_total_qa - total_qa_done
            max_qa_this_doc = min(max_qa_this_doc, remaining) if max_qa_this_doc else remaining

        logger.info("[%d/%d] %s / %s", i, len(docs), config_name, doc_id)
        doc_result = infer_doc(
            config_name=config_name,
            doc_id=doc_id,
            md_path=md_path,
            quest_path=quest_path,
            cfg=cfg,
            clients=clients,
            per_doc_dir=per_doc_dir,
            max_qa=max_qa_this_doc,
        )
        all_results.append(doc_result)
        total_qa_done += len(doc_result["qa_results"])

    # Write final summary
    logger.info("Writing summary over %d doc(s)...", len(all_results))
    write_summary(all_results, cfg.output_dir, [m.name for m in cfg.models])
    logger.info("Done. Results in: %s", cfg.output_dir)


if __name__ == "__main__":
    main()
