# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""PDF-native inference + judge evaluation pipeline.

Overview
--------
For each document in the v2 dataset:

  Pre-load (cheap, main thread):
    - Extract text from every page using PyMuPDF (~0.1 s for 200 pages).
      This page-text list drives question-guided page selection.

  Per question (parallel worker threads):
    1. Select relevant pages for this question:
       a. Extract element title from question text (regex).
       b. Search page texts for that title -> return matching page + window.
       c. If not found, fall back to BM25 ranking of all pages.
       d. If still nothing, use first N pages.
    2. Render only the selected pages as images (on-demand fitz rendering).
       For a 200-page PDF and a question answered on page 47, this sends
       ~5-10 images instead of 200.
    3. Call the QA model with (selected page images + question) -> answer.
    4. Call the judge model with (answer + GT assertion) -> PASS/FAIL.

  Outputs (written incrementally):
    <results-dir>/
      per_doc/   {config}__{doc_id}.json
      summary.json
      summary.xlsx   (Summary | By Element Type | QA Details)

Usage
-----
  py -3.12 -m inference.run_inference \\
      --models-config inference_models.yaml --dataset-dir <dataset>

  # Smoke-test on one config, 5 questions:
  py -3.12 -m inference.run_inference \\
      --models-config inference_models.yaml --dataset-dir <dataset> \\
      --config-filter "cfg_001*" --max-qa 5

  # Force re-run:
  py -3.12 -m inference.run_inference \\
      --models-config inference_models.yaml --dataset-dir <dataset> --force

  # Collect answers only (skip judging):
  py -3.12 -m inference.run_inference \\
      --models-config inference_models.yaml --dataset-dir <dataset> --no-judge
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from .config import (
    INFERENCE_RESULTS_DIR, VARIANT_CONFIGS_V2,
    InferenceConfig, ModelSpec, load_models_config,
)
from .judge_runner import judge_answer
from .page_selection import explain_page_selection, select_relevant_pages
from .pdf_loader import PDFPageCache
from .qa_runner import build_client, run_qa
from .results import save_per_doc, write_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_docs(
    config_filter: Optional[str] = None,
    dataset_dir: Path = VARIANT_CONFIGS_V2,
) -> list[tuple[str, str, Path, Path]]:
    """Return (config, doc_id, pdf_path, quest_path) for all available docs."""
    docs: list[tuple[str, str, Path, Path]] = []

    if not dataset_dir.exists():
        logger.error("Dataset dir not found: %s", dataset_dir)
        return docs

    for config_dir in sorted(dataset_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        if config_filter and not fnmatch.fnmatch(config_dir.name, config_filter):
            continue
        quest_dir = config_dir / "quest"
        pdf_dir = config_dir / "pdfs"
        if not quest_dir.exists():
            continue
        for quest_path in sorted(quest_dir.glob("*.json")):
            doc_id = quest_path.stem
            pdf_path = pdf_dir / f"{doc_id}.pdf"
            if pdf_path.exists():
                docs.append((config_dir.name, doc_id, pdf_path, quest_path))
            else:
                logger.debug("PDF missing for %s/%s — skipping", config_dir.name, doc_id)

    return docs


def _load_quest(quest_path: Path) -> list[dict]:
    try:
        items = json.loads(quest_path.read_text(encoding="utf-8"))
        return items if isinstance(items, list) else []
    except Exception as exc:
        logger.error("Failed to load quest %s: %s", quest_path, exc)
        return []


# ---------------------------------------------------------------------------
# Per-question worker
# ---------------------------------------------------------------------------

def _process_question(
    item: dict,
    qa_specs: list[ModelSpec],
    qa_clients: dict[str, Any],
    judge_spec: Optional[ModelSpec],
    judge_client: Optional[Any],
    pdf_cache: PDFPageCache,
    doc_label: str,
    log_page_selection: bool = False,
) -> dict:
    """Run all QA models + judge for one question.

    Called from worker threads.  pdf_cache.page_texts is read-only (safe).
    Images are rendered on-demand per question via a fresh fitz handle.
    """
    question = item.get("query", "")
    q_id = item.get("Id", "")
    refs = item.get("refs", [])
    element_type = refs[0].get("element_type", "unknown") if refs else "unknown"
    assertions = item.get("assertions", [])
    primary_assertion = assertions[0]["text"] if assertions else ""

    answers: dict[str, str] = {}
    judgments: dict[str, dict] = {}
    page_selection_info: dict[str, Any] = {}

    # Shared page selection (same for all image-mode models — saves repeated BM25)
    # We compute once and reuse; text-mode models use their own slicing.
    selected_pages: Optional[list[int]] = None

    for spec in qa_specs:
        label = f"{doc_label}/{q_id}/{spec.name}"

        if spec.input_mode == "images":
            # Select relevant pages on first vision model; reuse for subsequent ones
            if selected_pages is None:
                selected_pages = select_relevant_pages(
                    page_texts=pdf_cache.page_texts,
                    question=question,
                    top_k=spec.top_k_pages,
                    window_before=spec.page_selection_window_before,
                    window_after=spec.page_selection_window_after,
                )
                if log_page_selection:
                    info = explain_page_selection(pdf_cache.page_texts, question)
                    page_selection_info[q_id] = info

            # Cap to this model's top_k (in case specs differ)
            pages_for_model = selected_pages[: spec.top_k_pages]
            logger.debug(
                "%s [%s] selected pages %s / %d total",
                spec.name, label, pages_for_model, pdf_cache.n_pages,
            )
            page_images = pdf_cache.render_images(pages_for_model)
            answer = run_qa(
                client=qa_clients[spec.name],
                spec=spec,
                page_images=page_images,
                pdf_text=None,
                question=question,
                label=label,
            )
        else:
            # Text mode: send extracted text from relevant pages (or full doc)
            if selected_pages is None:
                selected_pages = select_relevant_pages(
                    page_texts=pdf_cache.page_texts,
                    question=question,
                    top_k=spec.top_k_pages,
                    window_before=spec.page_selection_window_before,
                    window_after=spec.page_selection_window_after,
                )
            relevant_text = pdf_cache.full_text(page_indices=selected_pages)
            answer = run_qa(
                client=qa_clients[spec.name],
                spec=spec,
                page_images=None,
                pdf_text=relevant_text,
                question=question,
                label=label,
            )

        answers[spec.name] = answer or ""

        if judge_spec and judge_client and answer:
            judgment = judge_answer(
                client=judge_client,
                spec=judge_spec,
                question=question,
                gt_assertion=primary_assertion,
                predicted=answer,
                label=label,
            )
        else:
            reason = "no answer produced" if not answer else "judge disabled"
            judgment = {"result": "SKIPPED", "reason": reason}

        judgments[spec.name] = judgment

    result = {
        "id": q_id,
        "question": question,
        "element_type": element_type,
        "assertion": primary_assertion,
        "selected_pages": selected_pages or [],
        "answers": answers,
        "judgments": judgments,
    }
    if page_selection_info:
        result["page_selection_debug"] = page_selection_info
    return result


# ---------------------------------------------------------------------------
# Per-document orchestration
# ---------------------------------------------------------------------------

def infer_doc(
    config: str,
    doc_id: str,
    pdf_path: Path,
    quest_path: Path,
    cfg: InferenceConfig,
    qa_clients: dict[str, Any],
    judge_client: Optional[Any],
) -> dict:
    """Run inference + judge for every question in one document."""
    t0 = time.perf_counter()
    doc_label = f"{config}/{doc_id}"

    items = _load_quest(quest_path)
    if cfg.max_qa_per_doc > 0:
        items = items[: cfg.max_qa_per_doc]

    if not items:
        logger.warning("No quest items for %s", doc_label)
        return {
            "config": config, "doc_id": doc_id,
            "pdf_path": str(pdf_path), "n_pages": 0,
            "n_questions": 0, "qa_items": [], "elapsed_seconds": 0.0,
        }

    # Pre-load page texts once in the main thread (shared read-only with workers)
    pdf_cache = PDFPageCache(pdf_path)
    try:
        _ = pdf_cache.page_texts   # trigger loading now; catch errors early
        logger.info(
            "  %s — %d pages, %d questions",
            doc_label, pdf_cache.n_pages, len(items),
        )
    except Exception as exc:
        logger.error("Failed to load PDF %s: %s", pdf_path, exc)
        return {
            "config": config, "doc_id": doc_id,
            "pdf_path": str(pdf_path), "error": str(exc),
            "n_questions": 0, "qa_items": [], "elapsed_seconds": 0.0,
        }

    worker_kwargs = {
        "qa_specs": cfg.qa_models,
        "qa_clients": qa_clients,
        "judge_spec": cfg.judge_model,
        "judge_client": judge_client,
        "pdf_cache": pdf_cache,
        "doc_label": doc_label,
    }

    qa_results: list[dict] = []

    if cfg.max_workers <= 1:
        for item in items:
            qa_results.append(_process_question(item, **worker_kwargs))
    else:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            futures = {
                pool.submit(_process_question, item, **worker_kwargs): item
                for item in items
            }
            for future in as_completed(futures):
                try:
                    qa_results.append(future.result())
                except Exception as exc:
                    q_item = futures[future]
                    logger.error(
                        "Question %s failed in %s: %s",
                        q_item.get("Id", "?"), doc_label, exc,
                    )

    elapsed = round(time.perf_counter() - t0, 1)

    # Log pass-rate summary
    n = len(qa_results)
    pass_counts = {
        m: sum(1 for r in qa_results if r["judgments"].get(m, {}).get("result") == "PASS")
        for m in (s.name for s in cfg.qa_models)
    }
    pass_str = ", ".join(f"{m}={c}/{n}" for m, c in pass_counts.items())
    logger.info("  %s done in %.1fs  [%s]", doc_label, elapsed, pass_str)

    return {
        "config": config,
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "n_pages": pdf_cache.n_pages,
        "n_questions": n,
        "qa_items": qa_results,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(cfg: InferenceConfig, force: bool = False) -> None:
    docs = discover_docs(cfg.config_filter, cfg.dataset_dir)
    if not docs:
        logger.error("No documents found in %s", cfg.dataset_dir)
        sys.exit(1)

    logger.info(
        "Found %d documents | %d QA model(s) | judge: %s",
        len(docs), len(cfg.qa_models),
        cfg.judge_model.name if cfg.judge_model else "disabled",
    )

    # Build clients once (auth happens here)
    qa_clients: dict[str, Any] = {}
    for spec in cfg.qa_models:
        try:
            qa_clients[spec.name] = build_client(spec)
            logger.info(
                "  QA: %-30s  client=%-12s  input=%-6s  pages=top-%d  deploy=%s",
                spec.name, spec.client_type, spec.input_mode,
                spec.top_k_pages, spec.deployment,
            )
        except Exception as exc:
            logger.error("Could not build client for %s: %s", spec.name, exc)

    cfg.qa_models = [s for s in cfg.qa_models if s.name in qa_clients]
    if not cfg.qa_models:
        logger.error("No QA model clients could be initialized. Aborting.")
        sys.exit(1)

    judge_client: Optional[Any] = None
    if cfg.judge_model:
        try:
            judge_client = build_client(cfg.judge_model)
            logger.info("  Judge: %s", cfg.judge_model.name)
        except Exception as exc:
            logger.warning("Judge client failed: %s — judging disabled", exc)

    # Resume support
    per_doc_dir = cfg.results_dir / "per_doc"
    done_keys: set[str] = set()
    all_results: list[dict] = []

    if cfg.resume and not force and per_doc_dir.exists():
        for f in sorted(per_doc_dir.glob("*.json")):
            done_keys.add(f.stem)
            try:
                all_results.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        if done_keys:
            logger.info("Resume: %d done, %d remaining", len(done_keys), len(docs) - len(done_keys))

    docs_todo = [
        (c, d, pp, qp) for c, d, pp, qp in docs
        if f"{c}__{d}" not in done_keys
    ]

    for config, doc_id, pdf_path, quest_path in docs_todo:
        result = infer_doc(config, doc_id, pdf_path, quest_path, cfg, qa_clients, judge_client)
        all_results.append(result)
        save_per_doc(result, per_doc_dir)

    write_summary(all_results, cfg.results_dir, [s.name for s in cfg.qa_models])
    logger.info("Done. Results: %s", cfg.results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models-config", type=Path, required=True,
        help="YAML file defining QA models and judge model",
    )
    p.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Dataset root containing per-config pdfs/ and quest/ directories",
    )
    p.add_argument(
        "--config-filter", type=str, default=None,
        help="Glob pattern to restrict dataset configs, e.g. 'cfg_001*'",
    )
    p.add_argument(
        "--max-qa", type=int, default=0,
        help="Max questions per document (0 = all)",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Parallel question workers per document (default 4)",
    )
    p.add_argument(
        "--results-dir", type=Path, default=INFERENCE_RESULTS_DIR,
        help="Override output directory",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run all docs even if results already exist",
    )
    p.add_argument(
        "--no-judge", action="store_true",
        help="Skip judging; collect QA model answers only",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.models_config.exists():
        logger.error("Models config not found: %s", args.models_config)
        sys.exit(1)

    cfg = load_models_config(args.models_config)
    cfg.dataset_dir = args.dataset_dir
    cfg.results_dir = args.results_dir
    cfg.max_qa_per_doc = args.max_qa
    cfg.max_workers = args.workers
    cfg.config_filter = args.config_filter

    if args.no_judge:
        cfg.judge_model = None

    run(cfg, force=args.force)


if __name__ == "__main__":
    main()
