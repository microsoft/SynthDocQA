# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
run_all.py -- Parse every PDF in a dataset directory with Docling, then
              evaluate each config against its ground-truth manifests.

Directory layout expected:
    <dataset-dir>/
        cfg_001_tables_basic_only/
            pdfs/          *.pdf
            manifests/     *.json
        cfg_002_.../
            pdfs/
            manifests/
        ...

Steps:
    1. Collect all PDFs under <dataset-dir>/*/pdfs/
    2. Skip any doc_id already present in <output-dir> (unless --reparse)
    3. Parse remaining PDFs with Docling
    4. For each config that has both manifests/ and at least one parsed doc,
       run the evaluator and write <results-dir>/<cfg_name>_eval.json
    5. Aggregate all per-config results into <results-dir>/_aggregate.json
       and print a combined summary table

Usage:
    python run_all.py \\
        --dataset-dir  Q:/path/to/variant_configs_v2/ \\
        --output-dir   pdf_parsers/docling/output/ \\
        --results-dir  pdf_parsers/docling/results/ \\
        [--device cpu] [--threads 4] [--reparse] [--eval-only]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_configs(dataset_dir: Path) -> list[Path]:
    """Return all cfg_* subdirectories that contain both pdfs/ and manifests/."""
    cfgs = sorted(
        d for d in dataset_dir.iterdir()
        if d.is_dir()
        and (d / "pdfs").is_dir()
        and (d / "manifests").is_dir()
    )
    return cfgs


def collect_pdfs_for_config(cfg_dir: Path) -> list[Path]:
    return sorted((cfg_dir / "pdfs").glob("*.pdf"))


def already_parsed(pdf_path: Path, output_dir: Path) -> bool:
    doc_id = pdf_path.stem
    return (output_dir / doc_id / f"{doc_id}.json").exists()


# ---------------------------------------------------------------------------
# Parse step
# ---------------------------------------------------------------------------

def parse_all(
    configs: list[Path],
    output_dir: Path,
    device: str,
    num_threads: int,
    reparse: bool,
) -> int:
    """Parse PDFs across all configs. Returns number of docs parsed."""
    from parse import parse_pdfs  # local parse.py

    to_parse = []
    skipped  = 0
    for cfg in configs:
        for pdf in collect_pdfs_for_config(cfg):
            if not reparse and already_parsed(pdf, output_dir):
                skipped += 1
            else:
                to_parse.append(pdf)

    if skipped:
        print(f"[parse] {skipped} document(s) already parsed — skipping (use --reparse to override)")

    if not to_parse:
        print("[parse] nothing to parse")
        return 0

    print(f"[parse] {len(to_parse)} document(s) to parse")
    parse_pdfs(
        pdf_paths=to_parse,
        output_dir=output_dir,
        device=device,
        num_threads=num_threads,
        force_ocr=False,
        no_ocr=False,
    )
    return len(to_parse)


# ---------------------------------------------------------------------------
# Evaluate step
# ---------------------------------------------------------------------------

def evaluate_all(
    configs: list[Path],
    output_dir: Path,
    results_dir: Path,
) -> list[dict]:
    """Evaluate every config that has at least one parsed document."""
    from evaluate import (  # local evaluate.py
        load_gt, load_docling, eval_document, aggregate,
        print_summary, _to_grouped,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    all_doc_results: list[dict] = []
    cfg_summaries:   list[dict] = []

    for cfg in configs:
        cfg_name     = cfg.name
        manifest_dir = cfg / "manifests"
        manifests    = sorted(manifest_dir.glob("*.json"))

        cfg_docs = []
        for mf in manifests:
            doc_id    = mf.stem
            pred_json = output_dir / doc_id / f"{doc_id}.json"
            if not pred_json.exists():
                continue

            manifest = load_gt(mf)
            doc      = load_docling(pred_json)
            res      = eval_document(manifest, doc)
            cfg_docs.append(res)
            all_doc_results.append(res)

        if not cfg_docs:
            continue

        # Per-config JSON
        cfg_agg = aggregate(cfg_docs)
        cfg_report = {
            "config":              cfg_name,
            "documents_evaluated": len(cfg_docs),
            "aggregate":           _to_grouped(cfg_agg),
            "per_document":        [_to_grouped(r) for r in cfg_docs],
        }
        out_path = results_dir / f"{cfg_name}_eval.json"
        out_path.write_text(
            json.dumps(cfg_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        cfg_summaries.append({"config": cfg_name, **cfg_agg})
        print(
            f"  [eval] {cfg_name:<55}  "
            f"docs={len(cfg_docs)}  "
            f"tbl_f1={cfg_agg.get('table_f1', 0):.3f}  "
            f"img_f1={cfg_agg.get('image_f1') or 0:.3f}"
        )

    return all_doc_results


# ---------------------------------------------------------------------------
# Aggregate summary across all configs
# ---------------------------------------------------------------------------

def write_aggregate(
    all_doc_results: list[dict],
    results_dir: Path,
    n_configs: int,
) -> None:
    from evaluate import aggregate, print_summary, _to_grouped

    if not all_doc_results:
        print("[aggregate] no results to aggregate")
        return

    agg = aggregate(all_doc_results)
    full_report = {
        "parser":              "docling",
        "configs_evaluated":   n_configs,
        "documents_evaluated": len(all_doc_results),
        "aggregate":           _to_grouped(agg),
    }
    agg_path = results_dir / "_aggregate.json"
    agg_path.write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[aggregate written to {agg_path}]")
    print_summary(agg, len(all_doc_results), Path("(all configs)"), results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse all dataset PDFs with Docling then evaluate all configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir", required=True, type=Path,
        help="Root dataset directory containing cfg_*/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Docling parse output directory (one sub-folder per doc_id).",
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Where to write per-config and aggregate evaluation JSON files.",
    )
    parser.add_argument(
        "--device", "-d", default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device for Docling (default: auto).",
    )
    parser.add_argument(
        "--threads", "-t", default=4, type=int,
        help="CPU threads for Docling model inference (default: 4).",
    )
    parser.add_argument(
        "--reparse", action="store_true",
        help="Re-parse documents even if output already exists.",
    )
    parser.add_argument(
        "--parse-only", action="store_true",
        help="Parse documents only; skip evaluation.",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip parsing; only run evaluation on existing output.",
    )
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        print(f"[error] dataset-dir not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    configs = collect_configs(args.dataset_dir)
    if not configs:
        print(f"[error] no cfg_* directories found in {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[run_all] dataset : {args.dataset_dir}")
    print(f"[run_all] output  : {args.output_dir}")
    print(f"[run_all] results : {args.results_dir}")
    print(f"[run_all] configs : {len(configs)}")
    print()

    t0 = time.perf_counter()

    # -- parse ----------------------------------------------------------------
    if not args.eval_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print("=== STEP 1: PARSE ===")
        parse_all(
            configs=configs,
            output_dir=args.output_dir,
            device=args.device,
            num_threads=args.threads,
            reparse=args.reparse,
        )
        print()

    if args.parse_only:
        elapsed = time.perf_counter() - t0
        print(f"\n[run_all] parse finished in {elapsed:.1f}s")
        return

    # -- evaluate -------------------------------------------------------------
    print("=== STEP 2: EVALUATE ===")
    all_doc_results = evaluate_all(
        configs=configs,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
    )
    print()

    # -- aggregate ------------------------------------------------------------
    n_evaluated = sum(
        1 for cfg in configs
        if any(
            (args.output_dir / mf.stem / f"{mf.stem}.json").exists()
            for mf in (cfg / "manifests").glob("*.json")
        )
    )
    write_aggregate(all_doc_results, args.results_dir, n_evaluated)

    elapsed = time.perf_counter() - t0
    print(f"\n[run_all] finished in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
