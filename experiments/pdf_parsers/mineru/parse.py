# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
MinerU PDF parser - single file or batch directory processing.

Usage:
    # Single PDF
    python parse.py --input path/to/doc.pdf --output path/to/output

    # Batch directory (all PDFs inside)
    python parse.py --input path/to/pdf_dir/ --output path/to/output

    # With explicit options
    python parse.py --input doc.pdf --output out/ --backend pipeline --method auto --lang en

Output structure per document:
    <output>/<doc_stem>/auto/
        <doc_stem>.md
        <doc_stem>_content_list.json
        <doc_stem>_content_list_v2.json
        <doc_stem>_middle.json
        images/

Backend choices:
    pipeline           CPU-compatible, good general accuracy (~85%)
    vlm-auto-engine    High accuracy, requires local GPU
    hybrid-auto-engine Highest accuracy, requires local GPU + torch
"""

import argparse
import sys
import time
from pathlib import Path


def collect_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print(f"[error] Not a PDF file: {input_path}", file=sys.stderr)
            sys.exit(1)
        return [input_path]
    elif input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            print(f"[error] No PDF files found in: {input_path}", file=sys.stderr)
            sys.exit(1)
        return pdfs
    else:
        print(f"[error] Path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)


def parse_pdfs(
    pdf_paths: list[Path],
    output_dir: Path,
    backend: str,
    method: str,
    lang: str,
    formula: bool,
    table: bool,
) -> None:
    from mineru.cli.common import do_parse, read_fn

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_file_names = [p.stem for p in pdf_paths]
    pdf_bytes_list = [read_fn(p) for p in pdf_paths]
    lang_list = [lang] * len(pdf_paths)

    print(f"[mineru] backend={backend}  method={method}  lang={lang}")
    print(f"[mineru] processing {len(pdf_paths)} document(s) -> {output_dir}")

    t0 = time.perf_counter()
    do_parse(
        output_dir=str(output_dir),
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        p_lang_list=lang_list,
        backend=backend,
        parse_method=method,
        formula_enable=formula,
        table_enable=table,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_orig_pdf=False,
        f_dump_md=True,
        f_dump_content_list=True,
        f_dump_middle_json=True,
        f_dump_model_output=False,
    )
    elapsed = time.perf_counter() - t0

    print(f"[mineru] done in {elapsed:.1f}s")
    print(f"[mineru] output written to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MinerU PDF parser — single file or batch directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to a single PDF file or a directory containing PDF files.",
    )
    default_output = Path(__file__).parent / "output"
    parser.add_argument(
        "--output", "-o",
        default=default_output,
        type=Path,
        help=f"Output directory. Sub-folders are created per document. (default: {default_output})",
    )
    parser.add_argument(
        "--backend", "-b",
        default="pipeline",
        choices=["pipeline", "vlm-auto-engine", "vlm-http-client",
                 "hybrid-auto-engine", "hybrid-http-client"],
        help="Parsing backend. Use 'pipeline' for CPU-only (default).",
    )
    parser.add_argument(
        "--method", "-m",
        default="auto",
        choices=["auto", "txt", "ocr"],
        help="Parsing method: auto (default), txt (text-based), ocr (image-based).",
    )
    parser.add_argument(
        "--lang", "-l",
        default="en",
        choices=["ch", "ch_server", "ch_lite", "en", "korean", "japan",
                 "chinese_cht", "ta", "te", "ka", "th", "el", "latin",
                 "arabic", "east_slavic", "cyrillic", "devanagari"],
        help="Primary document language for OCR (default: en).",
    )
    parser.add_argument(
        "--no-formula",
        action="store_true",
        help="Disable formula (LaTeX) parsing.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Disable table parsing.",
    )

    args = parser.parse_args()

    pdf_paths = collect_pdfs(args.input)
    parse_pdfs(
        pdf_paths=pdf_paths,
        output_dir=args.output,
        backend=args.backend,
        method=args.method,
        lang=args.lang,
        formula=not args.no_formula,
        table=not args.no_table,
    )


if __name__ == "__main__":
    main()
