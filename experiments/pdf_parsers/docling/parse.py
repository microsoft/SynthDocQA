# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
Docling PDF parser - single file or batch directory processing.

Usage:
    # Single PDF
    python parse.py --input path/to/doc.pdf

    # Batch directory (all PDFs inside)
    python parse.py --input path/to/pdf_dir/

    # With explicit output and options
    python parse.py --input doc.pdf --output out/ --device cpu --threads 4

Output per document (written to <output>/<doc_stem>/):
    <doc_stem>.md              clean markdown
    <doc_stem>.json            full docling document JSON (lossless)

Backend:
    Docling uses its own StandardPdfPipeline which runs layout analysis,
    table structure recognition, and OCR via embedded models (downloaded
    on first run from HuggingFace).

Device choices (--device):
    auto   detect best available (default)
    cpu    force CPU only
    cuda   NVIDIA GPU
"""

import argparse
import json
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
    device: str,
    num_threads: int,
    force_ocr: bool = False,
    no_ocr: bool = False,
) -> None:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import OcrAutoOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = AcceleratorOptions(device=device, num_threads=num_threads)
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.accelerator_options = accelerator
    if no_ocr:
        pipeline_opts.do_ocr = False
        ocr_label = "off"
    else:
        pipeline_opts.do_ocr = True
        pipeline_opts.ocr_options = OcrAutoOptions(force_full_page_ocr=force_ocr)
        ocr_label = "force" if force_ocr else "auto"
    # generate page images for figures; disable if memory is tight
    pipeline_opts.generate_page_images = False
    pipeline_opts.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )

    print(f"[docling] device={device}  threads={num_threads}  ocr={ocr_label}")
    print(f"[docling] processing {len(pdf_paths)} document(s) -> {output_dir}")

    t0 = time.perf_counter()
    successes, failures = 0, 0

    for result in converter.convert_all(pdf_paths, raises_on_error=False):
        stem = Path(result.input.file).stem
        doc_dir = output_dir / stem
        doc_dir.mkdir(parents=True, exist_ok=True)

        if result.status == ConversionStatus.SUCCESS:
            # Markdown
            md_path = doc_dir / f"{stem}.md"
            md_path.write_text(
                result.document.export_to_markdown(), encoding="utf-8"
            )
            # JSON
            json_path = doc_dir / f"{stem}.json"
            json_path.write_text(
                json.dumps(result.document.export_to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  [ok] {stem}")
            successes += 1
        else:
            errors = "; ".join(e.error_message for e in result.errors) if result.errors else "unknown"
            print(f"  [fail] {stem}: {errors}", file=sys.stderr)
            failures += 1

    elapsed = time.perf_counter() - t0
    print(f"[docling] done in {elapsed:.1f}s  ({successes} ok, {failures} failed)")
    print(f"[docling] output written to: {output_dir.resolve()}")


def main() -> None:
    default_output = Path(__file__).parent / "output"

    parser = argparse.ArgumentParser(
        description="Docling PDF parser — single file or batch directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to a single PDF file or a directory containing PDF files.",
    )
    parser.add_argument(
        "--output", "-o",
        default=default_output,
        type=Path,
        help=f"Output directory. Sub-folders are created per document. (default: {default_output})",
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device: auto (default), cpu, cuda, mps.",
    )
    parser.add_argument(
        "--threads", "-t",
        default=4,
        type=int,
        help="Number of CPU threads for model inference (default: 4).",
    )
    parser.add_argument(
        "--force-ocr", action="store_true",
        help="Force full-page OCR on every page (even vector PDFs).",
    )
    parser.add_argument(
        "--no-ocr", action="store_true",
        help="Disable OCR entirely.",
    )

    args = parser.parse_args()
    pdf_paths = collect_pdfs(args.input)
    parse_pdfs(
        pdf_paths=pdf_paths,
        output_dir=args.output,
        device=args.device,
        num_threads=args.threads,
        force_ocr=args.force_ocr,
        no_ocr=args.no_ocr,
    )


if __name__ == "__main__":
    main()
