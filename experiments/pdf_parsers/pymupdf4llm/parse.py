# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
pymupdf4llm PDF parser - single file or batch directory processing.

Usage:
    # Single PDF
    python parse.py --input path/to/doc.pdf

    # Batch directory (all PDFs inside)
    python parse.py --input path/to/pdf_dir/

    # With explicit output and OCR options
    python parse.py --input doc.pdf --output out/ --dpi 150 --force-ocr

Output per document (written to <output>/<doc_stem>/):
    <doc_stem>.md              clean markdown (all pages joined)
    <doc_stem>_pages.json      page chunks with structure metadata

Backend:
    Uses pymupdf4llm.to_markdown() with page_chunks=True.
    The layout backend (document_layout.py) is active by default when
    pymupdf.layout is importable; otherwise falls back to the legacy
    pymupdf_rag backend.  Both produce compatible _pages.json output
    that evaluate.py handles automatically.

    Layout backend page_chunks structure per page:
        metadata, toc_items, page_boxes [{index, class, bbox, pos}], text

    Legacy (rag) backend page_chunks structure per page:
        metadata, toc_items, tables, images, graphics, words, text

    evaluate.py detects the active backend via the "page_boxes" key.
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


def _safe_json(obj):
    """
    Recursively convert pymupdf4llm page-chunk objects to JSON-serializable
    Python types.

    Handles:
      - PyMuPDF geometry objects (Rect, IRect, Point, Matrix) via list()
      - bytes / binary image data  → dropped entirely
      - dicts that contain a raw "image" key → that key is skipped
      - everything else            → passed through unchanged
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, bytes):
        return None
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items() if k != "image"}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    # PyMuPDF geometry types (Rect, IRect, Point, Matrix, Color, …)
    try:
        return list(obj)
    except TypeError:
        return repr(obj)


def parse_pdfs(
    pdf_paths: list[Path],
    output_dir: Path,
    dpi: int,
    use_ocr: bool,
    force_ocr: bool,
) -> None:
    import pymupdf4llm

    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_label = "force" if force_ocr else ("yes" if use_ocr else "no")
    print(f"[pymupdf4llm] dpi={dpi}  ocr={ocr_label}")
    print(f"[pymupdf4llm] processing {len(pdf_paths)} document(s) -> {output_dir}")

    t0 = time.perf_counter()
    successes, failures = 0, 0

    for pdf_path in pdf_paths:
        stem = pdf_path.stem
        doc_dir = output_dir / stem
        doc_dir.mkdir(parents=True, exist_ok=True)

        try:
            chunks = pymupdf4llm.to_markdown(
                str(pdf_path),
                page_chunks=True,
                write_images=False,   # no image files on disk; metadata is kept
                dpi=dpi,
                use_ocr=use_ocr,
                force_ocr=force_ocr,
                show_progress=False,
            )

            # Markdown: join all page texts
            md_text = "\n\n".join(c.get("text", "") for c in chunks)
            (doc_dir / f"{stem}.md").write_text(md_text, encoding="utf-8")

            # Pages JSON: structured metadata for evaluate.py
            safe_chunks = _safe_json(chunks)
            (doc_dir / f"{stem}_pages.json").write_text(
                json.dumps(safe_chunks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"  [ok] {stem}  ({len(chunks)} page(s))")
            successes += 1

        except Exception as exc:
            print(f"  [fail] {stem}: {exc}", file=sys.stderr)
            failures += 1

    elapsed = time.perf_counter() - t0
    print(f"[pymupdf4llm] done in {elapsed:.1f}s  ({successes} ok, {failures} failed)")
    print(f"[pymupdf4llm] output written to: {output_dir.resolve()}")


def main() -> None:
    default_output = Path(__file__).parent / "output"

    parser = argparse.ArgumentParser(
        description="pymupdf4llm PDF parser — single file or batch directory",
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
        "--dpi",
        default=150,
        type=int,
        help="Rendering DPI (default: 150).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback entirely (text-layer only).",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on every page even if a text layer exists.",
    )

    args = parser.parse_args()

    if args.force_ocr and args.no_ocr:
        print("[error] --force-ocr and --no-ocr are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    pdf_paths = collect_pdfs(args.input)
    parse_pdfs(
        pdf_paths=pdf_paths,
        output_dir=args.output,
        dpi=args.dpi,
        use_ocr=not args.no_ocr,
        force_ocr=args.force_ocr,
    )


if __name__ == "__main__":
    main()
