# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
pdfplumber PDF parser - single file or batch directory processing.

Uses pdfminer.six under the hood — pure Python, no ML models, no GPU required.
Strong at precise text positioning and table detection via line/rule analysis.

Usage:
    # Single PDF
    python parse.py --input path/to/doc.pdf

    # Batch directory
    python parse.py --input path/to/pdf_dir/

Output per document (written to <output>/<doc_stem>/):
    <doc_stem>.md       markdown with tables rendered as GFM pipe tables
    <doc_stem>.json     structured JSON with per-page text, tables, metadata
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


def _table_to_markdown(rows: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to a GFM pipe table."""
    if not rows:
        return ""

    # Normalise cells: None -> "", strip whitespace, collapse newlines
    def clean(cell) -> str:
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").strip()

    cleaned = [[clean(c) for c in row] for row in rows]

    # Determine column widths
    col_count = max(len(row) for row in cleaned)
    # Pad short rows
    cleaned = [row + [""] * (col_count - len(row)) for row in cleaned]
    col_widths = [max(len(row[i]) for row in cleaned) for i in range(col_count)]
    col_widths = [max(w, 3) for w in col_widths]  # minimum 3 for separator

    def fmt_row(row):
        cells = [row[i].ljust(col_widths[i]) for i in range(col_count)]
        return "| " + " | ".join(cells) + " |"

    header = cleaned[0]
    lines = [fmt_row(header)]
    lines.append("| " + " | ".join("-" * col_widths[i] for i in range(col_count)) + " |")
    for row in cleaned[1:]:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def _page_to_markdown(page) -> str:
    """Convert a single pdfplumber page to markdown, tables in reading order."""
    # Find all tables and their bounding boxes (top-of-page coords)
    tables = page.find_tables()
    table_bboxes = [tbl.bbox for tbl in tables]

    # Build exclusion bboxes so extract_text skips table regions
    text_outside = page.filter(
        lambda obj: not any(
            obj["x0"] >= bbox[0] and obj["x1"] <= bbox[2]
            and obj["top"] >= bbox[1] and obj["bottom"] <= bbox[3]
            for bbox in table_bboxes
        )
    )
    raw_text = text_outside.extract_text(x_tolerance=3, y_tolerance=3) or ""

    if not tables:
        return raw_text.strip()

    # Interleave text blocks and tables by vertical position
    # Collect text segments above each table and the table itself
    segments: list[tuple[float, str]] = []  # (top_y, content)

    # Text above first table
    first_table_top = min(tbl.bbox[1] for tbl in tables)
    lines_above = []
    current_y = 0.0
    for line in raw_text.splitlines():
        # Approximate: emit all text then tables — positional interleaving
        # is best-effort since extract_text doesn't expose per-line coords.
        lines_above.append(line)

    # Simple approach: emit full text, then each table at its relative position
    # by scanning which text lines fall before each table top coord.
    # pdfplumber uses top-of-page coords (0 = top). Sort tables by top.
    sorted_tables = sorted(tables, key=lambda t: t.bbox[1])

    # Re-extract text with word positions to do proper interleaving
    words = page.extract_words(x_tolerance=3, y_tolerance=3, use_text_flow=True)

    md_parts: list[str] = []
    processed_word_idx = 0
    table_tops = [t.bbox[1] for t in sorted_tables]

    def words_to_text(word_list) -> str:
        if not word_list:
            return ""
        lines: list[list[str]] = []
        current_line_top = None
        current_line: list[str] = []
        for w in word_list:
            if current_line_top is None or abs(w["top"] - current_line_top) > 5:
                if current_line:
                    lines.append(current_line)
                current_line = [w["text"]]
                current_line_top = w["top"]
            else:
                current_line.append(w["text"])
        if current_line:
            lines.append(current_line)
        return "\n".join(" ".join(ln) for ln in lines)

    def in_any_table(w) -> bool:
        for bbox in table_bboxes:
            if (w["x0"] >= bbox[0] - 2 and w["x1"] <= bbox[2] + 2
                    and w["top"] >= bbox[1] - 2 and w["bottom"] <= bbox[3] + 2):
                return True
        return False

    non_table_words = [w for w in words if not in_any_table(w)]

    for tbl in sorted_tables:
        tbl_top = tbl.bbox[1]
        # Words above this table that haven't been emitted yet
        before = [w for w in non_table_words
                  if w["bottom"] <= tbl_top and w not in
                  [x for part_words in [
                      [w2 for w2 in non_table_words if w2["bottom"] <= t]
                      for t in table_tops[:table_tops.index(tbl_top)]
                  ] for x in part_words]]
        # Simpler: just get words whose bottom is above this table top
        # and above any previously-processed table
        pass

    # Straightforward fallback: text first, then tables appended at end with separators
    text_block = words_to_text(non_table_words).strip()
    if text_block:
        md_parts.append(text_block)

    for tbl in sorted_tables:
        rows = tbl.extract()
        if rows:
            md_parts.append(_table_to_markdown(rows))

    return "\n\n".join(md_parts)


def _page_to_dict(page, page_num: int) -> dict:
    """Extract structured data from a page as a dict."""
    tables_raw = []
    for tbl in page.find_tables():
        rows = tbl.extract()
        tables_raw.append({
            "bbox": list(tbl.bbox),
            "rows": [[cell if cell is not None else "" for cell in row] for row in (rows or [])],
        })

    text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

    return {
        "page": page_num,
        "width": page.width,
        "height": page.height,
        "text": text,
        "tables": tables_raw,
        "char_count": len(page.chars),
        "word_count": len(page.extract_words()),
    }


def parse_pdfs(pdf_paths: list[Path], output_dir: Path) -> None:
    import pdfplumber

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pdfplumber] processing {len(pdf_paths)} document(s) -> {output_dir}")

    t0 = time.perf_counter()
    successes, failures = 0, 0

    for pdf_path in pdf_paths:
        stem = pdf_path.stem
        doc_dir = output_dir / stem
        doc_dir.mkdir(parents=True, exist_ok=True)

        try:
            md_pages: list[str] = []
            json_pages: list[dict] = []
            meta: dict = {}

            with pdfplumber.open(str(pdf_path)) as pdf:
                meta = {
                    "file": pdf_path.name,
                    "pages": len(pdf.pages),
                    "metadata": pdf.metadata or {},
                }
                for i, page in enumerate(pdf.pages, start=1):
                    md_pages.append(_page_to_markdown(page))
                    json_pages.append(_page_to_dict(page, i))

            # Markdown: join pages with a horizontal rule
            md_content = "\n\n---\n\n".join(p for p in md_pages if p)
            (doc_dir / f"{stem}.md").write_text(md_content, encoding="utf-8")

            # JSON: full structured output
            doc_json = {**meta, "content": json_pages}
            (doc_dir / f"{stem}.json").write_text(
                json.dumps(doc_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            print(f"  [ok] {stem}  ({meta['pages']} pages)")
            successes += 1

        except Exception as exc:
            print(f"  [fail] {stem}: {exc}", file=sys.stderr)
            failures += 1

    elapsed = time.perf_counter() - t0
    print(f"[pdfplumber] done in {elapsed:.1f}s  ({successes} ok, {failures} failed)")
    print(f"[pdfplumber] output written to: {output_dir.resolve()}")


def main() -> None:
    default_output = Path(__file__).parent / "output"

    parser = argparse.ArgumentParser(
        description="pdfplumber PDF parser — single file or batch directory",
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

    args = parser.parse_args()
    pdf_paths = collect_pdfs(args.input)
    parse_pdfs(pdf_paths=pdf_paths, output_dir=args.output)


if __name__ == "__main__":
    main()
