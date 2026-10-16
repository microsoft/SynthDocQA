# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Public entry point for unconditional batch generation."""
from __future__ import annotations

from pathlib import Path

import click


@click.group()
def cli() -> None:
    """Generate synthetic PDFs and question-answer artifacts."""


@cli.command("generate")
@click.option("--seed", type=click.IntRange(0, 2**31 - 1), default=None)
@click.option("--num-documents", "-n", type=click.IntRange(min=1), default=None)
@click.option("--pages", "-p", type=click.IntRange(min=1), default=None)
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None)
def generate(seed: int | None, num_documents: int | None,
             pages: int | None, output_dir: Path | None) -> None:
    """Sample a random document configuration and run the LLM-backed pipeline."""
    from synthpdf.document_generator import generate_documents

    try:
        result = generate_documents(
            num_pages=pages, num_pdfs=num_documents, output_dir=output_dir, seed=seed
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Output: {result['output_dir']}")
    click.echo(f"Documents: {len(result['documents'])}")
