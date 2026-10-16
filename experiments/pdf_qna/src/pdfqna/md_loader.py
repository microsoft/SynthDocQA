# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
Markdown loader - reads and locates markdown files for a given document.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_markdown(path: Path) -> str:
    """Read a markdown file and return its full text content."""
    return path.read_text(encoding="utf-8", errors="replace")


def find_markdown(
    base_dir: Path,
    config_name: str,
    doc_id: str,
    source: str = "dataset",
) -> Optional[Path]:
    """
    Locate the markdown file for a given doc.

    source="dataset"  -> <base_dir>/<config_name>/markdown/<doc_id>.md
    source="<parser>" -> <base_dir>/<config_name>/<parser>/<doc_id>.md
                         falls back to the dataset markdown if not found.
    """
    if source == "dataset":
        candidate = base_dir / config_name / "markdown" / f"{doc_id}.md"
    else:
        candidate = base_dir / config_name / source / f"{doc_id}.md"
        if not candidate.exists():
            logger.warning(
                "Markdown not found for source=%r at %s; falling back to dataset markdown",
                source,
                candidate,
            )
            candidate = base_dir / config_name / "markdown" / f"{doc_id}.md"

    if not candidate.exists():
        return None
    return candidate
