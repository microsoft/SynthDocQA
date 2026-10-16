# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Core data models for the synthpdf package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Variant:
    """A single generated content variant (table, image, text, or annotation)."""

    id: str
    kind: Literal["table", "image", "text", "annotation"]
    metadata: dict[str, Any]
    render_payload: dict[str, Any]
    ground_truth: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "metadata": self.metadata,
            "render_payload": self.render_payload,
            "ground_truth": self.ground_truth,
        }


@dataclass
class SectionSpec:
    """Specifies one section within a document."""

    section_type: Literal["table", "image", "text"]
    variant_ids: list[str]
    layout: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_type": self.section_type,
            "variant_ids": self.variant_ids,
            "layout": self.layout,
        }


@dataclass
class DocumentSpec:
    """Full specification for one synthetic document."""

    doc_id: str
    seed: int
    recipe_name: str
    sections: list[SectionSpec]
    title: str = ""
    subtitle: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "seed": self.seed,
            "recipe_name": self.recipe_name,
            "title": self.title,
            "subtitle": self.subtitle,
            "sections": [s.to_dict() for s in self.sections],
        }
