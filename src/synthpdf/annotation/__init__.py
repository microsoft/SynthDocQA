# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Inline annotation package — post-content PDF annotation pass."""
from synthpdf.annotation.text_picker import AnnotationSpec, pick_annotation_specs
from synthpdf.annotation.pdf_layer import apply_inline_annotations

__all__ = ["AnnotationSpec", "pick_annotation_specs", "apply_inline_annotations"]
