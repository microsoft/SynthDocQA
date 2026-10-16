# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Jinja2 HTML template engine for synthetic PDF generation.

Pipeline
--------
1. TemplateIndex   — loads 100 spec JSONs; selects templates by page_role / industry.
2. extract_variables — discovers all Jinja2 variables in an HTML template.
3. fill_template_variables — generates synthetic content for every variable.
4. build_mixed_document — orchestrates multi-page document assembly.
"""
from synthpdf.template_engine.index import TemplateIndex, TemplateEntry
from synthpdf.template_engine.extractor import extract_variables
from synthpdf.template_engine.adapters import fill_template_variables
from synthpdf.template_engine.doc_builder import build_mixed_document

__all__ = [
    "TemplateIndex",
    "TemplateEntry",
    "extract_variables",
    "fill_template_variables",
    "build_mixed_document",
]
