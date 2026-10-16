# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Variant generator sub-package."""
from synthpdf.variants.tables import (
    generate_table_variants,
    paginate_table,
    generate_multipage_table_variants,
    generate_llm_multipage_table_variants,
)
from synthpdf.variants.complex_tables import generate_complex_table_variants
from synthpdf.variants.images import generate_image_variants
from synthpdf.variants.text_annotations import generate_text_variants, generate_annotation_variants
from synthpdf.variants.forms import (
    generate_checklist_variants,
    generate_button_variants,
    generate_form_variants,
    generate_fillable_form_variants,
    generate_form_pool,
)
from synthpdf.variants.scanned import (
    generate_scanned_text_variants,
    generate_scanned_table_variants,
    generate_handwriting_variants,
    generate_stamp_mark_variants,
    generate_scan_pool,
)
from synthpdf.variants.infographics import (
    generate_flow_variants,
    generate_org_chart_variants,
    generate_plotly_variants,
    generate_dashboard_variants,
    generate_floor_plan_variants,
    generate_infographic_pool,
    generate_stat_dashboard_variants,
    generate_timeline_variants,
    generate_comparison_variants,
    generate_process_step_variants,
    generate_hub_spoke_variants,
    generate_magazine_variants,
    generate_pictograph_variants,
)
from synthpdf.variants.annotations import (
    generate_highlight_variants,
    generate_callout_variants,
    generate_label_variants,
    generate_sticky_note_variants,
    generate_freetext_variants,
    generate_review_variants,
    generate_annotated_report_variants,
    generate_annotation_pool,
)

__all__ = [
    "generate_table_variants",
    "paginate_table",
    "generate_multipage_table_variants",
    "generate_llm_multipage_table_variants",
    "generate_image_variants",
    "generate_text_variants",
    "generate_annotation_variants",
    "generate_checklist_variants",
    "generate_button_variants",
    "generate_form_variants",
    "generate_fillable_form_variants",
    "generate_form_pool",
    "generate_scanned_text_variants",
    "generate_scanned_table_variants",
    "generate_handwriting_variants",
    "generate_stamp_mark_variants",
    "generate_scan_pool",
    "generate_flow_variants",
    "generate_org_chart_variants",
    "generate_plotly_variants",
    "generate_dashboard_variants",
    "generate_floor_plan_variants",
    "generate_infographic_pool",
    "generate_stat_dashboard_variants",
    "generate_timeline_variants",
    "generate_comparison_variants",
    "generate_process_step_variants",
    "generate_hub_spoke_variants",
    "generate_magazine_variants",
    "generate_pictograph_variants",
    "generate_highlight_variants",
    "generate_callout_variants",
    "generate_label_variants",
    "generate_sticky_note_variants",
    "generate_freetext_variants",
    "generate_review_variants",
    "generate_annotated_report_variants",
    "generate_annotation_pool",
    "generate_complex_table_variants",
]
