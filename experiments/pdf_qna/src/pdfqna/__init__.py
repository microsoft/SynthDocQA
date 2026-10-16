# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
pdfqna - PDF QnA evaluation pipeline.

Given a markdown representation of a PDF and a set of questions (with ground
truth answers and assertions), runs multiple QA models and judges their answers.

Quick start:
    python -m pdfqna --config models.yaml
    python -m pdfqna --config models.yaml --config-filter "cfg_001*" --max-docs 5
"""
