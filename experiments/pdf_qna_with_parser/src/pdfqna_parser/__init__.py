# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""QnA inference over markdown produced by PDF parsers."""

from .run_inference import discover_docs, infer_doc, main

__all__ = ["discover_docs", "infer_doc", "main"]