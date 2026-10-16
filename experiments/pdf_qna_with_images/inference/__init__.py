# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""PDF-native inference + judge evaluation pipeline.

Sends raw PDF files to configurable QA models, then judges the
predicted answers against ground-truth assertions.

Quick start
-----------
  # 1. Fill in your Azure AI Foundry endpoints:
  #    cp examples/inference_models.yaml my_models.yaml && edit my_models.yaml

  # 2. Run:
  py -3 -m synthpdf.inference.run_inference --models-config my_models.yaml

  # 3. Limit to a subset of questions for a quick smoke-test:
  py -3 -m synthpdf.inference.run_inference --models-config my_models.yaml \\
      --config-filter "cfg_001*" --max-qa 5
"""
