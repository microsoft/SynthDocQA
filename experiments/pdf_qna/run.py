# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
Run the PDF QnA evaluation pipeline without installing the package.

Usage:
    python run.py
    python run.py --config-filter "cfg_001*" --max-docs 1 --max-qa 2
    python run.py --markdown-source mineru --config-filter "cfg_00[12]*"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdfqna.run_eval import main

main()
