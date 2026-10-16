# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Run parser-result judge evaluation without installing the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdfqna_parser.run_judges import main

main()