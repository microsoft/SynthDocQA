# Local PDF parsers

Implemented execution routes: [Docling](docling/README.md),
[MinerU](mineru/README.md), [pdfplumber](pdfplumber/README.md), and
[pymupdf4llm](pymupdf4llm/README.md). Each retains `parse.py`, `evaluate.py`,
and `run_all.py`. Comparison, plotting, paper scripts and outputs are excluded.

Install each parser's requirements in a separate environment; backend/model
licenses, supported devices and any model downloads need independent review.
No weights are bundled and validation does not download them. Runtime model
caches are external artifacts, not release content.

From a chosen parser's directory:

```powershell
python parse.py --input <absolute-pdf-or-pdf-folder> --output <absolute-parser-output>
python evaluate.py --gt-dir <absolute-dataset>\cfg_001 --pred-dir <absolute-parser-output> --output <absolute-evaluation-output>\report.json
python run_all.py --dataset-dir <absolute-dataset> --output-dir <absolute-parser-output> --results-dir <absolute-evaluation-output>
```

`run_all.py` discovers `cfg_*` directories containing PDFs and manifests.
Use a dataset root containing only the desired configs for bounded runs;
`parse.py` accepts a single PDF. `--parse-only` skips evaluation;
`--eval-only` evaluates existing output; `--reparse` overrides existing parse
output. Parser results and evaluation reports are generated locally.
Use `--help` for parser-specific backend/device/OCR controls.
The parser-QA sibling consumes these generated Markdown files separately.
