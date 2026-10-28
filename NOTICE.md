# Third-party software notices

SynthDocQA incorporates the third-party font assets identified below and uses separately installed open-source libraries. The [MIT License](LICENSE.txt) for SynthDocQA does not replace the licenses of those components.

For each third-party software component or asset included in this repository, describe the component, its purpose and location, and its license in this file. Preserve the component's original copyright, license text, attribution, and reserved-name notices. Do not remove license information from third-party code or assets.

## Bundled fonts

The following ten font files are included, unmodified, in [`src/synthpdf/resources/fonts`](src/synthpdf/resources/fonts) for handwriting and filled-form rendering. Each is licensed under the **SIL Open Font License, Version 1.1**. Full license text, copyright statements, and applicable reserved font names accompany the binaries in the linked per-family notices.

| Font family and files | Copyright attribution | License and notice |
|---|---|---|
| Caveat: `Caveat-Regular.ttf`, `Caveat-Bold.ttf` | Copyright 2014 The Caveat Project Authors; font metadata credits Pablo Impallari. | [SIL OFL 1.1](src/synthpdf/resources/fonts/Caveat-OFL.txt) |
| Dancing Script: `DancingScript-Regular.ttf` | Copyright 2016 The Dancing Script Project Authors; Reserved Font Name “Dancing Script”. | [SIL OFL 1.1](src/synthpdf/resources/fonts/DancingScript-OFL.txt) |
| Gochi Hand: `Gochi-Hand.ttf` | Copyright 2011 Huerta Tipografica; Reserved Font Names “Gochi” and “Gochi Hand”. | [SIL OFL 1.1](src/synthpdf/resources/fonts/GochiHand-OFL.txt) |
| Grand Hotel: `GrandHotel.ttf` | Copyright 2012 Brian J. Bonislawsky DBA Astigmatic (AOETI); Reserved Font Name “Grand Hotel”. | [SIL OFL 1.1](src/synthpdf/resources/fonts/GrandHotel-OFL.txt) |
| Indie Flower: `IndieFlower-Regular.ttf` | Copyright 2010 Kimberly Geswein. | [SIL OFL 1.1](src/synthpdf/resources/fonts/IndieFlower-OFL.txt) |
| Kalam: `Kalam-Regular.ttf`, `Kalam-Bold.ttf` | Copyright 2014 Indian Type Foundry. | [SIL OFL 1.1](src/synthpdf/resources/fonts/Kalam-OFL.txt) |
| Patrick Hand: `PatrickHand-Regular.ttf` | Copyright 2012 Patrick Wagesreiter. | [SIL OFL 1.1](src/synthpdf/resources/fonts/PatrickHand-OFL.txt) |
| Shadows Into Light: `ShadowsIntoLight.ttf` | Copyright 2010 Kimberly Geswein. | [SIL OFL 1.1](src/synthpdf/resources/fonts/ShadowsIntoLight-OFL.txt) |

The attributions above summarize the notices in the font files. Retain the accompanying full notices when redistributing the fonts. Changes to font software remain subject to its own license and any reserved font names.

## Repository policy templates

`LICENSE.txt`, `SECURITY.md`, and `CODE_OF_CONDUCT.md` use the corresponding Microsoft repository templates from [`microsoft/repo-templates`](https://github.com/microsoft/repo-templates/tree/e9c5e27ff1fa70dd4cff5690ad6da3c7b6472b81), revision `e9c5e27ff1fa70dd4cff5690ad6da3c7b6472b81`.

- **Copyright:** Microsoft Corporation.
- **License:** MIT; see [LICENSE.txt](LICENSE.txt).
- **Source paths:** `shared/LICENSE`, `shared/SECURITY.md`, and `shared/CODE_OF_CONDUCT.md`.

## Separately installed dependencies

The following libraries are referenced by the generator's [`pyproject.toml`](pyproject.toml); their source distributions are not vendored in this repository. These descriptions identify their role and upstream licensing, not a change to their terms. Refer to the installed version's license files for the complete terms and any bundled subcomponents.

| Component | Use | Upstream license |
|---|---|---|
| Pydantic | Configuration models and validation. | MIT |
| NumPy | Random sampling and numeric arrays. | BSD-3-Clause; distributions also include separately licensed components |
| Jinja2 | HTML and Markdown template rendering. | BSD-3-Clause |
| Python-Markdown | Markdown-to-HTML conversion. | BSD-3-Clause |
| Matplotlib | Chart rendering. | Matplotlib license, based on the Python Software Foundation license |
| Pillow | Image generation, perturbations, and font rendering. | MIT-CMU / historical PIL license |
| pypdf | PDF page splitting, merging, and inspection. | BSD-3-Clause |
| PyYAML | Configuration and question export serialization. | MIT |
| Click | Command-line interface. | BSD-3-Clause |
| OpenAI Python SDK | Requests to configured model deployments. | Apache-2.0 |
| HTTPX | HTTP transport. | BSD-3-Clause |
| Azure Identity | Entra ID authentication. | MIT |
| python-dotenv | Environment-file loading. | BSD-3-Clause |
| openpyxl | Spreadsheet question/assertion exports. | MIT |
| pandas | Tabular data for chart generation. | BSD-3-Clause |
| seaborn | Statistical chart rendering. | BSD-3-Clause |
| SciPy | Scanned-document and image transformations. | BSD-3-Clause; distributions also include separately licensed components |
| ReportLab | PDF rendering support. | BSD license |
| PyMuPDF | PDF image/text access and annotations. | GNU AGPL v3 or an Artifex commercial license |
| Playwright | Browser automation for HTML/Markdown PDF rendering. | Apache-2.0 |

Optional rendering dependencies are **WeasyPrint** (BSD-3-Clause) and **xhtml2pdf** (Apache-2.0). Development/build tools include **pytest**, **build**, and **Hatchling** (MIT). Playwright's browser installation downloads Chromium separately; Chromium and its dependencies carry their own notices.

### Experiment dependencies

The [experiments](experiments/README.md) install their dependencies independently; parser libraries and model weights are not bundled here.

| Component | Use | Upstream license |
|---|---|---|
| Docling | Document parsing, layout analysis, and OCR integration. | MIT for the Docling package; backend libraries and model assets have separate terms |
| MinerU `3.4.5` | PDF parsing through the pipeline API used by this repository. | MinerU Open Source License (`LicenseRef-MinerU-Open-Source-License`): Apache-2.0 with additional commercial-use and online-service attribution terms; see the installed version's `LICENSE.md` |
| pdfplumber | PDF text and table extraction. | MIT |
| pymupdf4llm | PDF-to-Markdown extraction. | GNU AGPL v3 or an Artifex commercial license |
| six | Compatibility dependency for the MinerU OCR runtime. | MIT |

Installing a parser may also install transitive dependencies or download model weights. Retain and follow their respective license notices; a parser's code license does not apply automatically to its models. In particular, do not treat PyMuPDF, pymupdf4llm, or MinerU as MIT-licensed because this repository uses MIT.

## Maintaining this notice

When adding or updating third-party code or assets, record the upstream project and version, repository paths, purpose, modifications, and applicable license here. Include the required upstream license and copyright files with any copied code or assets. For dependencies installed separately, review the licenses and notices associated with the resolved versions and any redistributed binary components.
