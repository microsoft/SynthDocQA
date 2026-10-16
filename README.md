# SynthDocQA

Packaging preview of a synthetic document and question-answer generator. It samples document content and layouts in an **unconditional batch**, calls a user-configured Azure OpenAI deployment, and renders PDFs using Chromium. This is not a network-free generator. Existing prompts and generation algorithms are preserved.

This preview retains preexisting QA limitations: generated questions, assertions, and answers can be inconsistent and require review before use. Packaging does not establish their correctness or repair those limitations.

## Quickstart

From this repository directory, using an activated Python 3.10+ virtual environment:

```powershell
python -m pip install -e .
python -m playwright install chromium
```

Chromium installation is external setup and may download browser binaries. Rendering resources are included with the package.

Configure your own Azure OpenAI deployment and credentials:

```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://<your-resource>.openai.azure.com/"
$env:AZURE_OPENAI_DEPLOYMENT = "<your-deployment>"
$env:AZURE_OPENAI_API_VERSION = "<your-supported-api-version>"
$env:AZURE_OPENAI_API_KEY = "<your-key>"
synthpdf generate --seed 42 --num-documents 1 --pages 4 --output-dir output
```

The equivalent module invocation is:

```powershell
python -m synthpdf generate --seed 42 --num-documents 1 --pages 4 --output-dir output
```

Optional deployment overrides are `AZURE_OPENAI_DEPLOYMENT_EASY`, `AZURE_OPENAI_DEPLOYMENT_MEDIUM`, and `AZURE_OPENAI_DEPLOYMENT_HARD`. Set `AZURE_TENANT_ID` when your authentication setup needs a tenant. Run `synthpdf generate --help` for available options. The tenant, deployment, access permissions, and service charges are your responsibility; none are provisioned by this package. Never commit credentials. A seed controls local random sampling, not deterministic responses from a remote model.

You can instead copy [.env.example](.env.example) to `.env` in your working directory and supply your own values. API keys are read externally and are not written to the generated configuration.

## Generated outputs

The selected output directory contains `pdfs/`, `manifests/`, `quest/` and a run summary in `manifest.jsonl`. Per-document manifests describe structured content, variants, render metadata and QA candidates. The `quest/` directory contains question/assertion exports in JSON, XLSX, TSV and YAML formats. Intermediate Markdown, local assets, internal configuration and timing information may also be written. These are runtime outputs, not bundled examples or committed results.

Generation uses the existing local HTML/Markdown templates and neutral logo placeholders. No external topic or source-document input is exposed by the generation CLI. Model endpoint and deployment settings remain user-configurable.

## Experiments

The separate [experiment guide](experiments/README.md) describes execution-only evaluation code with its own prerequisites. It tests generated QnAs against configured models using existing input layouts: a `sourceMarkdown` baseline, native PDF or page-image inputs, and parser-produced `parserMarkdown`. No historical results, benchmark scores, datasets, or parser outputs are included. Generator installation does not require the experiment dependencies.

Install each parser's own requirements: the existing [MinerU runner](experiments/pdf_parsers/mineru/README.md) requires the verified `mineru[pipeline]==3.4.5` and `six>=1.16`, not MinerU 4.x. The example experiment configuration defines one judge; use `--judges judge_1` with the separate [parser-QA judge command](experiments/pdf_qna_with_parser/README.md), or explicitly select additional configured judges.

## Data and service calls

Local metrics are retained in memory or written locally. No project outbound telemetry collector is included. Generation sends prompts to the Azure OpenAI service you configure; experiment service calls depend on your configuration. SDKs and providers have their own terms and data-handling behavior. This statement is not a claim that every dependency disables telemetry.

## Contributing and conduct

See [CONTRIBUTING.md](CONTRIBUTING.md) for the CLA process and development guidance, [SUPPORT.md](SUPPORT.md) for community help, and [SECURITY.md](SECURITY.md) for private vulnerability reporting.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
See also [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License and release status

Authorized first-party code is covered by [LICENSE.txt](LICENSE.txt). That MIT license does not relicense dependencies or third-party assets; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

This staging preview is not approval to publish. Before publication, maintainers must confirm organizational and legal release authorization, bundled asset provenance and redistribution rights, final archive contents, and configured CLA/security/support processes. Policy templates were copied from [microsoft/repo-templates revision e9c5e27ff1fa70dd4cff5690ad6da3c7b6472b81](https://github.com/microsoft/repo-templates/tree/e9c5e27ff1fa70dd4cff5690ad6da3c7b6472b81); check for approved updates before release.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
