# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the project's conduct policy.

## Development

Use an activated virtual environment and install development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Follow the [generator quickstart](README.md#quickstart) for browser and service setup. Offline packaging checks must not require credentials; live generation requires your own authorized deployment and may incur costs. Consult the separate [experiment guide](experiments/README.md) before running evaluation jobs.

Open a community GitHub issue to discuss a change, then submit a focused pull request explaining its behavior and validation. Preserve existing sampling, prompts, rendering, and question/assertion relationships unless the change explicitly addresses them. Add focused tests for changed behavior using the existing test tooling. Do not claim live generation was tested when only import or CLI checks were run.

Never include credentials, private identifiers, customer documents, generated result payloads, or service logs containing sensitive data. Use sanitized, minimal fixtures. Report vulnerabilities through [SECURITY.md](SECURITY.md), not public issues.

## Source headers and third-party additions

Authorized first-party Python sources, entry points, retained scripts, and tests must start with:

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# See LICENSE.txt in the project root for license information.
```

Use an equivalent comment for other source languages. Apply ownership headers only where ownership and licensing authority are established. Preserve third-party copyright and license notices.

For any new dependency or bundled asset, describe its upstream source/version, purpose, affected paths, modifications, and license in [NOTICE.md](NOTICE.md). Preserve original copyright, attribution, and license information in third-party code and assets. The project's MIT license does not replace the licenses of upstream fonts, templates, libraries, or model weights.
