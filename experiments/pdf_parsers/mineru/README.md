# MinerU

Use this folder's isolated `requirements.txt` and the commands in the
[parser README](../README.md). Markdown and structured extraction output are
written under document/backend subfolders. `--help` lists backend, method,
language, formula and table controls. Backend/device requirements and model
downloads are external prerequisites; no backend defaults were changed.

The unchanged runner uses the MinerU 3.x Python API. Its requirements pin the
verified `mineru[pipeline]==3.4.5` and include `six`, needed by that OCR runtime.
MinerU 4.x removes the imported `mineru.cli.common` module and is not compatible
with this runner. Install this folder's requirements rather than bare `mineru`.
