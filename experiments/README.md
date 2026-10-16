# Execution-only experiments

These optional scripts evaluate generated questions against configured models
using source Markdown, native PDFs, selected page images, or parser-produced
Markdown. No datasets, predictions, parser outputs, model weights, historical
scores, or credentials are included. They do not import the generator package.
Install dependencies in a separate virtual environment for each route; they
are not part of the generator's default installation.

## Inputs and order

Prepare this existing per-config layout yourself; flat `ALL_queries` /
`grounding_pdfs` exports are not automatically interchangeable:

```text
dataset\
  cfg_<name>\
    pdfs\<doc_id>.pdf
    manifests\<doc_id>.json
    markdown\<doc_id>.md
    quest\<doc_id>.json
```

Question files contain records with `Id`, `query`, `refs`, and `assertions`.
Keep document IDs consistent across folders. Manifests are required by parser
extraction evaluation. No layout adapter is provided.

1. Generate or prepare PDFs, Markdown, questions/assertions, and manifests.
2. [Source Markdown baseline](pdf_qna/README.md): predictions and judges.
3. [Native PDF / page images](pdf_qna_with_images/README.md): the native-PDF
   runner requires freshly produced baseline `per_doc` JSONL question records;
   the independent page-image runner reads raw `quest` files.
4. [Parsers](pdf_parsers/README.md): parse PDFs and evaluate extraction.
5. [Parser Markdown QA](pdf_qna_with_parser/README.md): predict from parsed
   Markdown, finish predictions, then judge those same JSONL files.

Use explicit absolute input/config/output paths in route commands. Relative
paths are resolved from the working directory. Start with one config/document
and a few questions; model calls and PDF uploads may incur cost and transmit
document content to the configured service. Authentication is external.
Do not commit runtime outputs, credentials, cached models, or logs.

## Installation and judge selection

Use each parser's requirements file rather than installing its package by name.
The unchanged MinerU runner is verified with `mineru[pipeline]==3.4.5` and
`six>=1.16`; MinerU 4.x lacks the Python module that this runner imports.
See the [MinerU guide](pdf_parsers/mineru/README.md).

The example env file configures only `judge_1`. For the separate parser-QA
judge pass, specify `--judges judge_1`: without an explicit selection, that
runner requests four judges and rejects missing configurations. To use more,
configure their numbered endpoint/deployment pairs and select them explicitly.
See the [parser-QA commands](pdf_qna_with_parser/README.md).

## Scope and limitations

Only release relocation, import wiring, dependency declarations, safe templates,
and documentation were changed. Prompts, model defaults, judging, selection,
fallbacks, retry/concurrency and result schemas are preserved; this is not the
separately scoped QA-remediation work. Judges assess answers against supplied
assertions, not the truth of those assertions. A non-dataset Markdown selection
in the baseline falls back to source Markdown when parser Markdown is missing;
do not report such answers as parser measurements.

The page-image `ModelSpec` retains the original `verify_ssl=False` default.
The supplied YAML explicitly sets `verify_ssl: true` for each model; retain
that setting and configure trusted certificates rather than disabling TLS.
Other authentication/TLS behavior is unchanged. Outputs may contain configured
endpoint identifiers, source paths, answers, and error details; review locally
before sharing. Parser packages/model weights have their own license terms;
first-party MIT headers do not license these dependencies or dataset content.

Commands are Windows-oriented. Offline compile/import/mock validation is
distinct from an authorized live API smoke or a full benchmark reproduction.
