# Native PDF and question-guided page images

Install this route's `requirements.txt` in its own isolated environment.
These are two distinct runners, not interchangeable implementations.

## Native uploaded PDF

First complete a bounded `pdf_qna` baseline run and retain its fresh
`<baseline-output>\per_doc\*.jsonl` files. `src\main.py` consumes these
question records, not raw `quest` JSON. Each line must include `question` and
normally `question_id`, `assertion`, `config`, `doc_id`, `md_path`, and
`element_type`; use the baseline writer's schema rather than hand-built
records. PDF resolution tries an existing `pdf_path`, the PDF sibling of
`md_path`, then `<dataset-root>\<config>\pdfs\<doc_id>.pdf`.
Question de-duplication and assertion selection are unchanged.

Populate a private env file using the adjacent
`..\pdf_qna\experiment.env.example`. QA deployments must support Responses
and PDF file input. Native Claude CLI mode instead uses `QA_<NAME>_ENDPOINT`
set to `claude-cli` and an authenticated local CLI.

From this directory:

```powershell
python src\main.py --env-file <absolute-env-file> --questions-dir <absolute-baseline-output>\per_doc --dataset-root <absolute-dataset> --output-dir <absolute-native-output> --file-pattern "cfg_001*.jsonl" --max-questions 2
python src\judge_eval.py --env-file <absolute-env-file> --results-dir <absolute-native-output>
```

Finish predictions before judging. The native runner overwrites matching
output files on rerun (no prediction-resume switch). It deletes uploaded PDFs
after each document unless `--keep-uploaded-files` is set. Judges update
JSONL in place and resume existing successful verdicts; `--help` describes
named judge reruns.

## Selected page-image inference

This independent runner reads `<dataset>\<config>\{pdfs,quest}\<doc_id>.*`
directly, extracts page text, selects question-relevant pages and renders
images on demand. It preserves title matching, BM25 and first-page fallbacks.
`input_mode: text` retains the existing text-only path.

Configure `inference_models.yaml` in a private external copy and export
`QA_ENDPOINT`, `QA_DEPLOYMENT`, `JUDGE_ENDPOINT`, `JUDGE_DEPLOYMENT`, and
`AZURE_OPENAI_API_KEY` in your shell (or use the existing Entra identity route).
The loader expands endpoint/deployment placeholders; it does not load `.env`.
Keep every model's `verify_ssl: true`: the original class default remains
false for preservation, not as a public recommendation.

Launch as the local package module from this directory, not by file path:

```powershell
python -m inference.run_inference --models-config <absolute-yaml> --dataset-dir <absolute-dataset> --results-dir <absolute-image-output> --config-filter "cfg_001*" --max-qa 2 --workers 1
```

Writes per-document JSON and summary JSON/XLSX. Existing document outputs are
skipped unless `--force`; `--no-judge` collects QA answers only.
CLI question/worker values override YAML inference settings as in the source.
For `client_type: serverless`, supply an OpenAI-compatible endpoint/deployment.
No calls, models, PDFs, or stored results are bundled with these examples.
