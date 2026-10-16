# Source-Markdown baseline

From this directory, install `requirements.txt` in an isolated environment.
Copy `experiment.env.example` to a location outside the release and populate
your endpoint, deployment and authentication values. At least one QA pair and
one judge pair are required. Entra authentication requires an authorized Azure
identity; API-key mode uses `USE_ENTRA_ID=false` and an external key.
Deployments containing `claude` use the separately authenticated local Claude
CLI, rather than the Azure client; this existing routing is unchanged.

```powershell
python run.py --env <absolute-env-file> --dataset-dir <absolute-dataset> --output-dir <absolute-baseline-output> --max-docs 1 --max-qa 2
```

Reads `cfg_*\markdown\<doc_id>.md` and `cfg_*\quest\<doc_id>.json`.
Writes incremental `<output>\per_doc\<config>__<doc_id>.jsonl` and summary
JSON/spreadsheets. Records include `config`, `doc_id`, `md_path`,
`markdown_source`, `question_id`, `question`, `element_type`, `assertion`,
and `model_results`. This runtime output is the prerequisite question-record
input to the direct-PDF runner; no historical records are shipped.

Use `--config-filter`, `--max-total-qa` and `--workers` to bound work.
Resume is enabled; `--no-resume` reprocesses documents. `--judge-only`
evaluates existing predictions without QA calls. Use `--help` for targeted
judge reruns. `--markdown-source <parser>` searches a per-config parser folder
but falls back to source Markdown if absent; use the separate parser-QA route
for explicit parser-output discovery and keep the measurements distinct.
