# Parser-Markdown QA, then judges

Install `requirements.txt` in an isolated environment. Its adjacent
`-r ../pdf_qna/requirements.txt` reference is intentional; runtime helpers are
reused from `..\pdf_qna\src\pdfqna` without importing the generator.
Use `..\pdf_qna\experiment.env.example` to prepare your private env file.

First run a parser and retain its newly generated Markdown. Parser QA joins
`<dataset>\<config>\quest\<doc_id>.json` to recursive `<parser-root>` Markdown
by document ID. Canonical `<doc_id>\<doc_id>.md` paths are preferred; ambiguous
IDs are rejected and missing parser Markdown is skipped, not substituted.

From this directory:

```powershell
python run.py --env <absolute-env-file> --dataset-dir <absolute-dataset> --parser-root <absolute-parser-output> --parser-name docling --output-dir <absolute-parser-qa-output> --max-docs 1 --max-qa 2
python run_judges.py --env <absolute-env-file> --per-doc-dir <absolute-parser-qa-output>\per_doc --judges judge_1 --max-files 1 --max-questions 2
```

Finish prediction runs before judges update the same JSONL files. Predictions
write `per_doc\<config>__<doc_id>.jsonl` and prediction summaries. Resume is
enabled; `--no-resume` reprocesses predictions. Judges retain valid existing
verdicts and retry missing/error verdicts; use `--help` for targeted judge
reruns. Both commands accept worker controls and config filters; prediction
also accepts `--max-total-qa`. Keep parser identity and outputs separate from
the source-Markdown baseline; no historical outputs or scores are included.

The example env file configures only `judge_1`, so the command above selects
it explicitly. Without `--judges`, this runner requests `judge_1` through
`judge_4` and rejects missing configurations. For additional judges, configure
their numbered endpoint/deployment pairs and list the desired names explicitly.
