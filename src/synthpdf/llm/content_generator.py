# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""LLM-driven content generator for synthpdf.

Provides topic-aware text sections, table data, and chart specifications
by calling Azure OpenAI via AzureOpenAIClient.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from synthpdf.llm.client import AzureOpenAIClient

logger = logging.getLogger(__name__)


@dataclass
class TopicSpec:
    """A single document topic with id and description."""

    id: str
    description: str


def load_topics(path: Path) -> list[TopicSpec]:
    """Load topics from a YAML file and return as a list of TopicSpec."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return [TopicSpec(id=entry["id"], description=entry["description"].strip()) for entry in raw]


class ContentGenerator:
    """Generates topic-aware content using Azure OpenAI."""

    def __init__(self, client: "AzureOpenAIClient", topic: TopicSpec) -> None:
        self._client = client
        self._topic = topic

    def generate_document_outline(self, n_sections: int) -> list[str]:
        """Generate a list of n_sections coherent section headings for this topic.

        Returns:
            A list of heading strings tailored to the document type.
        """
        prompt = (
            f"You are planning the structure of a formal document.\n"
            f"Document topic: {self._topic.description}\n"
            f"Generate exactly {n_sections} section headings that form a logical, "
            f"professional document outline.\n"
            f"Return a JSON object with a single key \"headings\" whose value is an array "
            f"of {n_sections} heading strings.\n"
            f"Make headings specific and realistic — not generic placeholders.\n"
            f"Example for an annual report: "
            f'{{\"headings\": [\"Executive Summary\", \"Financial Highlights\", '
            f'\"Revenue Analysis by Business Unit\", \"Operating Expenses\", '
            f'\"Capital Allocation Strategy\", \"Risk Factors\"]}}'
        )
        raw = self._client.complete(prompt, json_mode=True, tier="easy", _label="outline")
        try:
            data = json.loads(raw)
            if "headings" in data and isinstance(data["headings"], list):
                return [str(h) for h in data["headings"]][:n_sections]
            # Fallback: look for any list value
            for val in data.values():
                if isinstance(val, list):
                    return [str(h) for h in val][:n_sections]
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        # Fallback to generic headings
        generic = [
            "Executive Summary", "Introduction", "Background", "Methodology",
            "Key Findings", "Analysis", "Results", "Discussion",
            "Recommendations", "Risk Assessment", "Implementation Plan",
            "Financial Overview", "Conclusion", "Appendix",
        ]
        return generic[:n_sections]

    def generate_document_brief(
        self,
        n_sections: int,
        n_tables: int,
        n_charts: int,
    ) -> dict[str, Any]:
        """Generate a coherent document brief tying all artifacts to one shared narrative.

        A single LLM call produces:
          - narrative_theme: one-sentence overarching story
          - sections: list of {heading, key_facts, data_references}
          - tables:   list of {id, caption, purpose, key_data}
          - charts:   list of {id, title, chart_type, x_label, y_label, purpose}

        Downstream generators consume their slice of this brief so that every
        table, chart, and prose section references the same underlying dataset.
        Falls back to a minimal stub (with a separate outline call) on parse failure.
        """
        table_ids = [f"table_{i}" for i in range(n_tables)]
        chart_ids = [f"chart_{i}" for i in range(n_charts)]
        chart_types = ["bar", "line", "scatter", "pie", "heatmap", "box"]

        example = {
            "narrative_theme": "One-sentence overarching story shared by all sections.",
            "sections": [
                {
                    "heading": "Section Title",
                    "key_facts": ["Specific fact with a number", "Another verifiable data point"],
                    "data_references": (["table_0"] if n_tables > 0 else []),
                }
            ] if n_sections > 0 else [],
            "tables": [
                {
                    "id": "table_0",
                    "caption": "Descriptive table caption",
                    "purpose": "What this table shows and why",
                    "key_data": {"Row label A": "Value 1", "Row label B": "Value 2"},
                }
            ] if n_tables > 0 else [],
            "charts": [
                {
                    "id": "chart_0",
                    "title": "Chart Title",
                    "chart_type": "bar",
                    "x_label": "X-axis label",
                    "y_label": "Y-axis label",
                    "purpose": "What this chart visualizes",
                }
            ] if n_charts > 0 else [],
        }

        prompt = (
            f"You are planning a coherent enterprise document.\n"
            f"Topic: {self._topic.description}\n\n"
            f"Generate a document brief with EXACTLY:\n"
            f"  - {n_sections} text section(s)\n"
            f"  - {n_tables} data table(s) with ids {table_ids}\n"
            f"  - {n_charts} chart(s) with ids {chart_ids}\n\n"
            f"Rules:\n"
            f"1. narrative_theme: one sentence — the overarching story connecting all artifacts\n"
            f"2. Each section heading must be specific and professional (not generic)\n"
            f"3. Each section key_facts: 2-3 concrete facts WITH specific numbers, dates, or names "
            f"that the section prose MUST state verbatim\n"
            f"4. Sections may reference tables/charts in data_references using their ids\n"
            f"5. Tables: caption and purpose must be specific to the narrative_theme; "
            f"key_data must contain 3-5 sample cell values the table MUST include\n"
            f"6. Charts: use one of {chart_types} for chart_type; axis labels must be "
            f"descriptive and coherent with the narrative\n"
            f"7. ALL artifacts must support the same narrative_theme — this is the most important rule\n\n"
            f"Return ONLY valid JSON matching this structure (with the exact counts above):\n"
            f"{json.dumps(example, indent=2)}"
        )

        raw = ""
        try:
            raw = self._client.complete(prompt, json_mode=True, tier="hard", _label="doc_brief")
            data = json.loads(raw)

            if not isinstance(data.get("sections"), list):
                raise ValueError("missing sections list")
            if not isinstance(data.get("tables"), list):
                data["tables"] = []
            if not isinstance(data.get("charts"), list):
                data["charts"] = []

            # Truncate to requested counts
            sections = data["sections"][:n_sections]
            while len(sections) < n_sections:
                sections.append({
                    "heading": f"Section {len(sections) + 1}",
                    "key_facts": [],
                    "data_references": [],
                })
            data["sections"] = sections
            data["tables"] = data["tables"][:n_tables]
            data["charts"] = data["charts"][:n_charts]

            # Ensure all required fields exist
            for t in data["tables"]:
                t.setdefault("id", "table_?")
                t.setdefault("caption", "Data Table")
                t.setdefault("purpose", "Summary data")
                t.setdefault("key_data", {})
            for c in data["charts"]:
                c.setdefault("id", "chart_?")
                c.setdefault("title", "Chart")
                c.setdefault("chart_type", "bar")
                c.setdefault("x_label", "Category")
                c.setdefault("y_label", "Value")
                c.setdefault("purpose", "Visualization")
            data.setdefault("narrative_theme", self._topic.description[:120])
            return data

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "generate_document_brief failed (%s: %s)  raw[:200]=%r",
                type(exc).__name__, exc, raw[:200],
            )
            # Fallback: reuse outline call for section headings; stubs for tables/charts
            fallback_headings = self.generate_document_outline(n_sections) if n_sections > 0 else []
            return {
                "narrative_theme": self._topic.description[:120],
                "sections": [
                    {"heading": h, "key_facts": [], "data_references": []}
                    for h in fallback_headings
                ],
                "tables": [
                    {
                        "id": f"table_{i}",
                        "caption": f"Table {i + 1}",
                        "purpose": f"summary statistics for {self._topic.description[:80]}",
                        "key_data": {},
                    }
                    for i in range(n_tables)
                ],
                "charts": [
                    {
                        "id": f"chart_{i}",
                        "title": f"Chart {i + 1}",
                        "chart_type": "bar",
                        "x_label": "Category",
                        "y_label": "Value",
                        "purpose": f"visualization for {self._topic.description[:80]}",
                    }
                    for i in range(n_charts)
                ],
            }

    def generate_text_section(
        self,
        heading: str,
        word_count: int,
        other_headings: list[str] | None = None,
        key_facts: list[str] | None = None,
        data_references: list[str] | None = None,
    ) -> str:
        """Generate a document section body as plain paragraphs (no heading line).

        Parameters
        ----------
        other_headings:
            Titles of the other sections in the same document.  Passed to the
            LLM so it knows to cover only the content specific to *this* section
            and avoid repeating material that belongs in a sibling section.

        Returns:
            Markdown text with 3-5 paragraphs. May include bullet or numbered lists
            where appropriate. Does NOT start with a heading line.
        """
        siblings = ""
        if other_headings:
            sibling_list = "\n".join(f"  - {h}" for h in other_headings)
            siblings = (
                f"\nThis document also contains these other sections (do NOT repeat "
                f"their content here — each section must cover distinct points):\n"
                f"{sibling_list}\n"
            )
        facts_block = ""
        if key_facts:
            facts_list = "\n".join(f"  - {f}" for f in key_facts)
            facts_block = (
                f"\nKey facts this section MUST state (use exact numbers/values from these):\n"
                f"{facts_list}\n"
            )
        prompt = (
            f"You are writing a section of a formal enterprise document.\n"
            f"Document topic: {self._topic.description}\n"
            f"Section title: \"{heading}\"\n"
            f"{siblings}"
            f"{facts_block}\n"
            f"Write {word_count} words as the body of this section.\n"
            f"Requirements:\n"
            f"- Start directly with the first paragraph — do NOT include the section title or any heading\n"
            f"- Write 3 to 5 paragraphs; each 80-180 words, developing a distinct point\n"
            f"- Each paragraph must cover a DIFFERENT aspect — no repeated ideas across paragraphs\n"
            f"- Use professional language with specific data, dates, percentages, and named entities\n"
            f"- You MAY use bullet lists (- item) or numbered lists (1. item) where they add clarity\n"
            f"- Do NOT use markdown headings (## or ###)\n"
            f"- Do NOT wrap entire paragraphs in **bold** — bold is only for key terms or short phrases\n"
            f"- Do NOT add any preamble, title, or metadata before the first paragraph"
        )
        return self._client.complete(prompt, tier="easy", _label=f"text:{heading[:30]}")

    def generate_table(
        self,
        purpose: str,
        n_rows: int,
        brief_slot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate table data as a dict with 'headers' and 'rows'.

        brief_slot (optional): a dict from a document brief with keys
          'caption', 'purpose', 'key_data' — used to pin specific cell values
          so the table's data is coherent with the rest of the document.

        Returns:
            {"headers": [str, ...], "rows": [[str, ...], ...]}
            Falls back to an empty dict on parse failure.
        """
        key_data_block = ""
        if brief_slot and brief_slot.get("key_data"):
            kd = brief_slot["key_data"]
            if isinstance(kd, dict) and kd:
                kd_str = "; ".join(f"{k}: {v}" for k, v in list(kd.items())[:6])
                key_data_block = (
                    f"\nThe table MUST include these specific values in the appropriate cells: {kd_str}\n"
                )
        prompt = (
            f"You are generating structured data for a document table.\n"
            f"Document topic: {self._topic.description}\n"
            f"Table purpose: \"{purpose}\" with {n_rows} data rows.\n"
            f"{key_data_block}"
            f"Return valid JSON only: "
            f'{{\"headers\": [\"Col1\", ...], \"rows\": [[\"val\", ...], ...]}}\n'
            f"Use realistic values — numbers, dates, statuses appropriate to the topic."
        )
        raw = self._client.complete(prompt, json_mode=True, tier="medium", _label=f"table:{purpose[:30]}")
        try:
            data = json.loads(raw)
            if "headers" in data and "rows" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return {}

    def generate_document_title(self) -> str:
        """Generate a realistic, professional document title for the topic.

        Returns:
            A concise title string with no quotes or explanation.
        """
        prompt = (
            f"Generate a realistic, professional document title for the following topic.\n"
            f"Topic: {self._topic.description}\n"
            f"Return only the title — no quotes, no explanation, no extra text."
        )
        return self._client.complete(prompt, tier="easy", _label="title").strip().strip('"').strip("'")

    def generate_document_subtitle(self) -> str:
        """Generate a short subtitle or date-line for the document (e.g. department, date, version).

        Returns:
            A one-line subtitle string.
        """
        prompt = (
            f"Generate a short document subtitle or byline (one line, no quotes) for this topic.\n"
            f"Examples: 'Prepared by the Finance Committee — Q1 2024', "
            f"'Version 2.1 | Confidential', 'Office of Strategic Planning'\n"
            f"Topic: {self._topic.description}\n"
            f"Return only the subtitle — no explanation."
        )
        return self._client.complete(prompt, tier="easy", _label="subtitle").strip().strip('"').strip("'")

    def generate_qa_candidates(
        self,
        content: str,
        content_type: str = "text",
        n: int = 10,
    ) -> list[dict[str, Any]]:
        """Generate *n* simple, direct Q&A pairs with a single clear assertion each.

        Purpose: verify (1) a PDF extractor correctly captured the content and
        (2) an LLM can retrieve specific facts from the extracted text.

        Each entry contains:
          - "question"   : a direct, fact-retrieval question
          - "answer"     : a short phrase or value directly present in the content
          - "assertions" : list with exactly one {"text": "The response must state ..."} dict

        content_type controls the prompt framing:
          "text"  — prose section: ask for specific named facts, numbers, dates
          "table" — structured data: ask for specific cell values, column names, row counts
          "image" — chart description: ask for title, axis labels, categories, trend direction

        Returns a list of dicts; falls back to an empty list on parse failure.
        """
        if content_type == "table":
            content_instruction = (
                "You are writing evaluation questions for a data table extracted from a PDF document.\n"
                "Goal: verify that (1) the PDF extractor correctly captured the table and "
                "(2) an LLM can retrieve specific values from the extracted content.\n\n"
                f"The table data is:\n{content}\n\n"
                f"Generate exactly {n} questions. Requirements:\n"
                "- Each question MUST identify the table by including its caption in the question "
                "(e.g. 'In the table titled \"[caption]\", what value does the Revenue column show for Q1?'). "
                "This makes the question unambiguous when a document contains multiple tables.\n"
                "- Each question asks for one specific, directly readable fact — a cell value, "
                "column name, row count, or caption detail\n"
                "- The answer must be a short phrase or number that appears verbatim in the table above\n"
                "- Do NOT ask for calculations, rankings, trends, or comparisons across rows\n"
                "- Good example: 'In the table titled \"Budget Overview\", what is the Q1 Marketing spend?'\n"
                "- Bad example: 'What value does the Revenue column show for Q1?' (no table anchor)"
            )
        elif content_type == "image":
            content_instruction = (
                "You are writing evaluation questions for a chart or diagram extracted from a PDF document.\n"
                "Goal: verify that (1) the PDF extractor correctly captured the chart metadata and "
                "(2) an LLM can retrieve specific observable facts from that description.\n\n"
                f"The chart description is:\n{content}\n\n"
                f"Generate exactly {n} questions. Requirements:\n"
                "- Each question MUST identify the chart by including its title in the question "
                "(e.g. 'In the chart titled \"[chart title]\", what does the Y-axis represent?'). "
                "This makes the question unambiguous when a document contains multiple charts.\n"
                "- Each question asks for one directly stated fact — chart title, axis label, "
                "category name, trend direction, or value range\n"
                "- The answer must be a short phrase directly present in the chart description above\n"
                "- Do NOT ask for interpretation, prediction, or reasoning beyond what is stated\n"
                "- Good example: 'In the \"Regional Sales\" chart, what does the Y-axis represent?'\n"
                "- Bad example: 'What does the Y-axis represent?' (no chart anchor)"
            )
        else:  # "text"
            # Extract the section heading — content is formatted as "**Heading**\n\nbody..."
            import re as _re
            _heading_match = _re.match(r"\*\*(.+?)\*\*", content.lstrip())
            _section_title = _heading_match.group(1) if _heading_match else "this section"
            content_instruction = (
                "You are writing evaluation questions for a text section extracted from a PDF document.\n"
                "Goal: verify that (1) the PDF extractor correctly captured the text and "
                "(2) an answering model — given only the full PDF with no other context — "
                "can locate this section and retrieve the specific fact.\n\n"
                f"The section title is: \"{_section_title}\"\n"
                f"The section content is:\n{content}\n\n"
                f"Generate exactly {n} questions. Requirements:\n"
                f"- Every question MUST identify this section by including its title "
                f"(e.g. 'According to the \"{_section_title}\" section, ...', "
                f"'In the \"{_section_title}\" section, what is ...'). "
                "This is mandatory — without the title the question is ambiguous in a multi-section document.\n"
                "- Each question asks for one specific fact explicitly stated in the text — "
                "a name, number, date, percentage, or short phrase\n"
                "- The answer must be a short phrase directly present in the section above\n"
                "- Do NOT ask for synthesis, inference, or comparison across sections\n"
                f"- Good example: 'In the \"{_section_title}\" section, what percentage is cited for X?'\n"
                "- Bad example: 'What percentage does the text cite for X?' (no section anchor)\n"
                "- Bad example: 'What is the ARR target mentioned in the section?' ('the section' is not specific enough)"
            )

        prompt = (
            f"{content_instruction}\n\n"
            "For each question provide:\n"
            "  1. The question itself\n"
            "  2. A short, specific answer — a phrase or value directly from the content "
            "(not a full sentence if a phrase suffices)\n"
            "  3. Exactly 1 assertion that verifies the answer is present, phrased as: "
            "\"The response must state [specific answer].\"\n\n"
            "Return valid JSON only:\n"
            '{"qa": [{"question": "...", "answer": "...", '
            '"assertions": [{"text": "The response must state ..."}]}, ...]}'
        )
        # QA is optional metadata — never let a failure here crash variant generation.
        raw = ""
        try:
            raw = self._client.complete(prompt, json_mode=True, tier="medium", _label=f"qa:{content_type}")
            if not raw or not raw.strip():
                logger.warning("QA: empty response from model (content_type=%s)", content_type)
                return []
            data = json.loads(raw)
            pairs = data.get("qa", data.get("questions", data.get("qa_pairs", [])))
            if not isinstance(pairs, list):
                logger.warning(
                    "QA parse: expected list, got %s. keys=%s  raw[:200]=%r",
                    type(pairs).__name__, list(data.keys()), raw[:200],
                )
                return []
            result = []
            for p in pairs:
                if not isinstance(p, dict) or not p.get("question"):
                    continue
                assertions = [
                    {"text": str(a.get("text", a) if isinstance(a, dict) else a)}
                    for a in p.get("assertions", [])
                    if a
                ]
                result.append({
                    "question":   str(p.get("question", "")),
                    "answer":     str(p.get("answer", "")),
                    "assertions": assertions,
                })
            if not result:
                logger.warning(
                    "QA parse: 0 items extracted. keys=%s  raw[:200]=%r",
                    list(data.keys()), raw[:200],
                )
            return result[:n]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "QA generation failed (%s: %s)  raw[:200]=%r",
                type(exc).__name__, exc, raw[:200],
            )
            return []

    def generate_large_table(
        self, purpose: str, n_rows: int, n_cols: int = 5, chunk_size: int = 15
    ) -> dict[str, Any]:
        """Generate a large table by issuing multiple small LLM calls (one per chunk).

        Splitting avoids slow/hanging single requests for large JSON payloads.
        - First call: establishes column headers + first ``chunk_size`` rows.
        - Subsequent calls: extend with more rows using the same headers.
        - Uses tier="medium" (fast) instead of tier="hard".

        Returns {"headers": [...], "rows": [[...], ...]} or {} if the first chunk fails.
        Partial results (some chunks fail) are still returned — caller decides whether
        the row count is sufficient.
        """
        n_chunks = math.ceil(n_rows / chunk_size)
        headers: list[str] = []
        all_rows: list[list[str]] = []

        for chunk_idx in range(n_chunks):
            rows_wanted = min(chunk_size, n_rows - len(all_rows))
            if rows_wanted <= 0:
                break

            if chunk_idx == 0:
                # First chunk: get headers + initial rows
                prompt = (
                    f"You are generating tabular data for a professional document.\n"
                    f"Document topic: {self._topic.description}\n"
                    f"Table purpose: \"{purpose}\".\n"
                    f"Generate exactly {rows_wanted} data rows with {n_cols} columns.\n"
                    f"Return valid JSON only:\n"
                    f'{{\"headers\": [\"Col1\", ...], \"rows\": [[\"val\", ...], ...]}}\n'
                    f"Use realistic, varied values. Ensure numeric columns are consistent."
                )
                raw = self._client.complete(
                    prompt, json_mode=True, tier="medium",
                    _label=f"lgtbl_c0:{purpose[:20]}",
                )
                try:
                    data = json.loads(raw)
                    if "headers" in data and "rows" in data:
                        headers = [str(h) for h in data["headers"]]
                        all_rows.extend(
                            [str(c) for c in row]
                            for row in data["rows"][:rows_wanted]
                        )
                except (json.JSONDecodeError, TypeError):
                    break  # No headers → cannot continue
            else:
                # Subsequent chunks: extend rows only (headers already fixed)
                prompt = (
                    f"Continue generating rows for the table below.\n"
                    f"Table purpose: \"{purpose}\".\n"
                    f"Column headers: {headers}\n"
                    f"Generate exactly {rows_wanted} MORE rows (do NOT repeat headers).\n"
                    f"Start from row {len(all_rows) + 1}. Use varied, realistic values.\n"
                    f"Return valid JSON only:\n"
                    f'{{\"rows\": [[\"val\", ...], ...]}}'
                )
                raw = self._client.complete(
                    prompt, json_mode=True, tier="medium",
                    _label=f"lgtbl_c{chunk_idx}:{purpose[:20]}",
                )
                try:
                    data = json.loads(raw)
                    if "rows" in data:
                        all_rows.extend(
                            [str(c) for c in row]
                            for row in data["rows"][:rows_wanted]
                        )
                except (json.JSONDecodeError, TypeError):
                    pass  # Skip failed chunk — keep rows collected so far

        if headers and all_rows:
            return {"headers": headers, "rows": all_rows}
        return {}

    def generate_chart_spec(
        self,
        chart_type: str,
        brief_slot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate chart metadata for a given chart type.

        brief_slot (optional): a dict from a document brief with keys
          'title', 'chart_type', 'x_label', 'y_label', 'purpose' — used to
          constrain the chart so it is coherent with the rest of the document.

        Returns:
            {
                "title": str,
                "x_label": str,
                "y_label": str,
                "categories": [str, ...],
                "series_names": [str, ...],
                "value_range": [float, float],
                "trend": "increasing"|"decreasing"|"volatile"|"uniform",
            }
            Falls back to an empty dict on parse failure.
        """
        constraint_block = ""
        if brief_slot:
            parts = []
            if brief_slot.get("title"):
                parts.append(f'title: "{brief_slot["title"]}"')
            if brief_slot.get("x_label"):
                parts.append(f'x_label: "{brief_slot["x_label"]}"')
            if brief_slot.get("y_label"):
                parts.append(f'y_label: "{brief_slot["y_label"]}"')
            if brief_slot.get("purpose"):
                parts.append(f'purpose: "{brief_slot["purpose"]}"')
            if parts:
                constraint_block = (
                    f"\nUse these pre-determined values (do NOT change them): "
                    + ", ".join(parts) + "\n"
                )
        prompt = (
            f"You are specifying a {chart_type} chart for a professional document.\n"
            f"Document topic: {self._topic.description}\n"
            f"{constraint_block}"
            f"Return JSON: "
            f'{{\"title\": \"...\", \"x_label\": \"...\", \"y_label\": \"...\", '
            f'\"categories\": [\"Cat1\", ...], \"series_names\": [\"Series1\", ...], '
            f'\"value_range\": [min, max], \"trend\": \"increasing|decreasing|volatile|uniform\"}}'
        )
        raw = self._client.complete(prompt, json_mode=True, tier="medium", _label=f"chart_spec:{chart_type}")
        try:
            data = json.loads(raw)
            # Apply brief overrides to ensure consistency
            if brief_slot:
                for key in ("title", "x_label", "y_label"):
                    if brief_slot.get(key):
                        data[key] = brief_slot[key]
            # Validate required keys are present
            required = {"title", "x_label", "y_label", "categories", "series_names",
                        "value_range", "trend"}
            if required.issubset(data.keys()):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return {}

    def generate_element_qa(
        self,
        element_type: str,
        ground_truth: dict[str, Any],
        style_examples: list[str],
        count: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate LLM-driven QnA for a structured element using few-shot style examples.

        Unlike ``generate_qa_candidates()`` which operates on raw text, this method
        takes structured ``ground_truth`` data (a dict of facts) and instructs the
        LLM to generate ``count`` questions with varied phrasing.

        ``style_examples`` are existing template question strings shown to the LLM as
        *phrasing inspiration only* — the LLM must NOT copy them verbatim, ensuring
        every document batch uses different question formulations.

        Args:
            element_type: Human-readable name (e.g. ``"timeline"``, ``"comparison table"``).
            ground_truth: Structured data dict representing the element's verifiable facts.
            style_examples: 2-4 template question strings used as few-shot phrasing guides.
            count: Number of QnA pairs to generate.

        Returns:
            List of ``{"question", "answer", "assertions"}`` dicts; empty list on failure.
        """
        title = (
            ground_truth.get("title", "")
            or ground_truth.get("caption", "")
            or "this element"
        )

        # Serialize ground truth — skip non-JSON-serializable values gracefully
        gt_clean: dict[str, Any] = {}
        for k, v in ground_truth.items():
            if k in ("image_path", "render_payload", "qa_candidates", "fill_style"):
                continue
            try:
                json.dumps(v)
                gt_clean[k] = v
            except (TypeError, ValueError):
                gt_clean[k] = str(v)

        facts_json = json.dumps(gt_clean, indent=2)
        examples_block = "\n".join(f'  - "{ex}"' for ex in style_examples[:4])

        prompt = (
            f"Generate evaluation questions for a document element.\n\n"
            f"ELEMENT TYPE: {element_type}\n\n"
            f"ELEMENT DATA (all answers must come directly from these facts):\n"
            f"{facts_json}\n\n"
            f"EXAMPLE question styles — use only as phrasing inspiration, do NOT copy verbatim:\n"
            f"{examples_block}\n\n"
            f"Generate exactly {count} evaluation questions. Rules:\n"
            f"1. Every question must anchor the element by including its title '{title}'\n"
            f"   (e.g. \"In the '{title}' ...\", \"According to the '{title}' ...\", \"On the '{title}' ...\")\n"
            f"2. Vary phrasing significantly — use different question openers and structures\n"
            f"3. Answers must be short phrases or values directly readable from the data above\n"
            f"4. Do NOT ask for information absent from the data or requiring inference\n"
            f"5. Phrase every question as a reader looking at the rendered PDF — "
            f"do NOT use code or metadata terminology such as 'sections array', "
            f"'fill_style', 'json key', 'field name', 'property', 'object', or any "
            f"term that only makes sense when reading raw data structures\n"
            f"6. Each assertion: \"The response must state '[answer]'.\"\n\n"
            "Return ONLY valid JSON:\n"
            '{{"qa": [{{"question": "...", "answer": "...", "assertions": [{{"text": "..."}}]}}]}}'
        )

        raw = ""
        try:
            raw = self._client.complete(
                prompt, json_mode=True, tier="easy", _label=f"elem_qa:{element_type[:20]}"
            )
            data = json.loads(raw)
            pairs = data.get("qa", data.get("questions", data.get("qa_pairs", [])))
            if not isinstance(pairs, list):
                return []
            result = []
            for p in pairs:
                if not isinstance(p, dict) or not p.get("question"):
                    continue
                assertions = [
                    {"text": str(a.get("text", a) if isinstance(a, dict) else a)}
                    for a in p.get("assertions", [])
                    if a
                ]
                result.append({
                    "question":   str(p.get("question", "")),
                    "answer":     str(p.get("answer", "")),
                    "assertions": assertions,
                })
            return result[:count]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "generate_element_qa failed (%s: %s) element_type=%s  raw[:200]=%r",
                type(exc).__name__, exc, element_type, raw[:200],
            )
            return []

    def generate_text_block(
        self,
        block_type: str,
    ) -> "tuple[str, dict[str, Any]] | None":
        """Generate a diverse structured text block (email, policy, spec, bullet_list, footnote).

        Uses the LLM to produce fresh content relevant to the current topic, avoiding
        the repetitive patterns of static templates.

        Args:
            block_type: One of ``"email"``, ``"policy"``, ``"spec"``,
                ``"bullet_list"``, ``"footnote"``.

        Returns:
            ``(text_markdown, key_facts_dict)`` on success, or ``None`` on failure.
            ``text_markdown`` is ready to embed in a document.
            ``key_facts_dict`` has string-valued fields for QnA generation.
        """
        _GUIDELINES: dict[str, str] = {
            "email": (
                "Write a realistic internal business email with all headers and a 4-5 sentence body.\n"
                "key_facts must include: sender (full name), recipient (full name), subject, date (YYYY-MM-DD), project.\n"
                "text format: '**From:** ...\\n**To:** ...\\n**Date:** ...\\n**Subject:** ...\\n\\n[body]'"
            ),
            "policy": (
                "Write a structured company policy document with version, effective date, owner, and 3-4 policy sections.\n"
                "key_facts must include: topic (policy name), version (e.g. '2.1'), effective_date (YYYY-MM-DD), owner (full name).\n"
                "text format: '## [Topic]\\n\\n**Version:** ...  \\n**Effective Date:** ...  \\n**Policy Owner:** ...\\n\\n### Purpose\\n...'"
            ),
            "spec": (
                "Write a technical specification with author, version, latency SLA (integer ms), and throughput (integer RPS).\n"
                "key_facts must include: topic (spec name), author (full name), version (e.g. '3.2.1'), latency_sla_ms (integer), throughput_rps (integer).\n"
                "text format: '## Technical Specification: [Topic]\\n\\n**Author:** ...  \\n**Version:** ...\\n\\n### Overview\\n...'"
            ),
            "bullet_list": (
                "Write a titled list with 5-8 specific, concrete bullet items.\n"
                "key_facts must include: topic (list title), item_count (integer), first_item (text of the first bullet).\n"
                "text format: '## [Topic]\\n\\n- [item]\\n- [item]\\n...'"
            ),
            "footnote": (
                "Write a paragraph about a technical or business topic with 3-5 inline footnote references [^1], [^2], etc.\n"
                "key_facts must include: topic (subject of the paragraph), footnote_count (integer).\n"
                "text format: '[paragraph with [^1] references]\\n\\n[^1]: [note text]\\n...'"
            ),
        }
        guideline = _GUIDELINES.get(block_type, _GUIDELINES["spec"])
        prompt = (
            f"Generate a realistic business document section of type '{block_type}' "
            f"for the topic: {self._topic.description}\n\n"
            f"Guidelines:\n{guideline}\n\n"
            "Return ONLY valid JSON:\n"
            '{{"block_type": "{block_type}", "text": "...full markdown text...", '
            '"key_facts": {{"field1": "value1", ...}}}}'
        ).replace("{block_type}", block_type)

        raw = ""
        try:
            raw = self._client.complete(
                prompt, json_mode=True, tier="easy", _label=f"text_block:{block_type}"
            )
            data = json.loads(raw)
            text = str(data.get("text", "")).strip()
            key_facts = data.get("key_facts", {})
            if not text or not isinstance(key_facts, dict):
                return None
            # Ensure all key_facts values are strings (for safe QnA generation)
            key_facts = {k: str(v) for k, v in key_facts.items()}
            return text, key_facts
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "generate_text_block failed (%s: %s) block_type=%s  raw[:200]=%r",
                type(exc).__name__, exc, block_type, raw[:200],
            )
            return None
