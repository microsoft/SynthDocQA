# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Text block and annotation variant generator.

Text block types: email, policy, spec, bullet list, footnote.
Each block has perturbations (typos, abbreviations, unicode punctuation, date formats).

Annotation types (Markdown-visible): callout, highlight, inline comment, footnote.

Also provides generate_llm_text_variants() for topic-specific LLM-driven prose sections.
"""
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

import click
import numpy as np

from synthpdf.rng import fork_rng
from synthpdf.types import Variant

if TYPE_CHECKING:
    from synthpdf.llm.client import AzureOpenAIClient
    from synthpdf.llm.content_generator import TopicSpec


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Text corpora
# ---------------------------------------------------------------------------

_EMAIL_SUBJECTS = [
    "Project Update - Q{q} Review",
    "Action Required: Approval Needed for {project}",
    "Re: {topic} Discussion",
    "Weekly Sync: {team} Team",
    "Follow-up on {item}",
]

_POLICY_TOPICS = [
    "Remote Work Policy",
    "Data Retention Policy",
    "Expense Reimbursement Policy",
    "Security Incident Response",
    "Code Review Guidelines",
]

_SPEC_TOPICS = [
    "API Authentication Requirements",
    "Database Schema Specification",
    "Service Level Agreement",
    "Deployment Checklist",
    "Performance Benchmarks",
]

_BULLET_TOPICS = [
    "Meeting Agenda",
    "Sprint Goals",
    "Risk Register",
    "Key Milestones",
    "Open Issues",
]

_NAMES = ["Alice Johnson", "Bob Smith", "Carol White", "David Lee", "Eve Martinez"]
_TEAMS = ["Engineering", "Product", "Design", "QA", "DevOps"]
_PROJECTS = ["Phoenix", "Orion", "Aurora", "Titan", "Nova"]


# ---------------------------------------------------------------------------
# Base text generators
# ---------------------------------------------------------------------------

def _email_block(rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    sender = _NAMES[int(rng.integers(0, len(_NAMES)))]
    recipient = _NAMES[int(rng.integers(0, len(_NAMES)))]
    q = int(rng.integers(1, 5))
    project = _PROJECTS[int(rng.integers(0, len(_PROJECTS)))]
    topic = _SPEC_TOPICS[int(rng.integers(0, len(_SPEC_TOPICS)))]
    subject_tmpl = _EMAIL_SUBJECTS[int(rng.integers(0, len(_EMAIL_SUBJECTS)))]
    subject = subject_tmpl.format(q=q, project=project, topic=topic, team=_TEAMS[0], item=topic)

    date_year = int(rng.integers(2023, 2026))
    date_month = int(rng.integers(1, 13))
    date_day = int(rng.integers(1, 29))
    date_str = f"{date_year:04d}-{date_month:02d}-{date_day:02d}"

    body_sentences = [
        f"I hope this message finds you well.",
        f"I wanted to follow up regarding the {topic} discussion we had last week.",
        f"Please review the attached document and provide your feedback by {date_str}.",
        f"We need to ensure compliance with the new requirements before the {project} launch.",
        f"Let me know if you have any questions or concerns.",
        f"Best regards,\n{sender}",
    ]
    body = "\n\n".join(body_sentences)
    text = f"**From:** {sender}\n**To:** {recipient}\n**Date:** {date_str}\n**Subject:** {subject}\n\n{body}"

    facts = {
        "sender": sender,
        "recipient": recipient,
        "date": date_str,
        "subject": subject,
        "project": project,
    }
    qa = [
        {
            "question": f"In the email with subject '{subject}', who is the sender?",
            "answer": sender,
            "assertions": [{"text": f"The response must state '{sender}'."}],
        },
        {
            "question": f"In the email with subject '{subject}', who is the recipient?",
            "answer": recipient,
            "assertions": [{"text": f"The response must state '{recipient}'."}],
        },
        {
            "question": f"What date was the email with subject '{subject}' sent?",
            "answer": date_str,
            "assertions": [{"text": f"The response must state '{date_str}'."}],
        },
    ]
    return text, {"key_facts": facts, "qa_candidates": qa}


def _policy_block(rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    topic = _POLICY_TOPICS[int(rng.integers(0, len(_POLICY_TOPICS)))]
    effective_year = int(rng.integers(2022, 2026))
    effective_month = int(rng.integers(1, 13))
    effective_date = f"{effective_year:04d}-{effective_month:02d}-01"
    owner = _NAMES[int(rng.integers(0, len(_NAMES)))]
    version = f"{int(rng.integers(1, 4))}.{int(rng.integers(0, 10))}"

    sections = [
        f"## {topic}\n\n**Version:** {version}  \n**Effective Date:** {effective_date}  \n**Policy Owner:** {owner}",
        "### Purpose\n\nThis policy establishes guidelines to ensure consistent, compliant, and efficient operations across the organization.",
        "### Scope\n\nThis policy applies to all employees, contractors, and third-party vendors with access to company systems and data.",
        "### Policy Statement\n\nAll personnel must adhere to the requirements outlined in this document. Non-compliance may result in disciplinary action.",
        f"### Review Cycle\n\nThis policy shall be reviewed annually. The next review is scheduled for {effective_year + 1:04d}-{effective_month:02d}-01.",
    ]
    text = "\n\n".join(sections)
    facts = {
        "topic": topic,
        "version": version,
        "effective_date": effective_date,
        "owner": owner,
    }
    qa = [
        {
            "question": f"What is the effective date of the '{topic}' policy?",
            "answer": effective_date,
            "assertions": [{"text": f"The response must state '{effective_date}'."}],
        },
        {
            "question": f"Who is the policy owner for the '{topic}' policy?",
            "answer": owner,
            "assertions": [{"text": f"The response must state '{owner}'."}],
        },
        {
            "question": f"What version number does the '{topic}' policy document show?",
            "answer": version,
            "assertions": [{"text": f"The response must state '{version}'."}],
        },
    ]
    return text, {"key_facts": facts, "qa_candidates": qa}


def _spec_block(rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    topic = _SPEC_TOPICS[int(rng.integers(0, len(_SPEC_TOPICS)))]
    version = f"{int(rng.integers(1, 5))}.{int(rng.integers(0, 10))}.{int(rng.integers(0, 10))}"
    author = _NAMES[int(rng.integers(0, len(_NAMES)))]
    sla_ms = int(rng.integers(50, 2000))
    throughput = int(rng.integers(100, 10000))

    text = (
        f"## Technical Specification: {topic}\n\n"
        f"**Author:** {author}  \n**Version:** {version}\n\n"
        "### Overview\n\n"
        f"This document specifies the requirements for {topic}. "
        "All implementing services must conform to this specification.\n\n"
        "### Requirements\n\n"
        f"- **Latency SLA:** p99 response time must not exceed {sla_ms} ms.\n"
        f"- **Throughput:** System must handle at least {throughput} requests per second.\n"
        "- **Availability:** 99.9% uptime measured monthly.\n"
        "- **Authentication:** OAuth 2.0 or mutual TLS required for all endpoints.\n\n"
        "### Out of Scope\n\n"
        "Legacy integrations and batch-mode operations are not covered by this specification."
    )
    facts = {
        "topic": topic,
        "author": author,
        "version": version,
        "latency_sla_ms": sla_ms,
        "throughput_rps": throughput,
    }
    qa = [
        {
            "question": f"In the '{topic}' technical specification, what is the p99 latency SLA?",
            "answer": f"{sla_ms} ms",
            "assertions": [{"text": f"The response must state '{sla_ms} ms'."}],
        },
        {
            "question": f"In the '{topic}' specification, what is the minimum throughput requirement?",
            "answer": f"{throughput} RPS",
            "assertions": [{"text": f"The response must state '{throughput} RPS'."}],
        },
        {
            "question": f"In the '{topic}' specification, who is listed as the author?",
            "answer": author,
            "assertions": [{"text": f"The response must state '{author}'."}],
        },
    ]
    return text, {"key_facts": facts, "qa_candidates": qa}


def _bullet_list_block(rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    topic = _BULLET_TOPICS[int(rng.integers(0, len(_BULLET_TOPICS)))]
    n_items = int(rng.integers(4, 9))
    items = [f"Item {i+1}: Action point or milestone description here." for i in range(n_items)]
    text = f"## {topic}\n\n" + "\n".join(f"- {item}" for item in items)
    facts = {"topic": topic, "item_count": n_items, "items": items}
    qa = [
        {
            "question": f"How many items are in the '{topic}' list?",
            "answer": str(n_items),
            "assertions": [{"text": f"The response must state {n_items}."}],
        },
        {
            "question": f"In the '{topic}' list, what is the first item listed?",
            "answer": items[0],
            "assertions": [{"text": f"The response must mention '{items[0]}'."}],
        },
    ]
    return text, {"key_facts": facts, "qa_candidates": qa}


def _footnote_block(rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    topic = _SPEC_TOPICS[int(rng.integers(0, len(_SPEC_TOPICS)))]
    ref_count = int(rng.integers(2, 6))
    main_text = (
        f"The {topic} framework[^1] provides a comprehensive approach to managing "
        f"system requirements[^2]. Implementation details vary by context[^3]."
    )
    footnotes = [f"[^{i+1}]: Reference note {i+1} explaining term or citation." for i in range(ref_count)]
    text = main_text + "\n\n" + "\n".join(footnotes)
    facts = {"topic": topic, "footnote_count": ref_count}
    qa = [
        {
            "question": f"In the '{topic}' section, how many footnotes are referenced?",
            "answer": str(ref_count),
            "assertions": [{"text": f"The response must state {ref_count}."}],
        },
    ]
    return text, {"key_facts": facts, "qa_candidates": qa}


_TEXT_GENERATORS = [
    ("email", _email_block),
    ("policy", _policy_block),
    ("spec", _spec_block),
    ("bullet_list", _bullet_list_block),
    ("footnote", _footnote_block),
]


# ---------------------------------------------------------------------------
# Text perturbations
# ---------------------------------------------------------------------------

def _perturb_typos(text: str, rng: np.random.Generator) -> str:
    """Apply random character-swap typos while preserving line structure."""
    result_lines = []
    for line in text.split("\n"):
        words = line.split()
        for i, word in enumerate(words):
            if len(word) > 4 and rng.random() < 0.04:
                pos = int(rng.integers(1, len(word) - 1))
                chars = list(word)
                chars[pos], chars[pos - 1] = chars[pos - 1], chars[pos]
                words[i] = "".join(chars)
        result_lines.append(" ".join(words))
    return "\n".join(result_lines)


def _perturb_abbreviations(text: str, rng: np.random.Generator) -> str:
    replacements = {
        "approximately": "approx.",
        "requirements": "reqs.",
        "document": "doc.",
        "organization": "org.",
        "information": "info.",
    }
    for full, abbrev in replacements.items():
        if rng.random() < 0.5:
            text = text.replace(full, abbrev)
    return text


def _perturb_unicode_punctuation(text: str, rng: np.random.Generator) -> str:
    if rng.random() < 0.5:
        text = text.replace('"', '\u201c').replace('"', '\u201d')
    if rng.random() < 0.5:
        text = text.replace("'", '\u2019')
    if rng.random() < 0.3:
        text = text.replace("...", '\u2026')
    return text


def _perturb_date_formats(text: str, rng: np.random.Generator) -> str:
    """Randomly reformat ISO dates to other common styles."""
    formats = [
        lambda y, m, d: f"{m:02d}/{d:02d}/{y:04d}",
        lambda y, m, d: f"{d:02d}-{m:02d}-{y:04d}",
        lambda y, m, d: f"{y:04d}/{m:02d}/{d:02d}",
    ]
    def replace_date(match: re.Match) -> str:
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        fmt = formats[int(rng.integers(0, len(formats)))]
        return fmt(y, m, d)

    return re.sub(r"(\d{4})-(\d{2})-(\d{2})", replace_date, text)


_TEXT_PERTURBATIONS = [
    ("typos", _perturb_typos),
    ("abbreviations", _perturb_abbreviations),
    ("unicode_punctuation", _perturb_unicode_punctuation),
    ("date_formats", _perturb_date_formats),
]


# ---------------------------------------------------------------------------
# Annotation generators (Markdown-visible)
# ---------------------------------------------------------------------------

def _callout_annotation(rng: np.random.Generator, target_text: str) -> tuple[str, dict[str, Any]]:
    types = ["NOTE", "WARNING", "TIP", "IMPORTANT"]
    ann_type = types[int(rng.integers(0, len(types)))]
    md = f"> **{ann_type}:** {target_text}"
    return md, {
        "annotation_type": "callout",
        "subtype": ann_type,
        "intent": f"Highlight important information as {ann_type}",
        "target_text": target_text,
    }


def _highlight_annotation(rng: np.random.Generator, target_text: str) -> tuple[str, dict[str, Any]]:
    md = f"==**{target_text}**=="
    return md, {
        "annotation_type": "highlight",
        "intent": "Draw visual attention to key phrase",
        "target_text": target_text,
    }


def _inline_comment_annotation(rng: np.random.Generator, target_text: str) -> tuple[str, dict[str, Any]]:
    comments = [
        "Verify this claim.",
        "Check with legal.",
        "Needs clarification.",
        "Update before release.",
        "TODO: confirm numbers.",
    ]
    comment = comments[int(rng.integers(0, len(comments)))]
    md = f"{target_text} <!-- {comment} -->"
    return md, {
        "annotation_type": "inline_comment",
        "intent": "Add reviewer note",
        "comment": comment,
        "target_text": target_text,
    }


def _footnote_annotation(rng: np.random.Generator, target_text: str) -> tuple[str, dict[str, Any]]:
    ref_id = int(rng.integers(10, 99))
    note = "Additional context or citation for this term."
    md = f"{target_text}[^{ref_id}]\n\n[^{ref_id}]: {note}"
    return md, {
        "annotation_type": "footnote",
        "intent": "Provide supplementary reference",
        "footnote_text": note,
        "target_text": target_text,
    }


_ANNOTATION_GENERATORS = [
    ("callout", _callout_annotation),
    ("highlight", _highlight_annotation),
    ("inline_comment", _inline_comment_annotation),
    ("footnote", _footnote_annotation),
]

_ANNOTATION_TARGETS = [
    "This section requires immediate attention.",
    "All deadlines are subject to change.",
    "Confidential — do not distribute.",
    "Version 1.0 — draft for review.",
    "Performance metrics are approximate.",
]


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

_SECTION_HEADINGS = [
    "Introduction",
    "Background",
    "Key Findings",
    "Methodology",
    "Results",
    "Discussion",
    "Recommendations",
    "Appendix",
    "Executive Summary",
    "Risk Assessment",
    "Scope and Objectives",
    "Technical Overview",
    "Financial Analysis",
    "Regulatory Context",
    "Implementation Plan",
    "Conclusion",
    "Data Sources",
    "Limitations",
    "Future Work",
    "Stakeholder Analysis",
]


def generate_llm_text_variants(
    client: "AzureOpenAIClient",
    topic: "TopicSpec",
    rng: np.random.Generator,
    count: int,
    outline: list[str] | None = None,
    max_workers: int = 1,
    section_briefs: "list[dict] | None" = None,
) -> list[Variant]:
    """Generate *count* text block variants with LLM-generated prose for the given topic.

    Each variant is a full document section: '## Heading' followed by 3-5 paragraphs.
    When max_workers > 1 all sections are generated concurrently (I/O-bound API calls).

    section_briefs (optional): list of brief dicts from generate_document_brief().
      Each dict may contain 'heading', 'key_facts', and 'data_references'.
      When provided, the section prose is constrained to reference those specific facts,
      making the document coherent across all artifacts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from synthpdf.llm.content_generator import ContentGenerator

    generator = ContentGenerator(client, topic)
    perturb_names = [p[0] for p in _TEXT_PERTURBATIONS]

    # Pre-compute all section headings so each section's prompt can list the
    # sibling headings and avoid repeating their content.
    # Brief headings take priority over outline, which takes priority over random.
    all_headings: list[str] = [
        (section_briefs[i]["heading"] if (section_briefs and i < len(section_briefs) and section_briefs[i].get("heading")) else None)
        or (outline[i] if (outline and i < len(outline)) else None)
        or _SECTION_HEADINGS[int(fork_rng(rng, f"llm_text_{i}").integers(0, len(_SECTION_HEADINGS)))]
        for i in range(count)
    ]

    def _generate_one(i: int) -> Variant:
        child_rng = fork_rng(rng, f"llm_text_{i}")
        heading = all_headings[i]
        sibling_headings = [all_headings[j] for j in range(count) if j != i]
        word_count = int(child_rng.integers(150, 351))

        # Extract brief-derived constraints for this section
        brief = section_briefs[i] if (section_briefs and i < len(section_briefs)) else {}
        key_facts: list[str] = brief.get("key_facts", []) if isinstance(brief, dict) else []
        data_references: list[str] = brief.get("data_references", []) if isinstance(brief, dict) else []

        t0 = time.perf_counter()
        body = generator.generate_text_section(
            heading, word_count,
            other_headings=sibling_headings,
            key_facts=key_facts if key_facts else None,
            data_references=data_references if data_references else None,
        )
        prose_time = time.perf_counter() - t0

        # Strip any spurious leading heading the LLM may have added despite instructions
        body = re.sub(r"^#+\s.*\n+", "", body.lstrip(), count=1)
        # Prepend heading as a bold label at normal font size (not h2 — avoids large bold blobs)
        text = f"**{heading}**\n\n{body.lstrip()}"

        n_perturb = int(child_rng.integers(0, 3))
        chosen = list(child_rng.choice(len(perturb_names), size=min(n_perturb, len(perturb_names)), replace=False))
        applied = []
        for p_idx in chosen:
            p_name, p_fn = _TEXT_PERTURBATIONS[p_idx]
            text = p_fn(text, child_rng)
            applied.append(p_name)

        t0 = time.perf_counter()
        qa_candidates = generator.generate_qa_candidates(text, content_type="text", n=3)
        qa_time = time.perf_counter() - t0

        click.echo(
            f"      text [{i+1}/{count}] {heading[:45]!r}: "
            f"prose({word_count}w)->{prose_time:.0f}s | QA({len(qa_candidates)})->{qa_time:.0f}s"
        )
        try:
            from synthpdf import stats_sink as _ss
            _ss.log_artifact(
                type="text", index=i + 1, total=count,
                caption=heading[:80], words=word_count,
                prose_time_s=round(prose_time, 1),
                qa_count=len(qa_candidates), qa_time_s=round(qa_time, 1),
            )
        except Exception:
            pass
        return Variant(
            id=f"text_{i:04d}_{_uid()[:8]}",
            kind="text",
            metadata={
                "text_type": "llm_prose",
                "heading": heading,
                "perturbations": applied,
                "char_count": len(text),
                "topic_id": topic.id,
            },
            render_payload={"markdown": text},
            ground_truth={
                "heading": heading,
                "word_count": word_count,
                "text_content": text,
                "key_facts": {"heading": heading, "topic_id": topic.id},
                "qa_candidates": qa_candidates,
                "text_type": "llm_prose",
                "perturbations": applied,
            },
        )

    if count == 0:
        return []
    results: list[Variant | None] = [None] * count
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, count)) as executor:
            future_to_idx = {executor.submit(_generate_one, i): i for i in range(count)}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
    else:
        for i in range(count):
            results[i] = _generate_one(i)
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Per-block-type style-example strings (used as few-shot phrasing inspiration)
# ---------------------------------------------------------------------------

_BLOCK_STYLE_EXAMPLES: dict[str, list[str]] = {
    "email": [
        "In the email with subject '{subject}', who is the sender?",
        "What date was the email with subject '{subject}' sent?",
        "Who is the recipient of the email with subject '{subject}'?",
    ],
    "policy": [
        "What is the effective date of the '{topic}' policy?",
        "Who is the policy owner for the '{topic}' policy?",
        "What version number does the '{topic}' policy document show?",
    ],
    "spec": [
        "In the '{topic}' technical specification, what is the p99 latency SLA?",
        "In the '{topic}' specification, what is the minimum throughput requirement?",
        "In the '{topic}' specification, who is listed as the author?",
    ],
    "bullet_list": [
        "How many items are in the '{topic}' list?",
        "In the '{topic}' list, what is the first item listed?",
    ],
    "footnote": [
        "In the '{topic}' section, how many footnotes are referenced?",
    ],
}


def generate_llm_structured_text_block_variants(
    client: "AzureOpenAIClient",
    topic: "TopicSpec",
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,  # noqa: F821
) -> list[Variant]:
    """Generate diverse structured text block variants using LLM for content and QnA.

    Replaces the static-template path (email/policy/spec/bullet_list/footnote)
    with LLM-generated content tailored to *topic*.  QnA is also LLM-generated
    via ``generate_element_qa()``, producing varied question styles across documents.

    Falls back to the static generator for any block where the LLM call fails.
    """
    from synthpdf.llm.content_generator import ContentGenerator

    gen = ContentGenerator(client, topic)
    perturb_names = [p[0] for p in _TEXT_PERTURBATIONS]
    block_cycle = ["email", "policy", "spec", "bullet_list", "footnote"]
    variants: list[Variant] = []

    for i in range(count):
        child_rng = fork_rng(rng, f"llm_text_block_{i}")
        block_type = block_cycle[i % len(block_cycle)]

        # Try LLM-generated content; fall back to static generator
        result = gen.generate_text_block(block_type)
        if result is not None:
            text, key_facts = result
        else:
            # Fallback: use static generator
            static_fn = dict(_TEXT_GENERATORS).get(block_type)
            if static_fn is None:
                static_fn = _TEXT_GENERATORS[i % len(_TEXT_GENERATORS)][1]
                block_type = _TEXT_GENERATORS[i % len(_TEXT_GENERATORS)][0]
            text, gt_extra = static_fn(child_rng)
            key_facts = gt_extra.get("key_facts", {})

        # Apply 0-2 perturbations
        n_perturb = int(child_rng.integers(0, 3))
        chosen = list(
            child_rng.choice(len(perturb_names), size=min(n_perturb, len(perturb_names)), replace=False)
        )
        applied = []
        for p_idx in chosen:
            p_name, p_fn = _TEXT_PERTURBATIONS[p_idx]
            text = p_fn(text, child_rng)
            applied.append(p_name)

        # LLM-generated QnA using few-shot style examples for this block type
        raw_examples = _BLOCK_STYLE_EXAMPLES.get(block_type, [])
        # Instantiate style examples with actual key_facts values where possible
        style_examples = []
        for ex in raw_examples:
            try:
                style_examples.append(ex.format(**key_facts))
            except (KeyError, ValueError):
                style_examples.append(ex)

        qa = gen.generate_element_qa(block_type, key_facts, style_examples, count=3) or []

        # If LLM QnA failed, derive simple fallback from key_facts
        if not qa and key_facts:
            anchor = key_facts.get("subject") or key_facts.get("topic") or block_type
            first_key = next(iter(key_facts))
            qa = [{
                "question": f"In the '{anchor}' {block_type}, what is the {first_key}?",
                "answer": str(key_facts[first_key]),
                "assertions": [{"text": f"The response must state '{key_facts[first_key]}'."}],
            }]

        variants.append(Variant(
            id=f"text_{i:04d}_{_uid()[:8]}",
            kind="text",
            metadata={
                "text_type": block_type,
                "perturbations": applied,
                "char_count": len(text),
                "topic_id": getattr(topic, "id", ""),
            },
            render_payload={"markdown": text},
            ground_truth={
                "text_type": block_type,
                "key_facts": key_facts,
                "perturbations": applied,
                "qa_candidates": qa,
            },
        ))

    return variants


def generate_text_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,  # noqa: F821
) -> list[Variant]:
    """Generate *count* text block variants."""
    variants: list[Variant] = []
    perturb_names = [p[0] for p in _TEXT_PERTURBATIONS]

    for i in range(count):
        child_rng = fork_rng(rng, f"text_{i}")
        text_name, text_fn = _TEXT_GENERATORS[i % len(_TEXT_GENERATORS)]
        text, gt_extra = text_fn(child_rng)

        # Apply 0-2 perturbations
        n_perturb = int(child_rng.integers(0, 3))
        chosen = list(child_rng.choice(len(perturb_names), size=min(n_perturb, len(perturb_names)), replace=False))
        applied = []
        for p_idx in chosen:
            p_name, p_fn = _TEXT_PERTURBATIONS[p_idx]
            text = p_fn(text, child_rng)
            applied.append(p_name)

        variant = Variant(
            id=f"text_{i:04d}_{_uid()[:8]}",
            kind="text",
            metadata={
                "text_type": text_name,
                "perturbations": applied,
                "char_count": len(text),
            },
            render_payload={
                "markdown": text,
            },
            ground_truth={
                **gt_extra,
                "text_type": text_name,
                "perturbations": applied,
            },
        )
        variants.append(variant)

    return variants


def generate_annotation_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: "Path | None" = None,  # noqa: F821
) -> list[Variant]:
    """Generate *count* Markdown-visible annotation variants."""
    variants: list[Variant] = []

    for i in range(count):
        child_rng = fork_rng(rng, f"annotation_{i}")
        ann_name, ann_fn = _ANNOTATION_GENERATORS[i % len(_ANNOTATION_GENERATORS)]
        target = _ANNOTATION_TARGETS[int(child_rng.integers(0, len(_ANNOTATION_TARGETS)))]
        md, ann_gt = ann_fn(child_rng, target)

        variant = Variant(
            id=f"ann_{i:04d}_{_uid()[:8]}",
            kind="annotation",
            metadata={
                "annotation_type": ann_name,
                "target_text": target,
            },
            render_payload={
                "markdown": md,
            },
            ground_truth=ann_gt,
        )
        variants.append(variant)

    return variants
