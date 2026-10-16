# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Judge model runner: evaluate (predicted answer + GT assertion) -> PASS/FAIL.

The judge receives:
  - question         : the original document question
  - gt_assertion     : the ground-truth assertion from the quest file
                       (e.g. "The response must state 60.0m (2026-08-27).")
  - gt_answer        : the expected answer string (from manifest qa_candidates),
                       or extracted from the assertion if not available
  - predicted_answer : the QA model's response

It replies PASS / FAIL + a one-line reason.

Note: the judge does NOT see the PDF.  It is evaluating whether the model's
answer is factually correct relative to the ground-truth, not whether it can
read the document.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from .config import ModelSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict factual evaluator assessing AI model answers to PDF document questions. "
    "You will be given the original question, the ground-truth expected answer, "
    "a ground-truth assertion that the answer must satisfy, "
    "and the model's predicted answer. "
    "Your task: determine whether the predicted answer satisfies the assertion. "
    "Reply with exactly PASS or FAIL on the first line, "
    "then one concise sentence explaining your decision."
)

_JUDGE_PROMPT = """\
QUESTION:
{question}

GROUND TRUTH EXPECTED ANSWER:
{gt_answer}

GROUND TRUTH ASSERTION (what the predicted answer must satisfy):
{gt_assertion}

MODEL'S PREDICTED ANSWER:
{predicted}

---
Evaluation criteria:
- PASS: The predicted answer contains the required information and satisfies the assertion.
  Minor phrasing differences or additional context are acceptable as long as the key
  facts are present and correct.
- FAIL: The predicted answer is missing the required information, states incorrect values,
  is too vague to verify, or contradicts the assertion.

Respond with:
PASS or FAIL
<one sentence reason>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_gt_answer_from_assertion(assertion: str) -> str:
    """Best-effort: pull the quoted or post-colon value from the assertion text.

    Example: "The response must state 60.0m (2026-08-27)."
         ->  "60.0m (2026-08-27)"
    Falls back to the full assertion string if nothing can be extracted.
    """
    m = re.search(r"must (?:state|say|contain|include)\s+(.+?)\.?\s*$", assertion, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return assertion.strip()


def _parse_verdict(text: str) -> str:
    """Return 'PASS', 'FAIL', or 'UNCLEAR' from the judge response."""
    first_line = text.split("\n")[0].upper()
    if "PASS" in first_line:
        return "PASS"
    if "FAIL" in first_line:
        return "FAIL"
    # Fallback: search full response
    if re.search(r"\bPASS\b", text.upper()):
        return "PASS"
    if re.search(r"\bFAIL\b", text.upper()):
        return "FAIL"
    return "UNCLEAR"


def _parse_reason(text: str) -> str:
    """Return the reason line(s) from the judge response (everything after line 1)."""
    lines = [l.strip() for l in text.split("\n")[1:] if l.strip()]
    return " ".join(lines)[:400]


# ---------------------------------------------------------------------------
# Main judge call
# ---------------------------------------------------------------------------

def judge_answer(
    client: Any,
    spec: ModelSpec,
    question: str,
    gt_assertion: str,
    predicted: str,
    gt_answer: Optional[str] = None,
    label: str = "",
) -> dict:
    """Evaluate one (predicted answer, assertion) pair.

    Returns a dict::

        {
            "result":  "PASS" | "FAIL" | "UNCLEAR" | "ERROR",
            "reason":  "<one-sentence explanation>",
        }
    """
    import openai

    # If no explicit gt_answer was supplied, extract it from the assertion text
    if not gt_answer:
        gt_answer = _extract_gt_answer_from_assertion(gt_assertion)

    prompt = _JUDGE_PROMPT.format(
        question=question,
        gt_answer=gt_answer,
        gt_assertion=gt_assertion,
        predicted=predicted[:3000],   # cap to avoid context explosion
    )

    kwargs: dict[str, Any] = {
        "model": spec.deployment,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": 300,
        "temperature": 0.0,
    }

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            content = (response.choices[0].message.content or "").strip()
            logger.debug("Judge %s [%s] %.1fs", spec.name, label, elapsed)
            return {
                "result": _parse_verdict(content),
                "reason": _parse_reason(content),
            }

        except openai.BadRequestError as exc:
            msg = str(exc).lower()
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            logger.error("Judge BadRequest [%s]: %s", label, exc)
            return {"result": "ERROR", "reason": str(exc)[:200]}

        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("Judge connection error, retrying in %ds [%s]", wait, label)
                time.sleep(wait)
            else:
                return {"result": "ERROR", "reason": str(exc)[:200]}

        except openai.RateLimitError as exc:
            if attempt < max_retries:
                wait = 15 * (attempt + 1)
                logger.warning("Judge rate limit, retrying in %ds [%s]", wait, label)
                time.sleep(wait)
            else:
                return {"result": "ERROR", "reason": str(exc)[:200]}

        except Exception as exc:
            logger.error("Judge call failed [%s]: %s", label, exc)
            return {"result": "ERROR", "reason": str(exc)[:200]}

    return {"result": "ERROR", "reason": "max retries exceeded"}
