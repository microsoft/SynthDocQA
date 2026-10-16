# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
Judge model runner.

Evaluates a predicted answer against the ground truth assertion and returns
structured metrics (PASS / PARTIAL / FAIL + confidence, quality, reasoning).
Uses the same shared AzureOpenAI client as the QA runner.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import openai

from .config import ModelSpec
from .qa_runner import _record_quirk, build_call_kwargs

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for document question-answering systems. "
    "Evaluate the predicted answer strictly and objectively based on how well "
    "it satisfies the assertion and matches the ground truth answer."
)

_JUDGE_USER_TEMPLATE = """\
Question: {question}

Ground truth answer: {gt_answer}

Assertion the answer must satisfy: {assertion}

Predicted answer from QA model: {predicted}

Evaluate the predicted answer and respond ONLY with a valid JSON object \
(no markdown fences) using this exact schema:
{{
  "verdict": "PASS" | "PARTIAL" | "FAIL",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<concise explanation, 1-3 sentences>",
  "missing_aspects": [<list of missing items, empty list if PASS>],
  "quality": "excellent" | "good" | "fair" | "poor"
}}

Verdict definitions:
  PASS    - predicted answer fully satisfies the assertion and matches the ground truth
  PARTIAL - some correct information but incomplete or imprecise
  FAIL    - does not satisfy the assertion or contradicts the ground truth
"""

_GT_PATTERN = re.compile(
    r"[Tt]he\s+response\s+must\s+state\s+(.+?)\.?\s*$", re.DOTALL
)


def extract_gt_answer(assertion_text: str) -> str:
    """Pull the expected value out of 'The response must state X.'"""
    m = _GT_PATTERN.search(assertion_text.strip())
    return m.group(1).strip() if m else assertion_text.strip()


def judge_answer(
    client: Any,
    spec: ModelSpec,
    question: str,
    assertion: str,
    predicted: str,
    label: str = "",
) -> dict[str, Any]:
    """
    Call the judge model to evaluate a predicted answer.

    Returns:
        verdict          str   - "PASS" | "PARTIAL" | "FAIL" | "ERROR"
        confidence       float - 0.0 to 1.0
        reasoning        str
        missing_aspects  list[str]
        quality          str   - "excellent" | "good" | "fair" | "poor" | "error"
        error            str | None
        judge_time       float
    """
    gt_answer = extract_gt_answer(assertion)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _JUDGE_USER_TEMPLATE.format(
                question=question,
                gt_answer=gt_answer,
                assertion=assertion,
                predicted=predicted or "(no answer provided)",
            ),
        },
    ]

    # Luna and other AI Foundry models can require the Responses API. Keep the
    # established chat-completions path unchanged for all other deployments.
    if spec.api_mode == "responses":
        kwargs = {
            "model": spec.deployment,
            "input": "\n\n".join(message["content"] for message in messages),
        }
    else:
        # Use the shared per-deployment quirk cache — once we've learned that a
        # deployment needs max_completion_tokens or no temperature, every subsequent
        # call (including judge calls) starts with the correct params immediately.
        kwargs = build_call_kwargs(
            spec.deployment, messages, spec.max_tokens, spec.temperature
        )

    for attempt in range(3):
        try:
            t0 = time.time()
            if spec.api_mode == "responses":
                resp = client.responses.create(**kwargs)
                raw = (resp.output_text or "").strip()
            else:
                resp = client.chat.completions.create(**kwargs)
                raw = (resp.choices[0].message.content or "").strip()
            elapsed = time.time() - t0
            result = _parse_judge_response(raw)
            result["error"] = None
            result["judge_time"] = round(elapsed, 3)
            return result

        except openai.BadRequestError as exc:
            err_str = str(exc)
            if "temperature" in err_str and "temperature" in kwargs:
                logger.warning("%s judge: dropping temperature (cached for deployment)", label)
                _record_quirk(spec.deployment, "drop_temperature")
                kwargs.pop("temperature")
                continue
            if "max_tokens" in err_str and "max_tokens" in kwargs:
                logger.warning("%s judge: max_tokens -> max_completion_tokens (cached for deployment)", label)
                _record_quirk(spec.deployment, "use_max_completion_tokens")
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                continue
            # Any other 400 — hard failure, no retry.
            logger.error("%s judge BadRequestError: %s", label, exc)
            return _err_judge(str(exc))

        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            # Authentication/RBAC failures are permanent for the current
            # identity. Retrying the same request only delays the run and can
            # flood every result with the same error.
            logger.error("%s judge authorization failed: %s", label, exc)
            return _err_judge(str(exc))

        except Exception as exc:
            if attempt < 2:
                wait = 4 * (attempt + 1)
                logger.warning(
                    "%s judge attempt %d failed (%s) — retrying in %ds",
                    label, attempt + 1, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("%s judge failed after 3 attempts: %s", label, exc)
                return _err_judge(str(exc))

    return _err_judge("max retries exceeded")


def _parse_judge_response(text: str) -> dict[str, Any]:
    """Extract and normalise a JSON judge response, with graceful fallback."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    obj: dict = {}
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    verdict = str(obj.get("verdict", "")).upper()
    if verdict not in ("PASS", "PARTIAL", "FAIL"):
        tl = text.lower()
        if "pass" in tl and "fail" not in tl and "partial" not in tl:
            verdict = "PASS"
        elif "partial" in tl:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

    quality = str(obj.get("quality", "fair")).lower()
    if quality not in ("excellent", "good", "fair", "poor"):
        quality = "fair"

    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": str(obj.get("reasoning", text[:400])),
        "missing_aspects": list(obj.get("missing_aspects", [])),
        "quality": quality,
    }


def _err_judge(msg: str) -> dict[str, Any]:
    return {
        "verdict": "ERROR",
        "confidence": 0.0,
        "reasoning": f"Judge call failed: {msg}",
        "missing_aspects": [],
        "quality": "error",
        "error": msg,
        "judge_time": 0.0,
    }
