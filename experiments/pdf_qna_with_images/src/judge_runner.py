# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""LLM judge calls for direct-PDF QA predictions."""
from __future__ import annotations

import json
import re
import time
from typing import Any

import openai

from judge_config import JudgeSpec


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
_PARAM_QUIRKS: dict[str, set[str]] = {}


def extract_gt_answer(assertion_text: str) -> str:
    """Pull the expected value out of 'The response must state X.'."""
    match = _GT_PATTERN.search(assertion_text.strip())
    return match.group(1).strip() if match else assertion_text.strip()


def _chat_kwargs(spec: JudgeSpec, messages: list[dict[str, str]]) -> dict[str, Any]:
    quirks = _PARAM_QUIRKS.get(spec.deployment, set())
    kwargs: dict[str, Any] = {"model": spec.deployment, "messages": messages}
    token_key = "max_completion_tokens" if "completion_tokens" in quirks else "max_tokens"
    kwargs[token_key] = spec.max_tokens
    if "drop_temperature" not in quirks:
        kwargs["temperature"] = spec.temperature
    return kwargs


def judge_answer(
    client: Any,
    spec: JudgeSpec,
    question: str,
    assertion: str,
    predicted: str,
    label: str = "",
) -> dict[str, Any]:
    """Evaluate one predicted answer with one configured judge."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _JUDGE_USER_TEMPLATE.format(
                question=question,
                gt_answer=extract_gt_answer(assertion),
                assertion=assertion,
                predicted=predicted or "(no answer provided)",
            ),
        },
    ]
    if spec.api_mode == "responses":
        kwargs: dict[str, Any] = {
            "model": spec.deployment,
            "input": "\n\n".join(message["content"] for message in messages),
        }
    else:
        kwargs = _chat_kwargs(spec, messages)

    for attempt in range(3):
        try:
            started = time.perf_counter()
            if spec.api_mode == "responses":
                response = client.responses.create(**kwargs)
                raw = (response.output_text or "").strip()
            else:
                response = client.chat.completions.create(**kwargs)
                raw = (response.choices[0].message.content or "").strip()
            result = parse_judge_response(raw)
            result["error"] = None
            result["judge_time"] = round(time.perf_counter() - started, 3)
            return result
        except openai.BadRequestError as error:
            message = str(error).lower()
            if "temperature" in message and "temperature" in kwargs:
                _PARAM_QUIRKS.setdefault(spec.deployment, set()).add("drop_temperature")
                kwargs.pop("temperature")
                continue
            if "max_tokens" in message and "max_tokens" in kwargs:
                _PARAM_QUIRKS.setdefault(spec.deployment, set()).add("completion_tokens")
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                continue
            return error_judge(str(error))
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            return error_judge(str(error))
        except Exception as error:
            if attempt < 2:
                time.sleep(4 * (attempt + 1))
            else:
                return error_judge(f"{label}: {error}")
    return error_judge("max retries exceeded")


def parse_judge_response(text: str) -> dict[str, Any]:
    """Extract and normalize a JSON judge response, with graceful fallback."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    obj: dict[str, Any] = {}
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    verdict = str(obj.get("verdict", "")).upper()
    if verdict not in {"PASS", "PARTIAL", "FAIL"}:
        lowered = text.lower()
        if "pass" in lowered and "fail" not in lowered and "partial" not in lowered:
            verdict = "PASS"
        elif "partial" in lowered:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

    quality = str(obj.get("quality", "fair")).lower()
    if quality not in {"excellent", "good", "fair", "poor"}:
        quality = "fair"
    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    missing = obj.get("missing_aspects", [])
    if not isinstance(missing, list):
        missing = [str(missing)]
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": str(obj.get("reasoning", text[:400])),
        "missing_aspects": missing,
        "quality": quality,
    }


def error_judge(message: str) -> dict[str, Any]:
    return {
        "verdict": "ERROR",
        "confidence": 0.0,
        "reasoning": f"Judge call failed: {message}",
        "missing_aspects": [],
        "quality": "error",
        "error": message,
        "judge_time": 0.0,
    }
