# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""QA model runner: send (PDF + question) -> predicted answer.

Supports two model client types:
  azure_openai -- openai.AzureOpenAI  (GPT deployments on Azure OpenAI Service)
  serverless   -- openai.OpenAI with custom base_url (Claude, Mistral, Phi, LLaMA,
                  MAI, and other Azure AI Foundry serverless endpoints)

PDF is presented either as a sequence of base64 PNG images (vision models)
or as extracted text (text-only models), depending on the model spec.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .config import ModelSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_QA_SYSTEM = (
    "You are an expert document analyst. "
    "You will be shown a PDF document and asked a question about it. "
    "Answer using only the information present in the document. "
    "Be concise and precise. "
    "If the answer cannot be determined from the document, say so explicitly."
)

_QA_QUESTION_SUFFIX = (
    "\n\nBased on the document above, answer the following question precisely:\n\n{question}"
)

_QA_TEXT_PREFIX = (
    "DOCUMENT CONTENT (extracted from PDF):\n"
    "---\n"
    "{text}\n"
    "---"
)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_client(spec: ModelSpec) -> Any:
    """Return an openai client (AzureOpenAI or OpenAI) for the given spec.

    Authentication priority (same as the existing AzureOpenAIClient):
      1. API key from environment variable named by spec.api_key_env
      2. Azure AD / DefaultAzureCredential (azure-identity must be installed)
    """
    import httpx
    import openai

    http_client = None
    if not spec.verify_ssl:
        http_client = httpx.Client(verify=False)

    common: dict[str, Any] = {"max_retries": 3, "timeout": 180.0}
    if http_client is not None:
        common["http_client"] = http_client

    api_key = spec.api_key

    if spec.client_type == "azure_openai":
        if api_key:
            return openai.AzureOpenAI(
                azure_endpoint=spec.endpoint,
                api_version=spec.api_version,
                api_key=api_key,
                **common,
            )
        # Fall back to Azure AD token
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:
            raise ImportError(
                "No API key found and 'azure-identity' is not installed. "
                f"Set the {spec.api_key_env} environment variable or "
                "install azure-identity: pip install azure-identity"
            ) from exc
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        return openai.AzureOpenAI(
            azure_endpoint=spec.endpoint,
            api_version=spec.api_version,
            azure_ad_token_provider=token_provider,
            **common,
        )

    # Serverless: Azure AI Foundry / OpenAI-compatible endpoint
    # Append /v1 if the endpoint doesn't already include it, since the
    # openai SDK appends /chat/completions to the base_url.
    base_url = spec.endpoint.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    return openai.OpenAI(
        base_url=base_url,
        api_key=api_key or "placeholder",  # some endpoints use Bearer token separately
        **common,
    )


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _image_messages(page_images: list[str], question: str) -> list[dict]:
    """Build multimodal user message: one image per page + question text."""
    content: list[dict] = []
    for b64 in page_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        })
    content.append({
        "type": "text",
        "text": _QA_QUESTION_SUFFIX.format(question=question),
    })
    return [{"role": "user", "content": content}]


def _text_messages(pdf_text: str, question: str) -> list[dict]:
    """Build plain-text user message with extracted PDF content."""
    # Cap text to avoid excessive context; keep the first 12 000 chars which
    # covers ~8-10 dense pages and fits within most models' context windows.
    truncated = pdf_text[:12_000]
    if len(pdf_text) > 12_000:
        truncated += "\n\n[... document truncated ...]"
    body = _QA_TEXT_PREFIX.format(text=truncated) + _QA_QUESTION_SUFFIX.format(question=question)
    return [{"role": "user", "content": body}]


# ---------------------------------------------------------------------------
# Main call
# ---------------------------------------------------------------------------

def run_qa(
    client: Any,
    spec: ModelSpec,
    page_images: Optional[list[str]],
    pdf_text: Optional[str],
    question: str,
    label: str = "",
) -> Optional[str]:
    """Send (PDF content + question) to the model and return the answer.

    Returns None on any unrecoverable error.
    """
    import openai

    if spec.input_mode == "images" and page_images:
        messages = _image_messages(page_images, question)
    elif pdf_text:
        messages = _text_messages(pdf_text, question)
    else:
        logger.error("No PDF content available for QA call [%s]", label)
        return None

    all_messages = [{"role": "system", "content": _QA_SYSTEM}] + messages

    kwargs: dict[str, Any] = {
        "model": spec.deployment,
        "messages": all_messages,
        "max_completion_tokens": spec.max_tokens,
        "temperature": spec.temperature,
    }

    return _call_with_retries(client, kwargs, spec, label)


def _call_with_retries(
    client: Any,
    kwargs: dict[str, Any],
    spec: ModelSpec,
    label: str,
    max_retries: int = 3,
) -> Optional[str]:
    """Execute the chat completion with automatic retry on transient errors."""
    import openai

    rejected_params: set[str] = set()

    for attempt in range(max_retries + 1):
        # Remove params that this model has already rejected
        call_kwargs = {k: v for k, v in kwargs.items() if k not in rejected_params}

        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(**call_kwargs)
            elapsed = time.perf_counter() - t0
            usage = response.usage
            tok = (
                f"prompt={usage.prompt_tokens} completion={usage.completion_tokens}"
                if usage else "tokens=unknown"
            )
            logger.info("  QA %s [%s] %.1fs — %s", spec.name, label, elapsed, tok)
            content = response.choices[0].message.content
            if not content:
                logger.warning("Empty response from %s [%s]", spec.name, label)
                return None
            return content.strip()

        except openai.BadRequestError as exc:
            msg = str(exc).lower()
            if "temperature" in msg and "temperature" in call_kwargs:
                logger.warning(
                    "Model %r does not accept 'temperature'; dropping it [%s]",
                    spec.deployment, label,
                )
                rejected_params.add("temperature")
                continue  # retry immediately without temperature
            if "response_format" in msg and "response_format" in call_kwargs:
                rejected_params.add("response_format")
                continue
            logger.error("QA BadRequest %s [%s]: %s", spec.name, label, exc)
            return None

        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "Connection error attempt %d/%d, retrying in %ds [%s]: %s",
                    attempt + 1, max_retries + 1, wait, label, exc,
                )
                time.sleep(wait)
            else:
                logger.error("All retries exhausted for %s [%s]: %s", spec.name, label, exc)
                return None

        except openai.RateLimitError as exc:
            if attempt < max_retries:
                wait = 15 * (attempt + 1)
                logger.warning(
                    "Rate limit attempt %d/%d, retrying in %ds [%s]",
                    attempt + 1, max_retries + 1, wait, label,
                )
                time.sleep(wait)
            else:
                logger.error("Rate limit not resolved for %s [%s]: %s", spec.name, label, exc)
                return None

        except Exception as exc:
            logger.error("QA call failed for %s [%s]: %s", spec.name, label, exc)
            return None

    return None
