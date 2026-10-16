# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
QA model runner.

Azure models share a single AzureOpenAI client (one endpoint from the .env file).
Claude models (deployment name contains "claude") are invoked via the local
claude CLI: `claude --model <name> -p`, with the prompt piped via stdin.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import httpx
import openai
from openai import AzureOpenAI

from .config import EvalConfig, ModelSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-deployment parameter quirk cache
# ---------------------------------------------------------------------------
# Maps deployment_name -> {"use_max_completion_tokens": bool, "drop_temperature": bool}
# Shared across threads so each deployment is probed at most once per process.
# After the first BadRequestError teaches us the right params, all subsequent
# calls to that deployment skip the failing params immediately.

_param_quirks: dict[str, dict[str, bool]] = {}
_param_quirks_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Shared bearer-token cache
# ---------------------------------------------------------------------------
# AzureCliCredential.get_token() shells out to `az account get-access-token`.
# When 15+ threads call it concurrently they all spawn parallel subprocesses
# and the CLI fails. This lock + cache ensures only one subprocess runs at a
# time and the resulting token is reused by all threads until near expiry.

_az_token_cache: dict[str, tuple[str, float]] = {}  # scope -> (token, expires_on)
_az_token_lock = threading.Lock()


def _get_bearer_token(cred: Any, scope: str) -> str:
    """Acquire a bearer token, serialising subprocess calls and caching the result."""
    with _az_token_lock:
        cached = _az_token_cache.get(scope)
        if cached:
            token, expires_on = cached
            if expires_on - time.time() > 300:   # reuse if >5 min remaining
                return token
        token_obj = cred.get_token(scope)
        expires_on = float(getattr(token_obj, "expires_on", time.time() + 3600))
        _az_token_cache[scope] = (token_obj.token, expires_on)
        return token_obj.token


def _record_quirk(deployment: str, key: str) -> None:
    with _param_quirks_lock:
        _param_quirks.setdefault(deployment, {})[key] = True


def build_call_kwargs(
    deployment: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Build API call kwargs using any cached quirks for this deployment."""
    with _param_quirks_lock:
        quirks = dict(_param_quirks.get(deployment, {}))
    kwargs: dict[str, Any] = {"model": deployment, "messages": messages}
    if quirks.get("use_max_completion_tokens"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if not quirks.get("drop_temperature"):
        kwargs["temperature"] = temperature
    return kwargs


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

_REFUSAL_PHRASES = (
    "not in the document",
    "not mentioned in the document",
    "not provided in the document",
    "not found in the document",
    "does not appear in the document",
    "not available in the document",
    "cannot find this information",
    "cannot find the answer",
    "i cannot answer",
    "i'm unable to answer",
    "i am unable to answer",
    "not stated in the document",
    "no information provided",
    "no information about",
    "the document does not",
    "document does not contain",
    "document does not mention",
    "document does not provide",
)


def _is_refusal(text: str) -> bool:
    """Return True if the model declined to answer or said the info isn't in the document."""
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


QA_SYSTEM_PROMPT = (
    "You are an expert document analyst. "
    "Read the document content carefully and answer the question precisely and concisely. "
    "Base your answer solely on the information present in the document. "
    "Do not speculate or add information not found in the document. "
    "If the answer is not in the document, say so explicitly."
)


# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------


def _is_serverless(endpoint: str) -> bool:
    """
    Serverless AI Foundry endpoints include /openai/v1/ in the path.
    They use the plain OpenAI-compatible API and do NOT accept an
    api-version query parameter.
    """
    return "/openai/v1" in endpoint


def _build_serverless_client(
    endpoint: str,
    use_entra_id: bool,
    tenant_id: str = "",
    auth_scope: str = "https://cognitiveservices.azure.com/.default",
) -> openai.OpenAI:
    """
    Build a plain openai.OpenAI client for serverless AI Foundry endpoints.

    AzureOpenAI always appends ?api-version=... which causes 404 on these
    endpoints.  Using openai.OpenAI with base_url avoids that.  Auth is
    injected via a custom httpx.Auth that fetches a fresh bearer token for
    every request (handles automatic token refresh).
    """
    if use_entra_id:
        try:
            from azure.identity import (
                AzureCliCredential,
                DefaultAzureCredential,
                get_bearer_token_provider,
            )
        except ImportError:
            raise RuntimeError(
                "azure-identity is not installed but USE_ENTRA_ID=true. "
                "Run: pip install azure-identity"
            )

        _scope = auth_scope
        if _scope == "https://ai.azure.com/.default":
            # AI Foundry's Responses API expects the OpenAI-compatible client
            # and bearer-token provider used by its documented authentication
            # pattern. In particular, Luna is hosted on an *.openai.azure.com
            # /openai/v1 endpoint and rejects Cognitive Services tokens.
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), _scope
            )
            return openai.OpenAI(base_url=endpoint, api_key=token_provider)

        # Use AzureCliCredential with explicit tenant_id — this passes --tenant to
        # `az account get-access-token` and avoids picking up a cached token from
        # a different (corporate) tenant. Fall back to DefaultAzureCredential when
        # no tenant is configured.
        _cred = (
            AzureCliCredential(tenant_id=tenant_id)
            if tenant_id
            else DefaultAzureCredential()
        )

        class _BearerAuth(httpx.Auth):
            def auth_flow(self, request: httpx.Request):
                token = _get_bearer_token(_cred, _scope)
                request.headers["Authorization"] = f"Bearer {token}"
                yield request

        return openai.OpenAI(
            base_url=endpoint,
            api_key="placeholder",          # required by SDK; overridden by _BearerAuth
            http_client=httpx.Client(auth=_BearerAuth()),
        )

    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("USE_ENTRA_ID=false but AZURE_OPENAI_API_KEY is not set.")
    return openai.OpenAI(base_url=endpoint, api_key=api_key)


def _build_one_client(
    endpoint: str,
    api_version: str,
    use_entra_id: bool,
    tenant_id: str = "",
    auth_scope: str = "https://cognitiveservices.azure.com/.default",
):
    """Build a client for one endpoint.

    Serverless AI Foundry endpoints (containing /openai/v1/) get a plain
    openai.OpenAI client — no api-version param, bearer token via httpx.Auth.
    Standard Azure OpenAI endpoints get an AzureOpenAI client.
    """
    if _is_serverless(endpoint):
        return _build_serverless_client(
            endpoint, use_entra_id, tenant_id, auth_scope
        )

    if use_entra_id:
        try:
            from azure.identity import AzureCliCredential, DefaultAzureCredential

            cred = (
                AzureCliCredential(tenant_id=tenant_id)
                if tenant_id
                else DefaultAzureCredential()
            )
            scope = "https://cognitiveservices.azure.com/.default"

            # AzureOpenAI may request a token concurrently from every worker.
            # Route these requests through the same process-wide lock/cache used
            # by serverless clients so only one Azure CLI subprocess is spawned.
            def token_provider() -> str:
                return _get_bearer_token(cred, scope)

            return AzureOpenAI(
                azure_ad_token_provider=token_provider,
                azure_endpoint=endpoint,
                api_version=api_version,
            )
        except ImportError:
            raise RuntimeError(
                "azure-identity is not installed but USE_ENTRA_ID=true. "
                "Run: pip install azure-identity"
            )

    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("USE_ENTRA_ID=false but AZURE_OPENAI_API_KEY is not set.")
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def build_clients(cfg: EvalConfig) -> dict[str, Any]:
    """
    Build one client per unique endpoint across all models + judges.

    Standard Azure OpenAI endpoints get an AzureOpenAI client.
    Serverless AI Foundry endpoints (/openai/v1/) get a plain openai.OpenAI client
    (no api-version query param, bearer token injected via httpx.Auth).
    Claude models are excluded — they use the local claude CLI instead.
    Returns a dict mapping endpoint URL -> client.
    """
    endpoints: dict[str, tuple[str, str]] = {}  # endpoint -> (api_version, auth_scope)
    for judge in cfg.judges:
        endpoints[judge.endpoint] = (judge.api_version, judge.auth_scope)
    for m in cfg.models:
        if not m.is_claude:
            endpoints.setdefault(m.endpoint, (m.api_version, m.auth_scope))

    use_entra_id = cfg.judges[0].use_entra_id  # same auth method for all Azure endpoints
    tenant_id = cfg.azure_tenant_id

    clients: dict[str, AzureOpenAI] = {}
    for ep, (ver, auth_scope) in endpoints.items():
        clients[ep] = _build_one_client(
            ep, ver, use_entra_id, tenant_id, auth_scope
        )
        logger.debug("Built Azure client for endpoint: %s", ep)

    return clients


# ---------------------------------------------------------------------------
# Claude CLI runner
# ---------------------------------------------------------------------------


def _resolve_claude_cmd() -> list[str]:
    """Return the command prefix for invoking the claude CLI.

    On Windows, claude is a .cmd wrapper and must be run through cmd.exe.
    """
    which = shutil.which("claude")
    if which is None:
        raise RuntimeError(
            "claude CLI not found in PATH. "
            "Install it with: npm install -g @anthropic-ai/claude-code"
        )
    if sys.platform == "win32" and which.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", which]
    return [which]


def _run_claude_cli(
    spec: ModelSpec,
    messages: list[dict[str, str]],
    label: str = "",
) -> dict[str, Any]:
    """
    Invoke the local claude CLI for a Claude model.

    Combines system + user messages into a single stdin prompt, then runs:
        claude --model <deployment> -p
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts = [m["content"] for m in messages if m["role"] == "user"]

    # Prepend system content into the prompt (most reliable across CLI versions)
    prompt = "\n\n".join(system_parts + user_parts)
    cmd = _resolve_claude_cmd() + ["--model", spec.deployment, "-p"]

    logger.debug("%s claude cmd=%s  prompt_len=%d", label, cmd, len(prompt))

    for attempt in range(3):
        try:
            t0 = time.time()
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=300,
            )
            elapsed = time.time() - t0

            if result.returncode != 0:
                err_msg = (
                    f"claude CLI exit {result.returncode}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
                # 429 from Claude CLI — skip immediately, no retry
                if "429" in err_msg or "rate limit" in err_msg.lower() or "overloaded" in err_msg.lower():
                    logger.warning("%s Claude rate limited — skipping", label)
                    return _skip("claude_rate_limit")
                if attempt < 2:
                    logger.warning("%s %s — retrying in %ds", label, err_msg, 4 * (attempt + 1))
                    time.sleep(4 * (attempt + 1))
                    continue
                return _err(err_msg)

            predicted = result.stdout.strip()
            out = {
                "predicted": predicted,
                "response_time": round(elapsed, 3),
                "prompt_tokens": 0,      # CLI does not expose token counts
                "completion_tokens": 0,
                "error": None,
            }
            if _is_refusal(predicted):
                out["model_refuse"] = True
            return out

        except subprocess.TimeoutExpired:
            return _err("Claude CLI timed out after 300s")

        except Exception as exc:
            return _err(str(exc))

    return _err("max retries exceeded")


# ---------------------------------------------------------------------------
# Unified QA entry point
# ---------------------------------------------------------------------------


def run_qa(
    clients: dict[str, Any],
    spec: ModelSpec,
    markdown_content: str,
    question: str,
    label: str = "",
) -> dict[str, Any]:
    """
    Ask one QA model a question given the markdown document as context.

    Routes to the Claude CLI for claude-* deployments.
    For Azure models, looks up the right client by spec.endpoint.

    Returns:
        predicted         str   - model answer
        response_time     float - seconds
        prompt_tokens     int   - 0 for Claude (CLI doesn't expose this)
        completion_tokens int   - 0 for Claude
        error             str | None
    """
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Document content:\n\n{markdown_content}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            ),
        },
    ]

    # --- Claude path ---
    if spec.is_claude:
        return _run_claude_cli(spec, messages, label)

    # --- Azure OpenAI path ---
    client = clients.get(spec.endpoint)
    if client is None:
        return _err(f"No Azure client found for endpoint: {spec.endpoint}")

    # Start with whatever params we know work for this deployment (may already
    # skip max_tokens or temperature if a prior call taught us to).
    kwargs = build_call_kwargs(spec.deployment, messages, spec.max_tokens, spec.temperature)

    for attempt in range(3):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            elapsed = time.time() - t0
            predicted = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            result = {
                "predicted": predicted,
                "response_time": round(elapsed, 3),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "error": None,
            }
            if _is_refusal(predicted):
                result["model_refuse"] = True
            return result

        except openai.RateLimitError:
            logger.warning("%s rate limited (429) — skipping", label)
            return _skip("rate_limit_429")

        except openai.BadRequestError as exc:
            err_str = str(exc)
            if "temperature" in err_str and "temperature" in kwargs:
                logger.warning("%s: dropping temperature (cached for deployment)", label)
                _record_quirk(spec.deployment, "drop_temperature")
                kwargs.pop("temperature")
                continue
            if "max_tokens" in err_str and "max_tokens" in kwargs:
                logger.warning("%s: max_tokens -> max_completion_tokens (cached for deployment)", label)
                _record_quirk(spec.deployment, "use_max_completion_tokens")
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                continue
            # Any other 400 — hard failure, do not retry.
            logger.error("%s BadRequestError: %s", label, exc)
            return _err(str(exc))

        except Exception as exc:
            if attempt < 2:
                wait = 4 * (attempt + 1)
                logger.warning(
                    "%s attempt %d failed (%s) — retrying in %ds",
                    label, attempt + 1, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("%s failed after 3 attempts: %s", label, exc)
                return _err(str(exc))

    return _err("max retries exceeded")


def _err(msg: str) -> dict[str, Any]:
    return {
        "predicted": "",
        "response_time": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": msg,
    }


def _skip(reason: str = "rate_limit_429") -> dict[str, Any]:
    return {
        "predicted": "",
        "response_time": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": None,
        "status": "skip",
        "skip_reason": reason,
    }
