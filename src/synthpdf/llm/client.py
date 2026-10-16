# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Azure OpenAI client wrapper for synthpdf.

Authentication priority:
1. API key — from config.api_key or AZURE_OPENAI_API_KEY env var.
2. Raw bearer token — from AZURE_OPENAI_ACCESS_TOKEN env var (set by run_configs.py
   after a one-time browser login so subprocesses never prompt again).
3. Entra ID (Azure AD) — via InteractiveBrowserCredential when no API key or token is present.
   tenant_id is read from config.tenant_id or AZURE_TENANT_ID env var.
   Requires: pip install azure-identity

Connectivity:
- verify_ssl=False  disables SSL verification (useful for corporate MITM proxies).
- ssl_ca_bundle     path to a custom CA bundle PEM file.
- http_proxy        explicit proxy URL; otherwise HTTPS_PROXY / HTTP_PROXY env vars
                    are picked up automatically by httpx.
  A custom httpx.Client is only constructed when ssl or proxy options are set;
  otherwise the default httpx client is used (same as the reference implementation).

Retry behaviour:
- APIConnectionError / APITimeoutError → exponential backoff (1 s, 2 s, 4 s, …).
- RateLimitError (if SDK retries are exhausted) → linear backoff (10 s, 20 s, …).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from synthpdf.config import AzureOpenAIConfig

logger = logging.getLogger(__name__)


def _build_http_client(config: AzureOpenAIConfig) -> Any | None:
    """Build an httpx.Client only when SSL or proxy options are configured.

    Returns None when no custom transport is needed so the default httpx
    client (used by the reference implementation) is used instead.
    """
    needs_custom = (
        not config.verify_ssl
        or config.ssl_ca_bundle
        or config.http_proxy
    )
    if not needs_custom:
        return None

    import httpx

    if not config.verify_ssl:
        ssl_context: bool | str = False
    elif config.ssl_ca_bundle:
        ssl_context = config.ssl_ca_bundle
    else:
        ssl_context = True

    proxy_url = config.http_proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy_url:
        return httpx.Client(verify=ssl_context, proxy=proxy_url)
    return httpx.Client(verify=ssl_context)


class AzureOpenAIClient:
    """Thin wrapper around openai.AzureOpenAI for content generation."""

    def __init__(self, config: AzureOpenAIConfig) -> None:
        from openai import AzureOpenAI

        api_key = config.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        http_client = _build_http_client(config)

        common_kwargs: dict[str, Any] = {
            "azure_endpoint": config.endpoint,
            "api_version": config.api_version,
            "max_retries": config.max_retries,
            "timeout": config.timeout_seconds,
        }
        if http_client is not None:
            common_kwargs["http_client"] = http_client

        access_token = os.environ.get("AZURE_OPENAI_ACCESS_TOKEN")

        if api_key:
            self._client = AzureOpenAI(api_key=api_key, **common_kwargs)
        elif access_token:
            # Raw bearer token injected by run_configs.py._preauth_azure_if_needed().
            # Subprocesses inherit this env var and skip the browser prompt entirely.
            self._client = AzureOpenAI(azure_ad_token=access_token, **common_kwargs)
        else:
            try:
                from azure.identity import InteractiveBrowserCredential, get_bearer_token_provider
            except ImportError as exc:
                raise ImportError(
                    "No API key found and 'azure-identity' is not installed. "
                    "Either set AZURE_OPENAI_API_KEY / config.api_key, "
                    "or install azure-identity: pip install azure-identity"
                ) from exc

            tenant_id = config.tenant_id or os.environ.get("AZURE_TENANT_ID")
            if not tenant_id:
                raise ValueError(
                    "Entra ID authentication requires a tenant_id. "
                    "Set AZURE_TENANT_ID environment variable or add tenant_id to the config."
                )

            credential = InteractiveBrowserCredential(tenant_id=tenant_id)
            # Force token acquisition now (main thread) so the cache is warm
            # before any worker threads call get_token() concurrently.
            # Without this, parallel workers each trigger a browser flow and
            # their OAuth states collide → "state mismatch" error.
            _scope = "https://cognitiveservices.azure.com/.default"
            credential.get_token(_scope)
            token_provider = get_bearer_token_provider(credential, _scope)
            self._client = AzureOpenAI(
                azure_ad_token_provider=token_provider, **common_kwargs
            )

        # Deployment selection per tier.  Each tier falls back to the next heavier one.
        # HARD → MEDIUM → EASY → base deployment (for backward compat)
        base = config.deployment
        self._deployment_hard   = config.deployment_hard   or base
        self._deployment_medium = config.deployment_medium or self._deployment_hard
        self._deployment_easy   = config.deployment_easy   or self._deployment_medium
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._max_retries = config.max_retries
        self._timeout_seconds = config.timeout_seconds
        # Tracks params rejected by each deployment so they are skipped on future calls.
        self._unsupported_params: dict[str, set[str]] = {}

    def _deployment_for_tier(self, tier: str) -> str:
        """Return the deployment name for the given complexity tier."""
        if tier == "easy":
            return self._deployment_easy
        if tier == "medium":
            return self._deployment_medium
        return self._deployment_hard  # "hard" or any unknown value

    def complete(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = False,
        tier: str = "hard",        # "easy" | "medium" | "hard"
        _label: str = "",          # optional caller label for timing logs
    ) -> str:
        """Call the chat completions endpoint and return the assistant message text."""
        import openai

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        deployment = self._deployment_for_tier(tier)
        rejected = self._unsupported_params.get(deployment, set())

        kwargs: dict[str, Any] = {"model": deployment, "messages": messages,
                                   "max_completion_tokens": self._max_tokens}
        if "temperature" not in rejected:
            kwargs["temperature"] = self._temperature
        if json_mode and "response_format" not in rejected:
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = self._client.chat.completions.create(**kwargs)
                elapsed = time.perf_counter() - t0
                label = f"[{_label}] " if _label else ""
                usage = response.usage
                tokens_info = (
                    f"prompt={usage.prompt_tokens} completion={usage.completion_tokens}"
                    if usage else "tokens=unknown"
                )
                logger.info("  LLM %s(%s) %s%.1fs — %s", tier, deployment, label, elapsed, tokens_info)
                content = response.choices[0].message.content
                if not content:
                    finish_reason = response.choices[0].finish_reason
                    raise RuntimeError(
                        f"Empty response from {deployment!r} "
                        f"(finish_reason={finish_reason!r}, label={_label!r}). "
                        f"This is usually caused by Azure content filtering."
                    )
                return content

            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                # Some models reject unsupported parameters — record once and retry
                # immediately without consuming a retry attempt.
                if "temperature" in msg and "temperature" in kwargs:
                    logger.warning(
                        "Model %r does not support 'temperature'; skipping it for all future calls.",
                        deployment,
                    )
                    self._unsupported_params.setdefault(deployment, set()).add("temperature")
                    kwargs.pop("temperature")
                    continue
                if "response_format" in msg and "response_format" in kwargs:
                    logger.warning(
                        "Model %r does not support 'response_format'; skipping it for all future calls.",
                        deployment,
                    )
                    self._unsupported_params.setdefault(deployment, set()).add("response_format")
                    kwargs.pop("response_format")
                    continue
                # Any other 400 error is not retryable
                raise

            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                last_exc = exc
                root_cause = exc.__cause__ or exc
                if attempt < self._max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Connection error (attempt %d/%d), retrying in %d s.\n"
                        "  Root cause: %s: %s",
                        attempt + 1, self._max_retries + 1, wait,
                        type(root_cause).__name__, root_cause,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "All %d attempts failed.\n"
                        "  Root cause: %s: %s\n"
                        "  Tip: if this is an SSL error, set verify_ssl: false in config.\n"
                        "  Tip: if you need a proxy, set http_proxy in config or "
                        "export HTTPS_PROXY.",
                        self._max_retries + 1,
                        type(root_cause).__name__, root_cause,
                    )
                    raise

            except openai.RateLimitError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = 10 * (attempt + 1)
                    logger.warning(
                        "Rate limit hit (attempt %d/%d), retrying in %d s.",
                        attempt + 1, self._max_retries + 1, wait,
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("complete() exited retry loop unexpectedly") from last_exc
