"""
OpenAI API client factory with support for relay endpoints.

This module provides centralized client creation and configuration
for OpenAI API calls, supporting both direct OpenAI access and
OpenAI-compatible relay platforms.
"""

import os
from openai import OpenAI


def make_client():
    """
    Create an OpenAI client instance.

    Reads configuration from environment variables:
    - OPENAI_API_KEY (required): API key for authentication
    - OPENAI_BASE_URL (optional): Custom base URL for relay endpoints

    Returns:
        OpenAI: Configured client instance

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    ignore_env_proxy = os.getenv("NO_ENV_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}
    proxy = None if ignore_env_proxy else (os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"))

    kwargs = {"api_key": api_key}
    if base_url and base_url.strip():
        kwargs["base_url"] = base_url
    if proxy and proxy.strip():
        import httpx
        kwargs["http_client"] = httpx.Client(proxy=proxy)
    elif ignore_env_proxy:
        import httpx
        kwargs["http_client"] = httpx.Client(trust_env=False)

    return OpenAI(**kwargs)


def get_request_timeout():
    """
    Get the request timeout value from environment.

    Returns:
        int: Timeout in seconds (default: 60)
    """
    timeout_str = os.getenv("REQUEST_TIMEOUT", "60")
    try:
        return int(timeout_str)
    except ValueError:
        return 60


def resolve_stage_model(stage_name, fallback_model):
    """
    Resolve the model to use for a specific pipeline stage.

    Priority order:
    1. Stage-specific env var (STAGE1_MODEL, STAGE2_MODEL, CONVERT_MODEL)
    2. Global OPENAI_MODEL env var
    3. Fallback model passed as argument

    Args:
        stage_name: One of "stage1", "stage2", or "convert"
        fallback_model: Default model to use if no env vars are set

    Returns:
        str: Model name to use
    """
    stage_env_map = {
        "stage1": "STAGE1_MODEL",
        "stage2": "STAGE2_MODEL",
        "convert": "CONVERT_MODEL"
    }

    # Try stage-specific model first
    stage_env = stage_env_map.get(stage_name)
    if stage_env:
        stage_model = os.getenv(stage_env)
        if stage_model:
            return stage_model

    # Try global OPENAI_MODEL
    global_model = os.getenv("OPENAI_MODEL")
    if global_model:
        return global_model

    # Fall back to provided default
    return fallback_model


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))

def build_token_limit_kwarg(model: str, max_tokens: int) -> dict:
    """Return correct token-limit kwarg; reasoning models also drop temperature."""
    if _is_reasoning_model(model):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}

def build_create_kwargs(model: str, max_tokens: int, temperature: float) -> dict:
    """Return all extra kwargs for chat.completions.create, handling model quirks."""
    kwargs = build_token_limit_kwarg(model, max_tokens)
    if not _is_reasoning_model(model):
        kwargs["temperature"] = temperature
    return kwargs


def format_usage(usage) -> str:
    """Format token usage from an OpenAI response for debug logs."""
    if usage is None:
        return "usage=unavailable"

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    parts = [
        f"prompt_tokens={prompt_tokens}",
        f"completion_tokens={completion_tokens}",
        f"total_tokens={total_tokens}",
    ]

    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
    if reasoning_tokens is not None:
        parts.append(f"reasoning_tokens={reasoning_tokens}")

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(prompt_details, "cached_tokens", None) if prompt_details else None
    if cached_tokens is not None:
        parts.append(f"cached_tokens={cached_tokens}")

    return ", ".join(parts)
