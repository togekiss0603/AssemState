from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VLMJSONParseError(ValueError):
    def __init__(self, message: str, response_text: str, cause: Exception | None = None):
        snippet = response_text[:1200].replace("\n", "\\n")
        super().__init__(f"{message}. response_snippet={snippet}")
        self.response_text = response_text
        self.cause = cause


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise VLMJSONParseError("Empty VLM response", text)

    last_error: Exception | None = None
    for candidate in json_text_candidates(text):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
            last_error = ValueError("VLM response JSON is not an object")
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error is None:
        raise VLMJSONParseError("No JSON object found in response", text)
    raise VLMJSONParseError(f"Invalid JSON object: {last_error}", text, cause=last_error)


def json_text_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stripped = strip_code_fence(text.strip())
    if stripped:
        candidates.append(stripped)

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        fenced = strip_code_fence(match.group(1).strip())
        if fenced:
            candidates.append(fenced)

    candidates.extend(extract_balanced_json_objects(text))

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = normalize_json_like_text(candidate)
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return out


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : idx + 1])
                start = None
    return objects


def normalize_json_like_text(text: str) -> str:
    # Keep smart quotes inside string values. Replacing curly double quotes with
    # ASCII quotes can turn otherwise-valid JSON prose into invalid syntax, e.g.
    # {"Reasoning": "the intended “inner rails” setup"}.
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def encode_image_data_uri(path: str | Path, max_side: int = 1200, jpeg_quality: int = 88) -> str:
    image_path = Path(path)
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            payload = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{payload}"
    except Exception:
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        suffix = image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64,{payload}"


@dataclass
class VLMRequest:
    system_prompt: str
    user_prompt: str
    image_paths: list[Path]
    temperature: float
    sample_id: int


class BaseVLMClient:
    def estimate_pose(self, request: VLMRequest) -> dict[str, Any]:
        raise NotImplementedError


class MockVLMClient(BaseVLMClient):
    """Dry-run client. The pipeline will replace this with geometry fallback."""

    def estimate_pose(self, request: VLMRequest) -> dict[str, Any]:
        return {
            "part_id": None,
            "confidence": 0.0,
            "rationale": "dry_run_mock_client",
        }


class OpenAIVLMClient(BaseVLMClient):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_image_side: int = 1200,
        max_completion_tokens: int | None = 1024,
        top_p: float | None = 0.95,
        stream: bool = False,
        stop: str | None = None,
        frequency_penalty: float | None = 0.0,
        presence_penalty: float | None = 0.0,
        api_retries: int | None = None,
        retry_delay: float | None = None,
        json_repair_retries: int | None = None,
        trust_env_proxy: bool = True,
        proxy_url: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("The openai package is required for VLM calls") from exc

        self.model = model or os.getenv("VLM_POSE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.2"
        self.max_image_side = int(max_image_side)
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.stream = bool(stream)
        self.stop = stop
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_retries = int(api_retries if api_retries is not None else os.getenv("VLM_API_RETRIES", "2"))
        self.retry_delay = float(retry_delay if retry_delay is not None else os.getenv("VLM_RETRY_DELAY", "2.0"))
        self.json_repair_retries = int(
            json_repair_retries
            if json_repair_retries is not None
            else os.getenv("VLM_JSON_REPAIR_RETRIES", "1")
        )
        kwargs: dict[str, Any] = {}
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.base_url = resolved_base_url
        if resolved_api_key:
            kwargs["api_key"] = resolved_api_key
        elif resolved_base_url:
            kwargs["api_key"] = "EMPTY"
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        if timeout or os.getenv("REQUEST_TIMEOUT"):
            kwargs["timeout"] = float(timeout or os.getenv("REQUEST_TIMEOUT", "60"))
        if proxy_url or not trust_env_proxy:
            import httpx

            kwargs["http_client"] = httpx.Client(
                proxy=proxy_url,
                trust_env=trust_env_proxy if proxy_url is None else False,
            )
        self.client = OpenAI(**kwargs)

    def _completion_kwargs(self, messages: list[dict[str, Any]], request: VLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_format": {"type": "json_object"},
        }
        if not (self._uses_openai_endpoint() and float(request.temperature) != 1.0):
            kwargs["temperature"] = request.temperature
        if self.top_p is not None and not (self._uses_openai_endpoint() and float(self.top_p) != 1.0):
            kwargs["top_p"] = self.top_p
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        return kwargs

    def _uses_openai_endpoint(self) -> bool:
        if not self.base_url:
            return True
        return "api.openai.com" in str(self.base_url).lower()

    def _create_completion(self, kwargs: dict[str, Any]) -> Any:
        attempts = 0
        while True:
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                message = str(exc)
                if "response_format" in message and "response_format" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs.pop("response_format")
                    continue
                if "max_completion_tokens" in message and "max_completion_tokens" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
                unsupported_sampling_param = next(
                    (
                        name
                        for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty")
                        if name in kwargs and name in message and "unsupported" in message.lower()
                    ),
                    None,
                )
                if unsupported_sampling_param:
                    kwargs = dict(kwargs)
                    kwargs.pop(unsupported_sampling_param)
                    continue
                if self._is_retryable_api_error(message) and attempts < self.api_retries:
                    attempts += 1
                    time.sleep(self.retry_delay * attempts)
                    continue
                raise

    def _is_retryable_api_error(self, message: str) -> bool:
        lowered = message.lower()
        retryable_markers = [
            "connection error",
            "connecterror",
            "unexpected_eof",
            "eof occurred",
            "timeout",
            "temporarily unavailable",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
        ]
        return any(marker in lowered for marker in retryable_markers)

    def estimate_pose(self, request: VLMRequest) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": request.user_prompt}]
        for image_path in request.image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_data_uri(image_path, max_side=self.max_image_side),
                        "detail": "high",
                    },
                }
            )
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": content},
        ]
        response = self._create_completion(self._completion_kwargs(messages, request))
        text = response.choices[0].message.content or ""
        try:
            return extract_json_object(text)
        except VLMJSONParseError as exc:
            repaired = self._repair_json_response(text, exc)
            return extract_json_object(repaired)

    def _repair_json_response(self, text: str, parse_error: Exception) -> str:
        if self.json_repair_retries <= 0:
            raise parse_error

        current_text = text
        last_error: Exception = parse_error
        for _ in range(self.json_repair_retries):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON repair tool. Return only one valid JSON object. "
                        "Do not add markdown. Do not change numeric values or field meanings."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "The following response should be a JSON object, but it is invalid. "
                        f"JSON parser error: {last_error}\n\n"
                        "Repair only JSON syntax: missing commas, trailing commas, quotes, or brackets. "
                        "Return the full corrected JSON object only.\n\n"
                        f"{current_text}"
                    ),
                },
            ]
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
                "response_format": {"type": "json_object"},
            }
            if self.max_completion_tokens is not None:
                kwargs["max_completion_tokens"] = max(int(self.max_completion_tokens), 2048)
            response = self._create_completion(kwargs)
            current_text = response.choices[0].message.content or ""
            try:
                extract_json_object(current_text)
                return current_text
            except VLMJSONParseError as exc:
                last_error = exc

        raise VLMJSONParseError(
            f"VLM returned invalid JSON and repair failed: {last_error}",
            current_text,
            cause=last_error,
        )
