from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import orthonormalize_rotation

from pose_estimation.vlm_pose.vlm_client import (
    VLMJSONParseError,
    encode_image_data_uri,
    extract_json_object,
)


class VisualAgentParseError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluatorResult:
    faithfulness: str
    image_number: int
    reasoning: str
    confidence: float
    ambiguity: float
    orientation: str
    contact: str
    contact_angle: str
    required_relation: str
    rotation_penalty: str
    target_residual: str
    raw: dict[str, Any]

    @property
    def is_yes(self) -> bool:
        return self.faithfulness == "Yes"

    def to_json(self) -> dict[str, Any]:
        return {
            "Faithfulness": self.faithfulness,
            "Image number": self.image_number,
            "Reasoning": self.reasoning,
            "Confidence": self.confidence,
            "Ambiguity": self.ambiguity,
            "Orientation": self.orientation,
            "Contact": self.contact,
            "Contact angle": self.contact_angle,
            "Required relation": self.required_relation,
            "Rotation penalty": self.rotation_penalty,
            "Target residual": self.target_residual,
            "raw": self.raw,
        }


@dataclass(frozen=True)
class ProposerResult:
    target_object: str
    instruction_target_relation: str
    current_error: str
    reasoning: str
    translation: np.ndarray
    axis: str
    angle: float
    rotation_matrix: np.ndarray | None
    confidence: float
    raw: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "Target object": self.target_object,
            "Instruction-derived target relation": self.instruction_target_relation,
            "Current error": self.current_error,
            "Reasoning": self.reasoning,
            "Translation": self.translation.round(6).tolist(),
            "Dominant rotation axis": self.axis,
            "Angle": round(float(self.angle), 6),
            "Rotation matrix": (
                self.rotation_matrix.round(6).tolist() if self.rotation_matrix is not None else None
            ),
            "Confidence": self.confidence,
            "raw": self.raw,
        }


class BaseVisualVLMClient:
    is_mock = False

    def evaluate(self, system_prompt: str, user_prompt: str, image_paths: list[Path], temperature: float, sample_id: int) -> EvaluatorResult:
        raise NotImplementedError

    def propose(self, system_prompt: str, user_prompt: str, image_paths: list[Path], temperature: float, sample_id: int) -> ProposerResult:
        raise NotImplementedError


class OpenAIVisualVLMClient(BaseVisualVLMClient):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_image_side: int = 1200,
        max_completion_tokens: int | None = None,
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
        modalities: dict[str, list[str]] | None = None,
        rotation_mode: str = "axis_angle",
        reasoning_effort: str | None = None,
        image_detail: str | None = None,
        client_role: str | None = None,
        usage_log_path: str | Path | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("The openai package is required for VLM calls") from exc

        self.model = model or os.getenv("VLM_POSE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.2"
        self.max_image_side = int(max_image_side)
        self.top_p = top_p
        self.stream = bool(stream)
        self.stop = stop
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_retries = int(api_retries if api_retries is not None else os.getenv("VLM_API_RETRIES", "2"))
        self.retry_delay = float(retry_delay if retry_delay is not None else os.getenv("VLM_RETRY_DELAY", "2.0"))
        self.rotation_mode = str(rotation_mode or "axis_angle").strip().lower()
        if self.rotation_mode not in {"axis_angle", "matrix"}:
            raise ValueError(f"rotation_mode must be 'axis_angle' or 'matrix', got {self.rotation_mode!r}")
        self.image_detail = (image_detail or os.getenv("VLM_IMAGE_DETAIL") or "high").strip().lower()
        if self.image_detail not in {"low", "high", "auto"}:
            raise ValueError(f"image_detail must be 'low', 'high', or 'auto', got {self.image_detail!r}")
        self.reasoning_effort = (reasoning_effort or os.getenv("OPENAI_REASONING_EFFORT") or "").strip() or None
        self.json_repair_retries = int(
            json_repair_retries
            if json_repair_retries is not None
            else os.getenv("VLM_JSON_REPAIR_RETRIES", "1")
        )
        self.client_role = str(client_role or os.getenv("VLM_CLIENT_ROLE") or "unknown")
        usage_log = usage_log_path or os.getenv("VLM_USAGE_LOG")
        self.usage_log_path = Path(usage_log) if usage_log else None
        client_kwargs: dict[str, Any] = {}
        resolved_api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )
        resolved_base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
        )
        self.base_url = resolved_base_url
        default_tokens = "4096" if self._uses_openai_endpoint() else "1024"
        self.max_completion_tokens = (
            int(max_completion_tokens)
            if max_completion_tokens is not None
            else int(os.getenv("VLM_MAX_COMPLETION_TOKENS", default_tokens))
        )
        self.modalities = modalities or {"input": ["text", "image"], "output": ["text"]}
        if resolved_api_key:
            client_kwargs["api_key"] = resolved_api_key
        elif resolved_base_url:
            client_kwargs["api_key"] = "EMPTY"
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        if timeout or os.getenv("REQUEST_TIMEOUT"):
            client_kwargs["timeout"] = float(timeout or os.getenv("REQUEST_TIMEOUT", "60"))
        if proxy_url or not trust_env_proxy:
            import httpx

            client_kwargs["http_client"] = httpx.Client(
                proxy=proxy_url,
                trust_env=trust_env_proxy if proxy_url is None else False,
            )
        self.client = OpenAI(**client_kwargs)

    def evaluate(self, system_prompt: str, user_prompt: str, image_paths: list[Path], temperature: float, sample_id: int) -> EvaluatorResult:
        raw = self._request_json(system_prompt, user_prompt, image_paths, temperature, sample_id, schema_name="evaluator")
        return parse_evaluator(raw)

    def propose(self, system_prompt: str, user_prompt: str, image_paths: list[Path], temperature: float, sample_id: int) -> ProposerResult:
        current_prompt = user_prompt
        last_error: Exception | None = None
        last_raw: dict[str, Any] | None = None
        attempts = max(1, self.json_repair_retries + 1)
        for attempt in range(attempts):
            raw = self._request_json(system_prompt, current_prompt, image_paths, temperature, sample_id, schema_name="proposer")
            try:
                return parse_proposer(raw)
            except VisualAgentParseError as exc:
                last_error = exc
                last_raw = raw
                if attempt >= attempts - 1:
                    break
                current_prompt = self._proposer_schema_retry_prompt(user_prompt, raw, exc)
        raise last_error or VisualAgentParseError(f"Proposer response did not match schema: {last_raw}")

    def _request_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path],
        temperature: float,
        sample_id: int,
        schema_name: str | None = None,
    ) -> dict[str, Any]:
        messages = self._messages(system_prompt, user_prompt, image_paths)
        kwargs = self._completion_kwargs(
            messages,
            temperature=temperature,
            has_images=bool(image_paths),
            schema_name=schema_name,
        )
        response = self._create_completion(kwargs)
        self._log_usage(
            response,
            schema_name=schema_name or "freeform_json",
            image_count=len(image_paths),
            input_text_chars=len(system_prompt) + len(user_prompt),
        )
        text = response.choices[0].message.content or ""
        try:
            return extract_json_object(text)
        except VLMJSONParseError as exc:
            repaired = self._repair_json_response(text, exc)
            return extract_json_object(repaired)

    def _messages(self, system_prompt: str, user_prompt: str, image_paths: list[Path]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_data_uri(image_path, max_side=self.max_image_side),
                        "detail": self.image_detail,
                    },
                }
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _proposer_schema_retry_prompt(self, original_prompt: str, raw: dict[str, Any], error: Exception) -> str:
        rotation_matrix_line = ""
        if self.rotation_mode == "matrix":
            rotation_matrix_line = (
                '  "Rotation matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],\n'
            )
        return (
            f"{original_prompt}\n\n"
            "# Schema Correction Required\n"
            f"Your previous JSON did not match the required proposer schema. Error: {error}\n"
            f"Previous JSON: {_compact_json(raw, limit=1000)}\n\n"
            "Return exactly one JSON object with these top-level fields and valid numeric values:\n"
            "{\n"
            '  "Target object": "moving part",\n'
            '  "Instruction-derived target relation": "what the instruction says the moving part should do relative to the fixed parts",\n'
            '  "Current error": "how the current view differs from that required relation",\n'
            '  "Reasoning": "why this incremental update improves the pose",\n'
            '  "Translation": [0.0, 0.0, 0.0],\n'
            f"{rotation_matrix_line}"
            '  "Dominant rotation axis": "x",\n'
            '  "Angle": 0.0,\n'
            '  "Confidence": 0.0\n'
            "}\n"
            "The Translation field is mandatory and must be a JSON array of exactly three numbers."
        )

    def _completion_kwargs(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        has_images: bool,
        schema_name: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        response_format = self._response_format(schema_name)
        if response_format is not None:
            kwargs["response_format"] = response_format
        if not (self._uses_openai_endpoint() and float(temperature) != 1.0):
            kwargs["temperature"] = temperature
        if self.top_p is not None and not (self._uses_openai_endpoint() and float(self.top_p) != 1.0):
            kwargs["top_p"] = self.top_p
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        if self.reasoning_effort and self._uses_openai_endpoint():
            kwargs["reasoning_effort"] = self.reasoning_effort
        if has_images and not self._uses_openai_endpoint():
            kwargs["extra_body"] = {"modalities": self.modalities}
        return kwargs

    def _response_format(self, schema_name: str | None) -> dict[str, Any]:
        if schema_name == "freeform_json":
            return None
        if not self._uses_openai_endpoint():
            return {"type": "json_object"}
        if schema_name == "evaluator":
            return _json_schema_response_format(
                "visual_assembly_evaluator",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "Faithfulness": {"type": "string", "enum": ["Yes", "No"]},
                        "Image number": {"type": "integer", "minimum": 1},
                        "Reasoning": {"type": "string"},
                        "Confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "Ambiguity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "Orientation": {"type": "string", "enum": ["Correct", "Incorrect", "Unclear"]},
                        "Contact": {"type": "string", "enum": ["Correct", "Incorrect", "Unclear"]},
                        "Contact angle": {"type": "string", "enum": ["Correct", "Incorrect", "Unclear"]},
                        "Required relation": {
                            "type": "string",
                            "enum": ["Perpendicular", "Parallel", "Horizontal", "End-to-end", "Unknown"],
                        },
                        "Rotation penalty": {"type": "string", "enum": ["None", "Mild", "Strong"]},
                        "Target residual": {"type": "string", "enum": ["Far", "Medium", "Close", "Unclear"]},
                    },
                    "required": [
                        "Faithfulness",
                        "Image number",
                        "Reasoning",
                        "Confidence",
                        "Ambiguity",
                        "Orientation",
                        "Contact",
                        "Contact angle",
                        "Required relation",
                        "Rotation penalty",
                        "Target residual",
                    ],
                },
            )
        if schema_name == "proposer":
            properties: dict[str, Any] = {
                "Target object": {"type": "string"},
                "Instruction-derived target relation": {"type": "string"},
                "Current error": {"type": "string"},
                "Reasoning": {"type": "string"},
                "Translation": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "Dominant rotation axis": {"type": "string", "enum": ["x", "y", "z"]},
                "Angle": {"type": "number"},
                "Confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            }
            required = [
                "Target object",
                "Instruction-derived target relation",
                "Current error",
                "Reasoning",
                "Translation",
                "Dominant rotation axis",
                "Angle",
                "Confidence",
            ]
            if self.rotation_mode == "matrix":
                properties["Rotation matrix"] = {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "minItems": 3,
                            "maxItems": 3,
                        },
                        {"type": "null"},
                    ]
                }
                required.insert(3, "Rotation matrix")
            return _json_schema_response_format(
                "visual_assembly_proposer",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": required,
                },
            )
        if schema_name == "choice_selector":
            return _json_schema_response_format(
                "visual_assembly_choice_selector",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "Selected option": {"type": "string"},
                        "Is acceptable": {"type": "string", "enum": ["Yes", "No"]},
                        "selected_3d_views": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                        "Reasoning": {"type": "string"},
                        "Confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["Selected option", "Is acceptable", "selected_3d_views", "Reasoning", "Confidence"],
                },
            )
        if schema_name == "choice_judger":
            return _json_schema_response_format(
                "visual_assembly_choice_judger",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "Accept": {"type": "string", "enum": ["Yes", "No"]},
                        "Error type": {"type": "string"},
                        "Residual direction": {"type": "string"},
                        "Contact": {"type": "string", "enum": ["Correct", "Incorrect", "Unclear"]},
                        "Orientation": {"type": "string", "enum": ["Correct", "Incorrect", "Unclear"]},
                        "Reasoning": {"type": "string"},
                        "Next candidate hints": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "Confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": [
                        "Accept",
                        "Error type",
                        "Residual direction",
                        "Contact",
                        "Orientation",
                        "Reasoning",
                        "Next candidate hints",
                        "Confidence",
                    ],
                },
            )
        if schema_name == "pose_command_candidates":
            return _json_schema_response_format(
                "visual_assembly_pose_command_candidates",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "selected_supporting_views": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                        "selected_supporting_view_reason": {"type": "string"},
                        "candidates": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "label": {"type": "string"},
                                    "translation_delta_world": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 3,
                                        "maxItems": 3,
                                    },
                                    "rotation_delta_axis": {"type": "string", "enum": ["x", "y", "z"]},
                                    "rotation_delta_deg": {"type": "number"},
                                    "semantic_reason": {"type": "string"},
                                    "expected_visual_change": {"type": "string"},
                                    "risk": {"type": "string"},
                                },
                                "required": [
                                    "label",
                                    "translation_delta_world",
                                    "rotation_delta_axis",
                                    "rotation_delta_deg",
                                    "semantic_reason",
                                    "expected_visual_change",
                                    "risk",
                                ],
                            },
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": [
                        "selected_supporting_views",
                        "selected_supporting_view_reason",
                        "candidates",
                        "reasoning",
                    ],
                },
            )
        if schema_name == "direct_pose_candidates":
            rotation_matrix_schema = {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "minItems": 3,
                "maxItems": 3,
            }
            return _json_schema_response_format(
                "visual_assembly_direct_pose_candidates",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "selected_supporting_views": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                        "selected_supporting_view_reason": {"type": "string"},
                        "candidates": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "label": {"type": "string"},
                                    "Translation": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 3,
                                        "maxItems": 3,
                                    },
                                    "Rotation matrix": rotation_matrix_schema,
                                    "Dominant rotation axis": {"type": "string", "enum": ["x", "y", "z"]},
                                    "Angle": {"type": "number"},
                                    "semantic_reason": {"type": "string"},
                                    "expected_visual_change": {"type": "string"},
                                    "risk": {"type": "string"},
                                },
                                "required": [
                                    "label",
                                    "Translation",
                                    "Rotation matrix",
                                    "Dominant rotation axis",
                                    "Angle",
                                    "semantic_reason",
                                    "expected_visual_change",
                                    "risk",
                                ],
                            },
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": [
                        "selected_supporting_views",
                        "selected_supporting_view_reason",
                        "candidates",
                        "reasoning",
                    ],
                },
            )
        if schema_name == "pose_token_candidates":
            return _json_schema_response_format(
                "visual_assembly_pose_token_candidates",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "label": {"type": "string"},
                                    "translation_steps_0p25": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "minItems": 3,
                                        "maxItems": 3,
                                    },
                                    "rotation_axis": {"type": "string", "enum": ["x", "y", "z"]},
                                    "rotation_deg": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": [
                                    "label",
                                    "translation_steps_0p25",
                                    "rotation_axis",
                                    "rotation_deg",
                                    "reason",
                                ],
                            },
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": ["candidates", "reasoning"],
                },
            )
        if schema_name == "semantic_primitive_candidates":
            return _json_schema_response_format(
                "visual_assembly_semantic_primitive_candidates",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "selected_supporting_views": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                        "selected_supporting_view_reason": {"type": "string"},
                        "candidates": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "label": {"type": "string"},
                                    "source_primitive": {"type": "string"},
                                    "semantic_action": {"type": "string"},
                                    "why_plausible": {"type": "string"},
                                    "risk": {"type": "string"},
                                },
                                "required": [
                                    "label",
                                    "source_primitive",
                                    "semantic_action",
                                    "why_plausible",
                                    "risk",
                                ],
                            },
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": [
                        "selected_supporting_views",
                        "selected_supporting_view_reason",
                        "candidates",
                        "reasoning",
                    ],
                },
            )
        return {"type": "json_object"}

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
                    if kwargs["response_format"].get("type") == "json_schema":
                        kwargs["response_format"] = {"type": "json_object"}
                    else:
                        kwargs.pop("response_format")
                    continue
                if "max_completion_tokens" in message and "max_completion_tokens" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
                if "reasoning_effort" in message and "reasoning_effort" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs.pop("reasoning_effort")
                    continue
                if (
                    "extra_body" in kwargs
                    and (
                        "modalities" in message.lower()
                        or "extra_body" in message.lower()
                        or "extra fields" in message.lower()
                        or "unknown parameter" in message.lower()
                    )
                ):
                    kwargs = dict(kwargs)
                    kwargs.pop("extra_body")
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
            self._log_usage(
                response,
                schema_name="json_repair",
                image_count=0,
                input_text_chars=sum(len(str(message.get("content", ""))) for message in messages),
            )
            current_text = response.choices[0].message.content or ""
            try:
                extract_json_object(current_text)
                return current_text
            except VLMJSONParseError as exc:
                last_error = exc
        raise last_error

    def _log_usage(
        self,
        response: Any,
        *,
        schema_name: str,
        image_count: int,
        input_text_chars: int,
    ) -> None:
        if self.usage_log_path is None:
            return
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            usage: dict[str, Any] = {}
        elif hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage = dict(usage_obj)
        else:
            usage = {
                key: getattr(usage_obj, key)
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "input_tokens",
                    "output_tokens",
                )
                if hasattr(usage_obj, key)
            }
        record = {
            "time": time.time(),
            "role": self.client_role,
            "schema_name": schema_name,
            "model": self.model,
            "base_url": self.base_url,
            "image_count": int(image_count),
            "image_detail": self.image_detail,
            "max_image_side": self.max_image_side,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
            "input_text_chars": int(input_text_chars),
            "usage": usage,
        }
        try:
            self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            return


class MockVisualVLMClient(BaseVisualVLMClient):
    is_mock = True

    def evaluate(self, system_prompt: str, user_prompt: str, image_paths: list[Path], temperature: float, sample_id: int) -> EvaluatorResult:
        return EvaluatorResult(
            faithfulness="No",
            image_number=1,
            reasoning="mock evaluator: visual faithfulness is decided by the geometric dry-run policy",
            confidence=0.1,
            ambiguity=0.9,
            orientation="Unclear",
            contact="Unclear",
            contact_angle="Unclear",
            required_relation="Unknown",
            rotation_penalty="Strong",
            target_residual="Far",
            raw={"mock": True},
        )

    def propose(self, system_prompt: str, user_prompt: str, image_paths: list[Path], temperature: float, sample_id: int) -> ProposerResult:
        return ProposerResult(
            target_object="mock",
            instruction_target_relation="mock relation",
            current_error="mock geometric displacement",
            reasoning="mock proposer placeholder; pipeline supplies deterministic dry-run updates",
            translation=np.zeros(3, dtype=np.float64),
            axis="z",
            angle=0.0,
            rotation_matrix=None,
            confidence=0.1,
            raw={"mock": True},
        )


def parse_evaluator(raw: dict[str, Any]) -> EvaluatorResult:
    payload = _find_mapping_with_field(
        raw,
        "Faithfulness",
        "faithfulness",
        "is_faithful",
        "is faithful",
        "correct",
        "is_correct",
        "assembly_correct",
        "matches_target",
    )
    faith_value = _field_value(
        payload,
        "Faithfulness",
        "faithfulness",
        "is_faithful",
        "is faithful",
        "correct",
        "is_correct",
        "assembly_correct",
        "matches_target",
        default=None,
    )
    faith = _yes_no_value(faith_value)
    missing_faith = faith is None
    if faith is None:
        faith = "No"
    elif faith.lower() == "yes":
        faith = "Yes"
    elif faith.lower() == "no":
        faith = "No"

    try:
        image_number = int(_field_value(payload, "Image number", "image_number", "view", "view_number", default=1))
    except Exception as exc:
        raise VisualAgentParseError("Evaluator Image number must be an integer") from exc
    reasoning = str(
        _field_value(payload, "Reasoning", "reasoning", "reason", "rationale", "explanation", default="")
    ).strip()
    if missing_faith:
        raw_text = _compact_json(raw)
        if reasoning:
            reasoning = f"{reasoning} Missing Faithfulness field; treated as No. raw={raw_text}"
        else:
            reasoning = f"Evaluator response omitted Faithfulness; treated as No. raw={raw_text}"
    if not reasoning:
        raise VisualAgentParseError("Evaluator Reasoning is empty")
    confidence = _confidence(payload, default=0.5 if faith == "Yes" else 0.35)
    if missing_faith:
        confidence = min(confidence, 0.2)
    ambiguity = _bounded_float(_field_value(payload, "Ambiguity", "ambiguity", default=1.0 - confidence), default=1.0 - confidence)
    orientation = _orientation(payload)
    contact = _correctness_field(payload, "Contact", "contact")
    contact_angle = _correctness_field(payload, "Contact angle", "Contact Angle", "contact_angle", "contact angle")
    required_relation = _required_relation(payload)
    rotation_penalty = _rotation_penalty(payload)
    target_residual = _target_residual(payload)
    return EvaluatorResult(
        faith,
        max(1, image_number),
        reasoning,
        confidence,
        ambiguity,
        orientation,
        contact,
        contact_angle,
        required_relation,
        rotation_penalty,
        target_residual,
        raw,
    )


def parse_proposer(raw: dict[str, Any]) -> ProposerResult:
    payload = _find_mapping_with_field(
        raw,
        "Translation",
        "translation",
        "translation_vector",
        "translation vector",
        "pose_update",
        "pose update",
        "update",
    )
    outer_payload = payload
    nested_update = _field_value(payload, "pose_update", "pose update", "update", default=None)
    if isinstance(nested_update, dict) and _field_value(
        nested_update, "Translation", "translation", "translation_vector", "translation vector", default=None
    ) is not None:
        payload = nested_update
    target = str(
        _field_value(
            payload,
            "Target object",
            "target_object",
            "target",
            "object",
            default=_field_value(outer_payload, "Target object", "target_object", "target", "object", default=""),
        )
    ).strip()
    reasoning = str(
        _field_value(
            payload,
            "Reasoning",
            "reasoning",
            "reason",
            "rationale",
            "explanation",
            default=_field_value(outer_payload, "Reasoning", "reasoning", "reason", "rationale", "explanation", default=""),
        )
    ).strip()
    instruction_target_relation = str(
        _field_value(
            payload,
            "Instruction-derived target relation",
            "instruction_target_relation",
            "target_relation",
            "required_relation",
            default=_field_value(
                outer_payload,
                "Instruction-derived target relation",
                "instruction_target_relation",
                "target_relation",
                "required_relation",
                default="",
            ),
        )
    ).strip()
    current_error = str(
        _field_value(
            payload,
            "Current error",
            "current_error",
            "error",
            "failure",
            "diagnosis",
            default=_field_value(
                outer_payload,
                "Current error",
                "current_error",
                "error",
                "failure",
                "diagnosis",
                default="",
            ),
        )
    ).strip()
    translation_value = _field_value(payload, "Translation", "translation", "translation_vector", "translation vector", default=None)
    translation = _parse_translation_vector(translation_value, payload)
    if translation is None:
        raw_text = _compact_json(raw)
        detail = f"reasoning={reasoning}; raw={raw_text}" if reasoning else f"raw={raw_text}"
        raise VisualAgentParseError(f"Proposer Translation missing or non-numeric; {detail}")
    try:
        angle = float(_field_value(payload, "Angle", "angle", "rotation_angle", "rotation angle", default=0.0))
    except Exception as exc:
        raise VisualAgentParseError("Proposer Angle must be numeric") from exc
    rotation_matrix = _parse_rotation_matrix(
        _field_value(
            payload,
            "Rotation matrix",
            "rotation_matrix",
            "R_delta",
            "r_delta",
            "rotation delta matrix",
            default=None,
        )
    )
    axis_raw = _field_value(payload, "Dominant rotation axis", "dominant_rotation_axis", "rotation_axis", "axis", default="")
    axis = str(axis_raw).strip().lower()
    if axis in {"", "none", "null", "no", "n/a", "na"}:
        if rotation_matrix is None and abs(angle) > 1e-6:
            raise VisualAgentParseError(f"Proposer axis must be x/y/z when Angle is non-zero, got {axis!r}")
        axis = "z"
    if axis not in {"x", "y", "z"}:
        if rotation_matrix is None:
            raise VisualAgentParseError(f"Proposer axis must be x/y/z, got {axis!r}")
        axis = "z"
    if not target:
        target = "moving part"
    if not reasoning:
        reasoning = "No reasoning provided."
    if not instruction_target_relation:
        instruction_target_relation = "Not specified."
    if not current_error:
        current_error = "Not specified."
    confidence = _confidence(payload, default=0.4)
    return ProposerResult(
        target,
        instruction_target_relation,
        current_error,
        reasoning,
        translation,
        axis,
        angle,
        rotation_matrix,
        confidence,
        raw,
    )


def _confidence(raw: dict[str, Any], default: float) -> float:
    value = _field_value(raw, "Confidence", "confidence", "confidence_score", default=default)
    return _bounded_float(value, default=default)


def _bounded_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def _orientation(raw: dict[str, Any]) -> str:
    value = str(_field_value(raw, "Orientation", "orientation", default="Unclear")).strip().lower()
    if value in {"correct", "yes", "true", "aligned", "right", "ok"}:
        return "Correct"
    if value in {"incorrect", "wrong", "flipped", "reversed", "upside-down", "upside down", "no", "false"}:
        return "Incorrect"
    return "Unclear"


def _field_value(raw: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    lowered = {_normalize_key(str(key)): value for key, value in raw.items()}
    for name in names:
        key = _normalize_key(name)
        if key in lowered:
            return lowered[key]
    return default


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if ch.isalnum())


def _yes_no_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "correct", "valid", "success", "successful", "faithful", "pass", "passes"}:
        return "Yes"
    if text in {"no", "n", "false", "incorrect", "invalid", "failure", "failed", "unfaithful", "fail", "fails"}:
        return "No"
    return None


def _parse_translation_vector(value: Any, payload: dict[str, Any]) -> np.ndarray | None:
    candidates: list[Any] = [value]
    for key in ("translation", "translationvector", "vector", "delta", "movement"):
        nested = _field_value(payload, key, default=None)
        if nested is not value:
            candidates.append(nested)
    tx = _field_value(payload, "Tx", "tx", "x", "delta_x", "delta x", default=None)
    ty = _field_value(payload, "Ty", "ty", "y", "delta_y", "delta y", default=None)
    tz = _field_value(payload, "Tz", "tz", "z", "delta_z", "delta z", default=None)
    if tx is not None or ty is not None or tz is not None:
        candidates.append([tx or 0.0, ty or 0.0, tz or 0.0])

    for candidate in candidates:
        parsed = _translation_candidate_to_array(candidate)
        if parsed is not None:
            return parsed
    return None


def _translation_candidate_to_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        tx = _field_value(value, "Tx", "tx", "x", "delta_x", "delta x", default=None)
        ty = _field_value(value, "Ty", "ty", "y", "delta_y", "delta y", default=None)
        tz = _field_value(value, "Tz", "tz", "z", "delta_z", "delta z", default=None)
        if tx is not None or ty is not None or tz is not None:
            return _translation_candidate_to_array([tx or 0.0, ty or 0.0, tz or 0.0])
        vector = _field_value(value, "vector", "values", "translation", "translation_vector", default=None)
        if vector is not None:
            return _translation_candidate_to_array(vector)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            decoded = json.loads(text)
            parsed = _translation_candidate_to_array(decoded)
            if parsed is not None:
                return parsed
        except Exception:
            pass
        cleaned = text.strip("[]()")
        parts = [part for part in cleaned.replace(";", ",").split(",") if part.strip()]
        if len(parts) == 3:
            return _translation_candidate_to_array(parts)
        return None
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        try:
            return np.asarray([float(x) for x in value], dtype=np.float64)
        except Exception:
            return None
    return None


def _parse_rotation_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "n/a", "na"}:
            return None
        try:
            return _parse_rotation_matrix(json.loads(text))
        except Exception:
            return None
    if isinstance(value, dict):
        nested = _field_value(
            value,
            "matrix",
            "values",
            "Rotation matrix",
            "rotation_matrix",
            "R_delta",
            "r_delta",
            default=None,
        )
        if nested is not None and nested is not value:
            return _parse_rotation_matrix(nested)
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    return orthonormalize_rotation(matrix)


def _find_mapping_with_field(raw: dict[str, Any], *names: str) -> dict[str, Any]:
    if _field_value(raw, *names, default=None) is not None:
        return raw
    for value in raw.values():
        if isinstance(value, dict):
            found = _find_mapping_with_field(value, *names)
            if found is not value or _field_value(value, *names, default=None) is not None:
                return found
    return raw


def _compact_json(raw: dict[str, Any], limit: int = 500) -> str:
    try:
        text = json.dumps(raw, ensure_ascii=True, sort_keys=True)
    except Exception:
        text = str(raw)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _correctness_field(raw: dict[str, Any], *names: str) -> str:
    value = str(_field_value(raw, *names, default="Unclear")).strip().lower()
    if value in {"correct", "yes", "true", "aligned", "right", "ok", "valid"}:
        return "Correct"
    if value in {"incorrect", "wrong", "bad", "invalid", "no", "false", "misaligned"}:
        return "Incorrect"
    return "Unclear"


def _required_relation(raw: dict[str, Any]) -> str:
    value = str(
        _field_value(raw, "Required relation", "Required Relation", "required_relation", "required relation", default="Unknown")
    ).strip().lower()
    value = value.replace("_", "-")
    if value in {"perpendicular", "vertical", "orthogonal", "right-angle", "right angle"}:
        return "Perpendicular"
    if value in {"parallel", "coplanar", "same-plane", "same plane"}:
        return "Parallel"
    if value in {"horizontal", "level", "support", "support-bar", "support bar"}:
        return "Horizontal"
    if value in {"end-to-end", "end to end", "end", "butt", "butt-joint", "butt joint"}:
        return "End-to-end"
    return "Unknown"


def _rotation_penalty(raw: dict[str, Any]) -> str:
    value = str(
        _field_value(raw, "Rotation penalty", "Rotation Penalty", "rotation_penalty", "rotation penalty", default="None")
    ).strip().lower()
    if value in {"none", "no", "false", "0", "off"}:
        return "None"
    if value in {"mild", "low", "weak", "small", "minor"}:
        return "Mild"
    return "Strong"


def _target_residual(raw: dict[str, Any]) -> str:
    value = str(
        _field_value(
            raw,
            "Target residual",
            "Target Residual",
            "target_residual",
            "target residual",
            "Residual",
            "residual",
            default="Unclear",
        )
    ).strip().lower()
    value = value.replace("_", "-")
    if value in {"far", "large", "high", "big", "major", "severe"}:
        return "Far"
    if value in {"medium", "moderate", "mid", "partial"}:
        return "Medium"
    if value in {"close", "near", "small", "minor", "low"}:
        return "Close"
    return "Unclear"
