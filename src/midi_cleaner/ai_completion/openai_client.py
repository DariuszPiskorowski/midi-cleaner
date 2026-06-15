from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from midi_cleaner.ai_completion.models import ai_output_json_schema


def calculate_max_output_tokens(max_completion_notes: int) -> int:
    return max(4000, min(12000, 1000 + (int(max_completion_notes) * 220)))


class OpenAIPatternCompletionClientError(Exception):
    """Raised when OpenAI completion cannot be obtained or parsed."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        raw_response_text: str | None = None,
        response_debug: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.raw_response_text = raw_response_text
        self.response_debug = dict(response_debug or {})


@dataclass(frozen=True)
class OpenAIPatternCompletionResult:
    raw_response_text: str
    parsed_payload: dict[str, object]
    response_debug: dict[str, object]
    max_output_tokens_used: int


class OpenAIPatternCompletionClient:
    def complete_pattern(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_completion_notes: int,
        max_output_tokens: int | None = None,
    ) -> OpenAIPatternCompletionResult:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - dependency import path
            raise OpenAIPatternCompletionClientError(
                "OpenAI package is not installed. Add dependency 'openai'."
            ) from exc

        client = OpenAI(api_key=api_key)
        max_output_tokens_used = self._resolve_max_output_tokens(
            max_completion_notes=max_completion_notes,
            max_output_tokens=max_output_tokens,
        )
        schema = ai_output_json_schema(max_completion_notes=max_completion_notes)
        input_messages = self._build_input_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        response: Any
        try:
            response = client.responses.create(
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens_used,
                input=input_messages,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "bass_ai_completion",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        except TypeError:
            # Fallback for SDK variants that do not yet expose response text schema format.
            try:
                response = client.responses.create(
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens_used,
                    input=input_messages,
                )
            except Exception as exc:
                raise OpenAIPatternCompletionClientError(
                    self._format_openai_error_message(exc)
                ) from exc
        except Exception as exc:
            raise OpenAIPatternCompletionClientError(
                self._format_openai_error_message(exc)
            ) from exc

        response_debug = self._build_response_debug(
            response,
            max_output_tokens=max_output_tokens_used,
        )
        raw_text, parsed = self._extract_response_payload(response)

        if parsed is None:
            message = (
                "OpenAI response was not valid JSON."
                if raw_text.strip()
                else "OpenAI response did not include text output."
            )
            code = "invalid_json" if raw_text.strip() else "empty_response"
            raise OpenAIPatternCompletionClientError(
                message,
                code=code,
                raw_response_text=raw_text,
                response_debug=response_debug,
            )

        if not isinstance(parsed, dict):
            raise OpenAIPatternCompletionClientError(
                "OpenAI JSON response must be an object.",
                code="invalid_json",
                raw_response_text=raw_text,
                response_debug=response_debug,
            )

        return OpenAIPatternCompletionResult(
            raw_response_text=raw_text,
            parsed_payload=parsed,
            response_debug=response_debug,
            max_output_tokens_used=max_output_tokens_used,
        )

    def _build_input_messages(self, *, system_prompt: str, user_prompt: str) -> list[dict[str, object]]:
        return [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ]

    def _format_openai_error_message(self, exc: Exception) -> str:
        status_code = getattr(exc, "status_code", None)
        prefix = "OpenAI Responses API request failed"
        if isinstance(status_code, int):
            prefix = f"{prefix} (HTTP {status_code})"

        details = self._sanitize_error_text(str(exc).strip())
        if not details:
            details = exc.__class__.__name__
        return f"{prefix}: {details}"

    def _sanitize_error_text(self, text: str) -> str:
        if not text:
            return ""

        redacted = re.sub(
            r"(?i)(incorrect api key provided:\s*)([^.]+)",
            r"\1[REDACTED]",
            text,
        )
        redacted = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[REDACTED_API_KEY]", redacted)
        redacted = re.sub(
            r"(?i)(api[-_ ]?key\s*[:=]\s*)['\"]?[A-Za-z0-9_-]{8,}['\"]?",
            r"\1[REDACTED]",
            redacted,
        )
        return redacted

    def _resolve_max_output_tokens(
        self,
        *,
        max_completion_notes: int,
        max_output_tokens: int | None,
    ) -> int:
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            return max_output_tokens
        return calculate_max_output_tokens(max_completion_notes=max_completion_notes)

    def _extract_response_payload(self, response: Any) -> tuple[str, object | None]:
        first_raw_text = ""

        for candidate in self._iter_response_candidates(response):
            if isinstance(candidate, dict):
                return json.dumps(candidate, ensure_ascii=False), candidate

            if isinstance(candidate, str):
                if not candidate.strip():
                    continue
                if not first_raw_text:
                    first_raw_text = candidate
                try:
                    parsed_candidate = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                return candidate, parsed_candidate

            if isinstance(candidate, (list, int, float, bool)):
                rendered = json.dumps(candidate, ensure_ascii=False)
                if not first_raw_text:
                    first_raw_text = rendered
                return rendered, candidate

        return first_raw_text, None

    def _iter_response_candidates(self, response: Any) -> Iterable[object]:
        output_parsed = self._read_attr(response, "output_parsed")
        if output_parsed is not None:
            yield output_parsed

        output = self._read_attr(response, "output")
        if isinstance(output, list):
            for item in output:
                content = self._read_attr(item, "content")
                if not isinstance(content, list):
                    continue
                for piece in content:
                    parsed = self._read_attr(piece, "parsed")
                    if parsed is not None:
                        yield parsed

                    value = self._read_attr(piece, "value")
                    if value is not None:
                        yield value

                    text = self._read_attr(piece, "text")
                    if isinstance(text, str) and text:
                        yield text

        output_text = self._read_attr(response, "output_text")
        if isinstance(output_text, str) and output_text.strip():
            yield output_text

    def _build_response_debug(self, response: Any, *, max_output_tokens: int) -> dict[str, object]:
        output_item_types: list[str] = []
        output_item_statuses: list[str] = []
        output_item_finish_reasons: list[str] = []
        output_item_incomplete_reasons: list[str] = []

        output = self._read_attr(response, "output")
        if isinstance(output, list):
            for item in output:
                item_type = self._coerce_string(self._read_attr(item, "type"))
                if item_type:
                    output_item_types.append(item_type)

                item_status = self._coerce_string(self._read_attr(item, "status"))
                if item_status:
                    output_item_statuses.append(item_status)

                finish_reason = self._coerce_string(self._read_attr(item, "finish_reason"))
                if finish_reason:
                    output_item_finish_reasons.append(finish_reason)

                item_incomplete_reason = self._extract_incomplete_reason(
                    self._read_attr(item, "incomplete_details")
                )
                if item_incomplete_reason:
                    output_item_incomplete_reasons.append(item_incomplete_reason)

        status = self._coerce_string(self._read_attr(response, "status"))
        response_incomplete_reason = self._extract_incomplete_reason(
            self._read_attr(response, "incomplete_details")
        )
        finish_reason = (
            next((value for value in output_item_finish_reasons if value), None)
            or response_incomplete_reason
        )

        debug: dict[str, object] = {
            "status": status,
            "finish_reason": finish_reason,
            "incomplete_reason": response_incomplete_reason,
            "output_item_types": output_item_types,
            "output_item_statuses": output_item_statuses,
            "output_item_finish_reasons": output_item_finish_reasons,
            "output_item_incomplete_reasons": output_item_incomplete_reasons,
            "max_output_tokens": int(max_output_tokens),
        }

        response_id = self._coerce_string(self._read_attr(response, "id"))
        if response_id:
            debug["response_id"] = response_id

        usage = self._extract_usage(self._read_attr(response, "usage"))
        if usage:
            debug["usage"] = usage

        return debug

    def _extract_usage(self, usage: Any) -> dict[str, int]:
        if usage is None:
            return {}

        usage_payload: dict[str, int] = {}
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            value = self._read_attr(usage, field)
            if isinstance(value, int):
                usage_payload[field] = value
        return usage_payload

    def _extract_incomplete_reason(self, details: Any) -> str | None:
        reason = self._read_attr(details, "reason")
        return self._coerce_string(reason)

    def _read_attr(self, value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    def _coerce_string(self, value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None
