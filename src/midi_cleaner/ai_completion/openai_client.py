from __future__ import annotations

import json
import re
from typing import Any

from midi_cleaner.ai_completion.models import ai_output_json_schema


class OpenAIPatternCompletionClientError(Exception):
    """Raised when OpenAI completion cannot be obtained or parsed."""


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
        max_output_tokens: int = 4000,
    ) -> tuple[str, dict[str, object]]:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - dependency import path
            raise OpenAIPatternCompletionClientError(
                "OpenAI package is not installed. Add dependency 'openai'."
            ) from exc

        client = OpenAI(api_key=api_key)
        schema = ai_output_json_schema(max_completion_notes=max_completion_notes)
        input_messages = self._build_input_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        response: Any
        raw_text: str
        try:
            response = client.responses.create(
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
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
            raw_text = self._extract_response_text(response)
        except TypeError:
            # Fallback for SDK variants that do not yet expose response text schema format.
            try:
                response = client.responses.create(
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    input=input_messages,
                )
                raw_text = self._extract_response_text(response)
            except Exception as exc:
                raise OpenAIPatternCompletionClientError(
                    self._format_openai_error_message(exc)
                ) from exc
        except Exception as exc:
            raise OpenAIPatternCompletionClientError(
                self._format_openai_error_message(exc)
            ) from exc

        if not raw_text.strip():
            raise OpenAIPatternCompletionClientError("OpenAI response did not include text output.")

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise OpenAIPatternCompletionClientError(
                "OpenAI response was not valid JSON."
            ) from exc

        if not isinstance(parsed, dict):
            raise OpenAIPatternCompletionClientError("OpenAI JSON response must be an object.")

        return raw_text, parsed

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

    def _extract_response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        # Fallback extraction for SDK object variants.
        output = getattr(response, "output", None)
        if isinstance(output, list):
            text_parts: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for piece in content:
                    text = getattr(piece, "text", None)
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)

        return ""
