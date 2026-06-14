from __future__ import annotations

import json
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

        response: Any
        raw_text: str
        try:
            response = client.responses.create(
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}],
                    },
                ],
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
            response = client.responses.create(
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}],
                    },
                ],
            )
            raw_text = self._extract_response_text(response)

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
